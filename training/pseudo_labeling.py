"""
M6：不确定性引导伪标签（Uncertainty-Guided Pseudo-Labeling）
=============================================================
在部分生命周期监督场景下，利用模型对未标注区间的高置信度预测作为伪标签，
扩展有效监督信号。

训练流程：
    1. Warmup 阶段（前 warmup_epochs）：仅用真实标签训练（正常流程）
    2. 每 update_every_k 个 epoch 刷新伪标签：
        a. model.eval() + MC Dropout 对所有无标签样本推理
        b. 计算 (μ, σ) 预测分布
        c. 按不确定性 σ 排序，取最低的 threshold_percentile% 为伪标签
        d. 伪标签权重 = 1 / (σ² + ε)（不确定性越小，权重越大）
    3. 伪标签训练（附加轮次）：
        - 用伪标签样本的加权 MSE 作为辅助损失
        - 总损失 = L_supervised + λ_pseudo * L_pseudo

设计原则：
    - 伪标签每 K epoch 完全重置（防止累积误差漂移）
    - 最多接受 max_pseudo_ratio * N_unlabeled 个伪标签（防止噪声淹没真实标签）
    - 独立的辅助 DataLoader，与主训练 DataLoader 解耦

论文说辞：
    "本文提出不确定性引导的自监督边界扩展策略：在 Warmup 训练后，
    利用 MC Dropout 对未标注循环区间的预测置信度动态筛选高质量伪标签，
    仅将不确定性低于自适应阈值的预测纳入辅助监督，伪标签权重与预测
    置信度正相关。该策略有效利用了部分监督场景中大量未标注数据，
    在低标注比例下尤为显著。"
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset


class PseudoLabelManager:
    """
    不确定性引导伪标签管理器。

    维护一个伪标签 TensorDataset，每 update_every_k epoch 通过 MC Dropout
    重新生成。训练时通过 get_pseudo_loader() 获取辅助数据加载器。

    Args:
        warmup_epochs:          Warmup 轮数（期间不使用伪标签）
        update_every_k:         每 K 轮刷新一次伪标签
        n_mc_samples:           MC Dropout 采样次数
        threshold_percentile:   接受不确定性最低的 P% 无标签样本（0-100）
        max_pseudo_ratio:       最多接受无标签样本中的 R 比例（防止噪声过多）
        lambda_pseudo:          伪标签损失权重（相对于主损失）
        epsilon:                数值稳定项（权重分母）
        inference_batch_size:   MC 推理时的批大小
    """

    def __init__(
        self,
        warmup_epochs: int = 30,
        update_every_k: int = 5,
        n_mc_samples: int = 50,
        threshold_percentile: float = 30.0,
        max_pseudo_ratio: float = 0.5,
        lambda_pseudo: float = 1.0,
        epsilon: float = 1e-6,
        inference_batch_size: int = 512,
    ):
        self.warmup_epochs        = warmup_epochs
        self.update_every_k       = update_every_k
        self.n_mc_samples         = n_mc_samples
        self.threshold_percentile = threshold_percentile
        self.max_pseudo_ratio     = max_pseudo_ratio
        self.lambda_pseudo        = lambda_pseudo
        self.epsilon              = epsilon
        self.inference_batch_size = inference_batch_size

        # 伪标签存储（在 update() 后填充）
        self.pseudo_X: torch.Tensor | None = None   # (N_accept, T, F)
        self.pseudo_y: torch.Tensor | None = None   # (N_accept, 1) — pseudo targets (μ)
        self.pseudo_w: torch.Tensor | None = None   # (N_accept, 1) — weights (1/σ²)
        self.n_pseudo: int = 0
        self._last_tau: float = 0.0                 # 最后一次更新的不确定性阈值
        # 诊断信息（每次 update() 后填充）
        self.last_diag: dict = {}
        self._update_count: int = 0                 # 已更新次数（用于打印标记）

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def should_update(self, epoch: int) -> bool:
        """当前 epoch 是否应刷新伪标签。"""
        if epoch < self.warmup_epochs:
            return False
        return (epoch - self.warmup_epochs) % self.update_every_k == 0

    def update(
        self,
        model: nn.Module,
        train_dataset,          # ConcatDataset（HUSTBatteryDatasetWithMetadata 组合）
        unlabeled_indices: list,
        device: torch.device,
        collate_fn=None,
    ) -> None:
        """
        刷新伪标签：对所有无标签样本运行 MC Dropout 推理，
        筛选高置信度样本存入 TensorDataset。

        Args:
            model:              已训练的模型（含 MCDropout 层）
            train_dataset:      训练集 ConcatDataset
            unlabeled_indices:  无标签样本在 train_dataset 中的全局索引列表
            device:             计算设备
            collate_fn:         与 train_loader 相同的 collate_fn（可选）
        """
        from models.modules.mc_dropout import mc_predict

        n_ul = len(unlabeled_indices)
        if n_ul == 0:
            print("  [PseudoLabel] 无无标签样本，跳过更新")
            self.n_pseudo = 0
            return

        # 1. 在无标签子集上运行 MC Dropout 推理
        subset = Subset(train_dataset, unlabeled_indices)
        loader = DataLoader(
            subset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        model.eval()
        all_features, all_means, all_stds, all_true_soh = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch['window'].to(device)
                    # target_soh 对所有样本（含无标签）均为真实 SOH，直接用于诊断
                    if 'target_soh' in batch:
                        all_true_soh.append(batch['target_soh'].cpu().float())
                else:
                    x = batch[0].to(device)

                mean, std = mc_predict(model, x, n_samples=self.n_mc_samples)
                all_features.append(x.cpu())
                all_means.append(mean.cpu())
                all_stds.append(std.cpu())

        features_all = torch.cat(all_features, dim=0)   # (N_ul, T, F)
        means_all    = torch.cat(all_means,    dim=0)   # (N_ul, 1)
        stds_all     = torch.cat(all_stds,     dim=0)   # (N_ul, 1)
        stds_flat    = stds_all.squeeze(1)               # (N_ul,)
        true_soh_all = (torch.cat(all_true_soh, dim=0).squeeze()
                        if all_true_soh else None)       # (N_ul,) 或 None

        # 2. 确定接受数量（取 percentile 和 max_ratio 的最小值）
        n_by_percentile = max(1, int(n_ul * self.threshold_percentile / 100.0))
        n_by_ratio      = max(1, int(n_ul * self.max_pseudo_ratio))
        n_accept        = min(n_by_percentile, n_by_ratio)

        # 3. 按不确定性升序排序，取前 n_accept 个
        sorted_idx = torch.argsort(stds_flat)[:n_accept]
        self._last_tau = float(stds_flat[sorted_idx[-1]].item())

        # 4. 构建伪标签 TensorDataset
        self.pseudo_X = features_all[sorted_idx].clone()          # (N_accept, T, F)
        self.pseudo_y = means_all[sorted_idx].clone()             # (N_accept, 1)
        sigma_sq      = stds_flat[sorted_idx] ** 2 + self.epsilon
        self.pseudo_w = (1.0 / sigma_sq).unsqueeze(1).clone()     # (N_accept, 1)
        self.n_pseudo = n_accept
        self._update_count += 1

        # 5. 诊断：σ 分布 + 伪标签质量（命中率）
        self.last_diag = self._compute_diag(
            stds_flat, sorted_idx, means_all.squeeze(),
            true_soh_all, n_ul, n_accept,
        )
        self._print_diag(self.last_diag)

    def get_pseudo_loader(self, batch_size: int = 256) -> Optional[DataLoader]:
        """
        返回伪标签 DataLoader（shuffle=True），供附加训练轮次使用。
        若当前没有伪标签，返回 None。

        Batch 格式：(features, pseudo_targets, pseudo_weights)
        """
        if self.n_pseudo == 0 or self.pseudo_X is None:
            return None

        dataset = TensorDataset(self.pseudo_X, self.pseudo_y, self.pseudo_w)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def compute_pseudo_loss(
        self,
        model: nn.Module,
        pseudo_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        """
        执行一个完整的伪标签训练轮次。

        Args:
            model:         模型（当前处于 train() 模式）
            pseudo_loader: get_pseudo_loader() 返回的 DataLoader
            optimizer:     与主训练共享的优化器
            device:        计算设备

        Returns:
            本轮平均伪标签损失（float，仅用于日志打印）
        """
        total_loss = 0.0
        n_batches = 0

        for p_features, p_targets, p_weights in pseudo_loader:
            p_features = p_features.to(device)
            p_targets  = p_targets.to(device)
            p_weights  = p_weights.to(device)

            optimizer.zero_grad()
            p_pred = model(p_features)

            # 加权 MSE：权重 = 1/(σ² + ε)，不确定性越低权重越高
            sq_err = (p_pred - p_targets) ** 2
            loss   = (sq_err * p_weights).mean()
            (self.lambda_pseudo * loss).backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_unlabeled_indices(train_dataset) -> list:
        """
        从 ConcatDataset 中提取无标签样本的全局索引。

        遍历各子 Dataset 的 supervision_mask，无需实际调用 __getitem__。

        Args:
            train_dataset: ConcatDataset（由 HUSTBatteryDatasetWithMetadata 组合）

        Returns:
            无标签样本的全局索引列表
        """
        unlabeled = []
        offset = 0

        # ConcatDataset.datasets 是子数据集列表
        sub_datasets = getattr(train_dataset, 'datasets', [train_dataset])

        for sub_ds in sub_datasets:
            if hasattr(sub_ds, 'supervision_mask'):
                mask = sub_ds.supervision_mask  # (N_sub,) bool tensor
                local_ul = (~mask).nonzero(as_tuple=True)[0].tolist()
                unlabeled.extend(offset + i for i in local_ul)
            offset += len(sub_ds)

        return unlabeled

    # ------------------------------------------------------------------ #
    #  诊断辅助方法
    # ------------------------------------------------------------------ #

    def _compute_diag(
        self,
        stds_flat:    torch.Tensor,   # (N_ul,)  所有无标签样本的 σ
        sorted_idx:   torch.Tensor,   # (n_accept,) 接受样本在 stds_flat 中的索引
        means_flat:   torch.Tensor,   # (N_ul,)  所有无标签样本的预测均值 μ
        true_soh_all: torch.Tensor | None,  # (N_ul,) 真实 SOH（可能为 None）
        n_ul:         int,
        n_accept:     int,
    ) -> dict:
        """计算伪标签诊断指标，返回 dict。"""
        n_reject = n_ul - n_accept

        # σ 分布（全部无标签样本）
        sigma_all = stds_flat.cpu().float()
        diag = {
            'update_count':      self._update_count,
            'n_unlabeled':       n_ul,
            'n_accepted':        n_accept,
            'n_rejected':        n_reject,
            'accept_rate':       n_accept / max(n_ul, 1),
            'sigma_threshold':   self._last_tau,
            'sigma_min':         float(sigma_all.min()),
            'sigma_max':         float(sigma_all.max()),
            'sigma_mean':        float(sigma_all.mean()),
            'sigma_p50':         float(sigma_all.median()),
            'sigma_p20':         float(torch.quantile(sigma_all, 0.20)),
            'w_mean':            float(self.pseudo_w.mean()),
            'w_max':             float(self.pseudo_w.max()),
        }

        if true_soh_all is not None:
            true_all = true_soh_all.cpu().float()
            mu_all   = means_flat.cpu().float()

            # 接受组 mask
            accept_mask = torch.zeros(n_ul, dtype=torch.bool)
            accept_mask[sorted_idx] = True

            # 接受组质量
            err_acc  = (mu_all[accept_mask] - true_all[accept_mask])
            mae_acc  = float(err_acc.abs().mean())
            bias_acc = float(err_acc.mean())
            std_acc  = float(err_acc.std())

            # 全无标签 MAE（不分接受/拒绝）
            err_all = (mu_all - true_all)
            mae_all = float(err_all.abs().mean())

            # 拒绝组 MAE（应高于接受组，否则筛选无效）
            mae_rej = float((mu_all[~accept_mask] - true_all[~accept_mask]).abs().mean()) \
                      if n_reject > 0 else float('nan')

            # σ 与误差的 Spearman 相关（越高说明 σ 越能反映真实误差）
            try:
                import scipy.stats as stats
                rho, pval = stats.spearmanr(
                    sigma_all.numpy(), err_all.abs().numpy()
                )
                spearman_rho = float(rho)
                spearman_p   = float(pval)
            except Exception:
                spearman_rho = float('nan')
                spearman_p   = float('nan')

            diag.update({
                'has_quality_metrics': True,
                'pseudo_mae_accepted': mae_acc,
                'pseudo_bias_accepted': bias_acc,
                'pseudo_std_accepted':  std_acc,
                'pseudo_mae_all_unlabeled': mae_all,
                'pseudo_mae_rejected':  mae_rej,
                'sigma_error_spearman_rho': spearman_rho,
                'sigma_error_spearman_p':   spearman_p,
            })
        else:
            diag['has_quality_metrics'] = False

        return diag

    def _print_diag(self, d: dict) -> None:
        """打印格式化的伪标签诊断表格。"""
        sep = "  " + "─" * 66
        print(f"\n  ── [M6 伪标签] 第 {d['update_count']} 次更新诊断 " + "─" * 36)

        # 接受情况
        print(f"  接受样本:    {d['n_accepted']:>5} / {d['n_unlabeled']:>5}"
              f"  ({d['accept_rate']*100:.1f}%)")

        # σ 分布
        print(f"  σ 分布:      min={d['sigma_min']:.5f}  "
              f"p20={d['sigma_p20']:.5f}  "
              f"p50={d['sigma_p50']:.5f}  "
              f"mean={d['sigma_mean']:.5f}  "
              f"max={d['sigma_max']:.5f}")
        print(f"  σ 阈值:      {d['sigma_threshold']:.5f}  "
              f"（仅接受 σ < 阈值的样本）")
        print(f"  伪标签权重:  mean={d['w_mean']:.2f}  max={d['w_max']:.2f}")

        if d.get('has_quality_metrics'):
            print(sep)
            print(f"  伪标签质量（vs 真实 SOH）:")
            print(f"    接受组 MAE:   {d['pseudo_mae_accepted']*100:.4f}%  "
                  f"偏差: {d['pseudo_bias_accepted']*100:+.4f}%  "
                  f"标准差: {d['pseudo_std_accepted']*100:.4f}%")
            print(f"    全无标签 MAE: {d['pseudo_mae_all_unlabeled']*100:.4f}%  "
                  f"拒绝组 MAE: {d['pseudo_mae_rejected']*100:.4f}%")
            rho = d['sigma_error_spearman_rho']
            p   = d['sigma_error_spearman_p']
            print(f"    σ-误差 Spearman ρ: {rho:.4f}  (p={p:.4f})"
                  f"  ← {'σ 能有效表征误差 ✅' if rho > 0.3 else 'σ 与误差相关性弱 ⚠️'}")

            # 自动预警
            sup_mae_est = d['pseudo_mae_all_unlabeled']  # 用全无标签 MAE 作参考
            noise_ratio = d['pseudo_mae_accepted'] / max(sup_mae_est, 1e-8)
            if d['pseudo_mae_accepted'] > 0.03:
                print(f"  ⚠️  【伪标签噪声过高】接受组 MAE={d['pseudo_mae_accepted']*100:.4f}%，"
                      f"超过 3%，伪标签信号质量差")
            if d['pseudo_mae_rejected'] < d['pseudo_mae_accepted']:
                print(f"  ⚠️  【筛选无效】拒绝组 MAE 低于接受组，"
                      f"σ 未能有效区分高/低质量预测")
            if abs(d['pseudo_bias_accepted']) > 0.01:
                print(f"  ⚠️  【系统性偏差】伪标签偏差={d['pseudo_bias_accepted']*100:+.4f}%，"
                      f"模型对无标签区间存在系统偏估")

        print()

    def __repr__(self):
        return (
            f"PseudoLabelManager("
            f"warmup={self.warmup_epochs}, "
            f"K={self.update_every_k}, "
            f"mc={self.n_mc_samples}, "
            f"pct={self.threshold_percentile}%, "
            f"λ={self.lambda_pseudo}, "
            f"n_pseudo={self.n_pseudo})"
        )
