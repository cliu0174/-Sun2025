"""
Transformer SOH 序列重建验证实验
================================

目标：验证 Transformer 能否从部分已知的 SOH 标签重建完整退化曲线。
完全独立于 PI-MS-CNN-LSTM，不修改任何现有代码。

实验设计：
  1. 加载 77 块电池的完整 SOH 序列
  2. 随机 mask 掉一部分标签（模拟 r=0.3/0.5/0.7）
  3. 训练 Transformer 从已知标签 + 位置编码重建被 mask 的标签
  4. 评估重建精度（MAE / RMSE / R2），并与线性插值对比

用法：
  python experiments/verify_transformer_reconstruction.py          # 完整实验
  python experiments/verify_transformer_reconstruction.py --quick  # 快速验证（少量电池）
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ============================================================
# 1. 数据准备：获取每块电池的完整 SOH 序列
# ============================================================

def load_battery_soh_sequences(data_dir=None):
    """
    加载所有电池的完整 SOH 序列（归一化后的 capacity）。

    Returns:
        sequences: dict {battery_id: np.array of shape (T,)}
    """
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'data', 'HUST data')

    from data_loaders.data_loader_hust import load_all_hust_batteries

    all_data = load_all_hust_batteries(
        data_dir=data_dir,
        train_ratio=1.0,       # 加载全部数据（不拆 train/test）
        normalize_target=True,  # 归一化为 SOH
        apply_cleaning=False,
    )

    sequences = {}
    for bid, data in all_data.items():
        soh = data['train_capacity']  # train_ratio=1.0 时全在 train 里
        if len(soh) >= 20:  # 过滤太短的序列
            sequences[bid] = soh.astype(np.float32)

    print(f'Loaded {len(sequences)} batteries')
    lengths = [len(v) for v in sequences.values()]
    print(f'Sequence lengths: min={min(lengths)}, max={max(lengths)}, '
          f'mean={np.mean(lengths):.0f}')

    return sequences


def mask_soh_sequence(soh, ratio, rng):
    """
    对单条 SOH 序列做 label masking。

    Args:
        soh:   (T,) 完整 SOH 序列
        ratio: float, 保留标签的比例（如 0.3 = 保留 30%）
        rng:   np.random.RandomState

    Returns:
        masked_soh:  (T,) 被 mask 的位置填 NaN
        mask:        (T,) bool, True=已知, False=被 mask
    """
    T = len(soh)
    n_keep = max(1, int(round(T * ratio)))
    keep_idx = rng.choice(T, size=n_keep, replace=False)

    mask = np.zeros(T, dtype=bool)
    mask[keep_idx] = True

    masked_soh = np.full(T, np.nan, dtype=np.float32)
    masked_soh[mask] = soh[mask]

    return masked_soh, mask


# ============================================================
# 2. Transformer 模型
# ============================================================

class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SOHReconstructionTransformer(nn.Module):
    """
    Masked SOH Reconstruction Transformer

    输入：已知 SOH 值 + 位置编码 + 已知/未知标记
    输出：重建的完整 SOH 序列

    类似 BERT 的 masked token prediction，但针对连续值回归。
    """

    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
                 dropout=0.1, max_len=2000):
        super().__init__()

        # 输入嵌入：SOH 值(1) + 已知标记(1) → d_model
        self.input_proj = nn.Linear(2, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出头：回归单个 SOH 值
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.d_model = d_model

    def forward(self, soh_values, known_mask):
        """
        Args:
            soh_values: (B, T) 已知位置填真值，未知位置填 0
            known_mask: (B, T) bool, True=已知

        Returns:
            reconstructed: (B, T) 重建的完整 SOH 序列
        """
        B, T = soh_values.shape

        # 构建输入：[soh_value, is_known_flag]
        soh_input = soh_values.unsqueeze(-1)                     # (B, T, 1)
        known_flag = known_mask.float().unsqueeze(-1)             # (B, T, 1)
        x = torch.cat([soh_input, known_flag], dim=-1)            # (B, T, 2)

        # 投影 + 位置编码
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_encoder(x)         # (B, T, d_model)

        # Transformer 编码
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # 输出
        out = self.output_head(x).squeeze(-1)  # (B, T)

        return out


# ============================================================
# 3. 基线方法：线性插值
# ============================================================

def linear_interpolation(masked_soh, known_mask):
    """
    线性插值重建：用已知标签的线性插值填充缺失位置。

    Args:
        masked_soh: (T,) 已知位置有值，其他为 NaN
        known_mask: (T,) bool

    Returns:
        interpolated: (T,) 插值后的完整序列
    """
    T = len(masked_soh)
    known_indices = np.where(known_mask)[0]
    known_values = masked_soh[known_mask]

    if len(known_indices) < 2:
        # 不够插值，用常数填充
        return np.full(T, known_values[0] if len(known_values) > 0 else 0.9)

    interpolated = np.interp(np.arange(T), known_indices, known_values)
    return interpolated


# ============================================================
# 4. 训练与评估
# ============================================================

def prepare_batch(soh_sequences, ratio, rng, max_len=None):
    """
    将多条 SOH 序列打包成一个 batch（padding 到等长）。

    Returns:
        soh_values:  (B, T_max) 已知位置填真值，未知填 0
        known_mask:  (B, T_max) bool
        full_soh:    (B, T_max) 完整真值（用于计算 loss）
        pad_mask:    (B, T_max) bool, True=有效位置
    """
    if max_len is None:
        max_len = max(len(s) for s in soh_sequences)

    B = len(soh_sequences)
    soh_values = np.zeros((B, max_len), dtype=np.float32)
    known_mask = np.zeros((B, max_len), dtype=bool)
    full_soh = np.zeros((B, max_len), dtype=np.float32)
    pad_mask = np.zeros((B, max_len), dtype=bool)

    for i, soh in enumerate(soh_sequences):
        T = len(soh)
        masked, mask = mask_soh_sequence(soh, ratio, rng)

        full_soh[i, :T] = soh
        soh_values[i, :T] = np.where(mask, soh, 0.0)
        known_mask[i, :T] = mask
        pad_mask[i, :T] = True

    return soh_values, known_mask, full_soh, pad_mask


def train_transformer(model, train_seqs, val_seqs, ratio, device,
                      epochs=200, lr=1e-3, seed=42):
    """训练 Transformer 重建模型"""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    best_state = None
    patience = 30
    no_improve = 0

    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        model.train()

        # 每个 epoch 重新 mask（数据增强效果：看到不同的 mask 模式）
        soh_values, known_mask, full_soh, pad_mask = prepare_batch(
            train_seqs, ratio, rng
        )

        soh_values = torch.FloatTensor(soh_values).to(device)
        known_mask = torch.BoolTensor(known_mask).to(device)
        full_soh = torch.FloatTensor(full_soh).to(device)
        pad_mask = torch.BoolTensor(pad_mask).to(device)

        optimizer.zero_grad()
        pred = model(soh_values, known_mask)

        # Loss：只在被 mask 掉且非 padding 的位置计算
        # 这样模型学的是"补全"，而不是"复制"
        loss_mask = pad_mask & ~known_mask
        if loss_mask.sum() > 0:
            loss = ((pred - full_soh) ** 2 * loss_mask.float()).sum() / loss_mask.sum()
        else:
            loss = ((pred - full_soh) ** 2 * pad_mask.float()).sum() / pad_mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                v_soh, v_mask, v_full, v_pad = prepare_batch(
                    val_seqs, ratio, rng
                )
                v_soh = torch.FloatTensor(v_soh).to(device)
                v_mask = torch.BoolTensor(v_mask).to(device)
                v_full = torch.FloatTensor(v_full).to(device)
                v_pad = torch.BoolTensor(v_pad).to(device)

                v_pred = model(v_soh, v_mask)
                v_loss_mask = v_pad & ~v_mask
                if v_loss_mask.sum() > 0:
                    val_loss = ((v_pred - v_full) ** 2 * v_loss_mask.float()).sum() / v_loss_mask.sum()
                else:
                    val_loss = ((v_pred - v_full) ** 2 * v_pad.float()).sum() / v_pad.sum()

                val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5

            if (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1:3d}/{epochs} | '
                      f'Train MSE: {loss.item():.6f} | '
                      f'Val MSE: {val_loss:.6f} | '
                      f'Best: {best_val_loss:.6f}')

            if no_improve >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_reconstruction(model, test_seqs, ratio, device, seed=42, n_trials=5):
    """
    评估重建效果：Transformer vs 线性插值。

    对每条测试序列做 n_trials 次随机 mask，取平均。

    Returns:
        results: dict with MAE/RMSE/R2 for both methods
    """
    model.eval()
    rng = np.random.RandomState(seed)

    transformer_errors = []
    interp_errors = []
    all_details = []

    for trial in range(n_trials):
        for bid, soh in test_seqs.items():
            masked_soh, mask = mask_soh_sequence(soh, ratio, rng)
            T = len(soh)

            # --- Transformer 预测 ---
            soh_input = np.where(mask, soh, 0.0).astype(np.float32)
            with torch.no_grad():
                inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
                m = torch.BoolTensor(mask.astype(bool)).unsqueeze(0).to(device)
                pred = model(inp, m).squeeze(0).cpu().numpy()

            # 只在 mask 掉的位置评估
            eval_mask = ~mask
            if eval_mask.sum() == 0:
                continue

            trans_mae = np.mean(np.abs(pred[eval_mask] - soh[eval_mask]))
            transformer_errors.append(trans_mae)

            # --- 线性插值 ---
            interp_result = linear_interpolation(masked_soh, mask)
            interp_mae = np.mean(np.abs(interp_result[eval_mask] - soh[eval_mask]))
            interp_errors.append(interp_mae)

            all_details.append({
                'battery': bid,
                'trial': trial,
                'n_known': int(mask.sum()),
                'n_total': T,
                'transformer_mae': float(trans_mae),
                'interp_mae': float(interp_mae),
            })

    # 汇总
    trans_mae_mean = np.mean(transformer_errors)
    interp_mae_mean = np.mean(interp_errors)

    # 逐电池平均 Transformer 优势
    battery_advantages = defaultdict(list)
    for d in all_details:
        adv = (d['interp_mae'] - d['transformer_mae']) / d['interp_mae'] * 100
        battery_advantages[d['battery']].append(adv)

    avg_advantage = np.mean([np.mean(v) for v in battery_advantages.values()])

    results = {
        'transformer_mae': float(trans_mae_mean),
        'interp_mae': float(interp_mae_mean),
        'advantage_pct': float(avg_advantage),
        'n_batteries': len(test_seqs),
        'n_trials': n_trials,
        'n_evaluations': len(all_details),
    }

    return results, all_details


# ============================================================
# 5. 置信度估计
# ============================================================

def estimate_confidence(model, soh, mask, device, n_masks=20, seed=42):
    """
    通过多次不同 mask 模式估计重建置信度。

    思路：对同一条序列用不同的 mask 做多次重建，
    预测值方差小 = 高置信度，方差大 = 低置信度。

    Args:
        model:   训练好的 Transformer
        soh:     (T,) 完整 SOH（实际使用时不可见，这里用于验证）
        mask:    (T,) bool, 原始已知位置
        n_masks: 额外 mask 次数

    Returns:
        mean_pred:  (T,) 多次预测的均值
        std_pred:   (T,) 多次预测的标准差（置信度指标）
    """
    model.eval()
    rng = np.random.RandomState(seed)
    T = len(soh)
    preds = []

    # 用原始 mask 做一次
    soh_input = np.where(mask, soh, 0.0).astype(np.float32)
    with torch.no_grad():
        inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
        m = torch.BoolTensor(mask).unsqueeze(0).to(device)
        pred = model(inp, m).squeeze(0).cpu().numpy()
    preds.append(pred)

    # 在已知标签中随机遮掉一部分，做多次预测
    known_indices = np.where(mask)[0]
    for _ in range(n_masks - 1):
        # 随机遮掉已知标签的 20%
        n_drop = max(1, int(len(known_indices) * 0.2))
        drop_idx = rng.choice(known_indices, size=n_drop, replace=False)
        aug_mask = mask.copy()
        aug_mask[drop_idx] = False

        soh_input = np.where(aug_mask, soh, 0.0).astype(np.float32)
        with torch.no_grad():
            inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
            m = torch.BoolTensor(aug_mask).unsqueeze(0).to(device)
            pred = model(inp, m).squeeze(0).cpu().numpy()
        preds.append(pred)

    preds = np.array(preds)  # (n_masks, T)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    return mean_pred, std_pred


# ============================================================
# 6. 主实验
# ============================================================

def run_experiment(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # 加载数据
    print('\n[1/4] Loading battery SOH sequences...')
    sequences = load_battery_soh_sequences()
    battery_ids = sorted(sequences.keys())

    if args.quick:
        battery_ids = battery_ids[:15]
        sequences = {k: sequences[k] for k in battery_ids}
        print(f'[Quick mode] Using {len(battery_ids)} batteries')

    # 划分：60% 训练 / 20% 验证 / 20% 测试
    rng = np.random.RandomState(42)
    rng.shuffle(battery_ids)
    n = len(battery_ids)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_ids = battery_ids[:n_train]
    val_ids = battery_ids[n_train:n_train + n_val]
    test_ids = battery_ids[n_train + n_val:]

    train_seqs = [sequences[bid] for bid in train_ids]
    val_seqs = [sequences[bid] for bid in val_ids]
    test_seqs = {bid: sequences[bid] for bid in test_ids}

    print(f'Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}')

    # 多个 ratio 测试
    ratios = [0.3, 0.5, 0.7]
    all_results = {}

    for ratio in ratios:
        print(f'\n{"="*60}')
        print(f'[2/4] Training Transformer for r={ratio}')
        print(f'{"="*60}')

        model = SOHReconstructionTransformer(
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model params: {n_params:,}')

        model = train_transformer(
            model, train_seqs, val_seqs, ratio, device,
            epochs=300 if not args.quick else 50,
            lr=1e-3,
            seed=42,
        )

        # 评估
        print(f'\n[3/4] Evaluating at r={ratio}...')
        results, details = evaluate_reconstruction(
            model, test_seqs, ratio, device, seed=42, n_trials=5
        )

        print(f'\n  === Results at r={ratio} ===')
        print(f'  Transformer MAE:  {results["transformer_mae"]*100:.4f}%')
        print(f'  Interpolation MAE: {results["interp_mae"]*100:.4f}%')
        print(f'  Advantage: {results["advantage_pct"]:+.2f}% '
              f'({"Transformer wins" if results["advantage_pct"] > 0 else "Interpolation wins"})')

        all_results[f'r={ratio}'] = results

        # 置信度分析（选一块测试电池）
        if len(test_ids) > 0:
            sample_bid = test_ids[0]
            sample_soh = sequences[sample_bid]
            _, sample_mask = mask_soh_sequence(sample_soh, ratio,
                                               np.random.RandomState(42))
            mean_pred, std_pred = estimate_confidence(
                model, sample_soh, sample_mask, device
            )

            # 置信度与实际误差的相关性
            actual_error = np.abs(mean_pred - sample_soh)
            eval_pos = ~sample_mask
            if eval_pos.sum() > 2:
                from scipy.stats import spearmanr
                corr, pval = spearmanr(std_pred[eval_pos], actual_error[eval_pos])
                print(f'\n  Confidence calibration (battery {sample_bid}):')
                print(f'    Spearman(std, error) = {corr:.3f} (p={pval:.4f})')
                print(f'    Mean std at masked positions: {std_pred[eval_pos].mean():.6f}')

                all_results[f'r={ratio}']['confidence_spearman'] = float(corr)
                all_results[f'r={ratio}']['confidence_pval'] = float(pval)

    # 保存结果
    print(f'\n[4/4] Saving results...')
    output_dir = os.path.join(ROOT, 'results', 'transformer_reconstruction')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'reconstruction_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'  Saved: {output_file}')

    # 汇总表
    print(f'\n{"="*60}')
    print(f'Summary: Transformer vs Linear Interpolation')
    print(f'{"="*60}')
    print(f'{"Ratio":<8} {"Trans MAE%":<14} {"Interp MAE%":<14} {"Advantage":<12}')
    print(f'{"-"*48}')
    for ratio in ratios:
        r = all_results[f'r={ratio}']
        print(f'{ratio:<8.1f} {r["transformer_mae"]*100:<14.4f} '
              f'{r["interp_mae"]*100:<14.4f} {r["advantage_pct"]:+.2f}%')

    print(f'\nConclusion:')
    avg_adv = np.mean([all_results[f'r={r}']['advantage_pct'] for r in ratios])
    if avg_adv > 5:
        print(f'  Transformer significantly outperforms interpolation ({avg_adv:+.1f}%)')
        print(f'  -> Worth integrating as label reconstruction module')
    elif avg_adv > 0:
        print(f'  Transformer slightly outperforms interpolation ({avg_adv:+.1f}%)')
        print(f'  -> Marginal benefit, consider if complexity is justified')
    else:
        print(f'  Interpolation outperforms Transformer ({avg_adv:+.1f}%)')
        print(f'  -> Transformer adds no value for this smooth degradation curve')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer SOH Reconstruction')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer batteries, fewer epochs')
    args = parser.parse_args()

    run_experiment(args)
