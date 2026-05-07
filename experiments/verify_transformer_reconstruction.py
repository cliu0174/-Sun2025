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
  5. 可视化重建效果

用法：
  python experiments/verify_transformer_reconstruction.py          # 完整实验
  python experiments/verify_transformer_reconstruction.py --quick  # 快速验证
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = os.path.join(ROOT, 'results', 'transformer_reconstruction')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 数据准备
# ============================================================

def load_battery_soh_sequences(data_dir=None):
    """加载所有电池的完整 SOH 序列"""
    if data_dir is None:
        data_dir = os.path.join(ROOT, 'data', 'HUST data')

    from data_loaders.data_loader_hust import load_all_hust_batteries

    all_data = load_all_hust_batteries(
        data_dir=data_dir,
        train_ratio=1.0,
        normalize_target=True,
        apply_cleaning=False,
    )

    sequences = {}
    for bid, data in all_data.items():
        soh = data['train_capacity']
        if len(soh) >= 20:
            sequences[bid] = soh.astype(np.float32)

    print(f'Loaded {len(sequences)} batteries')
    lengths = [len(v) for v in sequences.values()]
    print(f'Sequence lengths: min={min(lengths)}, max={max(lengths)}, '
          f'mean={np.mean(lengths):.0f}')

    return sequences


def mask_soh_sequence(soh, ratio, rng):
    """对单条 SOH 序列做 label masking"""
    T = len(soh)
    n_keep = max(2, int(round(T * ratio)))  # 至少保留 2 个点（插值需要）
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

    def __init__(self, d_model, max_len=5000):
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
    """

    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
                 dropout=0.1, max_len=5000):
        super().__init__()

        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

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

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.d_model = d_model

    def forward(self, soh_values, known_mask, pad_mask=None):
        """
        Args:
            soh_values: (B, T) 已知位置填真值，未知位置填 0
            known_mask: (B, T) bool, True=已知
            pad_mask:   (B, T) bool, True=有效位置, False=padding
                        传给 Transformer 的 src_key_padding_mask 需要反转

        Returns:
            reconstructed: (B, T) 重建的完整 SOH 序列
        """
        B, T = soh_values.shape

        # 构建输入：[soh_value, is_known_flag]
        soh_input = soh_values.unsqueeze(-1)            # (B, T, 1)
        known_flag = known_mask.float().unsqueeze(-1)    # (B, T, 1)
        x = torch.cat([soh_input, known_flag], dim=-1)   # (B, T, 2)

        # 投影 + 位置编码
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Transformer 编码（传入 padding mask）
        # src_key_padding_mask: True 表示忽略该位置，所以需要反转 pad_mask
        if pad_mask is not None:
            key_padding_mask = ~pad_mask  # True = padding 位置需要被忽略
        else:
            key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        out = self.output_head(x).squeeze(-1)  # (B, T)
        return out


# ============================================================
# 3. 基线方法：线性插值
# ============================================================

def linear_interpolation(masked_soh, known_mask):
    """线性插值重建"""
    T = len(masked_soh)
    known_indices = np.where(known_mask)[0]
    known_values = masked_soh[known_mask]

    if len(known_indices) < 2:
        return np.full(T, known_values[0] if len(known_values) > 0 else 0.9)

    interpolated = np.interp(np.arange(T), known_indices, known_values)
    return interpolated


# ============================================================
# 4. 训练（逐电池迭代，避免长序列 padding 问题）
# ============================================================

def train_transformer(model, train_seqs, val_seqs, ratio, device,
                      epochs=200, lr=1e-3, seed=42, batch_size=8):
    """
    训练 Transformer 重建模型。
    改用 mini-batch：按长度分组，减少 padding 浪费。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    best_state = None
    patience = 40
    no_improve = 0

    rng = np.random.RandomState(seed)

    # 按长度排序，相近长度的放一个 batch，减少 padding
    sorted_seqs = sorted(train_seqs, key=len)
    n_batches = max(1, len(sorted_seqs) // batch_size)

    for epoch in range(epochs):
        model.train()

        # 每 epoch 打乱 batch 顺序（但同一 batch 内的序列长度相近）
        batch_indices = list(range(n_batches))
        rng.shuffle(batch_indices)

        epoch_loss = 0.0
        epoch_count = 0

        for bi in batch_indices:
            start = bi * batch_size
            end = min(start + batch_size, len(sorted_seqs))
            batch_seqs = sorted_seqs[start:end]

            # 每次重新 mask（数据增强）
            max_len = max(len(s) for s in batch_seqs)
            B = len(batch_seqs)

            soh_values = np.zeros((B, max_len), dtype=np.float32)
            known_mask = np.zeros((B, max_len), dtype=bool)
            full_soh = np.zeros((B, max_len), dtype=np.float32)
            pad_mask = np.zeros((B, max_len), dtype=bool)

            for i, soh in enumerate(batch_seqs):
                T = len(soh)
                _, mask = mask_soh_sequence(soh, ratio, rng)
                full_soh[i, :T] = soh
                soh_values[i, :T] = np.where(mask, soh, 0.0)
                known_mask[i, :T] = mask
                pad_mask[i, :T] = True

            soh_t = torch.FloatTensor(soh_values).to(device)
            known_t = torch.BoolTensor(known_mask).to(device)
            full_t = torch.FloatTensor(full_soh).to(device)
            pad_t = torch.BoolTensor(pad_mask).to(device)

            optimizer.zero_grad()
            pred = model(soh_t, known_t, pad_t)

            # Loss：被 mask 掉且非 padding 的位置
            loss_mask = pad_t & ~known_t
            if loss_mask.sum() > 0:
                loss = ((pred - full_t) ** 2 * loss_mask.float()).sum() / loss_mask.sum()
            else:
                loss = ((pred - full_t) ** 2 * pad_t.float()).sum() / pad_t.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * loss_mask.sum().item()
            epoch_count += loss_mask.sum().item()

        scheduler.step()
        avg_train_loss = epoch_loss / max(epoch_count, 1)

        # 验证（每 5 个 epoch）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_loss = _evaluate_val(model, val_seqs, ratio, device, rng)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5

            if (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1:3d}/{epochs} | '
                      f'Train MSE: {avg_train_loss:.6f} | '
                      f'Val MSE: {val_loss:.6f} | '
                      f'Best: {best_val_loss:.6f}')

            if no_improve >= patience:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _evaluate_val(model, val_seqs, ratio, device, rng):
    """验证集评估（逐条序列，不 padding）"""
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for soh in val_seqs:
            T = len(soh)
            _, mask = mask_soh_sequence(soh, ratio, rng)

            soh_input = np.where(mask, soh, 0.0).astype(np.float32)
            inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(device)

            pred = model(inp, m).squeeze(0)  # (T,)
            target = torch.FloatTensor(soh).to(device)

            eval_mask = ~torch.BoolTensor(mask).to(device)
            if eval_mask.sum() > 0:
                mse = ((pred - target) ** 2 * eval_mask.float()).sum()
                total_loss += mse.item()
                total_count += eval_mask.sum().item()

    return total_loss / max(total_count, 1)


# ============================================================
# 5. 评估
# ============================================================

def evaluate_reconstruction(model, test_seqs, ratio, device, seed=42, n_trials=5):
    """评估重建效果：Transformer vs 线性插值"""
    model.eval()
    rng = np.random.RandomState(seed)

    transformer_errors = []
    interp_errors = []
    all_details = []

    for trial in range(n_trials):
        for bid, soh in test_seqs.items():
            masked_soh, mask = mask_soh_sequence(soh, ratio, rng)
            T = len(soh)

            # --- Transformer 预测（单条，无 padding）---
            soh_input = np.where(mask, soh, 0.0).astype(np.float32)
            with torch.no_grad():
                inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
                m = torch.BoolTensor(mask.astype(bool)).unsqueeze(0).to(device)
                pred = model(inp, m).squeeze(0).cpu().numpy()

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

    trans_mae_mean = np.mean(transformer_errors)
    interp_mae_mean = np.mean(interp_errors)

    battery_advantages = defaultdict(list)
    for d in all_details:
        if d['interp_mae'] > 1e-10:
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
# 6. 置信度估计
# ============================================================

def estimate_confidence(model, soh, mask, device, n_masks=20, seed=42):
    """多次不同 mask 模式估计重建置信度"""
    model.eval()
    rng = np.random.RandomState(seed)
    T = len(soh)
    preds = []

    soh_input = np.where(mask, soh, 0.0).astype(np.float32)
    with torch.no_grad():
        inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
        m = torch.BoolTensor(mask).unsqueeze(0).to(device)
        pred = model(inp, m).squeeze(0).cpu().numpy()
    preds.append(pred)

    known_indices = np.where(mask)[0]
    for _ in range(n_masks - 1):
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

    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


# ============================================================
# 7. 可视化
# ============================================================

def plot_reconstruction(model, test_seqs, ratio, device, output_dir, seed=42):
    """
    可视化重建效果：选 4 块代表性电池，展示真值 / 已知点 / Transformer重建 / 插值
    """
    model.eval()
    rng = np.random.RandomState(seed)

    # 选 4 块电池（不同长度）
    sorted_bids = sorted(test_seqs.keys(), key=lambda b: len(test_seqs[b]))
    n = len(sorted_bids)
    if n >= 4:
        indices = [0, n // 3, 2 * n // 3, n - 1]
    else:
        indices = list(range(n))
    selected = [sorted_bids[i] for i in indices]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, bid in enumerate(selected):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        soh = test_seqs[bid]
        T = len(soh)
        masked_soh, mask = mask_soh_sequence(soh, ratio, rng)
        cycles = np.arange(T)

        # Transformer 预测
        soh_input = np.where(mask, soh, 0.0).astype(np.float32)
        with torch.no_grad():
            inp = torch.FloatTensor(soh_input).unsqueeze(0).to(device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(device)
            pred = model(inp, m).squeeze(0).cpu().numpy()

        # 线性插值
        interp = linear_interpolation(masked_soh, mask)

        # 只在 masked 位置计算 MAE
        eval_mask = ~mask
        trans_mae = np.mean(np.abs(pred[eval_mask] - soh[eval_mask])) * 100
        interp_mae = np.mean(np.abs(interp[eval_mask] - soh[eval_mask])) * 100

        # 画图
        ax.plot(cycles, soh, 'k-', linewidth=1.5, alpha=0.4, label='True SOH')
        ax.scatter(cycles[mask], soh[mask], c='blue', s=12, zorder=5,
                   label=f'Known ({mask.sum()}/{T})', alpha=0.7)
        ax.plot(cycles, pred, 'r-', linewidth=1.2, alpha=0.8,
                label=f'Transformer (MAE={trans_mae:.3f}%)')
        ax.plot(cycles, interp, 'g--', linewidth=1.0, alpha=0.7,
                label=f'Interpolation (MAE={interp_mae:.3f}%)')

        ax.set_xlabel('Cycle')
        ax.set_ylabel('SOH')
        ax.set_title(f'Battery {bid} (T={T})', fontweight='bold')
        ax.legend(fontsize=7, loc='lower left')

    plt.suptitle(f'SOH Reconstruction at r={ratio}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, f'reconstruction_r{ratio}.png')
    plt.savefig(path)
    plt.close()
    print(f'  [PLOT] Saved: {path}')


def plot_summary(all_results, output_dir):
    """汇总柱状图：各 ratio 下 Transformer vs 插值"""
    ratios = sorted([float(k.split('=')[1]) for k in all_results.keys()])

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(ratios))
    width = 0.35

    trans_maes = [all_results[f'r={r}']['transformer_mae'] * 100 for r in ratios]
    interp_maes = [all_results[f'r={r}']['interp_mae'] * 100 for r in ratios]

    bars1 = ax.bar(x - width/2, trans_maes, width, label='Transformer', color='#E53935')
    bars2 = ax.bar(x + width/2, interp_maes, width, label='Linear Interpolation', color='#43A047')

    # 数值标注
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Supervision Ratio')
    ax.set_ylabel('MAE (%)')
    ax.set_title('SOH Reconstruction: Transformer vs Interpolation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'r={r}' for r in ratios])
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f'  [PLOT] Saved: {path}')


# ============================================================
# 8. 主实验
# ============================================================

def run_experiment(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('\n[1/5] Loading battery SOH sequences...')
    sequences = load_battery_soh_sequences()
    battery_ids = sorted(sequences.keys())

    if args.quick:
        battery_ids = battery_ids[:15]
        sequences = {k: sequences[k] for k in battery_ids}
        print(f'[Quick mode] Using {len(battery_ids)} batteries')

    # 划分
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

    ratios = [0.3, 0.5, 0.7]
    all_results = {}

    for ratio in ratios:
        print(f'\n{"="*60}')
        print(f'[2/5] Training Transformer for r={ratio}')
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
            epochs=300 if not args.quick else 80,
            lr=1e-3,
            seed=42,
            batch_size=8,
        )

        # 评估
        print(f'\n[3/5] Evaluating at r={ratio}...')
        results, details = evaluate_reconstruction(
            model, test_seqs, ratio, device, seed=42, n_trials=5
        )

        print(f'\n  === Results at r={ratio} ===')
        print(f'  Transformer MAE:   {results["transformer_mae"]*100:.4f}%')
        print(f'  Interpolation MAE: {results["interp_mae"]*100:.4f}%')
        print(f'  Advantage: {results["advantage_pct"]:+.2f}% '
              f'({"Transformer wins" if results["advantage_pct"] > 0 else "Interpolation wins"})')

        all_results[f'r={ratio}'] = results

        # 可视化
        print(f'\n[4/5] Plotting reconstruction at r={ratio}...')
        plot_reconstruction(model, test_seqs, ratio, device, OUTPUT_DIR, seed=42)

        # 置信度
        if len(test_ids) > 0:
            sample_bid = test_ids[0]
            sample_soh = sequences[sample_bid]
            _, sample_mask = mask_soh_sequence(sample_soh, ratio,
                                               np.random.RandomState(42))
            mean_pred, std_pred = estimate_confidence(
                model, sample_soh, sample_mask, device
            )

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

    # 汇总图
    print(f'\n[5/5] Summary...')
    plot_summary(all_results, OUTPUT_DIR)

    # 保存 JSON
    output_file = os.path.join(OUTPUT_DIR, 'reconstruction_results.json')
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
        print(f'  -> SOH curve may be too smooth for Transformer advantage')

    print(f'\nPlots saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer SOH Reconstruction')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer batteries, fewer epochs')
    args = parser.parse_args()

    run_experiment(args)
