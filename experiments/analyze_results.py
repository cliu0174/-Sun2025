"""
消融实验 vs Full Stack 对比分析脚本
=====================================
从各实验结果目录读取 result.json，生成统一对比表，
并对 Full Stack vs E0_baseline 做配对 t 检验。

使用方法：
    python experiments/analyze_results.py

前提：
    - run_ablation_single_module.py 已运行（含 E7_full_stack，seeds=[42,123,34,999,1024]）
    - run_exp07_full_stack.py 仅用于 M7 指标（PICP/MPIW/Spearman），MAE/违规率从消融脚本读取

对比设计：
    所有消融实验（含 Full Stack）使用相同 seeds，可做配对 t 检验。
    E7 的 MAE + 违规率来自消融脚本的 E7_full_stack。
"""

import os
import sys
import json
import math
from collections import defaultdict
from typing import Optional

import numpy as np

# ── scipy 可选（用于 t 检验） ──────────────────────────────────────────
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# 路径配置
# ================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(BASE_DIR, 'ablation_single_module')
EXP07_DIR    = os.path.join(BASE_DIR, 'exp07_full_stack')

SEEDS   = [42, 123, 34, 999, 1024]
RATIOS  = [1.0, 0.7, 0.5, 0.3]

# ================================================================
# 消融实验定义（与 run_ablation_single_module.py 保持一致）
# ================================================================
ABLATION_EXPS = [
    {'id': 'Eneg1_no_physics', 'label': '无物理约束 (Eneg1)'},
    {'id': 'E0_baseline',      'label': 'Baseline (E0)'},
    {'id': 'E1_m1_attention',  'label': '+M1 注意力 (E1)'},
    {'id': 'E2_m2_mc_dropout', 'label': '+M2 MC Dropout (E2)'},
    {'id': 'E3_m4_smoothness', 'label': '+M4 速率连续性 (E3)'},
    {'id': 'E4_m5_adaptive',   'label': '+M5 自适应权重 (E4)'},
    {'id': 'E5_m2_m6_pseudo',  'label': '+M2+M6 伪标签 (E5)'},
    {'id': 'E7_full_stack',    'label': 'Full Stack M2+M4 (E7)'},  # 已整合进消融脚本
]

# E7 现在也在消融脚本里，FULLSTACK 数据优先从消融脚本读取
# run_exp07_full_stack.py 仍保留，用于输出 PICP/MPIW/Spearman（M7 专项）
FULLSTACK_ID    = 'E7_full_stack'     # 消融脚本里的 id
FULLSTACK_LABEL = 'Full Stack M2+M4 (E7)'
BASELINE_ID     = 'E0_baseline'


# ================================================================
# 数据加载
# ================================================================
def load_ablation_results() -> dict:
    """
    加载消融实验结果。
    返回：{exp_id: {ratio: [result_dict, ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    for exp in ABLATION_EXPS:
        exp_id = exp['id']
        for ratio in RATIOS:
            ratio_tag = f"{ratio:.1f}".replace('.', 'p')
            for seed in SEEDS:
                fp = os.path.join(
                    ABLATION_DIR, exp_id,
                    f"ratio{ratio_tag}", f"seed{seed}", 'result.json'
                )
                if os.path.exists(fp):
                    with open(fp, encoding='utf-8') as f:
                        data[exp_id][ratio].append(json.load(f))
    return data


def load_exp07_results() -> dict:
    """
    加载 Exp-07 Full Stack 结果。
    返回：{ratio: [result_dict, ...]}
    """
    data = defaultdict(list)
    for ratio in RATIOS:
        ratio_str = f"{ratio:.1f}".replace('.', 'p')
        for seed in SEEDS:
            run_id = f"seed{seed}_ratio{ratio_str}"
            fp = os.path.join(EXP07_DIR, run_id, 'result.json')
            if os.path.exists(fp):
                with open(fp, encoding='utf-8') as f:
                    data[ratio].append(json.load(f))
    return data


# ================================================================
# 统计汇总
# ================================================================
def compute_stats(runs: list) -> Optional[dict]:
    """对一组 result.json 列表计算均值/标准差。"""
    if not runs:
        return None
    maes  = [r['test_mae']  for r in runs]
    rmses = [r['test_rmse'] for r in runs]
    r2s   = [r['test_r2']   for r in runs]

    viols   = [r['mono_violation_rate'] for r in runs if 'mono_violation_rate' in r]
    picps   = [r['picp']     for r in runs if 'picp'     in r and not math.isnan(r['picp'])]
    mpiws   = [r['mpiw']     for r in runs if 'mpiw'     in r and not math.isnan(r['mpiw'])]
    spearmans = [r['spearman'] for r in runs if 'spearman' in r and not math.isnan(r['spearman'])]

    n = len(runs)
    return {
        'n':          n,
        'mae_mean':   float(np.mean(maes)),
        'mae_std':    float(np.std(maes,  ddof=1) if n > 1 else 0.0),
        'rmse_mean':  float(np.mean(rmses)),
        'rmse_std':   float(np.std(rmses, ddof=1) if n > 1 else 0.0),
        'r2_mean':    float(np.mean(r2s)),
        'viol_mean':  float(np.mean(viols))    if viols    else None,
        'viol_std':   float(np.std(viols, ddof=1) if len(viols) > 1 else 0.0) if viols else None,
        'picp_mean':  float(np.mean(picps))    if picps    else None,
        'mpiw_mean':  float(np.mean(mpiws))    if mpiws    else None,
        'spearman_mean': float(np.mean(spearmans)) if spearmans else None,
        'raw_maes':   maes,          # 用于配对 t 检验
        'raw_viols':  viols,
    }


# ================================================================
# 配对 t 检验
# ================================================================
def paired_ttest(vals_a: list, vals_b: list, label_a: str, label_b: str) -> str:
    """配对 t 检验，返回格式化字符串。"""
    if len(vals_a) != len(vals_b) or len(vals_a) < 2:
        return f"  {label_a} vs {label_b}: 样本数不足，无法检验"
    if not _HAS_SCIPY:
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        mean_d = np.mean(diffs)
        std_d  = np.std(diffs, ddof=1)
        t_stat = mean_d / (std_d / math.sqrt(len(diffs))) if std_d > 0 else 0.0
        return (f"  {label_a} vs {label_b}: mean_diff={mean_d*100:+.4f}%  "
                f"t={t_stat:.3f}  (需 scipy 计算 p 值)")
    t_stat, p_val = scipy_stats.ttest_rel(vals_a, vals_b)
    mean_d = np.mean([a - b for a, b in zip(vals_a, vals_b)])
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    return (f"  {label_a} vs {label_b}: mean_diff={mean_d*100:+.4f}%  "
            f"t={t_stat:.3f}  p={p_val:.4f}  {sig}")


# ================================================================
# 打印工具
# ================================================================
SEP  = '═' * 88
SEP2 = '─' * 88
COL  = 18   # 每个 ratio 列宽


def _cell(val, fmt, fallback='—'):
    return fmt.format(val) if val is not None else fallback


def print_header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    hdr = f"  {'方法':<30}"
    for r in RATIOS:
        hdr += f"  r={r:.1f}".center(COL)
    print(hdr)
    print(SEP2)


def print_row(label: str, stats_by_ratio: dict,
              val_key='mae_mean', std_key='mae_std',
              scale=100, suffix='%', fallback='—'):
    row = f"  {label:<30}"
    for r in RATIOS:
        s = stats_by_ratio.get(r)
        if s and s.get(val_key) is not None:
            v   = s[val_key] * scale
            std = (s.get(std_key) or 0.0) * scale
            cell = f"{v:.3f}±{std:.3f}{suffix}"
        else:
            cell = fallback
        row += cell.center(COL)
    print(row)


def print_delta_row(label: str, stats_by_ratio: dict,
                    baseline_by_ratio: dict, val_key='mae_mean'):
    """相对 Baseline 的改善行（正 = 改善 = 降低）。"""
    row = f"    {'↳ vs Baseline':<28}"
    for r in RATIOS:
        s  = stats_by_ratio.get(r)
        bs = baseline_by_ratio.get(r)
        if s and bs and s.get(val_key) and bs.get(val_key):
            delta = (bs[val_key] - s[val_key]) / bs[val_key] * 100
            arrow = '↓' if delta > 0 else '↑'
            cell  = f"{arrow}{abs(delta):.1f}%"
        else:
            cell = '—'
        row += cell.center(COL)
    print(row)


# ================================================================
# 主分析
# ================================================================
def main():
    print(f"\n{'╔' + '═'*66 + '╗'}")
    print(f"║  消融实验 vs Full Stack 对比分析{' '*34}║")
    print(f"║  seeds = {SEEDS}{' '*24}║")
    print(f"╚{'═'*66}╝")

    # ── 加载数据 ─────────────────────────────────────────────────
    abl_raw  = load_ablation_results()
    fs_raw   = load_exp07_results()

    # ── 计算统计量 ───────────────────────────────────────────────
    abl_stats = {}
    for exp in ABLATION_EXPS:
        exp_id = exp['id']
        abl_stats[exp_id] = {}
        for ratio in RATIOS:
            runs = abl_raw[exp_id].get(ratio, [])
            abl_stats[exp_id][ratio] = compute_stats(runs)

    fs_stats = {}
    for ratio in RATIOS:
        runs = fs_raw.get(ratio, [])
        fs_stats[ratio] = compute_stats(runs)

    baseline_stats = abl_stats[BASELINE_ID]

    # ── 数据可用性报告 ────────────────────────────────────────────
    print(f"\n  [数据可用性]  seeds={SEEDS}  ratios={RATIOS}")
    for exp in ABLATION_EXPS:
        counts = [len(abl_raw[exp['id']].get(r, [])) for r in RATIOS]
        print(f"    {exp['label']:<35}  runs/ratio: {counts}")
    counts_fs = [len(fs_raw.get(r, [])) for r in RATIOS]
    print(f"    {FULLSTACK_LABEL:<35}  runs/ratio: {counts_fs}")

    # ================================================================
    # 表 1：MAE（主要精度指标）
    # ================================================================
    print_header("表1：MAE 均值 ± 标准差（越低越好）")

    for exp in ABLATION_EXPS:
        exp_id = exp['id']
        s_by_r = abl_stats[exp_id]
        print_row(exp['label'], s_by_r)
        if exp_id != BASELINE_ID:
            print_delta_row(exp['label'], s_by_r, baseline_stats)
        print(SEP2)

    # Full Stack
    print_row(FULLSTACK_LABEL, fs_stats)
    print_delta_row(FULLSTACK_LABEL, fs_stats, baseline_stats)
    print(SEP2)

    # ================================================================
    # 表 2：单调违规率（物理一致性）
    # ================================================================
    print_header("表2：单调违规率 % ± std（越低越好）")

    for exp in ABLATION_EXPS:
        exp_id = exp['id']
        s_by_r = abl_stats[exp_id]
        print_row(exp['label'], s_by_r,
                  val_key='viol_mean', std_key='viol_std',
                  scale=1, suffix='%')
        print(SEP2)

    # Full Stack 违规率
    print_row(FULLSTACK_LABEL, fs_stats,
              val_key='viol_mean', std_key='viol_std',
              scale=1, suffix='%')
    print(SEP2)

    # ================================================================
    # 表 3：M7 不确定性指标（仅 Full Stack）
    # ================================================================
    has_m7 = any(fs_stats[r] and fs_stats[r].get('picp_mean') is not None for r in RATIOS)
    if has_m7:
        print_header("表3：M7 不确定性指标（Full Stack 专属）")
        print_row('PICP (理想≥0.95)', fs_stats,
                  val_key='picp_mean', std_key=None,
                  scale=1, suffix='')
        print(SEP2)
        print_row('MPIW %', fs_stats,
                  val_key='mpiw_mean', std_key=None,
                  scale=100, suffix='%')
        print(SEP2)
        print_row('Spearman(|err|,σ)', fs_stats,
                  val_key='spearman_mean', std_key=None,
                  scale=1, suffix='')
        print(SEP2)

    # ================================================================
    # 配对 t 检验：Full Stack vs E0_baseline（按 ratio 分别检验）
    # ================================================================
    print(f"\n{SEP}")
    print("  配对 t 检验：Full Stack vs E0_baseline（MAE）")
    print(f"  使用相同种子 {SEEDS} 的逐 seed 配对")
    print(SEP2)

    for ratio in RATIOS:
        bl_runs = abl_raw[BASELINE_ID].get(ratio, [])
        fs_runs = fs_raw.get(ratio, [])

        # 按 seed 对齐配对
        bl_by_seed = {r['seed']: r['test_mae'] for r in bl_runs}
        fs_by_seed = {r['seed']: r['test_mae'] for r in fs_runs}
        common = sorted(set(bl_by_seed.keys()) & set(fs_by_seed.keys()))

        if len(common) < 2:
            print(f"  ratio={ratio:.1f}: 共同 seed 不足 ({len(common)})，跳过")
            continue

        a_vals = [bl_by_seed[s] for s in common]
        b_vals = [fs_by_seed[s] for s in common]
        result_str = paired_ttest(a_vals, b_vals,
                                  f"Baseline(r={ratio:.1f})",
                                  f"FullStack(r={ratio:.1f})")
        print(f"  ratio={ratio:.1f}  n={len(common)} pairs  {result_str.strip()}")

    print(SEP)
    if not _HAS_SCIPY:
        print("  ⚠  未找到 scipy，p 值未计算。安装：pip install scipy")

    # ================================================================
    # 逐 ratio 简明对比（便于快速看差异）
    # ================================================================
    print(f"\n{SEP}")
    print("  快速对比：E0_Baseline vs Full Stack（MAE 与违规率）")
    print(SEP2)
    print(f"  {'ratio':>6}  {'Baseline MAE':>14}  {'FullStack MAE':>14}  "
          f"{'MAE改善':>8}  {'Base违规率':>11}  {'FS违规率':>9}")
    print(SEP2)
    for ratio in RATIOS:
        bs = baseline_stats.get(ratio)
        fs = fs_stats.get(ratio)
        if bs and fs:
            b_mae  = bs['mae_mean']  * 100
            f_mae  = fs['mae_mean']  * 100
            b_viol = bs.get('viol_mean')
            f_viol = fs.get('viol_mean')
            delta  = (b_mae - f_mae) / b_mae * 100
            arrow  = '↓' if delta > 0 else '↑'
            viol_b = f"{b_viol:.2f}%" if b_viol is not None else '—'
            viol_f = f"{f_viol:.2f}%" if f_viol is not None else '—'
            print(f"  {ratio:>6.1f}  {b_mae:>13.4f}%  {f_mae:>13.4f}%  "
                  f"{arrow}{abs(delta):>6.1f}%  {viol_b:>11}  {viol_f:>9}")
        else:
            avail_b = 'OK' if bs else 'missing'
            avail_f = 'OK' if fs else 'missing'
            print(f"  {ratio:>6.1f}  [Baseline={avail_b}  FullStack={avail_f}]")
    print(SEP)


if __name__ == '__main__':
    main()
