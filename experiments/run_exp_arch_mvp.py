"""
架构 MVP 对比实验：多尺度 CNN（MS）+ Per-Window LayerNorm（M3）
================================================================

目标：在「无物理约束」条件下，对比 4 种架构变体与已有 Baseline 的差距。

实验列表：
  ┌──────┬──────────────────────────────────┬──────────┬──────────┐
  │  ID  │ 描述                             │ MS (多尺度)│ M3 (归一化)│
  ├──────┼──────────────────────────────────┼──────────┼──────────┤
  │  A0  │ 原 CNN-LSTM，无物理约束（参照组）  │    ✗     │    ✗     │
  │  A1  │ 多尺度 CNN-LSTM                  │    ✓     │    ✗     │
  │  A2  │ Per-Window LayerNorm             │    ✗     │    ✓     │
  │  A3  │ 多尺度 + LayerNorm（联合）        │    ✓     │    ✓     │
  └──────┴──────────────────────────────────┴──────────┴──────────┘

参照组（已有结果，从 ablation_single_module/ 加载，不重跑）：
  Eneg1 (cnn_lstm, 无物理)   ← 与 A0 完全等价，直接复用
  E0    (cnn_lstm, 软单调)   ← Baseline with physics

验证配置：
  seeds  = [929, 2262, 7]
  ratios = [1.0, 0.7, 0.5, 0.3]
  总运行次数 = 4 × 4 × 3 = 48 次

结果目录：experiments/ablation_arch_mvp/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from train_cross_battery import train_cross_battery_model
from evaluation.physics_viz import compute_physics_violations

# ================================================================
# 全局配置
# ================================================================
SEEDS              = [929, 2262, 7]
SUPERVISION_RATIOS = [1.0, 0.7, 0.5, 0.3]
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'ablation_arch_mvp')
ABLATION_DIR       = os.path.join(os.path.dirname(__file__), 'ablation_single_module')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 新架构实验配置（全部禁用物理约束，纯架构对比）
# ================================================================
EXPERIMENTS = [
    {
        'id':         'A0_baseline_no_physics',
        'label':      'A0: 原 CNN-LSTM（无物理）',
        'ms':         False,
        'norm':       False,
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'physics_constraints': {'enabled': False},
            'architecture':        {'use_multiscale': False, 'per_window_norm': False},
        },
        'note': '单路 CNN-LSTM，无物理约束。等价于 Eneg1，用于验证 V2 模型框架本身无误差。',
    },
    {
        'id':         'A1_multiscale',
        'label':      'A1: 多尺度 CNN（MS）',
        'ms':         True,
        'norm':       False,
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'physics_constraints': {'enabled': False},
            'architecture':        {'use_multiscale': True, 'per_window_norm': False},
        },
        'note': '3 路并行 CNN（kernel=3/7/15），无物理约束，无 LayerNorm。',
    },
    {
        'id':         'A2_norm',
        'label':      'A2: Per-Window LayerNorm（M3）',
        'ms':         False,
        'norm':       True,
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'physics_constraints': {'enabled': False},
            'architecture':        {'use_multiscale': False, 'per_window_norm': True},
        },
        'note': '单路 CNN-LSTM + 输入端 LayerNorm，无物理约束，无多尺度。',
    },
    {
        'id':         'A3_ms_norm',
        'label':      'A3: MS + M3 联合',
        'ms':         True,
        'norm':       True,
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'physics_constraints': {'enabled': False},
            'architecture':        {'use_multiscale': True, 'per_window_norm': True},
        },
        'note': '多尺度 CNN + Per-Window LayerNorm 联合，无物理约束。',
    },
]


# ================================================================
# 单次运行（带断点续跑）
# ================================================================
def run_single(exp: dict, seed: int, ratio: float) -> Optional[dict]:
    exp_id    = exp['id']
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id    = f"{exp_id}_ratio{ratio_tag}_seed{seed}"
    run_dir   = os.path.join(OUTPUT_DIR, exp_id, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id:<60} MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n  {'─'*68}")
    print(f"  [RUN]  {run_id}")
    ms_flag   = '✓ MS' if exp['ms']   else '✗ MS'
    norm_flag = '✓ M3' if exp['norm'] else '✗ M3'
    print(f"         {ms_flag}  {norm_flag}  ratio={ratio}  seed={seed}  device={DEVICE}")
    print(f"  {'─'*68}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=exp['model_type'],
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
            config_override=exp['override'],
        )
        elapsed = time.time() - t0

        phys_metrics = {}
        preds = results.get('predictions')
        tgts  = results.get('targets')
        bids  = results.get('battery_ids')
        if preds is not None and bids is not None and len(preds) > 0 and len(bids) > 0:
            phys_metrics = compute_physics_violations(preds, tgts, bids, tolerance=0.01)

        record = {
            'run_id':            run_id,
            'exp_id':            exp_id,
            'label':             exp['label'],
            'use_multiscale':    exp['ms'],
            'per_window_norm':   exp['norm'],
            'seed':              seed,
            'supervision_ratio': ratio,
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
            'note':              exp['note'],
            **phys_metrics,
        }

        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e), 'traceback': traceback.format_exc()},
                      f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 加载已有 Eneg1 / E0 结果作为参照
# ================================================================
def load_reference_results() -> dict:
    """
    从 ablation_single_module/ 加载 Eneg1 和 E0 的结果，
    返回 {exp_id: {ratio_str: {mean/std}}} 格式。
    """
    refs = {}
    for ref_id in ('Eneg1_no_physics', 'E0_baseline'):
        ref_data = defaultdict(list)
        for seed in SEEDS:
            for ratio in SUPERVISION_RATIOS:
                tag = f"{ratio:.1f}".replace('.', 'p')
                fp  = os.path.join(ABLATION_DIR, ref_id, f"ratio{tag}", f"seed{seed}", 'result.json')
                if os.path.exists(fp):
                    with open(fp, encoding='utf-8') as f:
                        r = json.load(f)
                    ref_data[tag].append(r)
        if ref_data:
            refs[ref_id] = ref_data
    return refs


# ================================================================
# 汇总 & 打印对比表
# ================================================================
def aggregate_and_print(all_results: list, ref_results: dict) -> dict:
    # 按 exp_id × ratio 聚合
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is None:
            continue
        tag = f"{r['supervision_ratio']:.1f}".replace('.', 'p')
        grouped[r['exp_id']][tag].append(r)

    summary = {}
    for exp_id, by_ratio in grouped.items():
        summary[exp_id] = {}
        for tag, records in by_ratio.items():
            maes   = [r['test_mae']  for r in records]
            rmses  = [r['test_rmse'] for r in records]
            r2s    = [r['test_r2']   for r in records]
            summary[exp_id][tag] = {
                'n':         len(records),
                'mae_mean':  float(np.mean(maes)),
                'mae_std':   float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std':  float(np.std(rmses)),
                'r2_mean':   float(np.mean(r2s)),
                'label':     records[0]['label'],
            }

    # 加入参照组到 summary（用于统一打印）
    for ref_id, by_ratio in ref_results.items():
        label_map = {'Eneg1_no_physics': '[REF] 原 CNN-LSTM（无物理）',
                     'E0_baseline':      '[REF] Baseline（软单调）'}
        summary[ref_id] = {}
        for tag, records in by_ratio.items():
            maes   = [r['test_mae']  for r in records]
            rmses  = [r['test_rmse'] for r in records]
            r2s    = [r['test_r2']   for r in records]
            summary[ref_id][tag] = {
                'n':         len(records),
                'mae_mean':  float(np.mean(maes)),
                'mae_std':   float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std':  float(np.std(rmses)),
                'r2_mean':   float(np.mean(r2s)),
                'label':     label_map.get(ref_id, ref_id),
            }

    # ── 打印各 ratio 对比表 ────────────────────────────────────────
    ratio_tags = [f"{r:.1f}".replace('.', 'p') for r in SUPERVISION_RATIOS]
    print_order = ['Eneg1_no_physics', 'E0_baseline',
                   'A0_baseline_no_physics', 'A1_multiscale', 'A2_norm', 'A3_ms_norm']

    for tag in ratio_tags:
        ratio_val = tag.replace('p', '.')
        print(f"\n{'='*80}")
        print(f"  ratio = {ratio_val}")
        print(f"  {'方法':<40} {'MAE%':>8} {'±':>4} {'RMSE%':>8} {'±':>4} {'R²':>7}  {'vs Eneg1':>10}")
        print(f"  {'-'*78}")

        eneg1_mae = summary.get('Eneg1_no_physics', {}).get(tag, {}).get('mae_mean', None)

        for exp_id in print_order:
            if exp_id not in summary or tag not in summary[exp_id]:
                continue
            s    = summary[exp_id][tag]
            mae  = s['mae_mean'] * 100
            mstd = s['mae_std']  * 100
            rms  = s['rmse_mean']* 100
            rstd = s['rmse_std'] * 100
            r2   = s['r2_mean']
            lbl  = s['label']

            if eneg1_mae is not None and eneg1_mae > 0 and exp_id != 'Eneg1_no_physics':
                delta = (eneg1_mae - s['mae_mean']) / eneg1_mae * 100
                vs_str = f"{delta:+.2f}%"
            else:
                vs_str = '—'

            flag = ' ◀' if exp_id in ('A1_multiscale', 'A2_norm', 'A3_ms_norm') else ''
            print(f"  {lbl:<40} {mae:>8.4f} {mstd:>4.4f} {rms:>8.4f} {rstd:>4.4f} {r2:>7.4f}  {vs_str:>10}{flag}")

    # ── 保存汇总 JSON ──────────────────────────────────────────────
    summary_fp = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 汇总结果 → {summary_fp}")
    return summary


# ================================================================
# 主入口
# ================================================================
def main():
    print("=" * 80)
    print("  架构 MVP 实验：多尺度 CNN（MS）+ Per-Window LayerNorm（M3）")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}  device={DEVICE}")
    print(f"  总新运行次数：{len(EXPERIMENTS)} × {len(SUPERVISION_RATIOS)} × {len(SEEDS)} = "
          f"{len(EXPERIMENTS) * len(SUPERVISION_RATIOS) * len(SEEDS)}")
    print("=" * 80)

    all_results = []
    for exp in EXPERIMENTS:
        for ratio in SUPERVISION_RATIOS:
            for seed in SEEDS:
                r = run_single(exp, seed, ratio)
                if r is not None:
                    all_results.append(r)

    print(f"\n\n{'='*80}")
    print("  对比汇总（含已有参照组 Eneg1 / E0）")
    print("=" * 80)

    ref_results = load_reference_results()
    if not ref_results:
        print("  [警告] 未找到 ablation_single_module/ 中的参照结果，"
              "请先运行 run_ablation_single_module.py 或忽略参照列。")

    aggregate_and_print(all_results, ref_results)


if __name__ == '__main__':
    main()
