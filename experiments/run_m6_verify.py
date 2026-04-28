"""
M6 三方向修复最小验证脚本
==========================
4 组 × 1 seed (42) × r=0.3，~1.5h

验证目的：
    E5_base    对照（直接读已有 ablation_single_module 结果，不重跑）
    E5_ema     仅 EMA 平滑（α=0.7）：第 2 次偏差是否被压制到 ≤0.2%
    E5_filter  仅单调性过滤：过滤率 > 0 且 mono_viol 下降
    E5_all     三合一（EMA + 过滤 + λ自适应）：MAE 是否回到 E0 ±1%

结果目录：experiments/m6_verify/
对比参照：experiments/ablation_single_module/E0_baseline/ratio0p3/seed42/result.json
         experiments/ablation_single_module/E5_m2_m6_pseudo/ratio0p3/seed42/result.json
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
from typing import Optional

import numpy as np
import torch

from train_cross_battery import train_cross_battery_model
from evaluation.physics_viz import compute_physics_violations

# ================================================================
# 全局配置
# ================================================================
SEED   = 42
RATIO  = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'm6_verify')
ABLATION_DIR = os.path.join(os.path.dirname(__file__), 'ablation_single_module')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验配置（4 组）
# ================================================================
EXPERIMENTS = [
    {
        'id':    'E5_ema',
        'label': 'E5 + EMA(α=0.7)',
        'note':  '仅开 EMA 平滑，观察第2次更新偏差是否被压制',
        'override': {
            'pseudo_labeling': {
                'use_ema':   True,
                'ema_alpha': 0.7,
            }
        },
    },
    {
        'id':    'E5_filter',
        'label': 'E5 + 单调过滤',
        'note':  '仅开单调性过滤，观察过滤率 > 0 且 mono_viol 下降',
        'override': {
            'pseudo_labeling': {
                'use_mono_filter': True,
            }
        },
    },
    {
        'id':    'E5_all',
        'label': 'E5 + EMA + 过滤 + λ自适应',
        'note':  '三合一组合，观察 MAE 是否回到 E0 ±1%',
        'override': {
            'pseudo_labeling': {
                'use_ema':         True,
                'ema_alpha':       0.7,
                'use_mono_filter': True,
                'lambda_adaptive': True,
            }
        },
    },
]


# ================================================================
# 读取已有基线结果
# ================================================================
def load_baseline_results() -> tuple[Optional[dict], Optional[dict]]:
    """读取 E0 和 E5_base 的 seed42/ratio0.3 结果（来自 ablation_single_module）。"""
    ratio_tag = f"{RATIO:.1f}".replace('.', 'p')

    e0_fp   = os.path.join(ABLATION_DIR, 'E0_baseline',
                           f'ratio{ratio_tag}', f'seed{SEED}', 'result.json')
    e5b_fp  = os.path.join(ABLATION_DIR, 'E5_m2_m6_pseudo',
                           f'ratio{ratio_tag}', f'seed{SEED}', 'result.json')

    e0  = json.load(open(e0_fp,  encoding='utf-8')) if os.path.exists(e0_fp)  else None
    e5b = json.load(open(e5b_fp, encoding='utf-8')) if os.path.exists(e5b_fp) else None

    if e0  is None: print(f"  ⚠️  E0 基线结果不存在: {e0_fp}")
    if e5b is None: print(f"  ⚠️  E5_base 结果不存在: {e5b_fp}")
    return e0, e5b


# ================================================================
# 单次运行
# ================================================================
def run_single(exp: dict) -> Optional[dict]:
    exp_id    = exp['id']
    run_dir   = os.path.join(OUTPUT_DIR, exp_id)
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        r = json.load(open(result_fp, encoding='utf-8'))
        print(f"  [SKIP] {exp_id:<30} MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n  {'─'*60}")
    print(f"  [RUN]  {exp_id}")
    print(f"         {exp['note']}")
    print(f"         ratio={RATIO}  seed={SEED}  device={DEVICE}")
    print(f"  {'─'*60}")

    t0 = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type        = 'cnn_lstm_pseudo_label',
            seed              = SEED,
            device            = DEVICE,
            supervision_ratio = RATIO,
            supervision_seed  = None,
            config_override   = exp['override'],
        )
        elapsed = time.time() - t0

        phys_metrics = {}
        preds_list = results.get('predictions')
        tgts_list  = results.get('targets')
        bids_list  = results.get('battery_ids')
        if preds_list is not None and bids_list is not None \
                and len(preds_list) > 0 and len(bids_list) > 0:
            phys_metrics = compute_physics_violations(
                preds_list, tgts_list, bids_list, tolerance=0.01
            )

        record = {
            'exp_id':            exp_id,
            'label':             exp['label'],
            'note':              exp['note'],
            'seed':              SEED,
            'supervision_ratio': RATIO,
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
            **phys_metrics,
        }
        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        viol_str = (f"  违规率={phys_metrics['mono_violation_rate']:.2f}%"
                    if phys_metrics else "")
        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min{viol_str}")
        return record

    except Exception as e:
        print(f"  [ERROR] {exp_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'exp_id': exp_id, 'error': str(e),
                       'traceback': traceback.format_exc()},
                      f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 打印对比表
# ================================================================
def print_comparison(e0: Optional[dict], e5b: Optional[dict],
                     new_results: list[Optional[dict]]) -> None:
    sep  = '═' * 72
    sep2 = '─' * 72

    print(f"\n\n{sep}")
    print(f"  M6 三方向修复验证对比表  (ratio={RATIO}, seed={SEED})")
    print(sep)
    print(f"  {'方法':<32}  {'MAE%':>8}  {'Δ vs E0':>9}  {'Δ vs E5b':>9}  {'违规率':>8}")
    print(sep2)

    e0_mae  = e0['test_mae']  if e0  else None
    e5b_mae = e5b['test_mae'] if e5b else None

    def fmt_delta(val, ref):
        if val is None or ref is None:
            return '—'
        d = (ref - val) / ref * 100
        return f"{'+' if d > 0 else ''}{d:.2f}%"

    def viol_str(r):
        if r and 'mono_violation_rate' in r:
            return f"{r['mono_violation_rate']:.2f}%"
        return '—'

    rows = [
        ('E0 Baseline (参照)',  e0,  True),
        ('E5_base (当前 M6)',   e5b, False),
    ] + [(exp['label'], r, False)
         for exp, r in zip(EXPERIMENTS, new_results)]

    for label, r, is_ref in rows:
        if r is None:
            print(f"  {label:<32}  {'(缺失)'}")
            continue
        mae_str = f"{r['test_mae']*100:.4f}%"
        d_e0    = '(基准)' if is_ref else fmt_delta(r['test_mae'], e0_mae)
        d_e5b   = '—'      if is_ref else fmt_delta(r['test_mae'], e5b_mae)
        v_str   = viol_str(r)
        print(f"  {label:<32}  {mae_str:>8}  {d_e0:>9}  {d_e5b:>9}  {v_str:>8}")
    print(sep2)

    # 决策建议
    all_r  = dict(zip([e['id'] for e in EXPERIMENTS], new_results))
    e5_all = all_r.get('E5_all')
    print("\n  [决策建议]")
    if e5_all and e0_mae:
        delta_pct = (e0_mae - e5_all['test_mae']) / e0_mae * 100
        viol_ok   = e5_all.get('mono_violation_rate', 99) < 10
        mae_ok    = abs(delta_pct) < 1.0
        if mae_ok and viol_ok:
            print(f"  ✅ E5_all 有效（MAE Δ={delta_pct:+.2f}%，违规率<10%）"
                  f" → 方案 C-enhanced：Full Stack = M2+M4+M6（修复版）")
        elif not mae_ok:
            print(f"  ⚠️  E5_all MAE Δ={delta_pct:+.2f}%（超出 ±1% 目标）")
            e5_ema = all_r.get('E5_ema')
            if e5_ema and e5b_mae:
                ema_delta = (e5b_mae - e5_ema['test_mae']) / e5b_mae * 100
                print(f"       E5_ema vs E5_base: {ema_delta:+.2f}%")
            print(f"  → 若偏差仍漂移：M6 在 r=0.3 不可靠 → 方案 C-clean（M6 仅 r≥0.5）")
        else:
            print(f"  ⚠️  E5_all 违规率仍高（{e5_all.get('mono_violation_rate', '?'):.2f}%）"
                  f"，考虑提高置信阈值 top10%→top5%")
    else:
        print("  （E5_all 结果缺失，无法判断）")
    print(sep)

    # 保存汇总
    summary = {
        'config': {'ratio': RATIO, 'seed': SEED},
        'E0_baseline': e0,
        'E5_base':     e5b,
        **{r['exp_id']: r for r in new_results if r},
    }
    out_fp = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总已保存: {out_fp}")


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('╔' + '═' * 60 + '╗')
    print('║  M6 三方向修复最小验证（4组 × 1seed × r=0.3）' + ' ' * 11 + '║')
    print('╠' + '═' * 60 + '╣')
    print(f'║  ratio={RATIO}  seed={SEED}  device={DEVICE}' + ' ' * 32 + '║')
    print(f'║  输出目录: experiments/m6_verify/' + ' ' * 26 + '║')
    print('╚' + '═' * 60 + '╝')

    e0_result, e5b_result = load_baseline_results()

    print(f"\n载入参照结果:")
    if e0_result:
        print(f"  E0  MAE={e0_result['test_mae']*100:.4f}%")
    if e5b_result:
        print(f"  E5b MAE={e5b_result['test_mae']*100:.4f}%  "
              f"违规率={e5b_result.get('mono_violation_rate', '?'):.2f}%")

    print(f"\n开始运行 {len(EXPERIMENTS)} 组实验...")
    new_results = []
    for exp in EXPERIMENTS:
        r = run_single(exp)
        new_results.append(r)

    print_comparison(e0_result, e5b_result, new_results)
    print("\n✅ M6 验证完成！")
