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
# 实验配置（5 组，warmup=10 使伪标签在 best_epoch 前生效）
# ================================================================
# 背景：第一轮验证发现 seed=42 的 best_epoch=15 < warmup_epochs=30，
# 导致伪标签从未影响最佳 checkpoint。此处统一将 warmup 降至 10，
# 并重新设一个 E5_base_w10 对照，保证五组在相同 warmup 下公平比较。
_W = {'warmup_epochs': 10}  # 公共 warmup override

EXPERIMENTS = [
    {
        'id':    'E5_base_w10',
        'label': 'E5_base (warmup=10)',
        'note':  '对照：warmup=10，无任何修复开关，确认伪标签已在 best_epoch 前生效',
        'override': {'pseudo_labeling': {**_W}},
    },
    {
        'id':    'E5_ema_w10',
        'label': 'E5 + EMA (warmup=10)',
        'note':  '仅开 EMA 平滑，观察第2次更新偏差是否被压制',
        'override': {'pseudo_labeling': {**_W, 'use_ema': True, 'ema_alpha': 0.7}},
    },
    {
        'id':    'E5_filter_w10',
        'label': 'E5 + 单调过滤 (warmup=10)',
        'note':  '仅开单调性过滤，观察过滤率 > 0 且 mono_viol 下降',
        'override': {'pseudo_labeling': {**_W, 'use_mono_filter': True}},
    },
    {
        'id':    'E5_lambda_w10',
        'label': 'E5 + λ自适应 (warmup=10)',
        'note':  '仅开 λ 自适应（r=0.3→0.02），观察降低伪标签权重是否单独有效',
        'override': {'pseudo_labeling': {**_W, 'lambda_adaptive': True}},
    },
    {
        'id':    'E5_all_w10',
        'label': 'E5 + 三合一 (warmup=10)',
        'note':  'EMA + 过滤 + λ自适应组合，观察 MAE 是否回到 E0 ±1%',
        'override': {'pseudo_labeling': {**_W, 'use_ema': True, 'ema_alpha': 0.7,
                                         'use_mono_filter': True, 'lambda_adaptive': True}},
    },
]


# ================================================================
# 读取已有基线结果
# ================================================================
def load_baseline_results() -> tuple[Optional[dict], Optional[dict]]:
    """读取 E0 的 seed42/ratio0.3 结果（来自 ablation_single_module）。
    E5_base_w10 由本脚本自己跑，不从缓存读。"""
    ratio_tag = f"{RATIO:.1f}".replace('.', 'p')

    e0_fp = os.path.join(ABLATION_DIR, 'E0_baseline',
                         f'ratio{ratio_tag}', f'seed{SEED}', 'result.json')
    e0 = json.load(open(e0_fp, encoding='utf-8')) if os.path.exists(e0_fp) else None

    if e0 is None:
        print(f"  ⚠️  E0 基线结果不存在: {e0_fp}")
    return e0


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
def print_comparison(e0: Optional[dict],
                     new_results: list[Optional[dict]]) -> None:
    sep  = '═' * 76
    sep2 = '─' * 76

    all_r    = {r['exp_id']: r for r in new_results if r}
    e5b_w10  = all_r.get('E5_base_w10')
    e0_mae   = e0['test_mae']   if e0      else None
    e5b_mae  = e5b_w10['test_mae'] if e5b_w10 else None

    def fmt_delta(val, ref):
        if val is None or ref is None:
            return '—'
        d = (ref - val) / ref * 100
        return f"{'+' if d > 0 else ''}{d:.2f}%"

    def viol_str(r):
        if r and 'mono_violation_rate' in r:
            return f"{r['mono_violation_rate']:.4f}"
        return '—'

    def best_ep(r):
        return str(r.get('best_epoch', '—')) if r else '—'

    print(f"\n\n{sep}")
    print(f"  M6 三方向修复验证对比表  (ratio={RATIO}, seed={SEED}, warmup=10)")
    print(sep)
    print(f"  {'方法':<36}  {'MAE%':>8}  {'Δ vs E0':>9}  {'Δ vs base_w10':>13}  {'违规率':>6}  {'bestEp':>6}")
    print(sep2)

    rows = [('E0 Baseline (参照)', e0, True)] + \
           [(exp['label'], all_r.get(exp['id']), exp['id'] == 'E5_base_w10')
            for exp in EXPERIMENTS]

    for label, r, is_base in rows:
        if r is None:
            print(f"  {label:<36}  {'(缺失)'}")
            continue
        mae_str = f"{r['test_mae']*100:.4f}%"
        d_e0    = '(基准)' if (label == 'E0 Baseline (参照)') else fmt_delta(r['test_mae'], e0_mae)
        d_e5b   = '(对照)' if is_base or label == 'E0 Baseline (参照)' \
                           else fmt_delta(r['test_mae'], e5b_mae)
        print(f"  {label:<36}  {mae_str:>8}  {d_e0:>9}  {d_e5b:>13}  {viol_str(r):>6}  {best_ep(r):>6}")
    print(sep2)

    # 决策建议
    e5_all = all_r.get('E5_all_w10')
    print("\n  [决策建议]")
    if e5b_w10:
        print(f"  E5_base_w10 best_epoch={e5b_w10.get('best_epoch','?')}  "
              f"← {'✅ 伪标签已生效' if e5b_w10.get('best_epoch',0) > 10 else '⚠️ 仍在 warmup 前收敛，需进一步降 warmup'}")
    if e5_all and e0_mae:
        delta_pct = (e0_mae - e5_all['test_mae']) / e0_mae * 100
        viol_ok   = e5_all.get('mono_violation_rate', 99) < 0.10
        mae_ok    = abs(delta_pct) < 1.0
        if mae_ok and viol_ok:
            print(f"  ✅ E5_all_w10 有效（MAE Δ={delta_pct:+.2f}%，违规率<10%）"
                  f" → 方案 C-enhanced：Full Stack = M2+M4+M6（修复版）")
        elif not mae_ok:
            print(f"  ⚠️  E5_all_w10 MAE Δ={delta_pct:+.2f}%（超出 ±1% 目标）")
            e5_ema = all_r.get('E5_ema_w10')
            if e5_ema and e5b_mae:
                print(f"       E5_ema vs E5_base_w10: {fmt_delta(e5_ema['test_mae'], e5b_mae)}")
            print(f"  → M6 在 r=0.3 不可靠 → 方案 C-clean（Full Stack = M2+M4，M6 仅 r≥0.5）")
        else:
            print(f"  ⚠️  E5_all_w10 违规率仍高（{e5_all.get('mono_violation_rate',0)*100:.2f}%）")
    else:
        print("  （E5_all_w10 结果缺失，无法判断）")
    print(sep)

    # 保存汇总
    summary = {
        'config':       {'ratio': RATIO, 'seed': SEED, 'warmup': 10},
        'E0_baseline':  e0,
        **{r['exp_id']: r for r in new_results if r},
    }
    out_fp = os.path.join(OUTPUT_DIR, 'summary_w10.json')
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总已保存: {out_fp}")


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('╔' + '═' * 62 + '╗')
    print('║  M6 三方向修复验证 Round 2（warmup=10，5组×1seed×r=0.3）' + ' ' * 4 + '║')
    print('╠' + '═' * 62 + '╣')
    print(f'║  ratio={RATIO}  seed={SEED}  warmup=10  device={DEVICE}' + ' ' * 28 + '║')
    print(f'║  输出目录: experiments/m6_verify/' + ' ' * 28 + '║')
    print('╚' + '═' * 62 + '╝')

    e0_result = load_baseline_results()

    print(f"\n载入参照结果:")
    if e0_result:
        print(f"  E0  MAE={e0_result['test_mae']*100:.4f}%  "
              f"best_epoch={e0_result.get('best_epoch','?')}")

    print(f"\n开始运行 {len(EXPERIMENTS)} 组实验（含新对照 E5_base_w10）...")
    new_results = []
    for exp in EXPERIMENTS:
        r = run_single(exp)
        new_results.append(r)

    print_comparison(e0_result, new_results)
    print("\n✅ M6 验证完成！")
