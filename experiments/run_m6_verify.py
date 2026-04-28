"""
M6 修复验证脚本（Round 4）
==========================
5 组 × 1 seed (42) × r=0.3

Round 4 改动：γ=0（纯 val_mae 优先）+ min_epoch=10 门控。
不再用 mono_viol 影响 model selection，彻底回归 MAE 目标。

结果目录：experiments/m6_verify/
对比参照：experiments/ablation_single_module/E0_baseline/ratio0p3/seed42/result.json
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
# 实验配置（Round 4：γ=0，纯 MAE + min_epoch 门控）
# ================================================================
# Round 3 诊断：γ=0.1 让 mono_viol 影响了 model selection，偏离 MAE 目标。
# Round 4 修正：γ=0（score = val_mae），min_epoch=10 仍保留（确保伪标签生效后才锁 best）。
# 目标：找到伪标签生效后 MAE 最低的 checkpoint，不引入任何 mono_viol 偏置。
_W   = {'warmup_epochs': 10}
_SEL = {'mode': 'composite', 'gamma': 0.0, 'min_epoch': 10}   # γ=0 → 纯 val_mae

EXPERIMENTS = [
    {
        'id':    'E5_base_g0',
        'label': 'E5_base (γ=0, min_ep=10)',
        'note':  'Round4 对照：仅开 min_epoch 门控，无其他修复，确认伪标签生效后 MAE 基线',
        'override': {
            'pseudo_labeling':  {**_W},
            'model_selection':  {**_SEL},
        },
    },
    {
        'id':    'E5_ema_g0',
        'label': 'E5 + EMA (γ=0)',
        'note':  'γ=0 + EMA α=0.7，观察 EMA 对 MAE 的净贡献',
        'override': {
            'pseudo_labeling':  {**_W, 'use_ema': True, 'ema_alpha': 0.7},
            'model_selection':  {**_SEL},
        },
    },
    {
        'id':    'E5_filter_g0',
        'label': 'E5 + 单调过滤 (γ=0)',
        'note':  'γ=0 + 单调过滤，观察过滤对 MAE 的净贡献',
        'override': {
            'pseudo_labeling':  {**_W, 'use_mono_filter': True},
            'model_selection':  {**_SEL},
        },
    },
    {
        'id':    'E5_lambda_g0',
        'label': 'E5 + λ自适应 (γ=0)',
        'note':  'γ=0 + λ自适应（Round3 已达标，此处去除 γ 偏置，看能否更优）',
        'override': {
            'pseudo_labeling':  {**_W, 'lambda_adaptive': True},
            'model_selection':  {**_SEL},
        },
    },
    {
        'id':    'E5_all_g0',
        'label': 'E5 + 三合一 (γ=0)',
        'note':  'γ=0 + EMA + 过滤 + λ自适应，Round4 组合上限',
        'override': {
            'pseudo_labeling':  {**_W, 'use_ema': True, 'ema_alpha': 0.7,
                                 'use_mono_filter': True, 'lambda_adaptive': True},
            'model_selection':  {**_SEL},
        },
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
    e5b      = all_r.get('E5_base_g0')
    e0_mae   = e0['test_mae']   if e0  else None
    e5b_mae  = e5b['test_mae']  if e5b else None

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
    print(f"  M6 验证 Round4：γ=0 纯 MAE + min_epoch 门控  (ratio={RATIO}, seed={SEED})")
    print(sep)
    print(f"  {'方法':<36}  {'MAE%':>8}  {'Δ vs E0':>9}  {'Δ vs base_g0':>13}  {'违规率':>6}  {'bestEp':>6}")
    print(sep2)

    rows = [('E0 Baseline (参照)', e0, True)] + \
           [(exp['label'], all_r.get(exp['id']), exp['id'] == 'E5_base_g0')
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
    e5_all = all_r.get('E5_all_g0')
    e5_lam = all_r.get('E5_lambda_g0')
    print("\n  [决策建议]")
    if e5b:
        be = e5b.get('best_epoch', 0)
        if be >= 10:
            print(f"  ✅ E5_base_g0 best_epoch={be} ≥ 10：门控有效，伪标签已在 best 前生效")
        else:
            print(f"  ⚠️ E5_base_g0 best_epoch={be} < 10：门控未生效，需检查")
    # 找所有实验中 MAE 最优的
    best_exp = min([(r['test_mae'], r['exp_id']) for r in all_r.values()], default=(None, None))
    if best_exp[0] and e0_mae:
        delta_pct = (e0_mae - best_exp[0]) / e0_mae * 100
        if delta_pct > 0:
            print(f"  ✅ 最优: {best_exp[1]}  MAE Δ={delta_pct:+.2f}%（优于 E0）")
        else:
            print(f"  ⚠️  最优: {best_exp[1]}  MAE Δ={delta_pct:+.2f}%（未超 E0，M6 在 r=0.3 场景有限）")
    if e5_all and e0_mae:
        delta_pct = (e0_mae - e5_all['test_mae']) / e0_mae * 100
        if delta_pct > 0:
            print(f"  ✅ 三合一 E5_all_g0 有效（MAE Δ={delta_pct:+.2f}% > 0）"
                  f" → Full Stack = M2+M4+M6（修复版）")
        elif abs(delta_pct) < 1.0:
            print(f"  〜 三合一与 E0 持平（Δ={delta_pct:+.2f}%），可报告为'不显著退化'")
        else:
            print(f"  ✗ 三合一 MAE Δ={delta_pct:+.2f}%，M6 在 r=0.3 不可靠，建议仅 r≥0.5 使用")
    else:
        print("  （E5_all_g0 结果缺失）")
    print(sep)

    # 保存汇总
    summary = {
        'config':       {'ratio': RATIO, 'seed': SEED, 'warmup': 10,
                         'selection': {'mode': 'composite', 'gamma': 0.0, 'min_epoch': 10}},
        'E0_baseline':  e0,
        **{r['exp_id']: r for r in new_results if r},
    }
    out_fp = os.path.join(OUTPUT_DIR, 'summary_g0.json')
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  汇总已保存: {out_fp}")


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('╔' + '═' * 62 + '╗')
    print('║  M6 验证 Round 4：γ=0 纯 MAE + min_epoch 门控（5组×1seed）' + ' ' * 2 + '║')
    print('╠' + '═' * 62 + '╣')
    print(f'║  ratio={RATIO}  seed={SEED}  warmup=10  γ=0.0  min_epoch=10' + ' ' * 18 + '║')
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
