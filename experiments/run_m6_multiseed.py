"""
M6 多 seed 稳定性验证
======================
目的：验证 E5_all_g0（三合一修复版）在 3 个 seed 下的 MAE 稳定性。
     用 mean ± std 确认 seed=42 的 +4.55% 不是偶然。

实验组合：
    E0_baseline  × [42, 123, 34]   ← 直接读 ablation_single_module 缓存
    E5_all_g0    × [42, 123, 34]   ← seed=42 读 m6_verify 缓存，其余新跑

总新运行：2 次（~14 min）
结果目录：experiments/m6_multiseed/
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
SEEDS  = [42, 123, 34]
RATIO  = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'm6_multiseed')
ABLATION_DIR = os.path.join(os.path.dirname(__file__), 'ablation_single_module')
VERIFY_DIR   = os.path.join(os.path.dirname(__file__), 'm6_verify')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# E5_all_g0 固定配置
_ALL_OVERRIDE = {
    'pseudo_labeling': {
        'warmup_epochs':  10,
        'use_ema':        True,
        'ema_alpha':      0.7,
        'use_mono_filter': True,
        'lambda_adaptive': True,
    },
    'model_selection': {
        'mode':       'composite',
        'gamma':      0.0,
        'min_epoch':  10,
    },
}


# ================================================================
# 加载 E0 缓存（来自 ablation_single_module）
# ================================================================
def load_e0(seed: int) -> Optional[dict]:
    ratio_tag = f"{RATIO:.1f}".replace('.', 'p')
    fp = os.path.join(ABLATION_DIR, 'E0_baseline',
                      f'ratio{ratio_tag}', f'seed{seed}', 'result.json')
    if os.path.exists(fp):
        r = json.load(open(fp, encoding='utf-8'))
        print(f"  [CACHE] E0_baseline  seed={seed}  MAE={r['test_mae']*100:.4f}%")
        return r
    # 缓存不存在则现跑（兜底）
    print(f"  [WARN] E0 缓存不存在 seed={seed}，现跑 baseline...")
    return _run_e0_fresh(seed)


def _run_e0_fresh(seed: int) -> Optional[dict]:
    """E0 缓存不存在时临时跑一次（兜底，正常不应触发）"""
    run_dir = os.path.join(OUTPUT_DIR, 'E0_baseline', f'seed{seed}')
    result_fp = os.path.join(run_dir, 'result.json')
    if os.path.exists(result_fp):
        r = json.load(open(result_fp, encoding='utf-8'))
        print(f"  [CACHE] E0_baseline  seed={seed}  MAE={r['test_mae']*100:.4f}% (local)")
        return r
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type='cnn_lstm', seed=seed, device=DEVICE,
            supervision_ratio=RATIO, supervision_seed=None,
        )
        elapsed = time.time() - t0
        phys = compute_physics_violations(
            results.get('predictions'), results.get('targets'),
            results.get('battery_ids'), tolerance=0.01,
        ) if results.get('predictions') is not None else {}
        record = {
            'exp_id': 'E0_baseline', 'seed': seed, 'supervision_ratio': RATIO,
            'test_mae':     float(results['test_mae']),
            'test_rmse':    float(results['test_rmse']),
            'test_mape':    float(results['test_mape']),
            'test_r2':      float(results['test_r2']),
            'best_val_mae': float(results['best_val_mae']),
            'best_epoch':   int(results['best_epoch']),
            'elapsed_sec':  elapsed, **phys,
        }
        json.dump(record, open(result_fp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"  [DONE] E0_baseline  seed={seed}  MAE={record['test_mae']*100:.4f}%")
        return record
    except Exception as e:
        print(f"  [ERROR] E0_baseline seed={seed}: {e}")
        traceback.print_exc()
        return None


# ================================================================
# 运行 E5_all_g0（单个 seed）
# ================================================================
def run_e5_all(seed: int) -> Optional[dict]:
    # seed=42 优先读 m6_verify 缓存
    if seed == 42:
        cached_fp = os.path.join(VERIFY_DIR, 'E5_all_g0', 'result.json')
        if os.path.exists(cached_fp):
            r = json.load(open(cached_fp, encoding='utf-8'))
            print(f"  [CACHE] E5_all_g0    seed={seed}  MAE={r['test_mae']*100:.4f}% (from m6_verify)")
            return r

    run_dir   = os.path.join(OUTPUT_DIR, 'E5_all_g0', f'seed{seed}')
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        r = json.load(open(result_fp, encoding='utf-8'))
        print(f"  [SKIP]  E5_all_g0    seed={seed}  MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n  {'─'*60}")
    print(f"  [RUN]  E5_all_g0  seed={seed}  ratio={RATIO}  device={DEVICE}")
    print(f"         三合一：filter + EMA(α=0.7) + λ自适应  warmup=10  γ=0  min_ep=10")
    print(f"  {'─'*60}")

    t0 = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type        = 'cnn_lstm_pseudo_label',
            seed              = seed,
            device            = DEVICE,
            supervision_ratio = RATIO,
            supervision_seed  = None,
            config_override   = _ALL_OVERRIDE,
        )
        elapsed = time.time() - t0

        phys = {}
        if results.get('predictions') is not None and results.get('battery_ids') is not None:
            phys = compute_physics_violations(
                results['predictions'], results['targets'],
                results['battery_ids'], tolerance=0.01,
            )

        record = {
            'exp_id':            'E5_all_g0',
            'seed':              seed,
            'supervision_ratio': RATIO,
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
            **phys,
        }
        json.dump(record, open(result_fp, 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)

        print(f"  [DONE] E5_all_g0  seed={seed}  MAE={record['test_mae']*100:.4f}%  "
              f"best_ep={record['best_epoch']}  elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"  [ERROR] E5_all_g0 seed={seed}: {e}")
        traceback.print_exc()
        json.dump({'exp_id': 'E5_all_g0', 'seed': seed, 'error': str(e),
                   'traceback': traceback.format_exc()},
                  open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8'),
                  indent=2, ensure_ascii=False)
        return None


# ================================================================
# 打印对比表（含 mean ± std）
# ================================================================
def print_multiseed_table(e0_list: list, e5_list: list) -> None:
    sep  = '═' * 74
    sep2 = '─' * 74

    def safe_mae(r):
        return r['test_mae'] if r else None

    def safe_ep(r):
        return r.get('best_epoch', '?') if r else '?'

    print(f"\n\n{sep}")
    print(f"  M6 多 seed 稳定性验证  (ratio={RATIO}, seeds={SEEDS})")
    print(sep)
    print(f"  {'方法':<20}  {'seed':>5}  {'MAE%':>8}  {'Δ vs E0':>9}  {'R²':>7}  {'bestEp':>6}")
    print(sep2)

    # 逐 seed 打印
    for i, seed in enumerate(SEEDS):
        e0 = e0_list[i]
        e5 = e5_list[i]
        e0_mae = safe_mae(e0)
        e5_mae = safe_mae(e5)

        e0_str = f"{e0_mae*100:.4f}%" if e0_mae else '—'
        e5_str = f"{e5_mae*100:.4f}%" if e5_mae else '—'

        delta_str = '—'
        if e0_mae and e5_mae:
            d = (e0_mae - e5_mae) / e0_mae * 100
            delta_str = f"{'+' if d > 0 else ''}{d:.2f}%"

        e0_r2 = f"{e0['test_r2']:.4f}" if e0 else '—'
        e5_r2 = f"{e5['test_r2']:.4f}" if e5 else '—'

        if i == 0:
            print(f"  {'E0 Baseline':<20}  {seed:>5}  {e0_str:>8}  {'(基准)':>9}  {e0_r2:>7}  {safe_ep(e0):>6}")
        else:
            print(f"  {'':20}  {seed:>5}  {e0_str:>8}  {'(基准)':>9}  {e0_r2:>7}  {safe_ep(e0):>6}")

        if i == 0:
            print(f"  {'E5_all_g0 (三合一)':<20}  {seed:>5}  {e5_str:>8}  {delta_str:>9}  {e5_r2:>7}  {safe_ep(e5):>6}")
        else:
            print(f"  {'':20}  {seed:>5}  {e5_str:>8}  {delta_str:>9}  {e5_r2:>7}  {safe_ep(e5):>6}")

        if i < len(SEEDS) - 1:
            print(f"  {'·'*72}")

    print(sep2)

    # 汇总统计
    e0_maes  = [r['test_mae'] for r in e0_list if r]
    e5_maes  = [r['test_mae'] for r in e5_list if r]
    e5_eps   = [r['best_epoch'] for r in e5_list if r]

    if e0_maes and e5_maes:
        e0_mean, e0_std = np.mean(e0_maes), np.std(e0_maes)
        e5_mean, e5_std = np.mean(e5_maes), np.std(e5_maes)
        delta_mean = (e0_mean - e5_mean) / e0_mean * 100
        deltas = [(e0 - e5) / e0 * 100 for e0, e5 in zip(e0_maes, e5_maes)]
        delta_std = np.std(deltas)

        print(f"\n  {'汇总统计':}")
        print(f"  E0  mean±std = {e0_mean*100:.4f}% ± {e0_std*100:.4f}%")
        print(f"  E5  mean±std = {e5_mean*100:.4f}% ± {e5_std*100:.4f}%")
        print(f"  Δ           = {delta_mean:+.2f}% ± {delta_std:.2f}%  "
              f"（正值 = E5 更好）")
        print(f"  best_epoch  = {e5_eps}  "
              f"mean={np.mean(e5_eps):.0f}  "
              f"（均 ≥ 10 → min_epoch 门控有效）")

        print(f"\n  [结论]")
        all_better = all(e5 < e0 for e0, e5 in zip(e0_maes, e5_maes))
        all_ep_ok  = all(ep >= 10 for ep in e5_eps)
        if all_better and all_ep_ok:
            print(f"  ✅ 3/3 seed 均优于 E0，Δ={delta_mean:+.2f}%±{delta_std:.2f}%")
            print(f"  ✅ 所有 best_epoch ≥ 10，min_epoch 门控全程有效")
            print(f"  → M6 三合一修复版稳定，可写入论文主表，更新 Exp-07 Full Stack")
        elif delta_mean > 0:
            n_better = sum(1 for e0, e5 in zip(e0_maes, e5_maes) if e5 < e0)
            print(f"  〜 {n_better}/3 seed 优于 E0，均值 Δ={delta_mean:+.2f}%，但不稳定")
            print(f"  → 建议扩展到 5 seed 后再决定是否写入主表")
        else:
            print(f"  ✗ E5_all_g0 均值不优于 E0（Δ={delta_mean:+.2f}%）")
            print(f"  → M6 在 r=0.3 不可靠，Full Stack 仅保留 M2+M4")

    print(sep)

    # 保存汇总
    summary = {
        'config':  {'ratio': RATIO, 'seeds': SEEDS,
                    'selection': {'mode': 'composite', 'gamma': 0.0, 'min_epoch': 10}},
        'E0_baseline': {str(SEEDS[i]): e0_list[i] for i in range(len(SEEDS))},
        'E5_all_g0':   {str(SEEDS[i]): e5_list[i] for i in range(len(SEEDS))},
        'stats': {
            'e0_mean_mae': float(np.mean(e0_maes)) if e0_maes else None,
            'e0_std_mae':  float(np.std(e0_maes))  if e0_maes else None,
            'e5_mean_mae': float(np.mean(e5_maes)) if e5_maes else None,
            'e5_std_mae':  float(np.std(e5_maes))  if e5_maes else None,
            'delta_mean_pct': float(delta_mean) if e0_maes and e5_maes else None,
            'delta_std_pct':  float(delta_std)  if e0_maes and e5_maes else None,
        } if e0_maes and e5_maes else {},
    }
    out_fp = os.path.join(OUTPUT_DIR, 'summary_multiseed.json')
    json.dump(summary, open(out_fp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n  汇总已保存: {out_fp}")


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    print('╔' + '═' * 62 + '╗')
    print('║  M6 多 seed 稳定性验证（E0 vs E5_all_g0，3 seeds）        ║')
    print('╠' + '═' * 62 + '╣')
    print(f'║  ratio={RATIO}  seeds={SEEDS}  device={DEVICE}' + ' ' * 20 + '║')
    print(f'║  新运行：E5_all_g0 × seed [123, 34]（~14 min）' + ' ' * 14 + '║')
    print(f'║  输出目录: experiments/m6_multiseed/' + ' ' * 24 + '║')
    print('╚' + '═' * 62 + '╝')

    e0_results = []
    e5_results = []

    for seed in SEEDS:
        print(f"\n{'='*62}")
        print(f"  Seed = {seed}")
        print(f"{'='*62}")
        e0_results.append(load_e0(seed))
        e5_results.append(run_e5_all(seed))

    print_multiseed_table(e0_results, e5_results)
    print("\n✅ 多 seed 验证完成！")
