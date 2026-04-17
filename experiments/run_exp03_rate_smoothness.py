"""
Exp-03：baseline-v2.2 + M4 速率连续性约束
==========================================
对比组：
    - baseline-v2.2 (CNN-LSTM + 软单调 + 边界，smoothness_weight=0)
    - Exp-03        (CNN-LSTM + 软单调 + 边界 + 速率连续性，smoothness_weight 扫描)

超参扫描：smoothness_weight ∈ {0.01, 0.05, 0.1, 0.2}
验证配置：supervision_ratio=1.0，seeds=[42, 123, 456, 789, 1024]

结果目录：experiments/exp03_rate_smoothness/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import copy
import traceback
import numpy as np
import torch

from train_cross_battery import train_cross_battery_model

# ===== 配置 =====
SEEDS              = [42, 123, 456, 789, 1024]
SUPERVISION_RATIO  = 1.0
MODEL_TYPE         = 'cnn_lstm'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_CONFIG_PATH   = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'models',
    'cnn_lstm_rate_smoothness_config.json'
)
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp03_rate_smoothness')
SMOOTHNESS_WEIGHTS = [0.01, 0.05, 0.1, 0.2]   # 超参扫描范围

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(seed: int, sw: float) -> dict | None:
    sw_str  = str(sw).replace('.', 'p')
    run_id  = f"seed{seed}_sw{sw_str}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}  MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"[RUN] Exp-03  seed={seed}  smoothness_weight={sw}  device={DEVICE}")
    print(f"{'='*65}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=SUPERVISION_RATIO,
            supervision_seed=None,
            config_override={'physics_constraints': {'smoothness_weight': sw}},
        )
        elapsed = time.time() - t0
        record = {
            'run_id':            run_id,
            'exp':               'exp03_rate_smoothness',
            'seed':              seed,
            'smoothness_weight': sw,
            'model_type':        MODEL_TYPE,
            'supervision_ratio': SUPERVISION_RATIO,
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  [DONE] {run_id}  MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"\n  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e)}, f, indent=2)
        return None


def aggregate(all_results: list):
    """按 smoothness_weight 分组汇总，找出最优超参。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        if r is not None and 'test_mae' in r:
            groups[r['smoothness_weight']].append(r)

    summary = {}
    print(f"\n{'='*65}")
    print("Exp-03 汇总（M4 速率连续性约束超参扫描）")
    print(f"{'='*65}")
    print(f"  {'sw':>6}  {'MAE mean':>10}  {'MAE std':>9}  {'RMSE mean':>11}  runs")

    for sw in sorted(groups.keys()):
        runs  = groups[sw]
        maes  = [r['test_mae']  for r in runs]
        rmses = [r['test_rmse'] for r in runs]
        r2s   = [r['test_r2']   for r in runs]
        entry = {
            'smoothness_weight': sw,
            'n_runs':    len(runs),
            'mae_mean':  float(np.mean(maes)),
            'mae_std':   float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std':  float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
            'r2_mean':   float(np.mean(r2s)),
            'all_runs':  runs,
        }
        summary[str(sw)] = entry
        print(f"  {sw:>6.3f}  {entry['mae_mean']*100:>9.4f}%  "
              f"{entry['mae_std']*100:>8.4f}%  "
              f"{entry['rmse_mean']*100:>10.4f}%  {len(runs)}")

    best_sw = min(summary.keys(), key=lambda k: summary[k]['mae_mean'])
    print(f"\n  ✅ 最优 smoothness_weight = {best_sw}  "
          f"MAE={summary[best_sw]['mae_mean']*100:.4f}%")

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp03_rate_smoothness', 'best_sw': best_sw,
                   'by_sw': summary}, f, indent=2, ensure_ascii=False)
    print(f"  保存至: {out}")


if __name__ == '__main__':
    print(f"Exp-03: CNN-LSTM + M4 Rate Smoothness  |  device={DEVICE}")
    print(f"  seeds={SEEDS}  smoothness_weights={SMOOTHNESS_WEIGHTS}")
    print(f"  总运行次数: {len(SEEDS) * len(SMOOTHNESS_WEIGHTS)}\n")

    all_results = []
    for sw in SMOOTHNESS_WEIGHTS:
        for seed in SEEDS:
            r = run_single(seed, sw)
            all_results.append(r)

    aggregate(all_results)
    print("\nExp-03 完成！")
