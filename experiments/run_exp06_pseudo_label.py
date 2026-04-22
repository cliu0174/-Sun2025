"""
Exp-06：baseline-v2.2 + M2 MC Dropout + M6 不确定性引导伪标签
===============================================================
⭐ Stage 3 核心实验

对比组：
    - baseline-v2.2 (CNN-LSTM + 物理约束，无伪标签)
    - Exp-02        (CNN-LSTM + M2 MC Dropout，无伪标签)
    - Exp-06        (CNN-LSTM + M2 + M6 伪标签)

M6 只在 supervision_ratio < 1.0 时有实质效果。
主要关注 ratio=0.5 和 ratio=0.3 的精度增益。

验证配置：
    supervision_ratios = [1.0, 0.7, 0.5, 0.3]
    seeds              = [42, 123, 456, 789, 1024]
    总运行次数          = 4 × 5 = 20

结果目录：experiments/exp06_pseudo_label/

论文关键对比：
    "部分监督场景下，M6 相比 baseline 的 MAE 改善幅度
     随 supervision_ratio 下降而增大" ← 这是 Table 1 的核心数据
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
from typing import Optional
import numpy as np
import torch

from train_cross_battery import train_cross_battery_model

# ===== 配置 =====
SEEDS              = [42, 123, 456, 789, 1024]
SUPERVISION_RATIOS = [1.0, 0.7, 0.5, 0.3]
MODEL_TYPE         = 'cnn_lstm_pseudo_label'   # M2 + M6
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp06_pseudo_label')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(seed: int, ratio: float) -> Optional[dict]:
    ratio_str   = f"{ratio:.1f}".replace('.', 'p')
    run_id      = f"seed{seed}_ratio{ratio_str}"
    run_dir     = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}  MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"[RUN] Exp-06  seed={seed}  supervision_ratio={ratio}  device={DEVICE}")
    print(f"{'='*65}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
        )
        elapsed = time.time() - t0
        record = {
            'run_id':            run_id,
            'exp':               'exp06_pseudo_label',
            'seed':              seed,
            'model_type':        MODEL_TYPE,
            'supervision_ratio': ratio,
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
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        if r is not None and 'test_mae' in r:
            groups[r['supervision_ratio']].append(r)

    summary = {}
    print(f"\n{'='*65}")
    print("Exp-06 汇总（M2+M6 伪标签 vs supervision_ratio）")
    print(f"{'='*65}")
    print(f"  {'ratio':>6}  {'MAE mean':>10}  {'MAE std':>9}  {'RMSE mean':>11}  runs")

    for ratio in sorted(groups.keys(), reverse=True):
        runs  = groups[ratio]
        maes  = [r['test_mae']  for r in runs]
        rmses = [r['test_rmse'] for r in runs]
        r2s   = [r['test_r2']   for r in runs]
        entry = {
            'supervision_ratio': ratio,
            'n_runs':    len(runs),
            'mae_mean':  float(np.mean(maes)),
            'mae_std':   float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std':  float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
            'r2_mean':   float(np.mean(r2s)),
            'all_runs':  runs,
        }
        summary[str(ratio)] = entry
        print(f"  {ratio:>6.1f}  {entry['mae_mean']*100:>9.4f}%  "
              f"{entry['mae_std']*100:>8.4f}%  "
              f"{entry['rmse_mean']*100:>10.4f}%  {len(runs)}")

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp06_pseudo_label', 'by_ratio': summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  保存至: {out}")
    print("\n  ── 论文关键对比（与 baseline 的相对改善）──")
    print("  需在 baseline 结果出来后对比各 ratio 的 MAE 降幅")


if __name__ == '__main__':
    total = len(SEEDS) * len(SUPERVISION_RATIOS)
    print(f"Exp-06: M2+M6 Pseudo-Labeling  |  device={DEVICE}")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}")
    print(f"  总运行次数: {total}\n")

    all_results = []
    for ratio in SUPERVISION_RATIOS:
        for seed in SEEDS:
            r = run_single(seed, ratio)
            all_results.append(r)

    aggregate(all_results)
    print("\nExp-06 完成！")
