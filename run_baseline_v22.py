"""
Stage 0 基线锁定脚本 — baseline-v2.2
=====================================
运行 5 seeds × 4 supervision_ratios = 20 次实验，
记录每次结果并最终汇总到 experiments/baseline_v2.2/metrics.json

配置：
  模型    : CNN-LSTM（物理约束开启：软单调 + 边界）
  划分    : 60/20/20（种子固定）
  种子组  : [42, 123, 456, 789, 1024]
  监督比例: [1.0, 0.7, 0.5, 0.3]

断点续跑：若某次结果文件已存在则自动跳过，支持中断后继续。
"""

import os
import json
import time
import traceback
import numpy as np
import torch

from train_cross_battery import train_cross_battery_model

# ===== 实验配置 =====
SEEDS = [42, 123, 456, 789, 1024]
SUPERVISION_RATIOS = [1.0, 0.7, 0.5, 0.3]
MODEL_TYPE = 'cnn_lstm'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = 'experiments/baseline_v2.2'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(seed, ratio):
    """运行单次实验，返回结果字典；若已存在则直接加载。"""
    run_id = f"seed{seed}_ratio{ratio:.1f}".replace('.', 'p')
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    # 断点续跑：已存在则跳过
    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id} already done  "
              f"MAE={r['test_mae']*100:.4f}%  RMSE={r['test_rmse']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"[RUN] seed={seed}  supervision_ratio={ratio:.1f}  device={DEVICE}")
    print(f"{'='*70}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,   # 与主 seed 一致
        )

        elapsed = time.time() - t0
        record = {
            'run_id': run_id,
            'seed': seed,
            'supervision_ratio': ratio,
            'test_mae': float(results['test_mae']),
            'test_rmse': float(results['test_rmse']),
            'test_mape': float(results['test_mape']),
            'test_r2': float(results['test_r2']),
            'best_val_mae': float(results['best_val_mae']),
            'best_epoch': int(results['best_epoch']),
            'elapsed_sec': elapsed,
            'device': DEVICE,
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  [DONE] {run_id}  "
              f"MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"\n  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        # 记录失败信息，不阻塞后续实验
        error_record = {
            'run_id': run_id,
            'seed': seed,
            'supervision_ratio': ratio,
            'error': str(e),
        }
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump(error_record, f, indent=2, ensure_ascii=False)
        return None


def aggregate_and_save(all_results):
    """按监督比例聚合统计，保存 metrics.json 并打印汇总表。"""
    # 过滤掉失败的结果
    valid = [r for r in all_results if r is not None and 'test_mae' in r]

    # 按监督比例分组
    from collections import defaultdict
    groups = defaultdict(list)
    for r in valid:
        groups[r['supervision_ratio']].append(r)

    summary = {}
    for ratio in sorted(groups.keys(), reverse=True):
        runs = groups[ratio]
        maes  = [r['test_mae']  for r in runs]
        rmses = [r['test_rmse'] for r in runs]
        r2s   = [r['test_r2']   for r in runs]

        summary[str(ratio)] = {
            'n_runs': len(runs),
            'mae_mean': float(np.mean(maes)),
            'mae_std':  float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
            'mae_min':  float(np.min(maes)),
            'mae_max':  float(np.max(maes)),
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std':  float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
            'r2_mean':  float(np.mean(r2s)),
            'seeds_done': [r['seed'] for r in runs],
        }

    metrics = {
        'experiment': 'baseline-v2.2',
        'model': 'CNN-LSTM',
        'physics': 'soft_monotonic + boundary (enabled)',
        'split': '60/20/20',
        'seeds': SEEDS,
        'supervision_ratios': SUPERVISION_RATIOS,
        'total_runs_attempted': len(all_results),
        'total_runs_valid': len(valid),
        'summary_by_ratio': summary,
        'all_runs': valid,
    }

    out_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 打印汇总表
    print('\n' + '='*70)
    print('baseline-v2.2  汇总结果')
    print('='*70)
    header = f"{'ratio':>6} | {'n':>3} | {'MAE mean':>10} | {'MAE std':>9} | {'RMSE mean':>10} | {'R2 mean':>8}"
    print(header)
    print('-' * len(header))
    for ratio in sorted(summary.keys(), key=float, reverse=True):
        s = summary[ratio]
        print(f"  {float(ratio):>4.1f} | {s['n_runs']:>3} | "
              f"{s['mae_mean']*100:>8.4f}%  | "
              f"{s['mae_std']*100:>7.4f}%  | "
              f"{s['rmse_mean']*100:>8.4f}%  | "
              f"{s['r2_mean']:>8.4f}")

    print(f"\n保存到: {out_path}")
    return metrics


if __name__ == '__main__':
    total = len(SEEDS) * len(SUPERVISION_RATIOS)
    print(f"baseline-v2.2  Stage 0 基线扫描")
    print(f"共 {total} 次实验 ({len(SEEDS)} seeds × {len(SUPERVISION_RATIOS)} ratios)")
    print(f"结果目录: {OUTPUT_DIR}")
    print(f"设备: {DEVICE}")

    all_results = []
    done = 0

    for ratio in SUPERVISION_RATIOS:
        for seed in SEEDS:
            done += 1
            print(f"\n进度: {done}/{total}  ratio={ratio:.1f}  seed={seed}")
            r = run_single(seed, ratio)
            all_results.append(r)

    # 最终汇总
    aggregate_and_save(all_results)
    print('\nStage 0 完成！')
