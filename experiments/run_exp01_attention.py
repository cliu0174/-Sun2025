"""
Exp-01：baseline-v2.2 + M1 循环级注意力
=========================================
对比组：
    - baseline-v2.2 (CNN-LSTM + 物理约束，无注意力)
    - Exp-01        (CNN-LSTM + 物理约束 + CycleAttention)

验证配置：
    - supervision_ratio = 1.0（全监督）
    - seeds = [42, 123, 456, 789, 1024]
    - 5 次运行，取 mean ± std

结果目录：experiments/exp01_attention/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
import numpy as np
import torch

from train_cross_battery import train_cross_battery_model

# ===== 配置 =====
SEEDS             = [42, 123, 456, 789, 1024]
SUPERVISION_RATIO = 1.0
MODEL_TYPE        = 'cnn_lstm_attention'
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), 'exp01_attention')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(seed):
    run_id      = f"seed{seed}"
    run_dir     = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}  MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*60}\n[RUN] Exp-01  seed={seed}  device={DEVICE}\n{'='*60}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=SUPERVISION_RATIO,
            supervision_seed=None,
        )
        elapsed = time.time() - t0
        record = {
            'run_id':      run_id,
            'exp':         'exp01_attention',
            'seed':        seed,
            'model_type':  MODEL_TYPE,
            'supervision_ratio': SUPERVISION_RATIO,
            'test_mae':    float(results['test_mae']),
            'test_rmse':   float(results['test_rmse']),
            'test_mape':   float(results['test_mape']),
            'test_r2':     float(results['test_r2']),
            'best_val_mae':float(results['best_val_mae']),
            'best_epoch':  int(results['best_epoch']),
            'elapsed_sec': elapsed,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  [DONE] seed={seed}  MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"\n  [ERROR] seed={seed}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'seed': seed, 'error': str(e)}, f, indent=2)
        return None


def aggregate(results):
    valid = [r for r in results if r is not None and 'test_mae' in r]
    if not valid:
        print("无有效结果，跳过汇总")
        return

    maes  = [r['test_mae']  for r in valid]
    rmses = [r['test_rmse'] for r in valid]
    r2s   = [r['test_r2']   for r in valid]

    summary = {
        'exp':      'exp01_attention',
        'model':    MODEL_TYPE,
        'n_runs':   len(valid),
        'seeds':    [r['seed'] for r in valid],
        'mae_mean': float(np.mean(maes)),
        'mae_std':  float(np.std(maes, ddof=1)),
        'rmse_mean':float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses, ddof=1)),
        'r2_mean':  float(np.mean(r2s)),
        'all_runs': valid,
    }

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Exp-01 汇总（{len(valid)} runs）")
    print(f"{'='*60}")
    print(f"  MAE  : {summary['mae_mean']*100:.4f}% ± {summary['mae_std']*100:.4f}%")
    print(f"  RMSE : {summary['rmse_mean']*100:.4f}% ± {summary['rmse_std']*100:.4f}%")
    print(f"  R2   : {summary['r2_mean']:.4f}")
    print(f"  保存至: {out}")


if __name__ == '__main__':
    print(f"Exp-01: CNN-LSTM + CycleAttention  |  {len(SEEDS)} seeds  |  device={DEVICE}")
    all_results = [run_single(s) for s in SEEDS]
    aggregate(all_results)
    print("\nExp-01 完成！")
