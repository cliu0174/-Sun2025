"""
Exp-07：Full Stack — M1+M2+M4+M5+M6+M7
=========================================
论文最终方法：将所有 P0/P1 模块全部启用的完整组合。

模块组合：
    M1  循环级注意力       — 增强时序表征
    M2  MC Dropout        — 不确定性量化基础
    M4  速率连续性约束     — 精化物理先验
    M5  自适应损失权重     — 替代人工调参
    M6  不确定性伪标签     — 利用未标注区间（依赖 M2）
    M7  置信区间评估       — PICP/MPIW 指标（依赖 M2）

验证配置：
    supervision_ratios = [1.0, 0.7, 0.5, 0.3]
    seeds              = [42, 123, 456, 789, 1024]
    总运行次数          = 4 × 5 = 20

结果目录：experiments/exp07_full_stack/

论文用途：
    - 消融表的最后一行（"Our Method"）
    - 与各单模块实验对比，验证组合增益
    - M7 指标（PICP/MPIW）在此实验中计算
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
MODEL_TYPE         = 'cnn_lstm_full_stack'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp07_full_stack')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_uncertainty_metrics(model, test_loader, device, n_mc_samples=50):
    """
    M7：对测试集计算 PICP / MPIW / Spearman 指标。
    仅在模型含 MCDropout 层时有效。
    """
    from models.modules.mc_dropout import mc_predict
    from evaluation.uncertainty_eval import uncertainty_report
    import numpy as np

    all_targets, all_means, all_stds = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if not isinstance(batch, dict) or 'window' not in batch:
                continue
            x = batch['window'].to(device)
            y = batch['target_soh'].cpu().numpy()
            mean, std = mc_predict(model, x, n_samples=n_mc_samples)
            all_targets.append(y)
            all_means.append(mean.cpu().numpy())
            all_stds.append(std.cpu().numpy())

    if not all_targets:
        return {}

    targets  = np.concatenate(all_targets).squeeze()
    means    = np.concatenate(all_means).squeeze()
    stds     = np.concatenate(all_stds).squeeze()

    report = uncertainty_report(targets, means, stds)
    return report


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
    print(f"[RUN] Exp-07  seed={seed}  ratio={ratio}  device={DEVICE}")
    print(f"{'='*65}")

    try:
        model, results, test_loader = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
        )
        elapsed = time.time() - t0

        # M7：计算置信区间指标
        uncertainty_metrics = {}
        try:
            uncertainty_metrics = compute_uncertainty_metrics(
                model, test_loader, device, n_mc_samples=50
            )
        except Exception as e:
            print(f"  [WARN] M7 指标计算失败: {e}")

        record = {
            'run_id':            run_id,
            'exp':               'exp07_full_stack',
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
            **{f'm7_{k}': v for k, v in uncertainty_metrics.items()
               if isinstance(v, (int, float))},
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  [DONE] {run_id}  MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  elapsed={elapsed/60:.1f}min")
        if uncertainty_metrics:
            print(f"  M7: PICP={uncertainty_metrics.get('picp_95',0):.3f}  "
                  f"MPIW={uncertainty_metrics.get('mpiw_95',0)*100:.4f}%  "
                  f"Spearman={uncertainty_metrics.get('spearman',0):.3f}")
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
    print("Exp-07 汇总（Full Stack M1+M2+M4+M5+M6+M7）")
    print(f"{'='*65}")
    print(f"  {'ratio':>6}  {'MAE mean':>10}  {'MAE std':>9}  "
          f"{'RMSE mean':>11}  {'PICP':>6}  runs")

    for ratio in sorted(groups.keys(), reverse=True):
        runs  = groups[ratio]
        maes  = [r['test_mae']  for r in runs]
        rmses = [r['test_rmse'] for r in runs]
        r2s   = [r['test_r2']   for r in runs]
        picps = [r.get('m7_picp_95', float('nan')) for r in runs]
        picps = [p for p in picps if not np.isnan(p)]

        entry = {
            'supervision_ratio': ratio,
            'n_runs':    len(runs),
            'mae_mean':  float(np.mean(maes)),
            'mae_std':   float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
            'rmse_mean': float(np.mean(rmses)),
            'rmse_std':  float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
            'r2_mean':   float(np.mean(r2s)),
            'picp_mean': float(np.mean(picps)) if picps else None,
            'all_runs':  runs,
        }
        summary[str(ratio)] = entry
        picp_str = f"{entry['picp_mean']:.3f}" if entry['picp_mean'] else '  N/A'
        print(f"  {ratio:>6.1f}  {entry['mae_mean']*100:>9.4f}%  "
              f"{entry['mae_std']*100:>8.4f}%  "
              f"{entry['rmse_mean']*100:>10.4f}%  {picp_str:>6}  {len(runs)}")

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp07_full_stack', 'by_ratio': summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  保存至: {out}")


if __name__ == '__main__':
    total = len(SEEDS) * len(SUPERVISION_RATIOS)
    print(f"Exp-07: Full Stack (M1+M2+M4+M5+M6+M7)  |  device={DEVICE}")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}")
    print(f"  总运行次数: {total}\n")

    all_results = []
    for ratio in SUPERVISION_RATIOS:
        for seed in SEEDS:
            r = run_single(seed, ratio)
            all_results.append(r)

    aggregate(all_results)
    print("\nExp-07 完成！")
