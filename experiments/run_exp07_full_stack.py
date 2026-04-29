"""
Exp-07：Full Stack — M2+M4+M7（最终版）
=========================================
论文最终方法。模块组合经消融验证确定：

    M2  MC Dropout        — 不确定性量化（r=1.0 +4.42%）
    M4  速率连续性约束     — 物理一致性（smoothness_weight=0.05）
    M7  置信区间评估       — PICP/MPIW/Spearman（依赖 M2）

已淘汰（不含）：
    M1  循环级注意力       — 低标签率有害，淘汰
    M5  自适应损失权重     — log_var 学歪，放弃
    M6  不确定性伪标签     — r≤0.5 不稳定，不进 Full Stack

验证配置：
    supervision_ratios = [1.0, 0.7, 0.5, 0.3]
    seeds              = [42, 123, 34, 999, 1024]
    总运行次数          = 4 × 5 = 20

结果目录：experiments/exp07_full_stack/

论文用途：
    - 消融表最后一行（"Our Method / Full Stack"）
    - 与 E0/E2/E3 对比，验证 M2+M4 组合增益
    - M7 指标（PICP/MPIW/Spearman）在此实验中计算
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
from evaluation.physics_viz import compute_physics_violations

# ===== 配置 =====
SEEDS              = [42, 123, 34, 999, 1024]
SUPERVISION_RATIOS = [1.0, 0.7, 0.5, 0.3]
MODEL_TYPE         = 'cnn_lstm_full_stack'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp07_full_stack')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_uncertainty_metrics(model, data_dict, device, n_mc_samples=50):
    """
    M7：对测试集计算 PICP / MPIW / Spearman 指标。
    仅在模型含 MCDropout 层时有效。

    Args:
        data_dict: train_cross_battery_model 返回的第三个值（包含 test_features/test_targets 的 numpy 数组字典）
    """
    from models.modules.mc_dropout import mc_predict
    from evaluation.uncertainty_eval import uncertainty_report

    test_feat = data_dict.get('test_features')
    test_targ = data_dict.get('test_targets')
    if test_feat is None or test_targ is None or len(test_feat) == 0:
        print("  [WARN] M7: data_dict 中缺少 test_features/test_targets，跳过")
        return {}

    x       = torch.tensor(np.array(test_feat), dtype=torch.float32).to(device)
    targets = np.array(test_targ).squeeze()

    # 分批推理，避免大测试集 OOM
    batch_size = 512
    all_means, all_stds = [], []
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        mean, std = mc_predict(model, x_batch, n_samples=n_mc_samples)
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())

    means = np.concatenate(all_means).squeeze()
    stds  = np.concatenate(all_stds).squeeze()

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
        # 若已有完整 M7 + 物理指标则跳过；否则重新训练补充
        if 'picp' in r and 'mono_violation_rate' in r:
            print(f"  [SKIP] {run_id}  MAE={r['test_mae']*100:.4f}%  "
                  f"PICP={r['picp']:.3f}  违规率={r['mono_violation_rate']:.2f}%")
            return r
        else:
            missing = []
            if 'picp' not in r:
                missing.append('M7(picp/mpiw/spearman)')
            if 'mono_violation_rate' not in r:
                missing.append('物理违规率')
            print(f"  [RERUN] {run_id}  缺少 {', '.join(missing)}，重新训练补充")

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"[RUN] Exp-07  seed={seed}  ratio={ratio}  device={DEVICE}")
    print(f"{'='*65}")

    try:
        model, results, data_dict = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
        )
        elapsed = time.time() - t0

        # 物理违规后验统计（与消融脚本对齐）
        phys_metrics = {}
        preds_list = results.get('predictions')
        tgts_list  = results.get('targets')
        bids_list  = results.get('battery_ids')
        if preds_list is not None and bids_list is not None \
                and len(preds_list) > 0 and len(bids_list) > 0:
            try:
                phys_metrics = compute_physics_violations(
                    preds_list, tgts_list, bids_list, tolerance=0.01
                )
            except Exception as e:
                print(f"  [WARN] 物理违规指标计算失败: {e}")

        # M7：计算置信区间指标（直接使用 data_dict 中的 numpy 数组）
        uncertainty_metrics = {}
        try:
            uncertainty_metrics = compute_uncertainty_metrics(
                model, data_dict, device, n_mc_samples=50
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
            # 物理指标（与消融脚本 key 名对齐）
            **phys_metrics,
            # M7 不确定性指标（统一 key 名，去掉 m7_ 前缀）
            'picp':     float(uncertainty_metrics.get('picp',     float('nan'))),
            'mpiw':     float(uncertainty_metrics.get('mpiw',     float('nan'))),
            'spearman': float(uncertainty_metrics.get('spearman', float('nan'))),
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        phys_line = ''
        if phys_metrics:
            phys_line = (f"  违规率={phys_metrics['mono_violation_rate']:.2f}%  "
                         f"ΔS均值={phys_metrics['delta_soh_mean']*100:.4f}%")
        print(f"\n  [DONE] {run_id}  MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  elapsed={elapsed/60:.1f}min"
              + phys_line)
        if uncertainty_metrics:
            print(f"  M7: PICP={record['picp']:.3f}  "
                  f"MPIW={record['mpiw']*100:.4f}%  "
                  f"Spearman={record['spearman']:.3f}")
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
    sep  = '=' * 80
    sep2 = '-' * 80
    print(f"\n{sep}")
    print("Exp-07 汇总（Full Stack M2+M4+M7）")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}")
    print(sep)
    print(f"  {'ratio':>6}  {'MAE mean':>10}  {'MAE std':>9}  "
          f"{'RMSE mean':>11}  {'违规率':>8}  {'PICP':>6}  runs")
    print(sep2)

    for ratio in sorted(groups.keys(), reverse=True):
        runs  = groups[ratio]
        maes  = [r['test_mae']  for r in runs]
        rmses = [r['test_rmse'] for r in runs]
        r2s   = [r['test_r2']   for r in runs]
        picps = [r.get('picp', float('nan')) for r in runs]
        picps = [p for p in picps if not np.isnan(p)]
        viols = [r.get('mono_violation_rate', float('nan')) for r in runs]
        viols = [v for v in viols if not np.isnan(v)]

        entry = {
            'supervision_ratio':    ratio,
            'n_runs':               len(runs),
            'mae_mean':             float(np.mean(maes)),
            'mae_std':              float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0),
            'rmse_mean':            float(np.mean(rmses)),
            'rmse_std':             float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
            'r2_mean':              float(np.mean(r2s)),
            'picp_mean':            float(np.mean(picps)) if picps else None,
            'mono_viol_rate_mean':  float(np.mean(viols)) if viols else None,
            'all_runs':             runs,
        }
        summary[str(ratio)] = entry
        picp_str = f"{entry['picp_mean']:.3f}"       if entry['picp_mean']           is not None else '  N/A'
        viol_str = f"{entry['mono_viol_rate_mean']:.2f}%" if entry['mono_viol_rate_mean'] is not None else '   N/A'
        print(f"  {ratio:>6.1f}  {entry['mae_mean']*100:>9.4f}%  "
              f"{entry['mae_std']*100:>8.4f}%  "
              f"{entry['rmse_mean']*100:>10.4f}%  {viol_str:>8}  {picp_str:>6}  {len(runs)}")

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp07_full_stack', 'by_ratio': summary},
                  f, indent=2, ensure_ascii=False)
    print(f"\n  保存至: {out}")


if __name__ == '__main__':
    total = len(SEEDS) * len(SUPERVISION_RATIOS)
    print(f"Exp-07: Full Stack (M2+M4+M7)  |  device={DEVICE}")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}")
    print(f"  总运行次数: {total}\n")

    all_results = []
    for ratio in SUPERVISION_RATIOS:
        for seed in SEEDS:
            r = run_single(seed, ratio)
            all_results.append(r)

    aggregate(all_results)
    print("\nExp-07 完成！")
