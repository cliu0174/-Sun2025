"""
Exp-10：PI-MS-CNN-LSTM 不确定性量化（M7 置信区间评估）
======================================================

目标：在 Exp09c（PI-MS-CNN-LSTM，软单调）基础上加入 M2 MC Dropout，
      计算 PICP / MPIW / Spearman 不确定性指标，用于论文不确定性量化章节。

实验组：
  ┌──────────┬───────────────────────────────────────────────────────────────┐
  │    ID    │ 描述                                                          │
  ├──────────┼───────────────────────────────────────────────────────────────┤
  │  Exp10   │ MS-CNN-LSTM + 软单调 w=0.3 + M2 MC Dropout（无 M4）          │
  └──────────┴───────────────────────────────────────────────────────────────┘

参照组（从已有结果加载，不重跑）：
  Exp09c  (PI-MS-CNN-LSTM, 仅软单调)  ← exp09_ms_full_stack/Exp09c_ms_pi_only/

验证配置：
  seeds  = [929, 2262, 7]
  ratios = [1.0, 0.7, 0.5, 0.3]
  总运行次数 = 1 × 4 × 3 = 12

结果目录：experiments/exp10_uncertainty/

论文用途：
  - 不确定性量化章节（Section 4.X）
  - PICP / MPIW / Spearman 指标表
  - 置信区间可视化图
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from train_cross_battery import train_cross_battery_model
from evaluation.physics_viz import compute_physics_violations

# ================================================================
# 全局配置
# ================================================================
SEEDS              = [929, 2262, 7]
SUPERVISION_RATIOS = [1.0, 0.7, 0.5, 0.3]
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp10_uncertainty')
EXP09_DIR          = os.path.join(os.path.dirname(__file__), 'exp09_ms_full_stack')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验配置：Exp09c + M2 MC Dropout（无 M4）
# ================================================================
EXPERIMENT = {
    'id':         'Exp10_ms_pi_mc',
    'label':      'Exp10: PI-MS-CNN-LSTM + M2 (不确定性)',
    'model_type': 'ms_cnn_lstm_v2',
    'override': {
        'architecture': {
            'use_multiscale': True,
            'per_window_norm': False,
            'dropout_rate': 0.1,
        },
        'mc_dropout': {
            'enabled': True,
            'n_samples': 50,
        },
        'physics_constraints': {
            'enabled': True,
            'base_loss_weight': 1.0,
            'monotonic_weight': 0.3,
            'boundary_weight': 0.0,
            'smoothness_weight': 0.0,
            'monotonic_tolerance': 0.005,
            'min_cycle': 300,
        },
        'training': {
            'early_stopping': {
                'enabled': True,
                'patience': 30,
                'min_delta': 1e-05,
            },
        },
    },
    'note': '多尺度CNN + 软单调(w=0.3) + M2 MC Dropout，用于不确定性量化',
}


# ================================================================
# M7 不确定性指标计算
# ================================================================
def compute_uncertainty_metrics(model, data_dict, device, n_mc_samples=50,
                                window_size=40):
    """M7：对测试集计算 PICP / MPIW / Spearman 指标。"""
    from models.modules.mc_dropout import mc_predict
    from evaluation.uncertainty_eval import uncertainty_report

    test_feat = data_dict.get('test_features')
    test_targ = data_dict.get('test_targets')
    if test_feat is None or test_targ is None or len(test_feat) == 0:
        print("  [WARN] M7: data_dict 中缺少 test_features/test_targets，跳过")
        return {}

    feat_arr = np.array(test_feat)
    targ_arr = np.array(test_targ)

    if window_size > 1 and len(feat_arr) >= window_size:
        wf, wt = [], []
        for i in range(len(feat_arr) - window_size + 1):
            wf.append(feat_arr[i:i + window_size])
            wt.append(targ_arr[i + window_size - 1])
        x_np    = np.array(wf)
        targets = np.array(wt).squeeze()
    else:
        x_np    = feat_arr[:, np.newaxis, :]
        targets = targ_arr.squeeze()

    x = torch.tensor(x_np, dtype=torch.float32).to(device)

    batch_size = 512
    all_means, all_stds = [], []
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        mean, std = mc_predict(model, x_batch, n_samples=n_mc_samples)
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())

    means = np.concatenate(all_means).squeeze()
    stds  = np.concatenate(all_stds).squeeze()

    min_len = min(len(means), len(targets))
    report  = uncertainty_report(targets[:min_len], means[:min_len], stds[:min_len])
    return report


# ================================================================
# 单次运行（带断点续跑）
# ================================================================
def run_single(seed: int, ratio: float) -> Optional[dict]:
    exp       = EXPERIMENT
    exp_id    = exp['id']
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id    = f"{exp_id}_ratio{ratio_tag}_seed{seed}"
    run_dir   = os.path.join(OUTPUT_DIR, exp_id, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id:<60} MAE={r['test_mae']*100:.4f}%  "
              f"PICP={r.get('picp', float('nan')):.3f}")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n  {'─'*68}")
    print(f"  [RUN]  {run_id}")
    print(f"         ratio={ratio}  seed={seed}  device={DEVICE}")
    print(f"         {exp['note']}")
    print(f"  {'─'*68}")

    try:
        model, results, data_dict = train_cross_battery_model(
            model_type=exp['model_type'],
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
            config_override=exp['override'],
        )
        elapsed = time.time() - t0

        # 物理违规后验统计
        phys_metrics = {}
        preds = results.get('predictions')
        tgts  = results.get('targets')
        bids  = results.get('battery_ids')
        if preds is not None and bids is not None and len(preds) > 0 and len(bids) > 0:
            try:
                phys_metrics = compute_physics_violations(preds, tgts, bids, tolerance=0.01)
            except Exception as e:
                print(f"  [WARN] 物理违规指标计算失败: {e}")

        # M7 置信区间指标（核心）
        uncertainty_metrics = {}
        try:
            uncertainty_metrics = compute_uncertainty_metrics(
                model, data_dict, DEVICE, n_mc_samples=50, window_size=40
            )
            if uncertainty_metrics:
                print(f"  [M7] PICP={uncertainty_metrics.get('picp', 0):.3f}  "
                      f"MPIW={uncertainty_metrics.get('mpiw', 0):.4f}  "
                      f"Spearman={uncertainty_metrics.get('spearman', 0):.3f}")
        except Exception as e:
            print(f"  [WARN] M7 指标计算失败: {e}")
            traceback.print_exc()

        record = {
            'run_id':            run_id,
            'exp_id':            exp_id,
            'label':             exp['label'],
            'seed':              seed,
            'supervision_ratio': ratio,
            'model_type':        exp['model_type'],
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
            'note':              exp['note'],
            **phys_metrics,
            'picp':     float(uncertainty_metrics.get('picp',     float('nan'))),
            'mpiw':     float(uncertainty_metrics.get('mpiw',     float('nan'))),
            'spearman': float(uncertainty_metrics.get('spearman', float('nan'))),
        }

        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min")
        return record

    except Exception as e:
        print(f"  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e),
                       'traceback': traceback.format_exc()},
                      f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 加载参照组：Exp09c
# ================================================================
def load_exp09c_reference() -> dict:
    """加载 Exp09c（PI-MS-CNN-LSTM，仅软单调，无 M2）作为精度参照。"""
    ref_data = defaultdict(list)
    for seed in SEEDS:
        for ratio in SUPERVISION_RATIOS:
            tag = f"{ratio:.1f}".replace('.', 'p')
            fp  = os.path.join(EXP09_DIR, 'Exp09c_ms_pi_only',
                               f"ratio{tag}", f"seed{seed}", 'result.json')
            if os.path.exists(fp):
                with open(fp, encoding='utf-8') as f:
                    ref_data[tag].append(json.load(f))
    return dict(ref_data)


# ================================================================
# 汇总 & 对比打印
# ================================================================
def aggregate_and_print(all_results: list, ref_09c: dict) -> dict:
    grouped = defaultdict(list)
    for r in all_results:
        if r is None:
            continue
        tag = f"{r['supervision_ratio']:.1f}".replace('.', 'p')
        grouped[tag].append(r)

    ratio_tags = [f"{r:.1f}".replace('.', 'p') for r in SUPERVISION_RATIOS]

    print(f"\n{'='*110}")
    print("  Exp-10 汇总：PI-MS-CNN-LSTM + M2 不确定性量化")
    print(f"{'='*110}")
    print(f"  {'ratio':<8} {'MAE%':>8} {'±':>6} {'R²':>7} {'违规率':>7} "
          f"{'PICP':>7} {'MPIW':>7} {'Spearman':>9} "
          f"{'vs Exp09c':>10}")
    print(f"  {'-'*108}")

    summary = {}
    for tag in ratio_tags:
        runs = grouped.get(tag, [])
        if not runs:
            continue

        maes  = [r['test_mae']  for r in runs]
        r2s   = [r['test_r2']   for r in runs]
        viols = [r.get('mono_violation_rate', float('nan')) for r in runs]
        viols = [v for v in viols if not np.isnan(v)]
        picps = [r.get('picp', float('nan')) for r in runs]
        picps = [p for p in picps if not np.isnan(p)]
        mpiws = [r.get('mpiw', float('nan')) for r in runs]
        mpiws = [m for m in mpiws if not np.isnan(m)]
        spears = [r.get('spearman', float('nan')) for r in runs]
        spears = [s for s in spears if not np.isnan(s)]

        mae_mean = float(np.mean(maes))
        mae_std  = float(np.std(maes))

        # vs Exp09c
        ref_runs = ref_09c.get(tag, [])
        if ref_runs:
            ref_mae = float(np.mean([r['test_mae'] for r in ref_runs]))
            delta = (ref_mae - mae_mean) / ref_mae * 100
            vs_str = f"{delta:+.2f}%"
        else:
            vs_str = 'N/A'

        ratio_val = tag.replace('p', '.')
        print(f"  {ratio_val:<8} "
              f"{mae_mean*100:>8.4f} {mae_std*100:>6.4f} "
              f"{float(np.mean(r2s)):>7.4f} "
              f"{float(np.mean(viols)) if viols else float('nan'):>7.2f}% "
              f"{float(np.mean(picps)) if picps else float('nan'):>7.3f} "
              f"{float(np.mean(mpiws)) if mpiws else float('nan'):>7.4f} "
              f"{float(np.mean(spears)) if spears else float('nan'):>9.3f} "
              f"{vs_str:>10}")

        summary[tag] = {
            'n':          len(runs),
            'mae_mean':   mae_mean,
            'mae_std':    mae_std,
            'r2_mean':    float(np.mean(r2s)),
            'viol_mean':  float(np.mean(viols)) if viols else None,
            'picp_mean':  float(np.mean(picps)) if picps else None,
            'mpiw_mean':  float(np.mean(mpiws)) if mpiws else None,
            'spearman_mean': float(np.mean(spears)) if spears else None,
        }

    summary_fp = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 汇总结果 → {summary_fp}")

    return summary


# ================================================================
# 主入口
# ================================================================
def main():
    total = len(SUPERVISION_RATIOS) * len(SEEDS)
    print("=" * 90)
    print("  Exp-10：PI-MS-CNN-LSTM 不确定性量化（M7 置信区间）")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}  device={DEVICE}")
    print(f"  总运行次数：{len(SUPERVISION_RATIOS)} × {len(SEEDS)} = {total}")
    print("=" * 90)

    all_results = []
    for ratio in SUPERVISION_RATIOS:
        for seed in SEEDS:
            r = run_single(seed, ratio)
            if r is not None:
                all_results.append(r)

    print(f"\n\n{'='*90}")
    print("  对比汇总（含参照组 Exp09c）")
    print("=" * 90)

    ref_09c = load_exp09c_reference()
    if not ref_09c:
        print("  [警告] 未找到 Exp09c 参照结果")

    aggregate_and_print(all_results, ref_09c)
    print("\nExp-10 完成！")


if __name__ == '__main__':
    main()
