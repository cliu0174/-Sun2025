"""
Exp-09：多尺度 CNN Full Stack — A1 + M2 + M4 + M7
====================================================

目标：验证多尺度 CNN 架构（A1）与物理约束模块（M2+M4）的联合增益，
      确定最终论文方法是 A1 standalone 还是 A1 + Full Stack。

实验列表：
  ┌──────────┬──────────────────────────────────────────────────────────┐
  │    ID    │ 描述                                                     │
  ├──────────┼──────────────────────────────────────────────────────────┤
  │  Exp09a  │ MS-CNN-LSTM + M2 + M4（monotonic_weight=0.3，同E7）     │
  │  Exp09b  │ MS-CNN-LSTM + M2 + M4（monotonic_weight=0.1，轻约束）   │
  │  Exp09c  │ MS-CNN-LSTM + 仅软单调（同E0配置，无M2/M4）             │
  └──────────┴──────────────────────────────────────────────────────────┘

参照组（从已有结果加载，不重跑）：
  A1      (ms_cnn_lstm_v2, 无物理)     ← ablation_arch_mvp/A1_multiscale/
  E0      (cnn_lstm + 软单调, 无M2/M4) ← ablation_single_module/E0_baseline/
  E7      (cnn_lstm_full_stack, M2+M4) ← exp07_full_stack/

验证配置：
  seeds  = [929, 2262, 7]
  ratios = [1.0, 0.7, 0.5, 0.3]
  总运行次数 = 3 × 4 × 3 = 36

结果目录：experiments/exp09_ms_full_stack/
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
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp09_ms_full_stack')
ARCH_MVP_DIR       = os.path.join(os.path.dirname(__file__), 'ablation_arch_mvp')
ABLATION_DIR       = os.path.join(os.path.dirname(__file__), 'ablation_single_module')
EXP07_DIR          = os.path.join(os.path.dirname(__file__), 'exp07_full_stack')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验配置：MS-CNN-LSTM + M2 MC Dropout + M4 速率连续性 + 物理约束
# ================================================================
EXPERIMENTS = [
    {
        'id':         'Exp09a_ms_full_stack_w03',
        'label':      'Exp09a: MS + M2 + M4 (w=0.3)',
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
                'smoothness_weight': 0.05,
                'monotonic_tolerance': 0.005,
                'min_cycle': 300,
                'temporal_decay': {
                    'enabled': True,
                    'max_step': 40,
                    'decay_type': 'exp',
                    'decay_alpha': 0.2,
                },
            },
            'training': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 30,
                    'min_delta': 1e-05,
                },
            },
        },
        'note': '多尺度CNN + M2 MC Dropout + M4 速率连续性，标准物理权重 w=0.3（同E7）',
    },
    {
        'id':         'Exp09b_ms_full_stack_w01',
        'label':      'Exp09b: MS + M2 + M4 (w=0.1)',
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
                'monotonic_weight': 0.1,
                'boundary_weight': 0.0,
                'smoothness_weight': 0.05,
                'monotonic_tolerance': 0.005,
                'min_cycle': 300,
                'temporal_decay': {
                    'enabled': True,
                    'max_step': 40,
                    'decay_type': 'exp',
                    'decay_alpha': 0.2,
                },
            },
            'training': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 30,
                    'min_delta': 1e-05,
                },
            },
        },
        'note': '多尺度CNN + M2 MC Dropout + M4 速率连续性，轻物理权重 w=0.1',
    },
    {
        'id':         'Exp09c_ms_pi_only',
        'label':      'Exp09c: PI-MS-CNN-LSTM (仅软单调)',
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'architecture': {
                'use_multiscale': True,
                'per_window_norm': False,
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
        },
        'note': '多尺度CNN + 仅软单调约束（同E0配置），无M2/M4，纯PI架构对比',
    },
]


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
def run_single(exp: dict, seed: int, ratio: float) -> Optional[dict]:
    exp_id    = exp['id']
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id    = f"{exp_id}_ratio{ratio_tag}_seed{seed}"
    run_dir   = os.path.join(OUTPUT_DIR, exp_id, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id:<60} MAE={r['test_mae']*100:.4f}%")
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

        # M7 置信区间指标
        uncertainty_metrics = {}
        try:
            uncertainty_metrics = compute_uncertainty_metrics(
                model, data_dict, DEVICE, n_mc_samples=50, window_size=40
            )
        except Exception as e:
            print(f"  [WARN] M7 指标计算失败: {e}")

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

        phys_line = ''
        if phys_metrics:
            phys_line = f"  违规率={phys_metrics.get('mono_violation_rate', 0):.2f}%"
        picp_line = ''
        if uncertainty_metrics:
            picp_line = f"  PICP={record['picp']:.3f}"

        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min{phys_line}{picp_line}")
        return record

    except Exception as e:
        print(f"  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e), 'traceback': traceback.format_exc()},
                      f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 加载参照组结果
# ================================================================
def load_reference_results() -> dict:
    """加载 A1 standalone、E0 PI-CNN-LSTM 和 E7 Full Stack 结果作为参照。"""
    refs = {}

    # A1 from arch_mvp (MS-CNN-LSTM, 无物理)
    a1_data = defaultdict(list)
    for seed in SEEDS:
        for ratio in SUPERVISION_RATIOS:
            tag = f"{ratio:.1f}".replace('.', 'p')
            fp  = os.path.join(ARCH_MVP_DIR, 'A1_multiscale', f"ratio{tag}", f"seed{seed}", 'result.json')
            if os.path.exists(fp):
                with open(fp, encoding='utf-8') as f:
                    a1_data[tag].append(json.load(f))
    if a1_data:
        refs['A1_multiscale'] = a1_data

    # E0 from ablation_single_module (PI-CNN-LSTM, 仅软单调)
    e0_data = defaultdict(list)
    for seed in SEEDS:
        for ratio in SUPERVISION_RATIOS:
            tag = f"{ratio:.1f}".replace('.', 'p')
            fp  = os.path.join(ABLATION_DIR, 'E0_baseline', f"ratio{tag}", f"seed{seed}", 'result.json')
            if os.path.exists(fp):
                with open(fp, encoding='utf-8') as f:
                    e0_data[tag].append(json.load(f))
    if e0_data:
        refs['E0_baseline'] = e0_data

    # E7 from exp07 (seeds 可能不同，取交集)
    e7_data = defaultdict(list)
    for seed in SEEDS:
        for ratio in SUPERVISION_RATIOS:
            tag = f"{ratio:.1f}".replace('.', 'p')
            run_id = f"seed{seed}_ratio{tag}"
            fp = os.path.join(EXP07_DIR, run_id, 'result.json')
            if os.path.exists(fp):
                with open(fp, encoding='utf-8') as f:
                    e7_data[tag].append(json.load(f))
    if e7_data:
        refs['E7_full_stack'] = e7_data

    return refs


# ================================================================
# 汇总 & 对比打印
# ================================================================
def aggregate_and_print(all_results: list, ref_results: dict) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is None:
            continue
        tag = f"{r['supervision_ratio']:.1f}".replace('.', 'p')
        grouped[r['exp_id']][tag].append(r)

    summary = {}

    # 新实验汇总
    for exp_id, by_ratio in grouped.items():
        summary[exp_id] = {}
        for tag, records in by_ratio.items():
            maes  = [r['test_mae']  for r in records]
            rmses = [r['test_rmse'] for r in records]
            r2s   = [r['test_r2']   for r in records]
            viols = [r.get('mono_violation_rate', float('nan')) for r in records]
            viols = [v for v in viols if not np.isnan(v)]
            picps = [r.get('picp', float('nan')) for r in records]
            picps = [p for p in picps if not np.isnan(p)]
            summary[exp_id][tag] = {
                'n':         len(records),
                'mae_mean':  float(np.mean(maes)),
                'mae_std':   float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'r2_mean':   float(np.mean(r2s)),
                'viol_mean': float(np.mean(viols)) if viols else None,
                'picp_mean': float(np.mean(picps)) if picps else None,
                'label':     records[0]['label'],
            }

    # 参照组汇总
    label_map = {
        'A1_multiscale': '[REF] A1: MS-CNN（无物理）',
        'E0_baseline':   '[REF] E0: PI-CNN-LSTM（仅软单调）',
        'E7_full_stack': '[REF] E7: CNN-LSTM Full Stack (M2+M4)',
    }
    for ref_id, by_ratio in ref_results.items():
        summary[ref_id] = {}
        for tag, records in by_ratio.items():
            maes  = [r['test_mae']  for r in records]
            rmses = [r['test_rmse'] for r in records]
            r2s   = [r['test_r2']   for r in records]
            viols = [r.get('mono_violation_rate', float('nan')) for r in records]
            viols = [v for v in viols if not np.isnan(v)]
            summary[ref_id][tag] = {
                'n':         len(records),
                'mae_mean':  float(np.mean(maes)),
                'mae_std':   float(np.std(maes)),
                'rmse_mean': float(np.mean(rmses)),
                'r2_mean':   float(np.mean(r2s)),
                'viol_mean': float(np.mean(viols)) if viols else None,
                'picp_mean': None,
                'label':     label_map.get(ref_id, ref_id),
            }

    # 打印对比表
    ratio_tags = [f"{r:.1f}".replace('.', 'p') for r in SUPERVISION_RATIOS]
    print_order = ['A1_multiscale', 'E0_baseline', 'E7_full_stack',
                   'Exp09c_ms_pi_only',
                   'Exp09a_ms_full_stack_w03', 'Exp09b_ms_full_stack_w01']

    for tag in ratio_tags:
        ratio_val = tag.replace('p', '.')
        print(f"\n{'='*100}")
        print(f"  ratio = {ratio_val}")
        print(f"  {'方法':<48} {'MAE%':>8} {'±':>6} {'RMSE%':>8} {'R²':>7} {'违规率':>7} {'vs E0':>8} {'vs A1':>8}")
        print(f"  {'-'*98}")

        a1_mae = summary.get('A1_multiscale', {}).get(tag, {}).get('mae_mean', None)
        e0_mae = summary.get('E0_baseline',   {}).get(tag, {}).get('mae_mean', None)

        for exp_id in print_order:
            if exp_id not in summary or tag not in summary[exp_id]:
                continue
            s   = summary[exp_id][tag]
            mae = s['mae_mean'] * 100
            std = s['mae_std']  * 100
            rms = s['rmse_mean'] * 100
            r2  = s['r2_mean']
            lbl = s['label']

            viol_str = f"{s['viol_mean']:.2f}%" if s['viol_mean'] is not None else '  N/A'

            if e0_mae is not None and e0_mae > 0 and exp_id != 'E0_baseline':
                delta_e0 = (e0_mae - s['mae_mean']) / e0_mae * 100
                vs_e0_str = f"{delta_e0:+.2f}%"
            else:
                vs_e0_str = '—'

            if a1_mae is not None and a1_mae > 0 and exp_id != 'A1_multiscale':
                delta_a1 = (a1_mae - s['mae_mean']) / a1_mae * 100
                vs_a1_str = f"{delta_a1:+.2f}%"
            else:
                vs_a1_str = '—'

            flag = ' ◀' if exp_id.startswith('Exp09') else ''
            print(f"  {lbl:<48} {mae:>8.4f} {std:>6.4f} {rms:>8.4f} {r2:>7.4f} {viol_str:>7} {vs_e0_str:>8} {vs_a1_str:>8}{flag}")

    # 保存汇总 JSON
    summary_fp = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 汇总结果 → {summary_fp}")

    # 结论提示
    print(f"\n{'='*100}")
    print("  决策参考：")
    print("  - Exp09c vs E0 → 纯 PI 对比：同样软单调约束下，多尺度架构是否优于基线？")
    print("  - Exp09c vs A1 → 多尺度架构加物理约束是否有增益？")
    print("  - Exp09a/b vs Exp09c → M2+M4 在多尺度架构上是否有额外增益？")
    print(f"{'='*100}")

    return summary


# ================================================================
# 主入口
# ================================================================
def main():
    total = len(EXPERIMENTS) * len(SUPERVISION_RATIOS) * len(SEEDS)
    print("=" * 90)
    print("  Exp-09：多尺度 CNN Full Stack (A1 + M2 + M4 + M7)")
    print(f"  seeds={SEEDS}  ratios={SUPERVISION_RATIOS}  device={DEVICE}")
    print(f"  总运行次数：{len(EXPERIMENTS)} × {len(SUPERVISION_RATIOS)} × {len(SEEDS)} = {total}")
    print("=" * 90)

    all_results = []
    for exp in EXPERIMENTS:
        print(f"\n{'━'*90}")
        print(f"  实验组：{exp['label']}")
        print(f"{'━'*90}")
        for ratio in SUPERVISION_RATIOS:
            for seed in SEEDS:
                r = run_single(exp, seed, ratio)
                if r is not None:
                    all_results.append(r)

    print(f"\n\n{'='*90}")
    print("  对比汇总（含参照组 A1 / E7）")
    print("=" * 90)

    ref_results = load_reference_results()
    if 'A1_multiscale' not in ref_results:
        print("  [警告] 未找到 ablation_arch_mvp/A1_multiscale/ 参照结果")
    if 'E0_baseline' not in ref_results:
        print("  [警告] 未找到 ablation_single_module/E0_baseline/ 参照结果")
    if 'E7_full_stack' not in ref_results:
        print("  [警告] 未找到 exp07_full_stack/ 参照结果（E7 seeds 可能不同）")

    aggregate_and_print(all_results, ref_results)
    print("\nExp-09 完成！")


if __name__ == '__main__':
    main()
