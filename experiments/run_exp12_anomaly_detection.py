"""
Exp-12：基于物理约束的电芯异常检测
====================================

目标：验证已训练的 PI-MS-CNN-LSTM 模型能否通过物理约束违反信号
      检测电池簇中突发失效的电芯（零额外训练成本）。

实验设计：
    1. 训练 E0 / Exp09c 模型（复用 Exp-11 流程）
    2. 对测试集每块电池：
       - Clean 版本：正常推理（负样本）
       - Fault 版本：在 50% 生命周期处注入电芯失效（正样本）
    3. 计算三维异常分数并评估 ROC-AUC / Detection Rate

异常注入：
    - 方式：跨电芯特征替换（正常电芯特征混入失效供体电芯末期特征）
    - 严重度：mild (alpha=0.2) / moderate (alpha=0.4) / severe (alpha=0.7)
    - 供体：退化最严重的 5 颗电芯

对比模型（7 个，形成完整对比梯队）：
    ── 纯序列基线（无 CNN、无物理）──
    B1_lstm      → 纯 LSTM
    B2_gru       → 纯 GRU
    ── CNN-LSTM 系列 ──
    Eneg1        → CNN-LSTM（无物理约束）
    E0           → PI-CNN-LSTM（软单调，基线）
    E2_mc        → PI-CNN-LSTM + MC Dropout（已知失效模块）
    ── 多尺度系列 ──
    A1_ms_nophys → MS-CNN-LSTM（无物理约束）
    Exp09c ★     → PI-MS-CNN-LSTM（最终方案）

对比维度：
    ① 架构升级价值：  LSTM / GRU < CNN-LSTM < MS-CNN-LSTM
    ② 物理约束价值：  无物理 < 有物理（异常检测信号来源）
    ③ 复杂度陷阱：    +MC Dropout 是否反而降低检测能力

检测信号：
    1. input_zscore    输入特征 z-score
    2. mono_violation  单调违规率（物理约束副产物）
    3. rate_anomaly    退化速率突变
    4. combined        综合分数

验证配置：
    seeds  = [929, 2262, 7]
    ratios = [0.5, 0.3]
    severity = ['mild', 'moderate', 'severe']

结果目录：experiments/exp12_anomaly_detection/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
import numpy as np
import torch
from collections import defaultdict

from train_cross_battery import train_cross_battery_model
from evaluation.anomaly_detection import (
    inject_cell_failure,
    select_donor_batteries,
    compute_anomaly_scores,
    evaluate_detection,
    run_anomaly_detection_for_battery,
)

# ================================================================
# 全局配置
# ================================================================
SEEDS              = [929, 2262, 7]
SUPERVISION_RATIOS = [0.5, 0.3]
SEVERITIES         = ['mild', 'moderate', 'severe']
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp12_anomaly_detection')
WINDOW_SIZE        = 40
N_DONORS           = 5       # 供体电芯数量
FAULT_POSITION     = 0.5     # 故障注入位置（生命周期的 50%）

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 物理约束公共配置
# ================================================================
_PI_COMMON = {
    'enabled': True,
    'monotonic_weight': 0.3,
    'boundary_weight': 0.0,
    'smoothness_weight': 0.0,
    'monotonic_tolerance': 0.005,
    'min_cycle': 300,
}

_NO_PHYSICS = {
    'enabled': False,
    'monotonic_weight': 0.0,
    'boundary_weight': 0.0,
    'smoothness_weight': 0.0,
}


# ================================================================
# 7 个对比模型
# ================================================================
MODELS = {
    # ── 纯序列基线（无 CNN、无物理）──
    'B1_lstm': {
        'label':      'B1: LSTM (no CNN, no physics)',
        'model_type': 'lstm',
        'override': {
            'physics_constraints': _NO_PHYSICS,
        },
    },
    'B2_gru': {
        'label':      'B2: GRU (no CNN, no physics)',
        'model_type': 'gru',
        'override': {
            'physics_constraints': _NO_PHYSICS,
        },
    },

    # ── CNN-LSTM 系列 ──
    'Eneg1_no_physics': {
        'label':      'Eneg1: CNN-LSTM (no physics)',
        'model_type': 'cnn_lstm',
        'override': {
            'physics_constraints': _NO_PHYSICS,
        },
    },
    'E0_baseline': {
        'label':      'E0: PI-CNN-LSTM (baseline)',
        'model_type': 'cnn_lstm',
        'override': {
            'physics_constraints': {**_PI_COMMON},
        },
    },
    'E2_mc_dropout': {
        'label':      'E2: PI-CNN-LSTM + MC Dropout',
        'model_type': 'cnn_lstm_mc',
        'override': {
            'physics_constraints': {**_PI_COMMON},
            'model': {
                'mc_dropout': {'enabled': True, 'rate': 0.1},
            },
        },
    },

    # ── 多尺度系列 ──
    'A1_ms_no_physics': {
        'label':      'A1: MS-CNN-LSTM (no physics)',
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'architecture': {
                'use_multiscale': True,
                'per_window_norm': False,
            },
            'physics_constraints': _NO_PHYSICS,
        },
    },
    'Exp09c_ms_pi': {
        'label':      'Exp09c: PI-MS-CNN-LSTM [Ours]',
        'model_type': 'ms_cnn_lstm_v2',
        'override': {
            'architecture': {
                'use_multiscale': True,
                'per_window_norm': False,
            },
            'physics_constraints': {
                **_PI_COMMON,
                'base_loss_weight': 1.0,
            },
        },
    },
}

# 运行顺序：从弱到强
MODEL_ORDER = [
    'B1_lstm', 'B2_gru',
    'Eneg1_no_physics', 'E0_baseline', 'E2_mc_dropout',
    'A1_ms_no_physics', 'Exp09c_ms_pi',
]


# ================================================================
# 从 data_dict 提取逐电池数据
# ================================================================
def extract_per_battery_data(data_dict):
    """从 data_dict 提取逐电池的特征和目标值。"""
    battery_data = {}

    for split in ['train', 'val', 'test']:
        features = data_dict[f'{split}_features']
        targets = data_dict[f'{split}_targets']
        battery_ids = data_dict[f'{split}_battery_ids']

        for bid in np.unique(battery_ids):
            mask = battery_ids == bid
            if bid not in battery_data:
                battery_data[bid] = {
                    'features': features[mask],
                    'targets': targets[mask],
                    'split': split,
                }
            else:
                # 合并（同一电芯可能跨 split，但实际不会）
                battery_data[bid]['features'] = np.concatenate(
                    [battery_data[bid]['features'], features[mask]])
                battery_data[bid]['targets'] = np.concatenate(
                    [battery_data[bid]['targets'], targets[mask]])

    return battery_data


# ================================================================
# 单次运行
# ================================================================
def run_single(model_key, seed, ratio):
    """训练模型 → 异常注入 → 评分 → 评估"""
    cfg = MODELS[model_key]
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id = f"{model_key}_ratio{ratio_tag}_seed{seed}"
    run_dir = os.path.join(OUTPUT_DIR, model_key, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    # 断点续跑
    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"[RUN] Exp-12  model={model_key}  ratio={ratio}  seed={seed}")
    print(f"      {cfg['label']}")
    print(f"{'='*70}")

    try:
        # 1. 训练模型
        model, results, data_dict = train_cross_battery_model(
            model_type=cfg['model_type'],
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
            config_override=cfg['override'],
        )
        elapsed_train = time.time() - t0
        print(f"  [TRAIN] MAE={results['test_mae']*100:.4f}%  "
              f"R2={results['test_r2']:.4f}  elapsed={elapsed_train/60:.1f}min")

        # 2. 提取逐电池数据
        battery_data = extract_per_battery_data(data_dict)
        test_batteries = data_dict['test_batteries']

        # 训练集特征统计量（用于 z-score 基准）
        train_feat = data_dict['train_features']
        feat_mean = train_feat.mean(axis=0)
        feat_std = train_feat.std(axis=0)

        # 3. 选择供体电芯（从训练集中选退化最严重的）
        train_battery_targets = {}
        for bid in data_dict['train_batteries']:
            mask = data_dict['train_battery_ids'] == bid
            train_battery_targets[bid] = data_dict['train_targets'][mask]

        train_battery_features = {}
        for bid in data_dict['train_batteries']:
            mask = data_dict['train_battery_ids'] == bid
            train_battery_features[bid] = data_dict['train_features'][mask]

        donor_ids = select_donor_batteries(
            train_battery_features, train_battery_targets, n_donors=N_DONORS
        )
        print(f"  [DONORS] {donor_ids}")

        # 4. 对每块测试电池 × 每个严重度 × 每个供体 做异常检测
        all_detection_results = []

        for test_bid in test_batteries:
            test_info = battery_data.get(test_bid)
            if test_info is None or len(test_info['features']) < WINDOW_SIZE + 10:
                continue

            normal_feat = test_info['features']
            T_raw = len(normal_feat)
            fault_cycle = int(T_raw * FAULT_POSITION)

            for severity in SEVERITIES:
                severity_results = []

                for donor_id in donor_ids:
                    donor_info = battery_data.get(donor_id)
                    if donor_info is None or len(donor_info['features']) < 50:
                        continue

                    det_result = run_anomaly_detection_for_battery(
                        model=model,
                        normal_features=normal_feat,
                        donor_features=donor_info['features'],
                        feature_mean=feat_mean,
                        feature_std=feat_std,
                        fault_cycle=fault_cycle,
                        severity=severity,
                        window_size=WINDOW_SIZE,
                        device=DEVICE,
                    )

                    if 'error' in det_result:
                        continue

                    severity_results.append({
                        'test_battery': test_bid,
                        'donor_battery': donor_id,
                        'severity': severity,
                        'metrics': det_result['metrics'],
                    })

                if severity_results:
                    all_detection_results.extend(severity_results)

        # 5. 汇总
        summary = aggregate_detection_results(all_detection_results)
        elapsed_total = time.time() - t0

        record = {
            'run_id':            run_id,
            'model_key':         model_key,
            'label':             cfg['label'],
            'seed':              seed,
            'supervision_ratio': ratio,
            'clean_mae':         float(results['test_mae']),
            'clean_r2':          float(results['test_r2']),
            'n_test_batteries':  len(test_batteries),
            'n_donors':          len(donor_ids),
            'donor_ids':         donor_ids,
            'fault_position':    FAULT_POSITION,
            'elapsed_sec':       elapsed_total,
            'summary':           summary,
            'detail_count':      len(all_detection_results),
        }

        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print_detection_summary(run_id, summary)
        return record

    except Exception as e:
        print(f"\n  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        error_fp = os.path.join(run_dir, 'error.json')
        with open(error_fp, 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e),
                       'traceback': traceback.format_exc()}, f, indent=2)
        return None


# ================================================================
# 汇总
# ================================================================
def aggregate_detection_results(results_list):
    """汇总所有电池×供体×严重度的检测结果"""
    summary = {}

    for severity in SEVERITIES:
        sev_results = [r for r in results_list if r['severity'] == severity]
        if not sev_results:
            continue

        summary[severity] = {}
        for score_key in ['input_zscore', 'mono_violation', 'rate_anomaly', 'combined']:
            aucs = [r['metrics'][score_key]['auc'] for r in sev_results
                    if not np.isnan(r['metrics'][score_key]['auc'])]
            det_rates = [r['metrics'][score_key]['det_rate_fpr5'] for r in sev_results
                        if not np.isnan(r['metrics'][score_key]['det_rate_fpr5'])]
            delays = [r['metrics'][score_key]['det_delay'] for r in sev_results
                     if r['metrics'][score_key]['det_delay'] >= 0]

            summary[severity][score_key] = {
                'auc_mean':        float(np.mean(aucs)) if aucs else float('nan'),
                'auc_std':         float(np.std(aucs)) if aucs else float('nan'),
                'det_rate_mean':   float(np.mean(det_rates)) if det_rates else float('nan'),
                'det_rate_std':    float(np.std(det_rates)) if det_rates else float('nan'),
                'delay_mean':      float(np.mean(delays)) if delays else float('nan'),
                'n_samples':       len(aucs),
            }

    return summary


def print_detection_summary(run_id, summary):
    """打印检测结果摘要"""
    print(f"\n  {'='*60}")
    print(f"  [RESULT] {run_id}")
    print(f"  {'─'*60}")
    print(f"  {'Severity':<10} {'Signal':<16} {'AUC':>8} {'DetRate@5%':>12} {'Delay':>8}")
    print(f"  {'─'*60}")

    for severity in SEVERITIES:
        if severity not in summary:
            continue
        for score_key in ['input_zscore', 'mono_violation', 'rate_anomaly', 'combined']:
            s = summary[severity][score_key]
            print(f"  {severity:<10} {score_key:<16} "
                  f"{s['auc_mean']:.3f}+/-{s['auc_std']:.3f} "
                  f"{s['det_rate_mean']*100:>7.1f}% "
                  f"{s['delay_mean']:>7.1f}")
    print(f"  {'='*60}")


# ================================================================
# 全局汇总 & 对比
# ================================================================
def final_comparison(all_results):
    """E0 vs Exp09c 最终对比"""
    print("\n" + "=" * 70)
    print("Exp-12 Final Comparison: E0 vs Exp09c")
    print("=" * 70)

    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is not None and 'summary' in r:
            groups[r['model_key']][r['supervision_ratio']].append(r)

    for severity in SEVERITIES:
        print(f"\n  --- Severity: {severity} ---")
        print(f"  {'Model':<38} {'Ratio':>6} {'AUC(combined)':>18} {'DetRate@5%':>12}")

        for model_key in MODEL_ORDER:
            for ratio in SUPERVISION_RATIOS:
                runs = groups[model_key][ratio]
                aucs = []
                det_rates = []
                for r in runs:
                    s = r.get('summary', {}).get(severity, {}).get('combined', {})
                    if s and not np.isnan(s.get('auc_mean', float('nan'))):
                        aucs.append(s['auc_mean'])
                        det_rates.append(s['det_rate_mean'])

                if aucs:
                    label = MODELS[model_key]['label']
                    print(f"  {label:<38} r={ratio:.1f}  "
                          f"{np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
                          f"{np.mean(det_rates)*100:.1f}%")


# ================================================================
# Main
# ================================================================
def main():
    print("Exp-12: Cell Anomaly Detection via Physics Constraints")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Ratios: {SUPERVISION_RATIOS}")
    print(f"Severities: {SEVERITIES}")
    print(f"Donors: {N_DONORS}")
    print(f"Fault position: {FAULT_POSITION*100:.0f}% lifecycle")
    print(f"Output: {OUTPUT_DIR}")
    print()

    total_runs = len(MODELS) * len(SUPERVISION_RATIOS) * len(SEEDS)
    print(f"Total training runs: {total_runs}")
    print()

    all_results = []
    run_count = 0

    for model_key in MODEL_ORDER:
        for ratio in SUPERVISION_RATIOS:
            for seed in SEEDS:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}]")
                result = run_single(model_key, seed, ratio)
                all_results.append(result)

    # 最终对比
    final_comparison(all_results)

    # 保存全局汇总
    summary_fp = os.path.join(OUTPUT_DIR, 'final_summary.json')
    summary_data = []
    for r in all_results:
        if r is not None:
            summary_data.append({
                'run_id':   r['run_id'],
                'model':    r['model_key'],
                'ratio':    r['supervision_ratio'],
                'seed':     r['seed'],
                'clean_mae': r['clean_mae'],
                'summary':  r.get('summary', {}),
            })

    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nFinal summary saved: {summary_fp}")


if __name__ == '__main__':
    main()
