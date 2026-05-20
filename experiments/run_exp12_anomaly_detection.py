"""
Exp-12：基于物理约束的电芯异常检测
====================================

目标：验证已训练的 PI-MS-CNN-LSTM 模型能否通过物理约束违反信号
      检测电池簇中不同类型的退化异常（零额外训练成本）。

退化场景（5 种）：
    sudden_aging    — 单电芯突发加速老化（跨电芯特征替换）       ← 原始场景
    knee_point      — 容量拐点突破（自身末期特征，无需供体）
    imbalance       — 簇内不均衡加剧（alpha 线性增长的渐进漂移）
    li_plating      — 析锂台阶突降（阶跃 + 持续混合）
    resistance_rise — 内阻渐进增长（特征向末期二次漂移，无需供体）

对比模型（7 个）：
    B1_lstm / B2_gru / Eneg1 / E0 / E2_mc / A1_ms_nophys / Exp09c★

检测信号（4 维）：
    input_zscore / mono_violation / rate_anomaly / combined

验证配置：
    seeds     = [929, 2262, 7]
    ratios    = [0.5, 0.3]
    severities = ['mild', 'moderate', 'severe']

结果目录：experiments/exp12_anomaly_detection/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import pickle
import traceback
import numpy as np
import torch
from collections import defaultdict

from train_cross_battery import train_cross_battery_model
from models.model_factory import UnifiedModelWrapper
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
N_DONORS           = 5
FAULT_POSITION     = 0.5   # 故障注入位置（生命周期的 50%）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 5 种退化场景
#   needs_donor: False → 无需供体（knee_point / resistance_rise）
#                True  → 需要供体（其余三种）
# ================================================================
SCENARIOS = {
    'sudden_aging': {
        'label':       '(1) Single-cell sudden aging',
        'needs_donor': True,
        'desc':        'Cross-cell feature mixing at fault cycle (original)',
    },
    'knee_point': {
        'label':       '(2) Capacity knee-point',
        'needs_donor': False,
        'desc':        'Self terminal-feature mixing; slope suddenly steepens',
    },
    'imbalance': {
        'label':       '(3) Inter-cell imbalance escalation',
        'needs_donor': True,
        'desc':        'Alpha linearly grows → gradual cluster-level drift',
    },
    'li_plating': {
        'label':       '(4) Lithium plating (step-drop)',
        'needs_donor': True,
        'desc':        'Large step at fault_cycle then continued mixing',
    },
    'resistance_rise': {
        'label':       '(5) Internal resistance rise',
        'needs_donor': False,
        'desc':        'Quadratic drift toward self terminal features',
    },
}

SCENARIO_ORDER = ['sudden_aging', 'knee_point', 'imbalance', 'li_plating', 'resistance_rise']

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
# 4 个对比模型（2×2 对比矩阵：架构 × 物理约束）
#
#              无物理约束      有物理约束
#  单路 CNN     Eneg1           E0
#  多尺度 CNN   A1              Exp09c★
#
# 覆盖核心问题：
#   ① 物理约束是否帮助检测？   Eneg1 vs E0  /  A1 vs Exp09c
#   ② 多尺度架构是否帮助检测？  Eneg1 vs A1  /  E0 vs Exp09c
# ================================================================
MODELS = {
    'Eneg1_no_physics': {
        'label':      'Eneg1: CNN-LSTM (no physics)',
        'model_type': 'cnn_lstm',
        'override':   {'physics_constraints': _NO_PHYSICS},
    },
    'E0_baseline': {
        'label':      'E0: PI-CNN-LSTM (baseline)',
        'model_type': 'cnn_lstm',
        'override':   {'physics_constraints': {**_PI_COMMON}},
    },
    'A1_ms_no_physics': {
        'label':      'A1: MS-CNN-LSTM (no physics)',
        'model_type': 'ms_cnn_lstm_v2',
        'override':   {
            'architecture': {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': _NO_PHYSICS,
        },
    },
    'Exp09c_ms_pi': {
        'label':      'Exp09c: PI-MS-CNN-LSTM [Ours]',
        'model_type': 'ms_cnn_lstm_v2',
        'override':   {
            'architecture': {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': {**_PI_COMMON, 'base_loss_weight': 1.0},
        },
    },
}

MODEL_ORDER = ['Eneg1_no_physics', 'E0_baseline', 'A1_ms_no_physics', 'Exp09c_ms_pi']


# ================================================================
# 辅助：从 data_dict 提取逐电池数据
# ================================================================
def extract_per_battery_data(data_dict):
    battery_data = {}
    for split in ['train', 'val', 'test']:
        features    = data_dict[f'{split}_features']
        targets     = data_dict[f'{split}_targets']
        battery_ids = data_dict[f'{split}_battery_ids']
        for bid in np.unique(battery_ids):
            mask = battery_ids == bid
            if bid not in battery_data:
                battery_data[bid] = {
                    'features': features[mask],
                    'targets':  targets[mask],
                    'split':    split,
                }
            else:
                battery_data[bid]['features'] = np.concatenate(
                    [battery_data[bid]['features'], features[mask]])
                battery_data[bid]['targets'] = np.concatenate(
                    [battery_data[bid]['targets'], targets[mask]])
    return battery_data


# ================================================================
# 单次运行（一个 model × ratio × seed 组合）
# 训练结果缓存：model.pth + data_cache.pkl 保存在 run_dir 下，
# 下次运行时直接加载，跳过训练（仅重跑异常检测推理部分）。
# ================================================================
def run_single(model_key, seed, ratio):
    cfg          = MODELS[model_key]
    ratio_tag    = f"{ratio:.1f}".replace('.', 'p')
    run_id       = f"{model_key}_ratio{ratio_tag}_seed{seed}"
    run_dir      = os.path.join(OUTPUT_DIR, model_key, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp    = os.path.join(run_dir, 'result.json')
    ckpt_fp      = os.path.join(run_dir, 'model.pth')        # ← 模型权重缓存
    datacache_fp = os.path.join(run_dir, 'data_cache.pkl')   # ← 数据统计量缓存

    # 完整结果已存在 → 全部跳过
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
        # ── 1. 训练或加载缓存 ──────────────────────────────────────
        if os.path.exists(ckpt_fp) and os.path.exists(datacache_fp):
            # 已有缓存：跳过训练，直接加载
            print(f"  [CACHE] 发现已有 checkpoint，跳过训练")
            wrapper = UnifiedModelWrapper.load_checkpoint(ckpt_fp, device=DEVICE)
            model   = wrapper          # run_anomaly_detection_for_battery 接受 wrapper
            with open(datacache_fp, 'rb') as f:
                cache = pickle.load(f)
            results = cache['results']
            data_dict = cache['data_dict']
            print(f"  [CACHE] MAE={results['test_mae']*100:.4f}%  R2={results['test_r2']:.4f}")
        else:
            # 首次运行：正常训练
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

            # 保存 checkpoint
            model.save_checkpoint(ckpt_fp, epoch=getattr(model, 'best_epoch', 0))

            # 保存数据统计量（不保存原始特征数组，只保存元信息）
            cache = {
                'results':   results,
                'data_dict': {k: v for k, v in data_dict.items()
                              if not k.endswith('_features')   # 不缓存大型特征矩阵
                              and not k.endswith('_targets')
                              and not k.endswith('_battery_ids')},
            }
            # 特征矩阵单独存（numpy，压缩）
            for key in ['train_features', 'train_targets', 'train_battery_ids',
                        'val_features',   'val_targets',   'val_battery_ids',
                        'test_features',  'test_targets',  'test_battery_ids']:
                if key in data_dict:
                    cache['data_dict'][key] = data_dict[key]
            with open(datacache_fp, 'wb') as f:
                pickle.dump(cache, f, protocol=4)
            print(f"  [CACHE] checkpoint 和数据缓存已保存 → {run_dir}")

        # 2. 提取逐电池数据
        battery_data  = extract_per_battery_data(data_dict)
        test_batteries = data_dict['test_batteries']

        train_feat = data_dict['train_features']
        feat_mean  = train_feat.mean(axis=0)
        feat_std   = train_feat.std(axis=0)

        # 3. 选择供体电芯（从训练集中选退化最严重的）
        train_battery_targets  = {bid: data_dict['train_targets'][data_dict['train_battery_ids'] == bid]
                                   for bid in data_dict['train_batteries']}
        train_battery_features = {bid: data_dict['train_features'][data_dict['train_battery_ids'] == bid]
                                   for bid in data_dict['train_batteries']}
        donor_ids = select_donor_batteries(train_battery_features, train_battery_targets, n_donors=N_DONORS)
        print(f"  [DONORS] {donor_ids}")

        # 4. 逐电池 × 逐场景 × 逐严重度 做异常检测
        all_detection_results = []

        for test_bid in test_batteries:
            test_info = battery_data.get(test_bid)
            if test_info is None or len(test_info['features']) < WINDOW_SIZE + 10:
                continue

            normal_feat = test_info['features']
            T_raw       = len(normal_feat)
            fault_cycle = int(T_raw * FAULT_POSITION)

            for scenario_key, scenario_cfg in SCENARIOS.items():
                needs_donor = scenario_cfg['needs_donor']
                # 需要供体的场景遍历所有供体；不需要的只跑一次（donor_id=None）
                donor_iter = donor_ids if needs_donor else [None]

                for donor_id in donor_iter:
                    donor_feat = battery_data[donor_id]['features'] if donor_id is not None else None

                    for severity in SEVERITIES:
                        det_result = run_anomaly_detection_for_battery(
                            model=model,
                            normal_features=normal_feat,
                            donor_features=donor_feat,
                            feature_mean=feat_mean,
                            feature_std=feat_std,
                            fault_cycle=fault_cycle,
                            severity=severity,
                            scenario=scenario_key,
                            window_size=WINDOW_SIZE,
                            device=DEVICE,
                        )

                        if 'error' in det_result:
                            continue

                        all_detection_results.append({
                            'test_battery': test_bid,
                            'donor_battery': donor_id,
                            'scenario': scenario_key,
                            'severity': severity,
                            'metrics': det_result['metrics'],
                        })

        # 5. 汇总
        summary       = aggregate_detection_results(all_detection_results)
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
# 汇总：按 scenario × severity × signal 维度聚合
# ================================================================
def aggregate_detection_results(results_list):
    """汇总所有电池 × 供体 × 场景 × 严重度的检测结果"""
    summary = {}

    for scenario_key in SCENARIO_ORDER:
        summary[scenario_key] = {}
        for severity in SEVERITIES:
            filtered = [r for r in results_list
                        if r['scenario'] == scenario_key and r['severity'] == severity]
            if not filtered:
                continue

            summary[scenario_key][severity] = {}
            for score_key in ['input_zscore', 'mono_violation', 'rate_anomaly', 'combined']:
                aucs      = [r['metrics'][score_key]['auc']
                             for r in filtered if not np.isnan(r['metrics'][score_key]['auc'])]
                det_rates = [r['metrics'][score_key]['det_rate_fpr5']
                             for r in filtered if not np.isnan(r['metrics'][score_key]['det_rate_fpr5'])]
                delays    = [r['metrics'][score_key]['det_delay']
                             for r in filtered if r['metrics'][score_key]['det_delay'] >= 0]

                summary[scenario_key][severity][score_key] = {
                    'auc_mean':      float(np.mean(aucs))      if aucs      else float('nan'),
                    'auc_std':       float(np.std(aucs))       if aucs      else float('nan'),
                    'det_rate_mean': float(np.mean(det_rates)) if det_rates else float('nan'),
                    'det_rate_std':  float(np.std(det_rates))  if det_rates else float('nan'),
                    'delay_mean':    float(np.mean(delays))    if delays    else float('nan'),
                    'n_samples':     len(aucs),
                }

    return summary


def print_detection_summary(run_id, summary):
    print(f"\n  {'='*70}")
    print(f"  [RESULT] {run_id}")
    header = f"  {'Scenario':<18} {'Sev':<10} {'Signal':<16} {'AUC':>8} {'Det@5%':>8} {'Delay':>7}"
    print(f"  {'─'*70}")
    print(header)
    print(f"  {'─'*70}")

    for scenario_key in SCENARIO_ORDER:
        if scenario_key not in summary:
            continue
        for severity in SEVERITIES:
            if severity not in summary[scenario_key]:
                continue
            for score_key in ['combined']:   # 只打 combined，简洁
                s = summary[scenario_key][severity][score_key]
                print(f"  {scenario_key:<18} {severity:<10} {score_key:<16} "
                      f"{s['auc_mean']:.3f}±{s['auc_std']:.3f}  "
                      f"{s['det_rate_mean']*100:>5.1f}%  "
                      f"{s['delay_mean']:>6.1f}")
    print(f"  {'='*70}")


# ================================================================
# 全局汇总：跨 seeds/ratios，对比 5 场景 × 7 模型
# ================================================================
def final_comparison(all_results):
    print("\n" + "=" * 75)
    print("Exp-12 Final Comparison  (combined score, mean over seeds)")
    print("=" * 75)

    # {model_key: {ratio: [records]}}
    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is not None and 'summary' in r:
            groups[r['model_key']][r['supervision_ratio']].append(r)

    for scenario_key in SCENARIO_ORDER:
        scenario_label = SCENARIOS[scenario_key]['label']
        print(f"\n  ── {scenario_label} ──")
        print(f"  {'Model':<40} {'Ratio':>5}  {'mild AUC':>9} {'mod AUC':>9} {'sev AUC':>9}  {'Det@5%(sev)':>12}")

        for model_key in MODEL_ORDER:
            model_label = MODELS[model_key]['label']
            for ratio in SUPERVISION_RATIOS:
                runs = groups[model_key][ratio]
                if not runs:
                    continue

                row_aucs = {}
                row_det  = {}
                for sev in SEVERITIES:
                    aucs     = []
                    det_rates = []
                    for r in runs:
                        s = (r.get('summary', {})
                               .get(scenario_key, {})
                               .get(sev, {})
                               .get('combined', {}))
                        if s and not np.isnan(s.get('auc_mean', float('nan'))):
                            aucs.append(s['auc_mean'])
                            det_rates.append(s['det_rate_mean'])
                    row_aucs[sev] = np.mean(aucs) if aucs else float('nan')
                    row_det[sev]  = np.mean(det_rates) * 100 if det_rates else float('nan')

                print(f"  {model_label:<40} r={ratio:.1f}  "
                      f"{row_aucs['mild']:>8.3f}  "
                      f"{row_aucs['moderate']:>8.3f}  "
                      f"{row_aucs['severe']:>8.3f}  "
                      f"{row_det['severe']:>10.1f}%")


# ================================================================
# Main
# ================================================================
def main():
    print("Exp-12: Cell Anomaly Detection via Physics Constraints")
    print(f"Device:     {DEVICE}")
    print(f"Seeds:      {SEEDS}")
    print(f"Ratios:     {SUPERVISION_RATIOS}")
    print(f"Severities: {SEVERITIES}")
    print(f"Scenarios:  {SCENARIO_ORDER}")
    print(f"Donors:     {N_DONORS}")
    print(f"Fault pos:  {FAULT_POSITION*100:.0f}% lifecycle")
    print(f"Output:     {OUTPUT_DIR}")

    total_runs = len(MODELS) * len(SUPERVISION_RATIOS) * len(SEEDS)
    print(f"\nTotal training runs: {total_runs}  "
          f"(each run covers all {len(SCENARIOS)} scenarios × {len(SEVERITIES)} severities)\n")

    all_results = []
    run_count   = 0

    for model_key in MODEL_ORDER:
        for ratio in SUPERVISION_RATIOS:
            for seed in SEEDS:
                run_count += 1
                print(f"\n[{run_count}/{total_runs}]")
                result = run_single(model_key, seed, ratio)
                all_results.append(result)

    final_comparison(all_results)

    # 保存全局汇总
    summary_fp   = os.path.join(OUTPUT_DIR, 'final_summary.json')
    summary_data = []
    for r in all_results:
        if r is not None:
            summary_data.append({
                'run_id':    r['run_id'],
                'model':     r['model_key'],
                'ratio':     r['supervision_ratio'],
                'seed':      r['seed'],
                'clean_mae': r['clean_mae'],
                'summary':   r.get('summary', {}),
            })

    with open(summary_fp, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nFinal summary saved: {summary_fp}")


if __name__ == '__main__':
    main()
