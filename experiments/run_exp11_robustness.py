"""
Exp-11：PI-MS-CNN-LSTM 鲁棒性验证（M8 特征缺失）
==================================================

目标：对比 E0（PI-CNN-LSTM）与 Exp09c（PI-MS-CNN-LSTM）在传感器故障/
      噪声/漂移场景下的性能衰减，验证多尺度架构的鲁棒性优势。

对比模型：
    E0       → cnn_lstm + 软单调 w=0.3（基线）
    Exp09c   → ms_cnn_lstm_v2 + 软单调 w=0.3（最终方案）

扰动场景（M8 三类）：
    Scene A  随机特征置零：n_mask ∈ {1, 2, 3}
    Scene B  高斯噪声：    σ ∈ {0.01, 0.05, 0.10}
    Scene C  系统性漂移：  drift ∈ {+0.05, +0.10, +0.20}

验证配置：
    supervision_ratios = [0.5, 0.3]  （核心部分监督场景）
    seeds              = [929, 2262, 7]
    总运行次数          = 2 (模型) × 2 (ratios) × 3 (seeds) = 12 训练 + 鲁棒性评估

结果目录：experiments/exp11_robustness/

论文用途：
    - 鲁棒性分析章节
    - 多尺度架构 vs 基线在传感器故障下的稳定性对比
    - 图：各扰动强度下 MAE 对比折线图
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
from torch.utils.data import TensorDataset, DataLoader

from train_cross_battery import train_cross_battery_model
from evaluation.robustness_eval import evaluate_robustness, print_robustness_report

# ================================================================
# 全局配置
# ================================================================
SEEDS              = [929, 2262, 7]
SUPERVISION_RATIOS = [0.5, 0.3]
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'exp11_robustness')
WINDOW_SIZE        = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 对比模型配置
# ================================================================
MODELS = {
    'E0_baseline': {
        'label':      'E0: PI-CNN-LSTM（基线）',
        'model_type': 'cnn_lstm',
        'override': {
            'physics_constraints': {
                'enabled': True,
                'monotonic_weight': 0.3,
                'boundary_weight': 0.0,
                'smoothness_weight': 0.0,
                'monotonic_tolerance': 0.005,
                'min_cycle': 300,
            },
        },
    },
    'Exp09c_ms_pi': {
        'label':      'Exp09c: PI-MS-CNN-LSTM（最终方案）',
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
    },
}


# ================================================================
# 从 data_dict 构建 test DataLoader
# ================================================================
def build_test_loader(data_dict, window_size=40, batch_size=512):
    """从 data_dict 构建窗口化的 test DataLoader。"""
    test_feat = np.array(data_dict['test_features'])
    test_targ = np.array(data_dict['test_targets'])

    if window_size > 1 and len(test_feat) >= window_size:
        wf, wt = [], []
        for i in range(len(test_feat) - window_size + 1):
            wf.append(test_feat[i:i + window_size])
            wt.append(test_targ[i + window_size - 1])
        x_np = np.array(wf)
        y_np = np.array(wt).squeeze()
    else:
        x_np = test_feat[:, np.newaxis, :]
        y_np = test_targ.squeeze()

    dataset = TensorDataset(
        torch.FloatTensor(x_np),
        torch.FloatTensor(y_np),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ================================================================
# 单次运行（带断点续跑）
# ================================================================
def run_single(model_key: str, seed: int, ratio: float) -> Optional[dict]:
    cfg       = MODELS[model_key]
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id    = f"{model_key}_ratio{ratio_tag}_seed{seed}"
    run_dir   = os.path.join(OUTPUT_DIR, model_key, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id:<55} clean_MAE={r['clean_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"[RUN] Exp-11  model={model_key}  ratio={ratio}  seed={seed}  device={DEVICE}")
    print(f"      {cfg['label']}")
    print(f"{'='*70}")

    try:
        model, results, data_dict = train_cross_battery_model(
            model_type=cfg['model_type'],
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
            config_override=cfg['override'],
        )
        elapsed_train = time.time() - t0

        print(f"\n  [TRAIN] MAE={results['test_mae']*100:.4f}%  "
              f"RMSE={results['test_rmse']*100:.4f}%  "
              f"R2={results['test_r2']:.4f}  "
              f"elapsed={elapsed_train/60:.1f}min")

        # 构建 test_loader 用于鲁棒性评估
        test_loader = build_test_loader(data_dict, window_size=WINDOW_SIZE)

        # M8：鲁棒性评估
        t1 = time.time()
        rob = evaluate_robustness(model, test_loader, device=DEVICE, seed=seed)
        print_robustness_report(rob, model_name=f"{model_key} ratio={ratio} seed={seed}")
        elapsed_rob = time.time() - t1

        # 展平鲁棒性结果
        def _flatten(prefix, d):
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    flat.update(_flatten(f"{prefix}_{k}", v))
                elif isinstance(v, (int, float)) and not np.isnan(v):
                    flat[f"{prefix}_{k}"] = float(v)
            return flat

        flat_rob = {}
        flat_rob.update(_flatten('scene_a', rob.get('scene_a', {})))
        flat_rob.update(_flatten('scene_b', rob.get('scene_b', {})))
        flat_rob.update(_flatten('scene_c', rob.get('scene_c', {})))

        record = {
            'run_id':            run_id,
            'exp':               'exp11_robustness',
            'model_key':         model_key,
            'model_type':        cfg['model_type'],
            'label':             cfg['label'],
            'seed':              seed,
            'supervision_ratio': ratio,
            'clean_mae':         float(results['test_mae']),
            'clean_rmse':        float(results['test_rmse']),
            'clean_r2':          float(results['test_r2']),
            'elapsed_train_sec': elapsed_train,
            'elapsed_rob_sec':   elapsed_rob,
            **flat_rob,
        }

        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"  [DONE] {run_id}  saved → {result_fp}")
        return record

    except Exception as e:
        print(f"\n  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e),
                       'traceback': traceback.format_exc()},
                      f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 汇总 & 对比打印
# ================================================================
def aggregate(all_results: list) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is not None:
            tag = f"{r['supervision_ratio']:.1f}".replace('.', 'p')
            groups[r['model_key']][tag].append(r)

    # 收集扰动指标 key
    perturb_keys = []
    for r in all_results:
        if r is not None:
            perturb_keys = [k for k in r.keys()
                            if k.startswith(('scene_a_', 'scene_b_', 'scene_c_'))
                            and k.endswith('_mae')]
            break

    summary = {}
    for ratio in SUPERVISION_RATIOS:
        tag = f"{ratio:.1f}".replace('.', 'p')
        print(f"\n{'='*80}")
        print(f"  Exp-11 鲁棒性对比  ratio={ratio}")
        print(f"  seeds={SEEDS}")
        print(f"{'='*80}")

        ratio_summary = {}
        for model_key in ['E0_baseline', 'Exp09c_ms_pi']:
            runs = groups.get(model_key, {}).get(tag, [])
            if not runs:
                continue

            clean_maes = [r['clean_mae'] for r in runs]
            entry = {
                'model_key':      model_key,
                'label':          runs[0]['label'],
                'n_runs':         len(runs),
                'clean_mae_mean': float(np.mean(clean_maes)),
                'clean_mae_std':  float(np.std(clean_maes, ddof=1) if len(clean_maes) > 1 else 0.0),
            }

            print(f"\n  [{model_key}]  clean MAE = "
                  f"{entry['clean_mae_mean']*100:.4f}% ± {entry['clean_mae_std']*100:.4f}%")

            for pk in perturb_keys:
                vals = [r[pk] for r in runs if pk in r]
                if not vals:
                    continue
                mean_mae = float(np.mean(vals)) * 100
                parts = pk.split('_')
                label = '_'.join(parts[:-1])
                delta_key = pk.replace('_mae', '_delta_mae')
                delta_vals = [r.get(delta_key, float('nan')) for r in runs]
                delta_vals = [v for v in delta_vals if not np.isnan(v)]
                mean_delta = float(np.mean(delta_vals)) * 100 if delta_vals else float('nan')
                print(f"    {label:<40} MAE={mean_mae:.4f}%  Δ={mean_delta:+.4f}%")

            entry['perturb_summary'] = {
                pk: {'mae_mean': float(np.mean([r[pk] for r in runs if pk in r]))}
                for pk in perturb_keys
                if any(pk in r for r in runs)
            }
            ratio_summary[model_key] = entry

        summary[tag] = ratio_summary

    out = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp11_robustness', 'by_ratio': summary}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n  保存至: {out}")


# ================================================================
# 主入口
# ================================================================
def main():
    total = len(MODELS) * len(SUPERVISION_RATIOS) * len(SEEDS)
    print("=" * 90)
    print("  Exp-11：PI-MS-CNN-LSTM 鲁棒性验证（M8 特征缺失）")
    print(f"  models={list(MODELS.keys())}")
    print(f"  ratios={SUPERVISION_RATIOS}  seeds={SEEDS}  device={DEVICE}")
    print(f"  总运行次数: {total}")
    print("=" * 90)

    all_results = []
    for model_key in MODELS:
        for ratio in SUPERVISION_RATIOS:
            for seed in SEEDS:
                r = run_single(model_key, seed, ratio)
                all_results.append(r)

    aggregate(all_results)
    print("\nExp-11 完成！")


if __name__ == '__main__':
    main()
