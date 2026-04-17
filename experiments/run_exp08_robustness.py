"""
Exp-08：M8 特征缺失鲁棒性验证
================================
在测试集上对 Baseline 和 Full Stack 模型分别施加三类扰动，
量化两者在传感器故障/噪声/漂移场景下的性能衰减差异。

对比模型：
    baseline       → model_type='cnn_lstm'           (configs/models/cnn_lstm_config.json)
    full_stack     → model_type='cnn_lstm_full_stack' (configs/models/cnn_lstm_full_stack_config.json)

扰动场景（M8 三类）：
    Scene A  随机特征置零：n_mask ∈ {1, 2, 3}
    Scene B  高斯噪声：    σ ∈ {0.01, 0.05, 0.10}
    Scene C  系统性漂移：  drift ∈ {+0.05, +0.10, +0.20}

验证配置：
    supervision_ratio = 0.5   （中等稀疏度，最具代表性）
    seeds             = [42, 123, 456, 789, 1024]
    总运行次数        = 2 (模型) × 5 (seeds) = 10  训练 + 鲁棒性评估

结果目录：experiments/exp08_robustness/

论文用途：
    - 鲁棒性分析（Section 4.5）
    - Full Stack 方法在传感器故障下的稳定性对比 Baseline
    - 图：各扰动强度下 MAE 对比折线图
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
import numpy as np
import torch

from train_cross_battery import train_cross_battery_model
from evaluation.robustness_eval import evaluate_robustness, print_robustness_report

# ===== 配置 =====
SEEDS             = [42, 123, 456, 789, 1024]
SUPERVISION_RATIO = 0.5          # 固定中等稀疏度
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), 'exp08_robustness')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    'baseline':   'cnn_lstm',
    'full_stack': 'cnn_lstm_full_stack',
}


def run_single(model_key: str, seed: int) -> dict | None:
    """训练一个模型并评估鲁棒性。"""
    model_type  = MODELS[model_key]
    run_id      = f"{model_key}_seed{seed}"
    run_dir     = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}  clean_MAE={r['clean_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"[RUN] Exp-08  model={model_key}  seed={seed}  device={DEVICE}")
    print(f"{'='*65}")

    try:
        model, results, test_loader = train_cross_battery_model(
            model_type=model_type,
            seed=seed,
            device=DEVICE,
            supervision_ratio=SUPERVISION_RATIO,
            supervision_seed=None,
        )
        elapsed_train = time.time() - t0

        print(f"\n  [TRAIN] MAE={results['test_mae']*100:.4f}%  "
              f"RMSE={results['test_rmse']*100:.4f}%  "
              f"elapsed={elapsed_train/60:.1f}min")

        # M8：鲁棒性评估
        t1 = time.time()
        rob = evaluate_robustness(model, test_loader, device=DEVICE, seed=seed)
        print_robustness_report(rob, model_name=f"{model_key} seed={seed}")
        elapsed_rob = time.time() - t1

        # ── 展平鲁棒性结果到 record ─────────────────────────
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
            'exp':               'exp08_robustness',
            'model_key':         model_key,
            'model_type':        model_type,
            'seed':              seed,
            'supervision_ratio': SUPERVISION_RATIO,
            'clean_mae':         float(results['test_mae']),
            'clean_rmse':        float(results['test_rmse']),
            'clean_r2':          float(results['test_r2']),
            'elapsed_train_sec': elapsed_train,
            'elapsed_rob_sec':   elapsed_rob,
            **flat_rob,
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"  [DONE] {run_id}  saved → {result_file}")
        return record

    except Exception as e:
        print(f"\n  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({'run_id': run_id, 'error': str(e)}, f, indent=2)
        return None


def aggregate(all_results: list) -> None:
    """汇总并打印两个模型的平均鲁棒性对比。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        if r is not None:
            groups[r['model_key']].append(r)

    # 构建对比表
    perturb_keys = []
    for r in all_results:
        if r is not None:
            perturb_keys = [k for k in r.keys()
                            if k.startswith(('scene_a_', 'scene_b_', 'scene_c_'))
                            and k.endswith('_mae')]
            break

    summary = {}
    print(f"\n{'='*65}")
    print("Exp-08 汇总（M8 鲁棒性对比：Baseline vs Full Stack）")
    print(f"  supervision_ratio={SUPERVISION_RATIO}  seeds={SEEDS}")
    print(f"{'='*65}")

    for model_key in ['baseline', 'full_stack']:
        runs = groups.get(model_key, [])
        if not runs:
            continue

        clean_maes = [r['clean_mae'] for r in runs]
        entry = {
            'model_key':   model_key,
            'n_runs':      len(runs),
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
            # scene label
            parts = pk.split('_')  # e.g. ['scene', 'a', 'n', 'mask', '1', 'mae']
            label = '_'.join(parts[:-1])
            delta_vals = [r.get(pk.replace('_mae', '_delta_mae'), float('nan')) for r in runs]
            delta_vals = [v for v in delta_vals if not np.isnan(v)]
            mean_delta = float(np.mean(delta_vals)) * 100 if delta_vals else float('nan')
            print(f"    {label:<40} MAE={mean_mae:.4f}%  Δ={mean_delta:+.4f}%")

        entry['perturb_summary'] = {
            pk: {
                'mae_mean': float(np.mean([r[pk] for r in runs if pk in r])),
            }
            for pk in perturb_keys
        }
        summary[model_key] = entry

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'exp': 'exp08_robustness', 'by_model': summary}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n  保存至: {out}")


if __name__ == '__main__':
    total = len(MODELS) * len(SEEDS)
    print(f"Exp-08: M8 鲁棒性验证  |  device={DEVICE}")
    print(f"  models={list(MODELS.keys())}  seeds={SEEDS}")
    print(f"  supervision_ratio={SUPERVISION_RATIO}")
    print(f"  总运行次数: {total}\n")

    all_results = []
    for model_key in MODELS:
        for seed in SEEDS:
            r = run_single(model_key, seed)
            all_results.append(r)

    aggregate(all_results)
    print("\nExp-08 完成！")
