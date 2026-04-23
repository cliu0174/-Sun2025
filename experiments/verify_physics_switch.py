"""
验证物理约束开关对 Baseline 的影响
====================================
2×2×2 = 8 次运行：
  physics: ON / OFF
  ratio:   1.0 / 0.5
  seed:    42  / 123
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from train_cross_battery import train_cross_battery_model

SEEDS   = [42, 123]
RATIOS  = [1.0, 0.5]
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT  = os.path.join(os.path.dirname(__file__), 'verify_physics_switch')
os.makedirs(OUTPUT, exist_ok=True)

CONFIGS = {
    'physics_ON': {
        'physics_constraints': {
            'enabled': True,
            'monotonic_weight': 0.1,
            'boundary_weight': 0.05,
        }
    },
    'physics_OFF': {
        'physics_constraints': {
            'enabled': False,
        }
    },
}

results = []

for physics_label, override in CONFIGS.items():
    for ratio in RATIOS:
        for seed in SEEDS:
            tag = f"{physics_label}_ratio{ratio}_seed{seed}"
            fp  = os.path.join(OUTPUT, f"{tag}.json")

            if os.path.exists(fp):
                r = json.load(open(fp))
                print(f"[SKIP] {tag}  MAE={r['mae']*100:.4f}%")
                results.append(r)
                continue

            print(f"\n[RUN]  {tag}")
            t0 = time.time()
            _, res, _ = train_cross_battery_model(
                model_type='cnn_lstm',
                seed=seed,
                device=DEVICE,
                supervision_ratio=ratio,
                supervision_seed=None,
                config_override=override,
            )
            record = {
                'tag': tag, 'physics': physics_label,
                'ratio': ratio, 'seed': seed,
                'mae':  float(res['test_mae']),
                'rmse': float(res['test_rmse']),
                'r2':   float(res['test_r2']),
                'elapsed': time.time() - t0,
            }
            json.dump(record, open(fp, 'w'), indent=2)
            print(f"[DONE] MAE={record['mae']*100:.4f}%  RMSE={record['rmse']*100:.4f}%  R2={record['r2']:.4f}")
            results.append(record)

# 汇总打印
print("\n" + "="*70)
print(f"{'配置':<22} {'ratio':>6} {'seed':>6} {'MAE%':>8} {'RMSE%':>8} {'R2':>7}")
print("-"*70)
for r in results:
    print(f"{r['physics']:<22} {r['ratio']:>6.1f} {r['seed']:>6}  "
          f"{r['mae']*100:>7.4f}  {r['rmse']*100:>7.4f}  {r['r2']:>7.4f}")

# 按 physics × ratio 统计均值
print("\n" + "="*70)
print("均值对比（2 seeds）")
print("-"*70)
print(f"{'配置':<22} {'ratio':>6}  {'MAE mean%':>10} {'RMSE mean%':>11}")
print("-"*70)
for physics_label in ['physics_ON', 'physics_OFF']:
    for ratio in RATIOS:
        subset = [r for r in results if r['physics'] == physics_label and r['ratio'] == ratio]
        if subset:
            mae_m  = np.mean([r['mae']  for r in subset]) * 100
            rmse_m = np.mean([r['rmse'] for r in subset]) * 100
            print(f"{physics_label:<22} {ratio:>6.1f}  {mae_m:>10.4f}  {rmse_m:>11.4f}")
