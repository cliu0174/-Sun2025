"""
Exp-02：baseline-v2.2 + M2 MC Dropout + M7 置信区间评估
=========================================================
对比组：
    - baseline-v2.2 (普通 Dropout，无不确定性估计)
    - Exp-02        (MC Dropout，50 次采样，PICP/MPIW 评估)

验证配置：
    - supervision_ratio = 1.0（全监督）
    - seeds = [42, 123, 456, 789, 1024]
    - 5 次运行，取 mean ± std

结果目录：experiments/exp02_mc_dropout/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_cross_battery import (
    train_cross_battery_model,
    prepare_cross_battery_data,
    create_dataloaders,
    load_all_batteries,
    split_batteries,
)
from models import ConfigLoader
from models.modules.mc_dropout import mc_predict
from evaluation import uncertainty_report

# ===== 配置 =====
SEEDS             = [42, 123, 456, 789, 1024]
SUPERVISION_RATIO = 1.0
MODEL_TYPE        = 'cnn_lstm_mc'
N_MC_SAMPLES      = 50
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR        = os.path.join(os.path.dirname(__file__), 'exp02_mc_dropout')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single(seed):
    run_id      = f"seed{seed}"
    run_dir     = os.path.join(OUTPUT_DIR, run_id)
    result_file = os.path.join(run_dir, 'result.json')

    if os.path.exists(result_file):
        with open(result_file, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id}  MAE={r['test_mae']*100:.4f}%  PICP={r.get('picp', 'N/A')}")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n{'='*60}\n[RUN] Exp-02  seed={seed}  device={DEVICE}\n{'='*60}")

    try:
        # 1. 训练
        wrapper, results, data_dict = train_cross_battery_model(
            model_type=MODEL_TYPE,
            seed=seed,
            device=DEVICE,
            supervision_ratio=SUPERVISION_RATIO,
            supervision_seed=None,
        )
        model = wrapper.model.to(DEVICE)

        # 2. 重建测试数据加载器（用于 MC 评估）
        config = ConfigLoader.load_model_config(MODEL_TYPE)
        window_size = config.get('data', {}).get('window_size', 40)
        batch_size  = config['training']['batch_size']

        _, _, test_loader, _ = create_dataloaders(
            data_dict,
            batch_size=batch_size,
            window_size=window_size,
            use_physics=True,
        )

        # 3. MC Dropout 推理（在测试集上）
        all_means, all_stds, all_targets = [], [], []
        model.eval()
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['window'].to(DEVICE)
                y = batch['target_soh'].cpu().numpy().squeeze()
            else:
                x, y_t = batch
                x = x.to(DEVICE)
                y = y_t.cpu().numpy().squeeze()

            mean, std = mc_predict(model, x, n_samples=N_MC_SAMPLES)
            all_means.append(mean.cpu().numpy().squeeze())
            all_stds.append(std.cpu().numpy().squeeze())
            all_targets.append(y)

        pred_mean = np.concatenate([np.atleast_1d(m) for m in all_means])
        pred_std  = np.concatenate([np.atleast_1d(s) for s in all_stds])
        targets   = np.concatenate([np.atleast_1d(t) for t in all_targets])

        # 4. 计算不确定性指标
        unc_report = uncertainty_report(targets, pred_mean, pred_std, verbose=True)

        elapsed = time.time() - t0
        record = {
            'run_id':      run_id,
            'exp':         'exp02_mc_dropout',
            'seed':        seed,
            'model_type':  MODEL_TYPE,
            'n_mc_samples': N_MC_SAMPLES,
            'supervision_ratio': SUPERVISION_RATIO,
            # 点估计指标（用 MC mean）
            'test_mae':    float(unc_report['mae']),
            'test_rmse':   float(unc_report['rmse']),
            'test_mape':   float(results['test_mape']),
            'test_r2':     float(results['test_r2']),
            'best_val_mae':float(results['best_val_mae']),
            'best_epoch':  int(results['best_epoch']),
            # 不确定性指标（M7）
            'picp':        float(unc_report['picp']),
            'mpiw':        float(unc_report['mpiw']),
            'spearman':    float(unc_report['spearman_corr']),
            'elapsed_sec': elapsed,
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        print(f"\n  [DONE] seed={seed}  MAE={record['test_mae']*100:.4f}%  "
              f"PICP={record['picp']:.4f}  MPIW={record['mpiw']*100:.4f}%  "
              f"elapsed={elapsed/60:.1f}min")
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

    maes    = [r['test_mae']  for r in valid]
    rmses   = [r['test_rmse'] for r in valid]
    picps   = [r['picp']      for r in valid]
    mpiws   = [r['mpiw']      for r in valid]

    summary = {
        'exp':         'exp02_mc_dropout',
        'model':       MODEL_TYPE,
        'n_mc_samples': N_MC_SAMPLES,
        'n_runs':      len(valid),
        'seeds':       [r['seed'] for r in valid],
        'mae_mean':    float(np.mean(maes)),
        'mae_std':     float(np.std(maes, ddof=1)),
        'rmse_mean':   float(np.mean(rmses)),
        'rmse_std':    float(np.std(rmses, ddof=1)),
        'picp_mean':   float(np.mean(picps)),
        'picp_std':    float(np.std(picps, ddof=1)),
        'mpiw_mean':   float(np.mean(mpiws)),
        'mpiw_std':    float(np.std(mpiws, ddof=1)),
        'all_runs':    valid,
    }

    out = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Exp-02 汇总（{len(valid)} runs）")
    print(f"{'='*60}")
    print(f"  MAE  : {summary['mae_mean']*100:.4f}% ± {summary['mae_std']*100:.4f}%")
    print(f"  RMSE : {summary['rmse_mean']*100:.4f}% ± {summary['rmse_std']*100:.4f}%")
    print(f"  PICP : {summary['picp_mean']:.4f} ± {summary['picp_std']:.4f}  (理想=0.95)")
    print(f"  MPIW : {summary['mpiw_mean']*100:.4f}% ± {summary['mpiw_std']*100:.4f}%")
    print(f"  保存至: {out}")


if __name__ == '__main__':
    print(f"Exp-02: CNN-LSTM + MC Dropout (n={N_MC_SAMPLES})  |  {len(SEEDS)} seeds  |  device={DEVICE}")
    all_results = [run_single(s) for s in SEEDS]
    aggregate(all_results)
    print("\nExp-02 完成！")
