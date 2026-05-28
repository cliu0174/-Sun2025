"""
图4-11 / 图4-12 / 表4-9：CRT 自建数据集上的 SOH 轨迹与异常检测
=================================================================
功能：
  1. 训练 CNN-LSTM 和 PI-MSCL（带缓存，支持断点续跑）
  2. 在 CRT 自建电池（来自 HUST 1-5/3-1/4-7/5-1）上推理
  3. 注入 5 种退化场景 × 3 种严重程度（self_trajectory 模式）
  4. 生成图4-11：4 块 CRT 电池的 SOH 预测轨迹（正常状态）
  5. 生成图4-12：典型场景下 trajectory_deviation 异常信号对比
  6. 打印表4-9：AUC / Det@FPR5% / Delay（按场景×严重程度）

用法：
    # 完整流程（训练 + 推理 + 出图）
    python scripts/plot_anomaly_detection.py

    # 仅出图（模型缓存已存在）
    python scripts/plot_anomaly_detection.py --plot-only

    # 强制重新训练
    python scripts/plot_anomaly_detection.py --retrain

注意：
    CRT 数据集为本项目自建验证集，使用 HUST 数据集中
    1-5/3-1/4-7/5-1 四块电池，覆盖率 38~50%（见图4-2）。
"""

import os
import sys
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE_DIR  = os.path.join(ROOT, 'figures', 'pred_cache')
OUTPUT_DIR = os.path.join(ROOT, 'figures')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 全局配置 ──────────────────────────────────────────────────────
SEED        = 42
RATIO       = 1.0       # 全监督
DEVICE      = 'cuda'
WINDOW_SIZE = 40        # 与 cnn_lstm_config.json 保持一致

# CRT 自建数据集（HUST 原始文件名 → CRT 标签）
CRT_BATTERIES = {
    'CRT-B02': '1-5',
    'CRT-B03': '3-1',
    'CRT-B04': '4-7',
    'CRT-B05': '5-1',
}

# 退化场景
SCENARIOS = ['sudden_aging', 'knee_point', 'imbalance', 'li_plating', 'resistance_rise']
SCENARIO_LABELS = {
    'sudden_aging':    'Sudden Aging',
    'knee_point':      'Knee-point',
    'imbalance':       'Cell Imbalance',
    'li_plating':      'Li Plating',
    'resistance_rise': 'Resistance Rise',
}
SEVERITIES = ['mild', 'moderate', 'severe']
SEV_LABELS  = {'mild': 'Mild', 'moderate': 'Moderate', 'severe': 'Severe'}

# ── 模型定义 ─────────────────────────────────────────────────────
_NO_PI = {
    'enabled'           : False,
    'monotonic_weight'  : 0.0,
    'boundary_weight'   : 0.0,
    'smoothness_weight' : 0.0,
}
_PI_CONFIG = {
    'enabled'           : True,
    'monotonic_weight'  : 0.3,
    'boundary_weight'   : 0.0,
    'smoothness_weight' : 0.0,
    'monotonic_tolerance': 0.005,
    'min_cycle'         : 300,
    'base_loss_weight'  : 1.0,
}

MODELS = {
    'cnn_lstm': {
        'model_type' : 'cnn_lstm',
        'override'   : {'physics_constraints': _NO_PI},
        'label'      : 'CNN-LSTM',
        'color'      : '#F5C542',
        'lw'         : 1.4,
        'ls'         : '--',
        'zorder'     : 3,
    },
    'pi_ms_cnn_lstm': {
        'model_type' : 'ms_cnn_lstm_v2',
        'override'   : {
            'architecture'       : {'use_multiscale': True, 'per_window_norm': False},
            'physics_constraints': _PI_CONFIG,
        },
        'label'      : 'PI-MSCL',
        'color'      : '#F4831F',
        'lw'         : 2.0,
        'ls'         : '-',
        'zorder'     : 5,
    },
}

# ── 图形样式 ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── 缓存路径 ─────────────────────────────────────────────────────
def _weight_path(exp_id):
    return os.path.join(CACHE_DIR, f'anomaly_{exp_id}_r{RATIO}_s{SEED}_weights.pt')

def _meta_path(exp_id):
    return os.path.join(CACHE_DIR, f'anomaly_{exp_id}_r{RATIO}_s{SEED}_meta.pkl')

def _scaler_path():
    return os.path.join(CACHE_DIR, f'anomaly_scaler_r{RATIO}_s{SEED}.pkl')


# ════════════════════════════════════════════════════════════════
# 1. 训练 & 缓存
# ════════════════════════════════════════════════════════════════

def train_and_cache(exp_id: str, cfg: dict, retrain: bool = False,
                    device: str = DEVICE):
    """
    训练模型并保存 state_dict + scaler。
    已存在缓存且 retrain=False 时直接跳过。
    返回：(model: nn.Module, scaler: StandardScaler)
    """
    import torch
    from train_cross_battery import train_cross_battery_model

    wp = _weight_path(exp_id)
    mp = _meta_path(exp_id)
    sp = _scaler_path()
    cached = os.path.exists(wp) and os.path.exists(mp)

    if cached and not retrain:
        print(f'  [CACHE HIT] {exp_id}')
        return _load_cached(exp_id, device=device)

    if cached and retrain:
        print(f'  [RETRAIN] {exp_id}')
    else:
        print(f'\n  [TRAIN] {exp_id}: {cfg["label"]}  (seed={SEED}, r={RATIO})')

    wrapper, _, data_dict = train_cross_battery_model(
        model_type        = cfg['model_type'],
        device            = device,
        seed              = SEED,
        supervision_ratio = RATIO,
        supervision_seed  = None,
        config_override   = cfg['override'],
    )

    # 保存 state_dict
    torch.save(wrapper.model.state_dict(), wp)

    # 保存模型元数据
    meta = {
        'model_type': cfg['model_type'],
        'input_size': data_dict['n_features'],
        'config'    : wrapper.config,
    }
    with open(mp, 'wb') as f:
        pickle.dump(meta, f)

    # 保存 scaler（仅第一个模型保存，后续共用）
    if not os.path.exists(sp):
        with open(sp, 'wb') as f:
            pickle.dump(data_dict['scaler'], f)

    print(f'  [SAVED] {wp}')
    return _load_cached(exp_id, device=device)


def _load_cached(exp_id: str, device: str = DEVICE):
    """从 state_dict 重建模型，返回 (model, scaler)。"""
    import torch
    from models.model_factory import UnifiedModelWrapper

    wp = _weight_path(exp_id)
    mp = _meta_path(exp_id)
    sp = _scaler_path()

    with open(mp, 'rb') as f:
        meta = pickle.load(f)

    wrapper = UnifiedModelWrapper(
        model_type = meta['model_type'],
        input_size = meta['input_size'],
        config     = meta['config'],
        device     = device,
    )
    state = torch.load(wp, map_location=device, weights_only=True)
    wrapper.model.load_state_dict(state)
    wrapper.model.eval()

    with open(sp, 'rb') as f:
        scaler = pickle.load(f)

    return wrapper.model, scaler


# ════════════════════════════════════════════════════════════════
# 2. 数据加载
# ════════════════════════════════════════════════════════════════

def load_crt_data(scaler):
    """
    加载 CRT 四块电池的标准化特征与 SOH。
    返回：{crt_label: {'features': (T,F), 'targets': (T,), 'bid': str}}
    """
    from train_cross_battery import load_all_batteries

    print('\n加载 CRT 电池数据...')
    _, all_data = load_all_batteries()

    crt_data = {}
    for crt_label, hust_id in CRT_BATTERIES.items():
        if hust_id not in all_data:
            print(f'  警告：{hust_id} 不在 HUST 数据集中，跳过')
            continue
        raw_features = all_data[hust_id]['train_features']   # (T, F)，未标准化
        targets      = all_data[hust_id]['train_capacity']   # (T,) SOH
        norm_features = scaler.transform(raw_features)        # 标准化
        crt_data[crt_label] = {
            'features': norm_features,
            'targets' : targets,
            'bid'     : hust_id,
        }
        print(f'  {crt_label} ({hust_id}): T={len(targets)}, '
              f'SOH range [{targets.min():.3f}, {targets.max():.3f}]')
    return crt_data


# ════════════════════════════════════════════════════════════════
# 3. 推理辅助
# ════════════════════════════════════════════════════════════════

def predict_sequence(model, features, window_size=WINDOW_SIZE,
                     device=DEVICE, batch_size=512):
    """
    对特征序列做窗口化推理。
    返回预测 SOH 序列，长度 = T - window_size + 1。
    """
    import torch
    raw = getattr(model, 'model', model)
    raw.eval()
    T = len(features)
    if T < window_size:
        return np.array([])

    windows = np.stack([features[i:i + window_size]
                        for i in range(T - window_size + 1)])
    x = torch.FloatTensor(windows).to(device)
    preds = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            out = raw(x[s:s + batch_size])
            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(out.squeeze(-1).cpu().numpy())
    return np.concatenate(preds)


# ════════════════════════════════════════════════════════════════
# 4. 异常检测评估
# ════════════════════════════════════════════════════════════════

def run_anomaly_eval(model, scaler, crt_data, exp_id, device=DEVICE):
    """
    对每块 CRT 电池 × 5 场景 × 3 严重程度运行异常检测。
    返回嵌套字典：
      results[crt_label][scenario][severity] = anomaly_detection result dict
    """
    from evaluation.anomaly_detection import run_anomaly_detection_for_battery

    feature_mean = scaler.mean_      # (F,) 训练集均值
    feature_std  = scaler.scale_     # (F,) 训练集标准差

    results = {}
    for crt_label, d in crt_data.items():
        print(f'\n  [{exp_id}] {crt_label}...')
        features = d['features']
        targets  = d['targets']
        T = len(features)
        fault_cycle = int(T * 0.50)   # 故障注入位置：50% 处

        results[crt_label] = {}
        for sc in SCENARIOS:
            results[crt_label][sc] = {}
            for sev in SEVERITIES:
                res = run_anomaly_detection_for_battery(
                    model            = model,
                    normal_features  = features,
                    donor_features   = None,     # self_trajectory 模式不需要供体
                    feature_mean     = feature_mean,
                    feature_std      = feature_std,
                    fault_cycle      = fault_cycle,
                    severity         = sev,
                    scenario         = sc,
                    injection_mode   = 'self_trajectory',
                    targets          = targets,
                    window_size      = WINDOW_SIZE,
                    device           = device,
                )
                results[crt_label][sc][sev] = res
    return results


# ════════════════════════════════════════════════════════════════
# 5. 图4-11：CRT 电池 SOH 预测轨迹（2×2）
# ════════════════════════════════════════════════════════════════

def plot_fig411(crt_data, model_preds):
    """
    4 块 CRT 电池的正常状态 SOH 轨迹对比。

    model_preds: {crt_label: {exp_id: np.ndarray}}  — 每个模型的预测序列
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (crt_label, d) in enumerate(crt_data.items()):
        ax = axes[idx]
        targets  = d['targets']
        T        = len(targets)
        x_true   = np.arange(T)

        # 真值
        ax.plot(x_true, targets,
                color='black', lw=1.8, ls='-', zorder=10,
                label='True SOH')

        for exp_id, cfg in MODELS.items():
            preds = model_preds[crt_label][exp_id]
            x_pred = np.arange(len(preds)) + WINDOW_SIZE - 1   # 对齐到原始 cycle
            nn = min(len(preds), T - WINDOW_SIZE + 1)
            err  = preds[:nn] - targets[WINDOW_SIZE - 1:WINDOW_SIZE - 1 + nn]
            rmse = np.sqrt(np.mean(err ** 2)) * 100
            ax.plot(x_pred[:nn], preds[:nn],
                    color   = cfg['color'],
                    lw      = cfg['lw'],
                    ls      = cfg['ls'],
                    zorder  = cfg['zorder'],
                    label   = f'{cfg["label"]} (RMSE={rmse:.3f}%)')

        ax.set_xlabel('Cycle Index', fontsize=9)
        ax.set_ylabel('SOH', fontsize=9)
        ax.set_title(f'{crt_label}  ({d["bid"]})', fontsize=10, fontweight='bold')
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.85, edgecolor='#CCCCCC')
        ax.tick_params(labelsize=8)

    plt.suptitle('CRT Dataset: SOH Prediction Trajectories (CNN-LSTM vs PI-MSCL)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_11_soh.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# 6. 图4-12：异常检测信号（trajectory_deviation）
# ════════════════════════════════════════════════════════════════

def plot_fig412(all_det_results, crt_data, rep_battery='CRT-B03'):
    """
    展示代表性电池在 3 种关键场景下的 trajectory_deviation 信号。

    布局：1 行 × 3 列（sudden_aging / li_plating / resistance_rise）
    每列：PI-MSCL clean(灰虚线) + CNN-LSTM fault(黄实线) + PI-MSCL fault(橙实线)
    """
    sel_scenarios = ['sudden_aging', 'li_plating', 'resistance_rise']
    severity      = 'moderate'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, sc in enumerate(sel_scenarios):
        ax = axes[col]

        for exp_id, cfg in MODELS.items():
            res = all_det_results[exp_id][rep_battery][sc][severity]
            if 'error' in res:
                print(f'  警告：{exp_id} {rep_battery} {sc} {severity}: {res["error"]}')
                continue

            fault_start = res['fault_start_out']
            traj_clean  = np.array(res['scores_clean']['trajectory_deviation'])
            traj_fault  = np.array(res['scores_fault']['trajectory_deviation'])

            x_clean = np.arange(len(traj_clean))
            x_fault = np.arange(len(traj_fault))

            # 正常（clean）只画 PI-MSCL，避免图面杂乱
            if exp_id == 'pi_ms_cnn_lstm':
                ax.plot(x_clean, traj_clean,
                        color='#AAAAAA', lw=1.0, ls='--', alpha=0.7,
                        label='Normal (PI-MSCL)', zorder=2)

            # 故障（fault）两个模型都画
            ax.plot(x_fault, traj_fault,
                    color   = cfg['color'],
                    lw      = cfg['lw'],
                    ls      = cfg['ls'],
                    alpha   = 0.9,
                    zorder  = cfg['zorder'],
                    label   = f'{cfg["label"]} Fault')

        # 故障注入线（取 pi_ms_cnn_lstm 的值）
        if 'pi_ms_cnn_lstm' in all_det_results:
            res0 = all_det_results['pi_ms_cnn_lstm'][rep_battery][sc][severity]
            if 'fault_start_out' in res0:
                ax.axvline(res0['fault_start_out'], color='red',
                           lw=1.2, ls=':', alpha=0.8, label='Fault Start')

        ax.set_xlabel('Window Index', fontsize=9)
        ax.set_ylabel('Trajectory Deviation', fontsize=9)
        ax.set_title(SCENARIO_LABELS[sc], fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, framealpha=0.85, edgecolor='#CCCCCC')
        ax.tick_params(labelsize=8)

    plt.suptitle(f'Anomaly Detection Signal: {rep_battery} (Moderate Severity)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_12_anomaly_signal.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# 7. 表4-9：Det@FPR5% 汇总
# ════════════════════════════════════════════════════════════════

def print_table49(all_det_results, save_csv=True):
    """
    表4-9：按场景 × 严重程度汇总 AUC / Det@FPR5% / Delay（平均所有 CRT 电池）。
    信号：trajectory_deviation（主要检测信号）
    """
    SIGNAL = 'trajectory_deviation'

    header = (
        f"{'场景':<20} {'严重程度':<8} "
        f"{'CNN-LSTM':>30}  {'PI-MSCL':>30}\n"
        f"{'':^28} "
        f"{'AUC':>6} {'Det@5%':>7} {'Delay':>6}  "
        f"{'AUC':>6} {'Det@5%':>7} {'Delay':>6}"
    )
    print('\n' + '='*80)
    print('表4-9  不同退化场景下的异常检测率（trajectory_deviation 信号）')
    print('='*80)
    print(header)
    print('-'*80)

    rows = []
    for sc in SCENARIOS:
        for sev in SEVERITIES:
            row = [SCENARIO_LABELS[sc], SEV_LABELS[sev]]
            for exp_id in ['cnn_lstm', 'pi_ms_cnn_lstm']:
                aucs, dets, delays = [], [], []
                for crt_label in CRT_BATTERIES:
                    if crt_label not in all_det_results[exp_id]:
                        continue
                    res = all_det_results[exp_id][crt_label][sc][sev]
                    if 'error' in res or 'metrics' not in res:
                        continue
                    m = res['metrics'].get(SIGNAL, {})
                    aucs.append(m.get('auc', float('nan')))
                    dets.append(m.get('det_rate_fpr5', float('nan')))
                    d = m.get('det_delay', -1)
                    if d >= 0:
                        delays.append(d)

                auc_m  = np.nanmean(aucs)  if aucs   else float('nan')
                det_m  = np.nanmean(dets)  if dets   else float('nan')
                dly_m  = np.nanmean(delays) if delays else float('nan')
                row += [auc_m, det_m, dly_m]

            rows.append(row)
            print(f'{row[0]:<20} {row[1]:<8} '
                  f'{row[2]:>6.3f} {row[3]:>6.1%}  {row[4]:>5.1f}  '
                  f'{row[5]:>6.3f} {row[6]:>6.1%}  {row[7]:>5.1f}')

    print('='*80)

    if save_csv:
        import csv
        csv_path = os.path.join(OUTPUT_DIR, 'table4_9_detection.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['场景', '严重程度',
                        'CNN-LSTM AUC', 'CNN-LSTM Det@FPR5%', 'CNN-LSTM Delay',
                        'PI-MSCL AUC',  'PI-MSCL Det@FPR5%',  'PI-MSCL Delay'])
            for r in rows:
                w.writerow([r[0], r[1],
                            f'{r[2]:.4f}', f'{r[3]:.4f}', f'{r[4]:.1f}',
                            f'{r[5]:.4f}', f'{r[6]:.4f}', f'{r[7]:.1f}'])
        print(f'[OK] CSV → {csv_path}')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot anomaly detection figures & Table 4-9')
    parser.add_argument('--plot-only', action='store_true',
                        help='跳过训练，直接从模型缓存出图（缓存必须已存在）')
    parser.add_argument('--retrain', action='store_true',
                        help='强制重新训练（忽略现有缓存）')
    parser.add_argument('--device', default=DEVICE,
                        help=f'计算设备（默认：{DEVICE}）')
    args = parser.parse_args()

    device = args.device

    # ── Step 1：训练/加载模型 ─────────────────────────────────
    print('\n[Step 1] 训练/加载模型...')
    models   = {}
    scalers  = {}
    for exp_id, cfg in MODELS.items():
        model, scaler = train_and_cache(exp_id, cfg,
                                        retrain=args.retrain,
                                        device=device)
        models[exp_id]  = model
        scalers[exp_id] = scaler

    # scaler 共用同一个（基于相同数据划分训练）
    shared_scaler = scalers['cnn_lstm']

    # ── Step 2：加载 CRT 电池数据 ────────────────────────────
    print('\n[Step 2] 加载 CRT 数据...')
    crt_data = load_crt_data(shared_scaler)
    if not crt_data:
        print('ERROR: 未找到任何 CRT 电池数据，请检查 data/HUST data/ 目录')
        return

    # ── Step 3：清洁推理（用于图4-11）────────────────────────
    print('\n[Step 3] 对 CRT 电池做正常推理...')
    model_preds = {crt_label: {} for crt_label in crt_data}
    for exp_id, model in models.items():
        for crt_label, d in crt_data.items():
            preds = predict_sequence(model, d['features'],
                                     window_size=WINDOW_SIZE, device=device)
            model_preds[crt_label][exp_id] = preds

    # ── Step 4：异常注入评估（用于图4-12 & 表4-9）────────────
    print('\n[Step 4] 运行异常检测评估...')
    all_det_results = {}
    for exp_id, model in models.items():
        print(f'\n  模型：{MODELS[exp_id]["label"]}')
        all_det_results[exp_id] = run_anomaly_eval(
            model, shared_scaler, crt_data, exp_id, device=device)

    # ── Step 5：出图 ──────────────────────────────────────────
    print('\n[Step 5] 生成图4-11...')
    plot_fig411(crt_data, model_preds)

    # 找一个 CRT-B03 是否存在
    rep_bat = 'CRT-B03' if 'CRT-B03' in crt_data else list(crt_data.keys())[0]
    print(f'\n[Step 5] 生成图4-12（代表电池：{rep_bat}）...')
    plot_fig412(all_det_results, crt_data, rep_battery=rep_bat)

    # ── Step 6：打印表4-9 ─────────────────────────────────────
    print('\n[Step 6] 生成表4-9...')
    print_table49(all_det_results)

    print(f'\n完成。输出目录：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
