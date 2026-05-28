"""
图4-11 / 图4-12 / 表4-9：CRT 自建数据集上的 SOH 轨迹与异常检测
=================================================================
功能：
  1. 训练 CNN-LSTM 和 PI-MSCL（带缓存，支持断点续跑）
  2. 在 CRT 自建电池（来自 HUST 1-5/3-1/4-7/5-1）上推理
  3. 注入 5 种退化场景 × 3 种严重程度（self_trajectory 模式）
  4. 生成图4-11：正常 vs 故障电池 SOH 轨迹对比（3 场景 × 1 列）
  5. 生成图4-12：trajectory_deviation 异常分数 + 阈值线 + N连击报警点
  6. 打印表4-9：AUC / Det@FPR5% / N-Hit Delay（按场景×严重程度）

检测逻辑（N连击）：
  - 用最近 SCORE_WINDOW 步预测做线性外推计算 trajectory_deviation
  - 阈值 = 正常段分数的 mean + THRESHOLD_K × std
  - 连续 N_HIT 个窗口超过阈值 → 触发报警

用法：
    python scripts/plot_anomaly_detection.py            # 完整流程
    python scripts/plot_anomaly_detection.py --plot-only  # 跳过训练
    python scripts/plot_anomaly_detection.py --retrain    # 强制重训
    python scripts/plot_anomaly_detection.py --device cpu
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
WINDOW_SIZE = 40        # 模型输入窗口（与 cnn_lstm_config.json 一致）

# ── 异常检测参数 ──────────────────────────────────────────────────
SCORE_WINDOW = 20       # trajectory_deviation 滑动拟合窗口大小
N_HIT        = 20       # N连击：连续 N 个窗口超阈值才触发报警（与 SCORE_WINDOW 一致）
THRESHOLD_K  = 2.0      # 阈值倍数：mean(clean) + K × std(clean)

# CRT 自建数据集（HUST 原始文件名 → CRT 标签）
CRT_BATTERIES = {
    'CRT-B02': '1-5',
    'CRT-B03': '3-1',
    'CRT-B04': '4-7',
    'CRT-B05': '5-1',
}

# 退化场景（全部 5 种，图示选前 3 种）
SCENARIOS = ['sudden_aging', 'knee_point', 'imbalance', 'li_plating', 'resistance_rise']
SCENARIO_LABELS = {
    'sudden_aging':    'Sudden Aging',
    'knee_point':      'Capacity Knee-point',
    'imbalance':       'Cell Imbalance',
    'li_plating':      'Li Plating',
    'resistance_rise': 'Resistance Rise',
}
# 图示场景（轨迹对比 + 分数图各用这 3 种）
PLOT_SCENARIOS = ['sudden_aging', 'knee_point', 'li_plating']

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

def train_and_cache(exp_id, cfg, retrain=False, device=DEVICE):
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

    print(f'\n  [{"RETRAIN" if cached else "TRAIN"}] {exp_id}: '
          f'{cfg["label"]}  (seed={SEED}, r={RATIO})')

    wrapper, _, data_dict = train_cross_battery_model(
        model_type        = cfg['model_type'],
        device            = device,
        seed              = SEED,
        supervision_ratio = RATIO,
        supervision_seed  = None,
        config_override   = cfg['override'],
    )

    torch.save(wrapper.model.state_dict(), wp)

    meta = {
        'model_type': cfg['model_type'],
        'input_size': data_dict['n_features'],
        'config'    : wrapper.config,
    }
    with open(mp, 'wb') as f:
        pickle.dump(meta, f)

    if not os.path.exists(sp):
        with open(sp, 'wb') as f:
            pickle.dump(data_dict['scaler'], f)

    print(f'  [SAVED] {wp}')
    return _load_cached(exp_id, device=device)


def _load_cached(exp_id, device=DEVICE):
    """从 state_dict 重建模型，返回 (model: nn.Module, scaler)。"""
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
        raw_features  = all_data[hust_id]['train_features']
        targets       = all_data[hust_id]['train_capacity']
        norm_features = scaler.transform(raw_features)
        crt_data[crt_label] = {
            'features': norm_features,
            'targets' : targets,
            'bid'     : hust_id,
        }
        print(f'  {crt_label} ({hust_id}): T={len(targets)}, '
              f'SOH [{targets.min():.3f}, {targets.max():.3f}]')
    return crt_data


# ════════════════════════════════════════════════════════════════
# 3. 推理辅助
# ════════════════════════════════════════════════════════════════

def predict_sequence(model, features, window_size=WINDOW_SIZE,
                     device=DEVICE, batch_size=512):
    """窗口化推理，返回预测 SOH 序列（长度 = T - window_size + 1）。"""
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
# 4. N连击检测辅助
# ════════════════════════════════════════════════════════════════

def compute_threshold(clean_scores, k=THRESHOLD_K):
    """
    基于正常段分数计算检测阈值。
    threshold = mean(clean) + k × std(clean)
    """
    arr = np.asarray(clean_scores, dtype=float)
    # 忽略前 SCORE_WINDOW 个点（线性拟合启动段，分数恒为 0）
    valid = arr[SCORE_WINDOW:]
    if len(valid) == 0:
        valid = arr
    return float(np.mean(valid) + k * np.std(valid))


def detect_first_alarm_n_hit(scores, threshold, n_hit=N_HIT):
    """
    N连击检测：连续 n_hit 个窗口分数 ≥ threshold → 触发报警。
    返回报警触发时的窗口索引（即第 n_hit 次连续命中处），
    若整段均未触发则返回 -1。
    """
    consecutive = 0
    for i, s in enumerate(scores):
        if s >= threshold:
            consecutive += 1
            if consecutive >= n_hit:
                return i        # 第 n_hit 次命中，此刻触发报警
        else:
            consecutive = 0
    return -1


# ════════════════════════════════════════════════════════════════
# 5. 异常检测评估
# ════════════════════════════════════════════════════════════════

def run_anomaly_eval(model, scaler, crt_data, exp_id, device=DEVICE):
    """
    对每块 CRT 电池 × 5 场景 × 3 严重程度运行异常检测。
    返回嵌套字典：
      results[crt_label][scenario][severity] = run_anomaly_detection_for_battery 的返回值
    """
    from evaluation.anomaly_detection import run_anomaly_detection_for_battery

    feature_mean = scaler.mean_
    feature_std  = scaler.scale_

    results = {}
    for crt_label, d in crt_data.items():
        print(f'\n  [{exp_id}] {crt_label}...')
        features    = d['features']
        targets     = d['targets']
        T           = len(features)
        fault_cycle = int(T * 0.50)

        results[crt_label] = {}
        for sc in SCENARIOS:
            results[crt_label][sc] = {}
            for sev in SEVERITIES:
                res = run_anomaly_detection_for_battery(
                    model            = model,
                    normal_features  = features,
                    donor_features   = None,
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
# 6. 图4-11：正常 vs 故障 SOH 轨迹对比（1×3）
# ════════════════════════════════════════════════════════════════

def plot_fig411(all_det_results, crt_data,
                rep_battery='CRT-B03', severity='moderate'):
    """
    图4-11：选取代表性电池，展示 3 种场景下正常与故障的 SOH 轨迹对比。

    布局：1 行 × 3 列（sudden_aging / knee_point / li_plating）
    每列：
      - 黑色实线：真实 SOH
      - 橙色虚线：PI-MSCL 正常预测
      - 深红色实线：PI-MSCL 故障预测
      - 灰色竖虚线：故障注入时刻
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    d       = crt_data[rep_battery]
    targets = d['targets']
    T       = len(targets)

    for col, sc in enumerate(PLOT_SCENARIOS):
        ax = axes[col]

        res = all_det_results['pi_ms_cnn_lstm'][rep_battery][sc][severity]
        if 'error' in res:
            ax.set_title(f'{SCENARIO_LABELS[sc]}\n(Error)', fontsize=10)
            continue

        preds_clean = np.asarray(res['preds_clean'])
        preds_fault = np.asarray(res['preds_fault'])
        fault_start = res['fault_start_out']      # 输出序列中的故障起始索引
        fault_cycle = res['fault_cycle']           # 原始 cycle 索引

        # x 轴对齐到原始 cycle（第 i 个输出 = cycle i + WINDOW_SIZE - 1）
        x_clean = np.arange(len(preds_clean)) + WINDOW_SIZE - 1
        x_fault = np.arange(len(preds_fault)) + WINDOW_SIZE - 1

        # 真实 SOH
        ax.plot(np.arange(T), targets,
                color='black', lw=1.8, ls='-', zorder=10, label='True SOH')

        # 正常预测
        ax.plot(x_clean, preds_clean,
                color='#F4831F', lw=1.6, ls='--', zorder=4, alpha=0.85,
                label='Normal Prediction')

        # 故障预测
        ax.plot(x_fault, preds_fault,
                color='#C0392B', lw=1.8, ls='-', zorder=5, alpha=0.90,
                label='Fault Prediction')

        # 故障注入竖线
        ax.axvline(fault_cycle, color='#777777', lw=1.0, ls=':',
                   zorder=2, label='Fault Injected')

        ax.set_xlabel('Cycle Index', fontsize=9)
        ax.set_ylabel('SOH', fontsize=9)
        ax.set_title(SCENARIO_LABELS[sc], fontsize=10, fontweight='bold')
        ax.set_xlim(0, T - 1)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.85,
                  edgecolor='#CCCCCC')
        ax.tick_params(labelsize=8)

    plt.suptitle(
        f'Normal vs Fault SOH Trajectories — {rep_battery} '
        f'({SEV_LABELS[severity]} Severity)',
        fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_11_traj_compare.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# 7. 图4-12：异常分数 + 阈值线 + N连击报警（1×3）
# ════════════════════════════════════════════════════════════════

def plot_fig412(all_det_results, crt_data,
                rep_battery='CRT-B03', severity='moderate'):
    """
    图4-12：3 种场景下 trajectory_deviation 异常分数，
    叠加阈值线与 N连击报警点。

    布局：1 行 × 3 列
    每列：
      - 灰色虚线：正常电池分数（基准）
      - 橙色实线（PI-MSCL）/ 黄色虚线（CNN-LSTM）：故障电池分数
      - 红色水平虚线：阈值（正常段 mean + K×std）
      - 灰色竖虚线：故障注入时刻
      - 红色竖实线：PI-MSCL N连击报警时刻（附延迟标注）
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for col, sc in enumerate(PLOT_SCENARIOS):
        ax = axes[col]

        # ── 取 PI-MSCL clean 分数计算阈值 ──────────────────
        res_pi_clean_ref = all_det_results['pi_ms_cnn_lstm'][rep_battery][sc][severity]
        if 'error' in res_pi_clean_ref:
            ax.set_title(f'{SCENARIO_LABELS[sc]}\n(Error)', fontsize=10)
            continue

        clean_traj = np.array(res_pi_clean_ref['scores_clean']['trajectory_deviation'])
        threshold  = compute_threshold(clean_traj)

        # ── 正常段基准线 ────────────────────────────────────
        ax.plot(np.arange(len(clean_traj)), clean_traj,
                color='#AAAAAA', lw=1.0, ls='--', alpha=0.7,
                zorder=2, label='Normal (PI-MSCL)')

        # ── 两个模型的故障分数 ───────────────────────────────
        pi_fault_traj = None
        pi_fault_start = None

        for exp_id, cfg in MODELS.items():
            res = all_det_results[exp_id][rep_battery][sc][severity]
            if 'error' in res:
                continue
            fault_traj  = np.array(res['scores_fault']['trajectory_deviation'])
            fault_start = res['fault_start_out']

            ax.plot(np.arange(len(fault_traj)), fault_traj,
                    color  = cfg['color'],
                    lw     = cfg['lw'],
                    ls     = cfg['ls'],
                    alpha  = 0.9,
                    zorder = cfg['zorder'],
                    label  = f'{cfg["label"]} Fault')

            if exp_id == 'pi_ms_cnn_lstm':
                pi_fault_traj  = fault_traj
                pi_fault_start = fault_start

        # ── 阈值水平线 ───────────────────────────────────────
        ax.axhline(threshold, color='#E74C3C', lw=1.4, ls='--',
                   zorder=6, label=f'Threshold (μ+{THRESHOLD_K:.0f}σ)')

        # ── 故障注入竖线 ─────────────────────────────────────
        if pi_fault_start is not None:
            ax.axvline(pi_fault_start, color='#777777', lw=1.0, ls=':',
                       zorder=3, label='Fault Start')

        # ── PI-MSCL N连击报警点 ──────────────────────────────
        if pi_fault_traj is not None and pi_fault_start is not None:
            alarm_idx = detect_first_alarm_n_hit(pi_fault_traj, threshold, N_HIT)
            if alarm_idx >= 0:
                delay = max(0, alarm_idx - pi_fault_start)
                ax.axvline(alarm_idx, color='#E74C3C', lw=1.8, ls='-',
                           zorder=7, label=f'Alarm (delay={delay})')
                ax.annotate(
                    f'+{delay}',
                    xy=(alarm_idx, ax.get_ylim()[1] if ax.get_ylim()[1] > threshold else threshold * 1.1),
                    xytext=(alarm_idx + max(2, len(pi_fault_traj) // 30), threshold * 1.05),
                    fontsize=8, color='#E74C3C',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=0.8),
                )

        ax.set_xlabel('Window Index', fontsize=9)
        ax.set_ylabel('Trajectory Deviation', fontsize=9)
        ax.set_title(SCENARIO_LABELS[sc], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7.5, framealpha=0.85, edgecolor='#CCCCCC',
                  loc='upper left')
        ax.tick_params(labelsize=8)

    plt.suptitle(
        f'Anomaly Score with N-Hit Detection ({rep_battery}, '
        f'{SEV_LABELS[severity]}, N={N_HIT})',
        fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fpath = os.path.join(OUTPUT_DIR, f'fig4_12_anomaly_threshold.{ext}')
        plt.savefig(fpath)
        print(f'[OK] → {fpath}')
    plt.close()


# ════════════════════════════════════════════════════════════════
# 8. 表4-9：检测指标汇总（含 N连击 Delay）
# ════════════════════════════════════════════════════════════════

def print_table49(all_det_results, save_csv=True):
    """
    表4-9：5 场景 × 3 严重程度，
    AUC / Det@FPR5%（ROC 框架）+ N-Hit Delay（N连击框架），
    4 块 CRT 电池取平均。
    """
    SIGNAL = 'trajectory_deviation'

    print('\n' + '='*92)
    print('表4-9  不同退化场景下的异常检测率')
    print('       信号：trajectory_deviation  |  Delay = FPR@5% 阈值下首次超过窗口数')
    print('='*92)
    print(f"{'场景':<22} {'严重程度':<9} "
          f"{'---- CNN-LSTM ----':^28}  {'---- PI-MSCL -----':^28}")
    print(f"{'':^31} "
          f"{'AUC':>5} {'Det@5%':>7} {'Delay':>6}  "
          f"{'AUC':>5} {'Det@5%':>7} {'Delay':>6}")
    print('-'*92)

    rows = []
    for sc in SCENARIOS:
        for sev in SEVERITIES:
            row = [SCENARIO_LABELS[sc], SEV_LABELS[sev]]
            for exp_id in ['cnn_lstm', 'pi_ms_cnn_lstm']:
                aucs, dets, nhit_delays = [], [], []
                for crt_label in CRT_BATTERIES:
                    if crt_label not in all_det_results.get(exp_id, {}):
                        continue
                    res = all_det_results[exp_id][crt_label][sc][sev]
                    if 'error' in res or 'metrics' not in res:
                        continue

                    # ROC 指标
                    m = res['metrics'].get(SIGNAL, {})
                    aucs.append(m.get('auc', float('nan')))
                    dets.append(m.get('det_rate_fpr5', float('nan')))

                    # ROC-based Delay（FPR=5% 阈值下首次超过的窗口数）
                    d = m.get('det_delay', -1)
                    if d >= 0:
                        nhit_delays.append(d)

                auc_m  = np.nanmean(aucs)      if aucs        else float('nan')
                det_m  = np.nanmean(dets)      if dets        else float('nan')
                dly_m  = np.mean(nhit_delays)  if nhit_delays else float('nan')
                row += [auc_m, det_m, dly_m]

            rows.append(row)
            dly0 = f'{row[4]:>5.1f}' if not np.isnan(row[4]) else '  N/D'
            dly1 = f'{row[7]:>5.1f}' if not np.isnan(row[7]) else '  N/D'
            print(f'{row[0]:<22} {row[1]:<9} '
                  f'{row[2]:>5.3f} {row[3]:>7.1%} {dly0}  '
                  f'{row[5]:>5.3f} {row[6]:>7.1%} {dly1}')

    print('='*92)

    if save_csv:
        import csv
        csv_path = os.path.join(OUTPUT_DIR, 'table4_9_detection.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['场景', '严重程度',
                        'CNN-LSTM AUC', 'CNN-LSTM Det@FPR5%', 'CNN-LSTM Delay',
                        'PI-MSCL AUC',  'PI-MSCL Det@FPR5%',  'PI-MSCL Delay'])
            for r in rows:
                def _fmt(v): return f'{v:.4f}' if not np.isnan(v) else 'N/D'
                def _fmt1(v): return f'{v:.1f}' if not np.isnan(v) else 'N/D'
                w.writerow([r[0], r[1],
                            _fmt(r[2]), _fmt(r[3]), _fmt1(r[4]),
                            _fmt(r[5]), _fmt(r[6]), _fmt1(r[7])])
        print(f'[OK] CSV → {csv_path}')


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot anomaly detection figures & Table 4-9')
    parser.add_argument('--plot-only', action='store_true',
                        help='跳过训练，直接从模型缓存出图')
    parser.add_argument('--retrain', action='store_true',
                        help='强制重新训练')
    parser.add_argument('--device', default=DEVICE,
                        help=f'计算设备（默认：{DEVICE}）')
    parser.add_argument('--battery', default='CRT-B03',
                        help='图示代表电池（默认：CRT-B03）')
    parser.add_argument('--severity', default='moderate',
                        choices=SEVERITIES,
                        help='图示严重程度（默认：moderate）')
    args = parser.parse_args()

    device   = args.device
    rep_bat  = args.battery
    severity = args.severity

    # ── Step 1：训练/加载模型 ─────────────────────────────────
    print('\n[Step 1] 训练/加载模型...')
    models  = {}
    scalers = {}
    for exp_id, cfg in MODELS.items():
        model, scaler = train_and_cache(exp_id, cfg,
                                        retrain=args.retrain,
                                        device=device)
        models[exp_id]  = model
        scalers[exp_id] = scaler

    shared_scaler = scalers['cnn_lstm']

    # ── Step 2：加载 CRT 数据 ─────────────────────────────────
    print('\n[Step 2] 加载 CRT 数据...')
    crt_data = load_crt_data(shared_scaler)
    if not crt_data:
        print('ERROR: 未找到任何 CRT 电池数据，请检查 data/HUST data/')
        return

    if rep_bat not in crt_data:
        rep_bat = list(crt_data.keys())[0]
        print(f'  指定电池不存在，改用 {rep_bat}')

    # ── Step 3：异常注入评估 ──────────────────────────────────
    print('\n[Step 3] 运行异常检测评估（5场景 × 3严重程度）...')
    all_det_results = {}
    for exp_id, model in models.items():
        print(f'\n  模型：{MODELS[exp_id]["label"]}')
        all_det_results[exp_id] = run_anomaly_eval(
            model, shared_scaler, crt_data, exp_id, device=device)

    # ── Step 4：图4-11 正常 vs 故障轨迹对比 ──────────────────
    print(f'\n[Step 4] 生成图4-11（{rep_bat}, {severity}）...')
    plot_fig411(all_det_results, crt_data,
                rep_battery=rep_bat, severity=severity)

    # ── Step 5：图4-12 异常分数 + 阈值 + N连击 ───────────────
    print(f'\n[Step 5] 生成图4-12（{rep_bat}, {severity}, N={N_HIT}）...')
    plot_fig412(all_det_results, crt_data,
                rep_battery=rep_bat, severity=severity)

    # ── Step 6：表4-9 ─────────────────────────────────────────
    print('\n[Step 6] 生成表4-9...')
    print_table49(all_det_results)

    print(f'\n完成。输出目录：{OUTPUT_DIR}')


if __name__ == '__main__':
    main()
