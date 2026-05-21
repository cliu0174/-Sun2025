"""
MVP 快速验证：Exp-12 异常检测（支持 legacy / self_trajectory 两种注入模式）
===========================================================================

不重新训练，直接加载 exp12_anomaly_detection/ 下已有的
model.pth + data_cache.pkl（之前跑 Exp-12 时自动保存的）。

用法：
  # self_trajectory 模式（新方案，推荐先跑）
  python experiments/mvp_anomaly_verify.py --injection_mode self_trajectory

  # legacy 模式（旧方案，供对比）
  python experiments/mvp_anomaly_verify.py --injection_mode legacy

  # 指定具体 checkpoint 目录
  python experiments/mvp_anomaly_verify.py --injection_mode self_trajectory \\
      --ckpt_dir experiments/exp12_anomaly_detection/E0_baseline/ratio0p5/seed929
"""

import os, sys, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model_factory import UnifiedModelWrapper
from evaluation.anomaly_detection import (
    select_donor_batteries,
    run_anomaly_detection_for_battery,
)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
WINDOW_SIZE = 40
EXP12_DIR   = os.path.join(os.path.dirname(__file__), 'exp12_anomaly_detection')

# 全部信号（含新增的 drop_anomaly / trajectory_deviation）
ALL_SIGNALS = [
    'input_zscore', 'rate_anomaly', 'drop_anomaly',
    'trajectory_deviation', 'mono_violation', 'combined',
]


# ─── 找可用的 checkpoint 目录 ────────────────────────────────────────
def find_cached_run(hint_dir=None):
    if hint_dir and os.path.isfile(os.path.join(hint_dir, 'model.pth')):
        return hint_dir
    for root, _, files in os.walk(EXP12_DIR):
        if 'model.pth' in files and 'data_cache.pkl' in files:
            return root
    return None


# ─── 从 data_dict 重建逐电池特征字典 ─────────────────────────────────
def build_battery_data(data_dict):
    battery_data = {}
    for split in ['train', 'val', 'test']:
        feats = data_dict[f'{split}_features']
        tgts  = data_dict[f'{split}_targets']
        bids  = data_dict[f'{split}_battery_ids']
        for bid in np.unique(bids):
            mask = bids == bid
            if bid not in battery_data:
                battery_data[bid] = {'features': feats[mask], 'targets': tgts[mask]}
    return battery_data


# ─── 主逻辑 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',       default=None,
                        help='指定 checkpoint 目录（可选）')
    parser.add_argument('--injection_mode', default='self_trajectory',
                        choices=['legacy', 'self_trajectory'],
                        help='注入模式：self_trajectory（新）或 legacy（旧）')
    parser.add_argument('--n_test',   type=int, default=8,
                        help='使用几块测试电池（默认8）')
    parser.add_argument('--n_donors', type=int, default=2,
                        help='legacy 模式下使用几个供体（默认2）')
    parser.add_argument('--severity', default='severe',
                        choices=['mild', 'moderate', 'severe'])
    args = parser.parse_args()

    # 1. 加载 checkpoint
    run_dir = find_cached_run(args.ckpt_dir)
    if run_dir is None:
        print("ERROR: 未找到缓存 checkpoint。")
        print(f"  请先运行一次 Exp-12 训练（{EXP12_DIR}/*/model.pth）")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"MVP 异常检测快速验证")
    print(f"{'='*65}")
    print(f"Checkpoint    : {run_dir}")
    print(f"Device        : {DEVICE}")
    print(f"Injection mode: {args.injection_mode}")
    print(f"Severity      : {args.severity}")
    print(f"Test batteries: {args.n_test}")

    wrapper = UnifiedModelWrapper.load_checkpoint(
        os.path.join(run_dir, 'model.pth'), device=DEVICE)
    wrapper.model.eval()

    with open(os.path.join(run_dir, 'data_cache.pkl'), 'rb') as f:
        cache = pickle.load(f)
    data_dict = cache['data_dict']
    results   = cache['results']
    print(f"Model MAE     : {results['test_mae']*100:.4f}%   R2: {results['test_r2']:.4f}")

    # 2. 重建数据
    battery_data = build_battery_data(data_dict)
    feat_mean    = data_dict['train_features'].mean(axis=0)
    feat_std     = data_dict['train_features'].std(axis=0)

    # 3. 供体（legacy 模式需要）
    if args.injection_mode == 'legacy':
        tb_tgts  = {b: data_dict['train_targets'][data_dict['train_battery_ids']==b]
                    for b in data_dict['train_batteries']}
        tb_feats = {b: data_dict['train_features'][data_dict['train_battery_ids']==b]
                    for b in data_dict['train_batteries']}
        donor_ids = select_donor_batteries(tb_feats, tb_tgts, n_donors=args.n_donors)
        print(f"Donors        : {donor_ids}")
    else:
        donor_ids = [None]

    print()

    # 4. 测试场景
    scenarios = [
        ('sudden_aging',    True,  'sudden_aging    [预期：易检测]'),
        ('knee_point',      False, 'knee_point      [预期：易检测]'),
        ('imbalance',       True,  'imbalance       [预期：中等]'),
        ('li_plating',      True,  'li_plating      [预期：易检测]'),
        ('resistance_rise', False, 'resistance_rise [预期：难检测]'),
    ]

    # 表头：逐信号 AUC
    sig_width = 7
    header = f"{'场景':<20}"
    for s in ALL_SIGNALS:
        header += f" {s[:sig_width]:>{sig_width}}"
    header += "  verdict"
    print(header)
    print('-' * (20 + len(ALL_SIGNALS) * (sig_width + 1) + 12))

    test_batteries = data_dict['test_batteries'][:args.n_test]

    # 诊断汇总（打印在表格后）
    diag_rows = []

    for scenario_key, needs_donor, desc in scenarios:
        if args.injection_mode == 'legacy':
            donor_iter = donor_ids if needs_donor else [None]
        else:
            donor_iter = [None]   # self_trajectory 不需要供体

        all_metrics   = []
        all_diags     = []

        for test_bid in test_batteries:
            info = battery_data.get(test_bid)
            if info is None or len(info['features']) < WINDOW_SIZE + 10:
                continue

            normal_feat = info['features']
            targets     = info['targets']
            fault_cycle = int(len(normal_feat) * 0.5)

            for donor_id in donor_iter:
                donor_feat = (battery_data[donor_id]['features']
                              if donor_id is not None else None)

                det = run_anomaly_detection_for_battery(
                    model=wrapper,
                    normal_features=normal_feat,
                    donor_features=donor_feat,
                    feature_mean=feat_mean,
                    feature_std=feat_std,
                    fault_cycle=fault_cycle,
                    severity=args.severity,
                    scenario=scenario_key,
                    injection_mode=args.injection_mode,
                    targets=targets,
                    window_size=WINDOW_SIZE,
                    device=DEVICE,
                )
                if 'error' not in det:
                    all_metrics.append(det['metrics'])
                    all_diags.append(det.get('diagnostics', {}))

        if not all_metrics:
            print(f"{scenario_key:<20}  N/A")
            continue

        # 每个信号的平均 AUC
        row = f"{scenario_key:<20}"
        aucs = {}
        for sig in ALL_SIGNALS:
            vals = [m[sig]['auc'] for m in all_metrics
                    if sig in m and not np.isnan(m[sig]['auc'])]
            auc = np.mean(vals) if vals else float('nan')
            aucs[sig] = auc
            row += f" {auc:>{sig_width}.3f}"

        combined_auc = aucs.get('combined', float('nan'))
        if   combined_auc > 0.70: verdict = 'GOOD'
        elif combined_auc > 0.58: verdict = 'WEAK'
        else:                     verdict = 'FAIL'
        row += f"  {verdict}"
        print(row)

        # 收集诊断量
        drops  = [d.get('pred_drop_at_fault', float('nan')) for d in all_diags]
        gaps   = [d.get('mean_pred_gap_after', float('nan')) for d in all_diags]
        l1s    = [d.get('feature_l1_after_fault', float('nan')) for d in all_diags]
        diag_rows.append((scenario_key,
                          np.nanmean(drops), np.nanmean(gaps), np.nanmean(l1s)))

    print('-' * (20 + len(ALL_SIGNALS) * (sig_width + 1) + 12))
    print("判断: AUC > 0.70 GOOD | 0.58~0.70 WEAK | < 0.58 FAIL")

    # 诊断信息
    print(f"\n{'='*65}")
    print("诊断量（验证注入是否有效）")
    print(f"{'场景':<20} {'pred_drop@fault':>16} {'mean_gap_after':>16} {'feat_L1_after':>14}")
    print('-'*70)
    for sc, drop, gap, l1 in diag_rows:
        drop_s = f"{drop:+.4f}" if not np.isnan(drop) else "  N/A"
        gap_s  = f"{gap:+.4f}"  if not np.isnan(gap)  else "  N/A"
        l1_s   = f"{l1:.4f}"    if not np.isnan(l1)   else "  N/A"
        print(f"{sc:<20} {drop_s:>16} {gap_s:>16} {l1_s:>14}")
    print()
    print("pred_drop_at_fault : 故障点模型预测 clean-fault 差值（正值=故障预测更低，好）")
    print("mean_gap_after     : 故障后 clean 与 fault 预测均值差（正=fault 预测更低，好）")
    print("feat_L1_after      : 故障后特征 L1 变化均值（越大=注入越有效）")


if __name__ == '__main__':
    main()
