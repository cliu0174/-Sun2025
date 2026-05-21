"""
MVP 快速验证：Exp-12 bug 修复后异常检测是否可行
================================================

核心思路：
  - 不重新训练，直接加载 exp12_anomaly_detection/ 下已有的
    model.pth + data_cache.pkl（之前跑 Exp-12 时自动保存的）
  - 只跑 3 个场景 × severe × 前 8 块测试电池 × 2 个供体
  - 全程推理，约 2~5 分钟出结论

用法：
  python experiments/mvp_anomaly_verify.py
  python experiments/mvp_anomaly_verify.py --ckpt_dir experiments/exp12_anomaly_detection/E0_baseline/ratio0p5/seed929
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


# ─── 找可用的 checkpoint 目录 ────────────────────────────────────────
def find_cached_run(hint_dir=None):
    """找到含 model.pth + data_cache.pkl 的目录"""
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


# ─── 主验证逻辑 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default=None, help='指定 checkpoint 目录')
    parser.add_argument('--n_test',   type=int, default=8,  help='使用几块测试电池（默认8）')
    parser.add_argument('--n_donors', type=int, default=2,  help='使用几个供体（默认2）')
    parser.add_argument('--severity', default='severe',     help='mild/moderate/severe')
    args = parser.parse_args()

    # 1. 加载 checkpoint
    run_dir = find_cached_run(args.ckpt_dir)
    if run_dir is None:
        print("ERROR: 未找到缓存 checkpoint。")
        print(f"  请确认服务器 Exp-12 已运行过至少一轮（{EXP12_DIR}/*/model.pth）")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"MVP 异常检测快速验证（bug 修复后）")
    print(f"{'='*65}")
    print(f"Checkpoint : {run_dir}")
    print(f"Device     : {DEVICE}")
    print(f"Severity   : {args.severity}")
    print(f"Test bats  : {args.n_test}")
    print(f"Donors     : {args.n_donors}")

    wrapper = UnifiedModelWrapper.load_checkpoint(
        os.path.join(run_dir, 'model.pth'), device=DEVICE)
    wrapper.model.eval()

    with open(os.path.join(run_dir, 'data_cache.pkl'), 'rb') as f:
        cache = pickle.load(f)
    data_dict = cache['data_dict']
    results   = cache['results']
    print(f"Model MAE  : {results['test_mae']*100:.4f}%   R² : {results['test_r2']:.4f}")

    # 2. 重建数据
    battery_data = build_battery_data(data_dict)
    feat_mean    = data_dict['train_features'].mean(axis=0)
    feat_std     = data_dict['train_features'].std(axis=0)

    # 3. 选供体
    tb_targets  = {bid: data_dict['train_targets'][data_dict['train_battery_ids'] == bid]
                   for bid in data_dict['train_batteries']}
    tb_features = {bid: data_dict['train_features'][data_dict['train_battery_ids'] == bid]
                   for bid in data_dict['train_batteries']}
    donor_ids = select_donor_batteries(tb_features, tb_targets, n_donors=args.n_donors)
    print(f"Donors     : {donor_ids}\n")

    # 4. 测试场景
    scenarios = [
        ('sudden_aging',    True,  '① 突发老化    [预期：易检测]'),
        ('knee_point',      False, '② 容量拐点    [预期：易检测]'),
        ('imbalance',       True,  '③ 不均衡加剧  [预期：中等]'),
        ('li_plating',      True,  '④ 析锂台阶    [预期：易检测]'),
        ('resistance_rise', False, '⑤ 内阻增长    [预期：难检测]'),
    ]

    print(f"{'场景':<14} {'AUC':>7} {'Det@5%':>9} {'Delay':>8}  判断")
    print("-" * 55)

    test_batteries = data_dict['test_batteries'][:args.n_test]

    for scenario_key, needs_donor, desc in scenarios:
        donor_iter = donor_ids if needs_donor else [None]
        all_metrics = []

        for test_bid in test_batteries:
            info = battery_data.get(test_bid)
            if info is None or len(info['features']) < WINDOW_SIZE + 10:
                continue

            normal_feat = info['features']
            fault_cycle = int(len(normal_feat) * 0.5)

            for donor_id in donor_iter:
                donor_feat = battery_data[donor_id]['features'] if donor_id else None

                det = run_anomaly_detection_for_battery(
                    model=wrapper,
                    normal_features=normal_feat,
                    donor_features=donor_feat,
                    feature_mean=feat_mean,
                    feature_std=feat_std,
                    fault_cycle=fault_cycle,
                    severity=args.severity,
                    scenario=scenario_key,
                    window_size=WINDOW_SIZE,
                    device=DEVICE,
                )
                if 'error' not in det:
                    all_metrics.append(det['metrics'])

        if not all_metrics:
            print(f"{scenario_key:<14}  {'N/A':>7}")
            continue

        for signal in ['combined']:
            aucs  = [m[signal]['auc']           for m in all_metrics if not np.isnan(m[signal]['auc'])]
            dets  = [m[signal]['det_rate_fpr5'] for m in all_metrics if not np.isnan(m[signal]['det_rate_fpr5'])]
            dlys  = [m[signal]['det_delay']     for m in all_metrics if m[signal]['det_delay'] >= 0]

            auc   = np.mean(aucs)  if aucs else float('nan')
            det   = np.mean(dets) * 100 if dets else float('nan')
            delay = np.mean(dlys)  if dlys else float('nan')

            if auc > 0.70:   verdict = '✅ 有效'
            elif auc > 0.58: verdict = '⚠️  弱效'
            else:             verdict = '❌ 无效'

            print(f"{scenario_key:<14} {auc:>7.3f} {det:>8.1f}%  {delay:>7.1f}  {verdict}  {desc}")

    print("-" * 55)
    print("判断标准: AUC > 0.70 有效 | 0.58~0.70 弱效 | < 0.58 无效")
    print("\n如果突发场景（①②④）AUC > 0.65，说明 bug 修复有效，可重跑完整 Exp-12。")
    print("如果仍然 ≈ 0.5，需要进一步排查特征是否真的在故障后发生变化。\n")


if __name__ == '__main__':
    main()
