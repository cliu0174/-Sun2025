"""
单模块消融实验：验证各创新模块的独立贡献
==========================================

实验设计：
    每次只开启一个模块，其余全部关闭，与基线进行对比。
    通过对比结果，可以直接量化每个模块带来的性能提升。

实验列表（共 6 组）：
    ┌──────────────────────────────────────────────────────────────┐
    │  ID   名称              开启的模块       模型类型             │
    │  E0   Baseline         无               cnn_lstm             │
    │  E1   + M1             循环级注意力      cnn_lstm_attention   │
    │  E2   + M2             MC Dropout       cnn_lstm_mc          │
    │  E3   + M4             速率连续性约束    cnn_lstm (override)  │
    │  E4   + M5             自适应损失权重    cnn_lstm_adaptive_weight│
    │  E5   + M2+M6          不确定性伪标签    cnn_lstm_pseudo_label│
    └──────────────────────────────────────────────────────────────┘

说明：
    - M6（伪标签）必须依赖 M2（MC Dropout）才能工作，两者视为一组
    - M5（自适应权重）在设计上自动覆盖所有损失项（含 M4 速率连续性），
      两者并列对比，可看出"固定 M4"与"自适应 M5"的差异
    - M4 的 smoothness_weight 固定为 0.05（可在跑完 Exp-03 后更新为最优值）

验证配置：
    supervision_ratios = [1.0, 0.7, 0.5, 0.3]  ← 4 个稀疏度
    seeds              = [42, 123, 456, 789, 1024]
    总运行次数          = 6 × 4 × 5 = 120 次

结果目录：experiments/ablation_single_module/

最终输出：
    - 每次运行的 result.json
    - 汇总的 metrics.json（含各方法在各 ratio 下的 mean ± std）
    - 控制台打印对比表（直接看出各模块提升幅度）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import contextlib
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from train_cross_battery import train_cross_battery_model
from evaluation.physics_viz import compute_physics_violations

# ================================================================
# 全局配置
# ================================================================
SEEDS              = [42, 123, 34, 999, 1024]
SUPERVISION_RATIOS = [1.0, 0.5, 0.3]
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR         = os.path.join(os.path.dirname(__file__), 'ablation_single_module')

# M4 的固定 smoothness_weight（在 Exp-03 跑完后可改为最优值）
M4_SMOOTHNESS_WEIGHT = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 实验配置表（每个 entry 代表一组消融实验）
# ================================================================
EXPERIMENTS = [
    {
        'id':          'Eneg1_no_physics',
        'label':       '无物理约束',
        'modules':     'None',
        'model_type':  'cnn_lstm',
        'override':    {
            'physics_constraints': {'enabled': False}
        },
        'note':        '纯 CNN-LSTM，无任何物理约束（对照组）',
    },
    {
        'id':          'E0_baseline',
        'label':       'Baseline',
        'modules':     '无',
        'model_type':  'cnn_lstm',
        'override':    {
            'physics_constraints': {
                'enabled': True,
                'monotonic_weight': 0.3,
                'boundary_weight': 0.0,
                'monotonic_tolerance': 0.005,
                'min_cycle': 300,
            }
        },
        'note':        'CNN-LSTM + 软单调（min_cycle=300, w=0.3, tol=0.005）',
    },
    {
        'id':          'E1_m1_attention',
        'label':       '+ M1 注意力',
        'modules':     'M1',
        'model_type':  'cnn_lstm_attention',
        'override':    None,
        'note':        'LSTM 输出后加入循环级注意力机制（CycleAttention）',
    },
    {
        'id':          'E2_m2_mc_dropout',
        'label':       '+ M2 MC Dropout',
        'modules':     'M2',
        'model_type':  'cnn_lstm_mc',
        'override':    None,
        'note':        '推理时保持 Dropout 激活，50次随机推理取均值',
    },
    {
        'id':          'E3_m4_smoothness',
        'label':       '+ M4 速率连续性',
        'modules':     'M4',
        'model_type':  'cnn_lstm',
        'override':    {
            'physics_constraints': {
                'enabled': True,
                'monotonic_weight': 0.3,
                'boundary_weight': 0.0,
                'monotonic_tolerance': 0.005,
                'min_cycle': 300,
                'smoothness_weight': M4_SMOOTHNESS_WEIGHT,
            }
        },
        'note':        f'速率连续性约束 smoothness_weight={M4_SMOOTHNESS_WEIGHT}，min_cycle=300',
    },
    {
        'id':          'E4_m5_adaptive',
        'label':       '+ M5 自适应权重',
        'modules':     'M5',
        'model_type':  'cnn_lstm_adaptive_weight',
        'override':    None,
        'note':        '4 个损失项权重全部可学习（自动替代人工调参，含 M4）',
    },
    {
        'id':          'E5_m2_m6_pseudo',
        'label':       '+ M2+M6 伪标签',
        'modules':     'M2+M6',
        'model_type':  'cnn_lstm_pseudo_label',
        'override':    None,
        'note':        'MC Dropout 估计不确定性，筛选高置信无标签样本作伪标签（ratio<1.0 有效）',
    },
]


# ================================================================
# 单次运行
# ================================================================
def run_single(exp: dict, seed: int, ratio: float) -> Optional[dict]:
    """
    运行一次实验，支持断点续跑。

    Args:
        exp:   实验配置 dict（来自 EXPERIMENTS 列表）
        seed:  随机种子
        ratio: 监督比例

    Returns:
        结果 dict，失败时返回 None
    """
    exp_id    = exp['id']
    ratio_tag = f"{ratio:.1f}".replace('.', 'p')
    run_id    = f"{exp_id}_ratio{ratio_tag}_seed{seed}"
    run_dir   = os.path.join(OUTPUT_DIR, exp_id, f"ratio{ratio_tag}", f"seed{seed}")
    result_fp = os.path.join(run_dir, 'result.json')

    # 断点续跑：已有结果直接跳过
    if os.path.exists(result_fp):
        with open(result_fp, encoding='utf-8') as f:
            r = json.load(f)
        print(f"  [SKIP] {run_id:<55} MAE={r['test_mae']*100:.4f}%")
        return r

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n  {'─'*65}")
    print(f"  [RUN]  {run_id}")
    print(f"         模块={exp['modules']}  ratio={ratio}  seed={seed}  device={DEVICE}")
    print(f"  {'─'*65}")

    try:
        _, results, _ = train_cross_battery_model(
            model_type=exp['model_type'],
            seed=seed,
            device=DEVICE,
            supervision_ratio=ratio,
            supervision_seed=None,
            config_override=exp['override'],
        )
        elapsed = time.time() - t0

        # 物理违规后验统计
        # 注意：use_physics=True 时 battery_ids 是 list；use_physics=False 时是 ndarray
        # 因此必须显式判 None / 长度，不能直接用 truthy 判断（ndarray 会抛 ambiguous）
        phys_metrics = {}
        preds_list = results.get('predictions')
        tgts_list  = results.get('targets')
        bids_list  = results.get('battery_ids')
        if preds_list is not None and bids_list is not None \
                and len(preds_list) > 0 and len(bids_list) > 0:
            phys_metrics = compute_physics_violations(
                preds_list, tgts_list, bids_list, tolerance=0.01
            )

        record = {
            'run_id':            run_id,
            'exp_id':            exp_id,
            'label':             exp['label'],
            'modules':           exp['modules'],
            'model_type':        exp['model_type'],
            'seed':              seed,
            'supervision_ratio': ratio,
            'test_mae':          float(results['test_mae']),
            'test_rmse':         float(results['test_rmse']),
            'test_mape':         float(results['test_mape']),
            'test_r2':           float(results['test_r2']),
            'best_val_mae':      float(results['best_val_mae']),
            'best_epoch':        int(results['best_epoch']),
            'elapsed_sec':       elapsed,
            'note':              exp['note'],
            **phys_metrics,      # mono_violation_rate / mono_violation_mean / boundary_violation_rate / delta_soh_mean / delta_soh_std
        }

        with open(result_fp, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        phys_line = ""
        if phys_metrics:
            phys_line = (f"  违规率={phys_metrics['mono_violation_rate']:.2f}%  "
                         f"ΔS均值={phys_metrics['delta_soh_mean']*100:.4f}%")
        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  "
              f"RMSE={record['test_rmse']*100:.4f}%  "
              f"R2={record['test_r2']:.4f}  "
              f"elapsed={elapsed/60:.1f}min"
              + phys_line)
        return record

    except Exception as e:
        print(f"  [ERROR] {run_id}: {e}")
        traceback.print_exc()
        with open(os.path.join(run_dir, 'error.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'run_id':    run_id,
                'error':     str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc(),
            }, f, indent=2, ensure_ascii=False)
        return None


# ================================================================
# 汇总 & 打印对比表
# ================================================================
def aggregate(all_results: list) -> None:
    """
    按 (exp_id, ratio) 分组，计算 mean ± std，
    以 Baseline 为参照计算各模块的相对提升，打印对比表并保存。
    """

    # ── 分组 ────────────────────────────────────────────────────
    # stats[exp_id][ratio] = {'mae_mean', 'mae_std', 'rmse_mean', ...}
    raw: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is not None and 'test_mae' in r:
            raw[r['exp_id']][r['supervision_ratio']].append(r)

    stats: dict = {}
    for exp_id, ratio_dict in raw.items():
        stats[exp_id] = {}
        for ratio, runs in ratio_dict.items():
            maes  = [r['test_mae']  for r in runs]
            rmses = [r['test_rmse'] for r in runs]
            r2s   = [r['test_r2']   for r in runs]

            # 物理违规指标（旧 result.json 可能没有，跳过）
            mono_viol_rates = [r['mono_violation_rate'] for r in runs
                               if 'mono_violation_rate' in r]
            mono_viol_means = [r['mono_violation_mean'] for r in runs
                               if 'mono_violation_mean' in r]
            delta_soh_means = [r['delta_soh_mean'] for r in runs
                               if 'delta_soh_mean' in r]

            phys_stats = {}
            if mono_viol_rates:
                phys_stats = {
                    'mono_viol_rate_mean': float(np.mean(mono_viol_rates)),
                    'mono_viol_rate_std':  float(np.std(mono_viol_rates, ddof=1) if len(mono_viol_rates) > 1 else 0.0),
                    'mono_viol_mean_mean': float(np.mean(mono_viol_means)),
                    'delta_soh_mean_mean': float(np.mean(delta_soh_means)),
                }

            stats[exp_id][ratio] = {
                'n':         len(runs),
                'mae_mean':  float(np.mean(maes)),
                'mae_std':   float(np.std(maes,  ddof=1) if len(maes)  > 1 else 0.0),
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std':  float(np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0),
                'r2_mean':   float(np.mean(r2s)),
                **phys_stats,
            }

    # ── 打印对比表 ───────────────────────────────────────────────
    ratios_sorted = sorted(SUPERVISION_RATIOS, reverse=True)
    col_w = 16  # 每列宽度

    sep   = '═' * (28 + col_w * len(ratios_sorted))
    sep2  = '─' * (28 + col_w * len(ratios_sorted))

    print(f"\n\n{sep}")
    print("  消融实验对比表（各模块单独贡献 vs Baseline）")
    print(f"  总 seed 数: {len(SEEDS)}  |  device: {DEVICE}")
    print(sep)

    # 表头
    header = f"  {'方法':<26}"
    for r in ratios_sorted:
        header += f"  ratio={r:.1f}".center(col_w)
    print(header)
    print(sep2)

    # Baseline 的 MAE mean（用于计算改善比）
    baseline_id  = 'E0_baseline'
    baseline_row = stats.get(baseline_id, {})

    for exp in EXPERIMENTS:
        exp_id   = exp['id']
        exp_stat = stats.get(exp_id, {})
        label    = exp['label']
        modules  = exp['modules']

        # ── MAE ± std 行 ─────────────────────────────────
        row_mae = f"  {label:<26}"
        for ratio in ratios_sorted:
            s = exp_stat.get(ratio)
            if s:
                cell = f"{s['mae_mean']*100:.3f}±{s['mae_std']*100:.3f}%"
            else:
                cell = '—'
            row_mae += cell.center(col_w)
        print(row_mae)

        # ── 相对改善行（只对非 Baseline 打印）───────────────
        if exp_id != baseline_id:
            row_delta = f"  {'':>4}{'改善 vs Baseline':<22}"
            for ratio in ratios_sorted:
                s  = exp_stat.get(ratio)
                bs = baseline_row.get(ratio)
                if s and bs and bs['mae_mean'] > 0:
                    delta_pct = (bs['mae_mean'] - s['mae_mean']) / bs['mae_mean'] * 100
                    arrow = '↓' if delta_pct > 0 else '↑'
                    cell  = f"{arrow}{abs(delta_pct):.1f}%"
                else:
                    cell = '—'
                row_delta += cell.center(col_w)
            print(row_delta)

        print(sep2)

    # ── 物理约束违规对比表 ───────────────────────────────────────
    # 只有至少一组实验有物理数据时才打印
    has_phys = any(
        'mono_viol_rate_mean' in stats.get(exp['id'], {}).get(ratio, {})
        for exp in EXPERIMENTS
        for ratio in ratios_sorted
    )
    if has_phys:
        print(f"\n\n{sep}")
        print("  物理约束违规分析（单调性违规率 / 预测变化率 ΔS均值）")
        print(f"  容忍量 tolerance=0.01  |  值越低 = 约束越有效")
        print(sep)

        # 子表头：违规率行
        hdr2 = f"  {'方法':<26}"
        for r in ratios_sorted:
            hdr2 += f"  ratio={r:.1f}".center(col_w)
        print(hdr2)
        print(sep2)

        baseline_phys = stats.get(baseline_id, {})

        for exp in EXPERIMENTS:
            exp_id   = exp['id']
            exp_stat = stats.get(exp_id, {})

            # ── 单调性违规率行 ────────────────────────────────
            row_viol = f"  {exp['label']:<26}"
            for ratio in ratios_sorted:
                s = exp_stat.get(ratio, {})
                if 'mono_viol_rate_mean' in s:
                    cell = f"{s['mono_viol_rate_mean']:.2f}%"
                else:
                    cell = '—'
                row_viol += cell.center(col_w)
            print(row_viol)

            # ── 相对 Baseline 的违规率变化（非 Baseline 才打印）
            if exp_id != baseline_id:
                row_rel = f"  {'':>4}{'vs Baseline':<22}"
                for ratio in ratios_sorted:
                    s  = exp_stat.get(ratio, {})
                    bs = baseline_phys.get(ratio, {})
                    if 'mono_viol_rate_mean' in s and 'mono_viol_rate_mean' in bs:
                        delta = s['mono_viol_rate_mean'] - bs['mono_viol_rate_mean']
                        sign  = '+' if delta > 0 else ''
                        cell  = f"{sign}{delta:.2f}%"
                    else:
                        cell = '—'
                    row_rel += cell.center(col_w)
                print(row_rel)

            # ── ΔS 均值行 ─────────────────────────────────────
            row_ds = f"  {'':>4}{'ΔS均值(per step)':<22}"
            for ratio in ratios_sorted:
                s = exp_stat.get(ratio, {})
                if 'delta_soh_mean_mean' in s:
                    cell = f"{s['delta_soh_mean_mean']*100:.4f}%"
                else:
                    cell = '—'
                row_ds += cell.center(col_w)
            print(row_ds)

            print(sep2)

    # ── 模块说明 ─────────────────────────────────────────────────
    print("\n  [模块说明]")
    for exp in EXPERIMENTS:
        print(f"  {exp['label']:<20}  {exp['note']}")

    # ── 保存 metrics.json ────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    save_obj = {
        'experiment':  'ablation_single_module',
        'seeds':       SEEDS,
        'ratios':      SUPERVISION_RATIOS,
        'device':      DEVICE,
        'experiments': [],
    }

    for exp in EXPERIMENTS:
        exp_id   = exp['id']
        exp_stat = stats.get(exp_id, {})
        entry    = {
            'exp_id':     exp_id,
            'label':      exp['label'],
            'modules':    exp['modules'],
            'model_type': exp['model_type'],
            'note':       exp['note'],
            'by_ratio':   {},
        }

        for ratio in ratios_sorted:
            s  = exp_stat.get(ratio)
            bs = baseline_row.get(ratio)
            if s:
                delta = None
                rel   = None
                if exp_id != baseline_id and bs and bs['mae_mean'] > 0:
                    delta = float(bs['mae_mean'] - s['mae_mean'])
                    rel   = float(delta / bs['mae_mean'] * 100)

                # 物理违规相对 Baseline 的变化
                phys_delta = None
                if (exp_id != baseline_id
                        and 'mono_viol_rate_mean' in s
                        and bs and 'mono_viol_rate_mean' in bs):
                    phys_delta = float(s['mono_viol_rate_mean'] - bs['mono_viol_rate_mean'])

                entry['by_ratio'][str(ratio)] = {
                    **s,
                    'delta_mae':               delta,
                    'rel_improvement':         rel,
                    'mono_viol_rate_delta':    phys_delta,
                }
        save_obj['experiments'].append(entry)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)

    print(f"\n  汇总结果已保存至: {out_path}")
    print(sep)


# ================================================================
# 进度追踪工具
# ================================================================
def count_done(exp_list=None) -> tuple[int, int]:
    """统计已完成和总运行次数（可传入过滤后的 exp_list）。"""
    if exp_list is None:
        exp_list = EXPERIMENTS
    total = len(exp_list) * len(SUPERVISION_RATIOS) * len(SEEDS)
    done  = 0
    for exp in exp_list:
        for ratio in SUPERVISION_RATIOS:
            ratio_tag = f"{ratio:.1f}".replace('.', 'p')
            for seed in SEEDS:
                fp = os.path.join(
                    OUTPUT_DIR, exp['id'],
                    f"ratio{ratio_tag}", f"seed{seed}", 'result.json'
                )
                if os.path.exists(fp):
                    done += 1
    return done, total


def load_all_results_from_disk() -> list:
    """
    扫描磁盘上所有 result.json，返回结果列表。
    用于在只跑部分实验时，让 aggregate 仍能读到其他组的历史结果。
    """
    results = []
    for exp in EXPERIMENTS:
        for ratio in SUPERVISION_RATIOS:
            ratio_tag = f"{ratio:.1f}".replace('.', 'p')
            for seed in SEEDS:
                fp = os.path.join(
                    OUTPUT_DIR, exp['id'],
                    f"ratio{ratio_tag}", f"seed{seed}", 'result.json'
                )
                if os.path.exists(fp):
                    with open(fp, encoding='utf-8') as f:
                        results.append(json.load(f))
    return results


# ================================================================
# 主入口
# ================================================================
if __name__ == '__main__':
    # ── CLI 参数解析 ────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='单模块消融实验')
    parser.add_argument(
        '--exp',
        type=str,
        default=None,
        help=(
            '只运行指定实验（逗号分隔），不影响汇总表。\n'
            '示例：--exp E5  或  --exp E4,E5\n'
            '可用 ID 前缀：E0 E1 E2 E3 E4 E5'
        ),
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=None,
        help='只运行指定监督比例（单个值），如 --ratio 1.0',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='只运行指定随机种子（单个值），如 --seed 42',
    )
    args = parser.parse_args()

    # 根据 --exp 过滤要运行的实验列表
    if args.exp:
        prefixes = [p.strip().upper() for p in args.exp.split(',')]
        run_experiments = [
            e for e in EXPERIMENTS
            if any(e['id'].upper().startswith(p) for p in prefixes)
        ]
        if not run_experiments:
            print(f"[ERROR] --exp '{args.exp}' 没有匹配到任何实验。"
                  f"可用 ID：{[e['id'] for e in EXPERIMENTS]}")
            sys.exit(1)
    else:
        run_experiments = EXPERIMENTS

    # --ratio / --seed 过滤（诊断模式，不影响汇总表读取历史结果）
    run_ratios = [args.ratio] if args.ratio is not None else SUPERVISION_RATIOS
    run_seeds  = [args.seed]  if args.seed  is not None else SEEDS

    done, total = count_done(run_experiments)

    print('╔' + '═' * 65 + '╗')
    print('║  单模块消融实验（各模块独立贡献验证）' + ' ' * 27 + '║')
    print('╠' + '═' * 65 + '╣')
    print(f'║  运行实验   : {[e["id"] for e in run_experiments]}' + ' ' * 10 + '║')
    print(f'║  监督比例   : {SUPERVISION_RATIOS}' + ' ' * 23 + '║')
    print(f'║  随机种子   : {SEEDS}' + ' ' * 22 + '║')
    print(f'║  本次运行数 : {total}（已完成 {done}，剩余 {total - done}）' + ' ' * 21 + '║')
    print(f'║  计算设备   : {DEVICE}' + ' ' * (51 - len(DEVICE)) + '║')
    print(f'║  输出目录   : experiments/ablation_single_module/' + ' ' * 15 + '║')
    print('╚' + '═' * 65 + '╝')

    if done == total:
        print("\n指定实验均已完成，直接生成对比表...\n")
    else:
        print(f"\n开始运行（剩余 {total - done} 次）...\n")

    # ── 按 ratio → experiment → seed 顺序运行 ──────────────────
    new_results = []

    for ratio in run_ratios:
        ratio_tag = f"ratio={ratio:.1f}"
        print(f"\n{'▶'*3}  开始 {ratio_tag}  {'▶'*3}")

        for exp in run_experiments:
            label_pad = f"{exp['label']}（{exp['modules']}）"
            print(f"\n  ┌── {label_pad} | {ratio_tag}")

            for seed in run_seeds:
                r = run_single(exp, seed, ratio)
                new_results.append(r)

            # 每个 (exp, ratio) 组合跑完后打印小结
            sub_results = [r for r in new_results
                           if r and r['exp_id'] == exp['id']
                           and r['supervision_ratio'] == ratio]
            if sub_results:
                maes = [r['test_mae'] for r in sub_results]
                print(f"  └── {label_pad} | {ratio_tag}  "
                      f"MAE = {np.mean(maes)*100:.4f}% ± {np.std(maes, ddof=1)*100:.4f}%")

    # ── 最终汇总：从磁盘读取全量结果，保证对比表包含所有实验组 ──
    all_results = load_all_results_from_disk()
    aggregate(all_results)

    label = "指定实验" if args.exp else "全部实验"
    print(f"\n✅ {label}运行完成！")
