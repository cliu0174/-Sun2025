"""
M6 r=0.5 快速探针（1 seed × 2 组，~15 min）
最后验证：E5_all_g0 在 r=0.5 是否有效。
结果决定 M6 命运：有效 → 写入论文"r≥0.5 有效"；无效 → Full Stack 彻底不含 M6。
"""
import os, sys, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from train_cross_battery import train_cross_battery_model
from evaluation.physics_viz import compute_physics_violations

SEED   = 42
RATIO  = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'm6_r05_probe')
ABLATION_DIR = os.path.join(os.path.dirname(__file__), 'ablation_single_module')
os.makedirs(OUTPUT_DIR, exist_ok=True)

_ALL_OVERRIDE = {
    'pseudo_labeling': {
        'warmup_epochs':   10,
        'use_ema':         True,
        'ema_alpha':       0.7,
        'use_mono_filter': True,
        'lambda_adaptive': True,
    },
    'model_selection': {'mode': 'composite', 'gamma': 0.0, 'min_epoch': 10},
}


def run_or_load(exp_id, model_type, override=None):
    # 优先读 ablation 缓存（E0）
    if exp_id == 'E0_baseline':
        ratio_tag = f"{RATIO:.1f}".replace('.', 'p')
        fp = os.path.join(ABLATION_DIR, 'E0_baseline',
                          f'ratio{ratio_tag}', f'seed{SEED}', 'result.json')
        if os.path.exists(fp):
            r = json.load(open(fp, encoding='utf-8'))
            print(f"  [CACHE] {exp_id}  MAE={r['test_mae']*100:.4f}%  best_ep={r.get('best_epoch','?')}")
            return r

    result_fp = os.path.join(OUTPUT_DIR, f'{exp_id}_result.json')
    if os.path.exists(result_fp):
        r = json.load(open(result_fp, encoding='utf-8'))
        print(f"  [SKIP]  {exp_id}  MAE={r['test_mae']*100:.4f}%  best_ep={r.get('best_epoch','?')}")
        return r

    print(f"\n  [RUN] {exp_id}  ratio={RATIO}  seed={SEED}  device={DEVICE}")
    t0 = time.time()
    try:
        _, results, _ = train_cross_battery_model(
            model_type=model_type, seed=SEED, device=DEVICE,
            supervision_ratio=RATIO, supervision_seed=None,
            config_override=override,
        )
        elapsed = time.time() - t0
        phys = compute_physics_violations(
            results.get('predictions'), results.get('targets'),
            results.get('battery_ids'), tolerance=0.01,
        ) if results.get('predictions') is not None else {}
        record = {
            'exp_id': exp_id, 'seed': SEED, 'supervision_ratio': RATIO,
            'test_mae':     float(results['test_mae']),
            'test_rmse':    float(results['test_rmse']),
            'test_r2':      float(results['test_r2']),
            'best_val_mae': float(results['best_val_mae']),
            'best_epoch':   int(results['best_epoch']),
            'elapsed_sec':  elapsed, **phys,
        }
        json.dump(record, open(result_fp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"  [DONE] MAE={record['test_mae']*100:.4f}%  R²={record['test_r2']:.4f}"
              f"  best_ep={record['best_epoch']}  elapsed={elapsed/60:.1f}min")
        return record
    except Exception as e:
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print('╔' + '═' * 52 + '╗')
    print('║  M6 r=0.5 探针（最后一次验证，1 seed）        ║')
    print(f'║  ratio={RATIO}  seed={SEED}  device={DEVICE}' + ' ' * 22 + '║')
    print('╚' + '═' * 52 + '╝')

    e0 = run_or_load('E0_baseline',  'cnn_lstm')
    e5 = run_or_load('E5_all_g0',    'cnn_lstm_pseudo_label', _ALL_OVERRIDE)

    print('\n' + '═' * 52)
    print(f'  r=0.5  seed={SEED}  M6 最终裁决')
    print('─' * 52)
    if e0 and e5:
        e0_mae, e5_mae = e0['test_mae'], e5['test_mae']
        delta = (e0_mae - e5_mae) / e0_mae * 100
        ep    = e5.get('best_epoch', '?')
        print(f"  E0  MAE = {e0_mae*100:.4f}%  best_ep={e0.get('best_epoch','?')}")
        print(f"  E5  MAE = {e5_mae*100:.4f}%  best_ep={ep}")
        print(f"  Δ       = {delta:+.2f}%")
        print()
        if delta > 1.0 and isinstance(ep, int) and ep > 10:
            print('  ✅ M6 在 r=0.5 有效（Δ>+1%，best_ep>10）')
            print('  → 论文结论：M6 在 r≥0.5 有效，r=0.3 不稳定')
            print('  → Full Stack 保留 M6（仅 r≥0.5 启用）')
        elif delta > 0:
            print(f'  〜 M6 在 r=0.5 略有改善（Δ={delta:+.2f}%），但幅度有限')
            print('  → 酌情报告，Full Stack 不强制含 M6')
        else:
            print(f'  ✗ M6 在 r=0.5 仍无效（Δ={delta:+.2f}%）')
            print('  → M6 彻底退出 Full Stack，仅在 r=1.0 消融表中报告')
    print('═' * 52)
    json.dump({'e0': e0, 'e5': e5}, open(
        os.path.join(OUTPUT_DIR, 'summary.json'), 'w', encoding='utf-8'),
        indent=2, ensure_ascii=False)
