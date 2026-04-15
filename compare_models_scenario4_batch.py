"""
批量对比 4 个模型在 Scenario 4（连续缺失）下的表现。

目标：
1) 批量训练/测试：xgboost_simple, lstm, cnn_lstm, pi_cnn_lstm
2) 固定退化参数：cycle_drop_rate=0.1, cycle_drop_num_gaps=10
3) 输出图表：
   - 全测试集散点图对比（各模型 Pred vs True）
   - 指标柱状图对比（RMSE/MAE/R²）
   - 指定电池（3-3, 5-2, 9-5）SOH 轨迹对比（CNN-LSTM vs PI-CNNLSTM）
   - 指定电池误差曲线对比（CNN-LSTM vs PI-CNNLSTM）

说明：
- 本脚本只调用现有训练函数，不修改原算法逻辑。
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_cross_battery import train_cross_battery_model
from train_xgboost_baseline import train_xgboost_baseline


# =========================
# Config
# =========================
MODELS_TO_TEST = [
    # "xgboost_simple",
    # "lstm",
    # "cnn_lstm",
    "pi_cnn_lstm",
]

RANDOM_SEED = 999
DEVICE = "auto"
APPLY_CLEANING = False

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DEGRADATION_SCENARIO = "scenario4"
CYCLE_DROP_RATE = 0.1
CYCLE_DROP_NUM_GAPS = 10

SELECTED_BATTERIES = ["10-5", "3-2"]
PLOT_COMPARE_MODELS = ["lstm", "pi_cnn_lstm"]
OUTPUT_DIR = "results/scenario4_model_comparison"

DISPLAY_NAME_MAP = {
    "xgboost_simple": "XGBoost",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "pi_cnn_lstm": "PI-CNNLSTM",
}


def display_name(model_name: str) -> str:
    return DISPLAY_NAME_MAP.get(model_name, model_name)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def validate_result(result: dict, model_name: str) -> None:
    keys = ["predictions", "targets", "battery_ids", "test_rmse", "test_mae", "test_r2"]
    missing = [k for k in keys if k not in result]
    if missing:
        raise ValueError(f"{model_name}: missing keys: {missing}")

    n_pred = len(result["predictions"])
    n_tgt = len(result["targets"])
    n_bid = len(result["battery_ids"])
    if not (n_pred == n_tgt == n_bid):
        raise ValueError(
            f"{model_name}: length mismatch predictions/targets/battery_ids="
            f"{n_pred}/{n_tgt}/{n_bid}"
        )


def train_models(device: str) -> dict:
    results = {}

    print("\n" + "=" * 80)
    print("Scenario 4 批量模型对比")
    print("=" * 80)
    print(f"Models: {MODELS_TO_TEST}")
    print(
        f"Scenario4: drop_rate={CYCLE_DROP_RATE}, num_gaps={CYCLE_DROP_NUM_GAPS}, "
        f"seed={RANDOM_SEED}"
    )

    for idx, model_type in enumerate(MODELS_TO_TEST, start=1):
        print(f"\n[{idx}/{len(MODELS_TO_TEST)}] Training {model_type} ...")
        try:
            if model_type in ["xgboost_simple", "xgboost_enhanced"]:
                _, result, _ = train_xgboost_baseline(
                    model_type=model_type,
                    train_ratio=TRAIN_RATIO,
                    val_ratio=VAL_RATIO,
                    test_ratio=TEST_RATIO,
                    device="cpu",
                    seed=RANDOM_SEED,
                    apply_cleaning=APPLY_CLEANING,
                    degradation_scenario=DEGRADATION_SCENARIO,
                    cycle_drop_rate=CYCLE_DROP_RATE,
                    cycle_drop_num_gaps=CYCLE_DROP_NUM_GAPS,
                )
            else:
                _, result, _ = train_cross_battery_model(
                    model_type=model_type,
                    train_ratio=TRAIN_RATIO,
                    val_ratio=VAL_RATIO,
                    test_ratio=TEST_RATIO,
                    device=device,
                    seed=RANDOM_SEED,
                    apply_cleaning=APPLY_CLEANING,
                    color_by_battery=True,
                    highlight_anomalies=False,
                    degradation_scenario=DEGRADATION_SCENARIO,
                    cycle_drop_rate=CYCLE_DROP_RATE,
                    cycle_drop_num_gaps=CYCLE_DROP_NUM_GAPS,
                )

            validate_result(result, model_type)
            results[model_type] = result
            print(
                f"  OK  RMSE={result['test_rmse']:.6f}, "
                f"MAE={result['test_mae']:.6f}, R²={result['test_r2']:.6f}"
            )
        except Exception as e:
            print(f"  FAIL {model_type}: {e}")

    if not results:
        raise RuntimeError("No model finished successfully.")
    return results


def plot_scatter_comparison(results: dict, save_dir: str) -> None:
    model_names = list(results.keys())
    n = len(model_names)
    n_cols = 2
    n_rows = (n + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        pred = np.array(results[model_name]["predictions"], dtype=float)
        tgt = np.array(results[model_name]["targets"], dtype=float)

        # 与 compare_models.py 保持一致的散点样式
        ax.scatter(tgt, pred, alpha=0.5, s=20, color=colors[i], edgecolors="none")
        lo = min(float(tgt.min()), float(pred.min()))
        hi = max(float(tgt.max()), float(pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=2, label="Ideal")

        ax.set_title(
            f"{display_name(model_name)}\n"
            f"RMSE={results[model_name]['test_rmse']*100:.3f}%, "
            f"R²={results[model_name]['test_r2']:.4f}"
        , fontsize=11, fontweight="bold")
        ax.set_xlabel("True SOH", fontsize=10)
        ax.set_ylabel("Predicted SOH", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(save_dir, "scatter_comparison_all_test.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")


def plot_metric_bars(results: dict, save_dir: str) -> pd.DataFrame:
    rows = []
    for m, r in results.items():
        rows.append(
            {
                "model": m,
                "display_name": display_name(m),
                "RMSE": float(r["test_rmse"]),
                "MAE": float(r["test_mae"]),
                "R2": float(r["test_r2"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("RMSE")

    # Plot order: CNN-LSTM on the left, PI-CNNLSTM on the right (others keep stable)
    preferred_order = ["CNN-LSTM", "PI-CNNLSTM"]
    df_plot = df.copy()
    df_plot["__order"] = df_plot["display_name"].apply(
        lambda name: preferred_order.index(name) if name in preferred_order else len(preferred_order)
    )
    df_plot = df_plot.sort_values(["__order", "display_name"]).drop(columns=["__order"])

    model_labels = df_plot["display_name"].tolist()
    rmse_values = (df_plot["RMSE"] * 100).tolist()
    mae_values = (df_plot["MAE"] * 100).tolist()
    r2_values = df_plot["R2"].tolist()

    colors = plt.cm.Set3(np.linspace(0, 1, len(model_labels)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RMSE??
    ax1 = axes[0]
    bars1 = ax1.bar(model_labels, rmse_values, color=colors)
    ax1.set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    ax1.set_ylabel("RMSE (%)")
    ax1.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

    # MAE??
    ax2 = axes[1]
    bars2 = ax2.bar(model_labels, mae_values, color=colors)
    ax2.set_title("MAE Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MAE (%)")
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}%", ha="center", va="bottom", fontsize=8)

    # R² 对比（与 compare_models.py 一致的文字显示）
    ax3 = axes[2]
    bars3 = ax3.bar(model_labels, r2_values, color=colors)
    ax3.set_title("R² Comparison", fontsize=12, fontweight="bold")
    ax3.set_ylabel("R²")
    ax3.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars3, r2_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "metric_bar_comparison_rmse_mae_r2.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

    df.to_csv(os.path.join(save_dir, "metric_summary.csv"), index=False)
    return df


def extract_battery_data(result: dict, battery_id: str) -> tuple[np.ndarray, np.ndarray]:
    battery_ids = np.array(result["battery_ids"]).astype(str)
    pred = np.array(result["predictions"], dtype=float)
    tgt = np.array(result["targets"], dtype=float)
    mask = battery_ids == battery_id
    return tgt[mask], pred[mask]


def plot_selected_battery_trajectories(results: dict, save_dir: str) -> None:
    available_models = [m for m in PLOT_COMPARE_MODELS if m in results]
    if len(available_models) == 0:
        print("[INFO] No models available for battery trajectory/error plots.")
        return

    # 与 analyze_unseen_batteries.py 风格一致：使用 Set2 调色板
    palette = plt.cm.Set2(np.linspace(0, 1, max(3, len(available_models))))
    pairs = [(m, palette[i % len(palette)]) for i, m in enumerate(available_models)]

    # SOH trajectory
    fig, axes = plt.subplots(len(SELECTED_BATTERIES), 1, figsize=(12, 3.8 * len(SELECTED_BATTERIES)), squeeze=False)
    axes = axes.flatten()
    battery_rows = []

    for i, bid in enumerate(SELECTED_BATTERIES):
        ax = axes[i]
        plotted_true = False

        for m, color in pairs:
            y_true, y_pred = extract_battery_data(results[m], bid)
            if len(y_true) == 0:
                continue
            x = np.arange(len(y_true))
            if not plotted_true:
                ax.plot(x, y_true, color="black", linewidth=2.2, label="True SOH", alpha=0.85)
                plotted_true = True

            cur_rmse = rmse(y_true, y_pred)
            cur_mae = mae(y_true, y_pred)
            ax.plot(
                x,
                y_pred,
                color=color,
                linewidth=1.6,
                alpha=0.9,
                label=f"{display_name(m)} (RMSE={cur_rmse*100:.3f}%, MAE={cur_mae*100:.3f}%)",
            )

            battery_rows.append(
                {
                    "battery": bid,
                    "model": m,
                    "display_name": display_name(m),
                    "count": int(len(y_true)),
                    "RMSE": cur_rmse,
                    "MAE": cur_mae,
                    "R2": r2(y_true, y_pred),
                }
            )

        if not plotted_true:
            ax.text(0.5, 0.5, f"Battery {bid} not found in current test split", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Battery {bid} (Unseen)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Window Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("SOH", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "selected_batteries_soh_trajectory_compare.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

    # Error curves
    fig, axes = plt.subplots(len(SELECTED_BATTERIES), 1, figsize=(12, 3.8 * len(SELECTED_BATTERIES)), squeeze=False)
    axes = axes.flatten()

    for i, bid in enumerate(SELECTED_BATTERIES):
        ax = axes[i]
        ax.axhline(0.0, color="red", linestyle="--", linewidth=2.0, alpha=0.8, label="Zero Error")
        has_data = False

        for m, color in pairs:
            y_true, y_pred = extract_battery_data(results[m], bid)
            if len(y_true) == 0:
                continue
            has_data = True
            x = np.arange(len(y_true))
            err = y_pred - y_true
            cur_rmse = rmse(y_true, y_pred)
            cur_mae = mae(y_true, y_pred)
            ax.plot(
                x,
                err,
                color=color,
                linewidth=1.6,
                alpha=0.9,
                label=f"{display_name(m)} (RMSE={cur_rmse*100:.3f}%, MAE={cur_mae*100:.3f}%)",
            )

        if not has_data:
            ax.text(0.5, 0.5, f"Battery {bid} not found in current test split", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Battery {bid} Error Curve (Pred - True)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Window Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Error", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "selected_batteries_error_curve_compare.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_path}")

    if battery_rows:
        pd.DataFrame(battery_rows).to_csv(os.path.join(save_dir, "selected_battery_metrics.csv"), index=False)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    device = "cuda" if (DEVICE == "auto" and torch.cuda.is_available()) else ("cpu" if DEVICE == "auto" else DEVICE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{OUTPUT_DIR}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Output dir: {save_dir}")
    results = train_models(device=device)

    # 保存原始结果
    with open(os.path.join(save_dir, "detailed_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    meta = {
        "models": MODELS_TO_TEST,
        "seed": RANDOM_SEED,
        "device": device,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "degradation_scenario": DEGRADATION_SCENARIO,
        "cycle_drop_rate": CYCLE_DROP_RATE,
        "cycle_drop_num_gaps": CYCLE_DROP_NUM_GAPS,
        "selected_batteries": SELECTED_BATTERIES,
    }
    with open(os.path.join(save_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    plot_scatter_comparison(results, save_dir)
    plot_metric_bars(results, save_dir)
    plot_selected_battery_trajectories(results, save_dir)

    print("\nAll done.")
    print("Generated files:")
    print("- scatter_comparison_all_test.png")
    print("- metric_bar_comparison_rmse_mae_r2.png")
    print("- selected_batteries_soh_trajectory_compare.png")
    print("- selected_batteries_error_curve_compare.png")
    print("- metric_summary.csv")
    print("- selected_battery_metrics.csv (if selected batteries exist)")
    print("- detailed_results.pkl")
    print("- run_meta.json")


if __name__ == "__main__":
    main()
