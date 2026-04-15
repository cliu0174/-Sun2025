import os
import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
SELECTED_BATTERIES = ["1-4", "9-7", "3-5"]

# Preferred: one compare_models output file containing all models
DETAILED_RESULTS_PKL = ""

# Fallback: per-model result files (from existing trained runs)
MODEL_RESULT_FILES = {
    "xgboost_simple": "results/cross_battery/xgboost_simple/results.pkl",
    "lstm": "results/cross_battery/lstm/results.pkl",
    "cnn_lstm": "results/cross_battery/cnn_lstm/results.pkl",
    "pi_cnn_lstm": "results/cross_battery/pi_cnn_lstm/results.pkl",
}

OUTPUT_DIR = "results/unseen_battery_analysis_443"
STAGE_BINS = [(0.0, 0.3, "Early"), (0.3, 0.7, "Mid"), (0.7, 1.0, "Late")]

# Display-only relabel for figures (does not change underlying metrics)
DISPLAY_NAME_MAP = {
    "xgboost_simple": "XGBoost",
    "lstm": "LSTM",
    "cnn_lstm": "CNN-LSTM",
    "pi_cnn_lstm": "PI-CNNLSTM",
}


# =========================
# Utilities
# =========================
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


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _validate_result(result: dict, model_name: str) -> None:
    needed = ["predictions", "targets", "battery_ids"]
    for k in needed:
        if k not in result:
            raise ValueError(f"{model_name}: missing key '{k}' in results")
    n_pred = len(result["predictions"])
    n_tgt = len(result["targets"])
    n_bid = len(result["battery_ids"])
    if not (n_pred == n_tgt == n_bid):
        raise ValueError(
            f"{model_name}: length mismatch predictions/targets/battery_ids = "
            f"{n_pred}/{n_tgt}/{n_bid}"
        )


def load_results() -> Dict[str, dict]:
    model_results: Dict[str, dict] = {}

    if DETAILED_RESULTS_PKL and os.path.exists(DETAILED_RESULTS_PKL):
        with open(DETAILED_RESULTS_PKL, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("DETAILED_RESULTS_PKL is not a model->result dict")
        for model_name, result in data.items():
            _validate_result(result, model_name)
            model_results[model_name] = result
        return model_results

    for model_name, pkl_path in MODEL_RESULT_FILES.items():
        if not os.path.exists(pkl_path):
            print(f"[WARN] skip {model_name}: file not found -> {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        _validate_result(result, model_name)
        model_results[model_name] = result

    if not model_results:
        raise FileNotFoundError(
            "No usable result file found. Set DETAILED_RESULTS_PKL or MODEL_RESULT_FILES correctly."
        )

    return model_results


def extract_battery_arrays(result: dict, battery_id: str) -> Tuple[np.ndarray, np.ndarray]:
    battery_ids = np.array(result["battery_ids"]).astype(str)
    preds = np.array(result["predictions"], dtype=float)
    tgts = np.array(result["targets"], dtype=float)
    mask = battery_ids == battery_id
    return tgts[mask], preds[mask]


def calc_stage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    n = len(y_true)
    if n == 0:
        return []
    if n == 1:
        norm_pos = np.array([0.0])
    else:
        norm_pos = np.arange(n) / (n - 1)

    rows = []
    for low, high, stage_name in STAGE_BINS:
        if stage_name == "Late":
            mask = (norm_pos >= low) & (norm_pos <= high)
        else:
            mask = (norm_pos >= low) & (norm_pos < high)
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) == 0:
            rows.append({"stage": stage_name, "count": 0, "rmse": np.nan, "mae": np.nan})
        else:
            rows.append(
                {
                    "stage": stage_name,
                    "count": int(len(yt)),
                    "rmse": rmse(yt, yp),
                    "mae": mae(yt, yp),
                }
            )
    return rows


def display_name(model_name: str) -> str:
    return DISPLAY_NAME_MAP.get(model_name, model_name)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_results = load_results()
    model_names = list(model_results.keys())
    print(f"[INFO] loaded models: {model_names}")

    # Verify selected batteries exist in at least one model
    available_battery_set = set()
    for result in model_results.values():
        available_battery_set.update(np.array(result["battery_ids"]).astype(str).tolist())

    valid_batteries = [b for b in SELECTED_BATTERIES if b in available_battery_set]
    missing_batteries = [b for b in SELECTED_BATTERIES if b not in available_battery_set]

    if missing_batteries:
        print(f"[WARN] selected batteries not found in loaded results: {missing_batteries}")
    if not valid_batteries:
        raise ValueError("None of SELECTED_BATTERIES exists in loaded model results")

    # Per-battery and stage metrics
    per_battery_rows = []
    stage_rows = []
    pooled_rows = []

    for model_name, result in model_results.items():
        pooled_true = []
        pooled_pred = []

        for battery_id in valid_batteries:
            y_true, y_pred = extract_battery_arrays(result, battery_id)
            if len(y_true) == 0:
                print(f"[WARN] {model_name}: no samples for battery {battery_id}")
                continue

            pooled_true.append(y_true)
            pooled_pred.append(y_pred)

            per_battery_rows.append(
                {
                    "model": model_name,
                    "battery": battery_id,
                    "count": int(len(y_true)),
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "r2": r2(y_true, y_pred),
                    "bias": bias(y_true, y_pred),
                }
            )

            for s in calc_stage_metrics(y_true, y_pred):
                stage_rows.append(
                    {
                        "model": model_name,
                        "battery": battery_id,
                        "stage": s["stage"],
                        "count": s["count"],
                        "rmse": s["rmse"],
                        "mae": s["mae"],
                    }
                )

        if pooled_true:
            all_true = np.concatenate(pooled_true)
            all_pred = np.concatenate(pooled_pred)
            pooled_rows.append(
                {
                    "model": model_name,
                    "count": int(len(all_true)),
                    "rmse": rmse(all_true, all_pred),
                    "mae": mae(all_true, all_pred),
                    "r2": r2(all_true, all_pred),
                    "bias": bias(all_true, all_pred),
                }
            )

    if not per_battery_rows:
        raise RuntimeError("No usable data found for selected batteries")

    per_battery_df = pd.DataFrame(per_battery_rows)
    stage_df = pd.DataFrame(stage_rows)
    pooled_df = pd.DataFrame(pooled_rows)

    # Macro metrics (average over batteries to avoid long-life battery dominance)
    macro_df = (
        per_battery_df.groupby("model", as_index=False)[["rmse", "mae", "r2", "bias"]]
        .mean()
        .rename(columns={"rmse": "rmse_macro", "mae": "mae_macro", "r2": "r2_macro", "bias": "bias_macro"})
    )

    summary_df = pooled_df.merge(macro_df, on="model", how="left").sort_values("rmse")

    stage_summary_df = (
        stage_df.groupby(["model", "stage"], as_index=False)[["rmse", "mae"]].mean()
    )

    # Save tables
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "overall_metrics_selected.csv"), index=False)
    per_battery_df.to_csv(os.path.join(OUTPUT_DIR, "per_battery_metrics.csv"), index=False)
    stage_df.to_csv(os.path.join(OUTPUT_DIR, "stage_metrics_per_battery.csv"), index=False)
    stage_summary_df.to_csv(os.path.join(OUTPUT_DIR, "stage_metrics_summary.csv"), index=False)

    meta = {
        "selected_batteries": valid_batteries,
        "missing_batteries": missing_batteries,
        "models": model_names,
        "stage_bins": [{"name": s[2], "range": [s[0], s[1]]} for s in STAGE_BINS],
    }
    with open(os.path.join(OUTPUT_DIR, "analysis_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Plot 1: trajectories on selected batteries
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(len(valid_batteries), 1, figsize=(12, 3.8 * len(valid_batteries)), squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(model_names))))

    for i, battery_id in enumerate(valid_batteries):
        ax = axes[i]
        true_plotted = False

        for m_idx, model_name in enumerate(model_names):
            y_true, y_pred = extract_battery_arrays(model_results[model_name], battery_id)
            if len(y_true) == 0:
                continue

            x = np.arange(len(y_true))
            if not true_plotted:
                ax.plot(x, y_true, color="black", linewidth=2.2, label="True SOH", alpha=0.85)
                true_plotted = True

            cur_rmse = rmse(y_true, y_pred)
            cur_rmse_pct = cur_rmse * 100.0
            ax.plot(
                x,
                y_pred,
                linewidth=1.6,
                alpha=0.9,
                color=colors[m_idx % len(colors)],
                label=f"{display_name(model_name)} (RMSE={cur_rmse_pct:.3f}%)",
            )

        ax.set_title(f"Battery {battery_id} (Unseen)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("SOH")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_4_8_unseen_battery_trajectories.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 1b: error curves on selected batteries
    fig, axes = plt.subplots(len(valid_batteries), 1, figsize=(12, 3.8 * len(valid_batteries)), squeeze=False)
    axes = axes.flatten()

    for i, battery_id in enumerate(valid_batteries):
        ax = axes[i]
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero Error")

        for m_idx, model_name in enumerate(model_names):
            y_true, y_pred = extract_battery_arrays(model_results[model_name], battery_id)
            if len(y_true) == 0:
                continue

            x = np.arange(len(y_true))
            err = y_pred - y_true
            cur_rmse = rmse(y_true, y_pred)
            cur_mae = mae(y_true, y_pred)
            cur_rmse_pct = cur_rmse * 100.0
            cur_mae_pct = cur_mae * 100.0
            ax.plot(
                x,
                err,
                linewidth=1.6,
                alpha=0.9,
                color=colors[m_idx % len(colors)],
                label=f"{display_name(model_name)} (RMSE={cur_rmse_pct:.3f}%, MAE={cur_mae_pct:.3f}%)",
            )

        ax.set_title(f"Battery {battery_id} Prediction Error (Unseen)")
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Pred - True")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_4_8b_unseen_battery_error_curves.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: stage RMSE heatmap (model x stage)
    stage_order = ["Early", "Mid", "Late"]
    heat_df = stage_summary_df.copy()
    heat_df["stage"] = pd.Categorical(heat_df["stage"], categories=stage_order, ordered=True)
    heat_df = heat_df.sort_values(["model", "stage"])
    pivot = heat_df.pivot(index="model", columns="stage", values="rmse") * 100.0
    y_labels = [display_name(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(pivot) + 2.2))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(y_labels)
    ax.set_title("Stage-wise RMSE on Unseen Batteries")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("RMSE (%)")

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.3f}%", ha="center", va="center", fontsize=9, color="white")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_4_9_stage_rmse_heatmap.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: bias bar chart (optional but useful for section C)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bias_plot = summary_df.sort_values("bias")
    bias_labels = [display_name(m) for m in bias_plot["model"]]
    bias_values = bias_plot["bias"] * 100.0
    ax.bar(bias_labels, bias_values, color="#4C78A8")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Bias on Selected Unseen Batteries")
    ax.set_ylabel("Bias (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "bias_bar_selected.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n[OK] Analysis completed.")
    print(f"[OK] Output dir: {OUTPUT_DIR}")
    print("[OK] Key files:")
    print("  - overall_metrics_selected.csv")
    print("  - per_battery_metrics.csv")
    print("  - stage_metrics_summary.csv")
    print("  - fig_4_8_unseen_battery_trajectories.png")
    print("  - fig_4_8b_unseen_battery_error_curves.png")
    print("  - fig_4_9_stage_rmse_heatmap.png")
    print("  - bias_bar_selected.png")


if __name__ == "__main__":
    main()
