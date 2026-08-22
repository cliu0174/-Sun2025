"""Generate same-protocol baseline tables from two complete paper archives.

The primary archive contains the four factorial PI-MSCL variants over four
label budgets.  The supporting archive contains four classical/modern
baselines at the full- and sparse-label endpoints.  Published values from
heterogeneous protocols are intentionally excluded from these tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_springer_results import (
    MODEL_LABELS as PRIMARY_LABELS,
    MODEL_ORDER as PRIMARY_ORDER,
    PIPELINE_REVISION as PRIMARY_REVISION,
    metric_stats,
    run_pair_key,
    validate_summary as validate_primary_summary,
)


BASELINE_REVISION = "v1_leakage_free_endpoints"
BASELINE_ORDER = ("xgboost", "lstm", "gru", "attn_cnn_lstm")
BASELINE_LABELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "attn_cnn_lstm": "Attn. CNN--LSTM",
}
RATIOS = (1.0, 0.3)
EXPECTED_REPEATS = 3


def formal_runs(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in summary.get("runs", []) if not row.get("smoke", False)]


def validate_baseline_summary(
    summary: Mapping[str, Any], *, expected_repeats: int = EXPECTED_REPEATS
) -> List[Dict[str, Any]]:
    if summary.get("pipeline_revision") != BASELINE_REVISION:
        raise ValueError(
            f"expected baseline pipeline_revision={BASELINE_REVISION!r}, "
            f"got {summary.get('pipeline_revision')!r}"
        )
    runs = formal_runs(summary)
    expected_total = len(BASELINE_ORDER) * len(RATIOS) * expected_repeats
    if len(runs) != expected_total:
        raise ValueError(
            f"baseline matrix is incomplete: expected {expected_total} runs, got {len(runs)}"
        )

    seen = set()
    for run in runs:
        run_id = run.get("run_id")
        if not run_id or run_id in seen:
            raise ValueError(f"missing or duplicate baseline run_id: {run_id!r}")
        seen.add(run_id)
        if run.get("pipeline_revision") != BASELINE_REVISION:
            raise ValueError(f"baseline run {run_id} has a mismatched revision")
        if run.get("model_id") not in BASELINE_ORDER:
            raise ValueError(f"unexpected baseline model in run {run_id}")
        if float(run.get("supervision_ratio")) not in RATIOS:
            raise ValueError(f"unexpected baseline ratio in run {run_id}")
        missing = [key for key in ("test_mae", "test_rmse", "test_r2") if key not in run]
        if missing:
            raise ValueError(f"baseline run {run_id} is missing metrics: {missing}")

    for model_id in BASELINE_ORDER:
        for ratio in RATIOS:
            cell = [
                run
                for run in runs
                if run["model_id"] == model_id
                and abs(float(run["supervision_ratio"]) - ratio) < 1e-12
            ]
            if len(cell) != expected_repeats:
                raise ValueError(
                    f"{model_id}/{ratio} has {len(cell)} runs; expected {expected_repeats}"
                )
    return runs


def validate_cross_archive_pairing(
    primary_runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> None:
    for ratio in RATIOS:
        reference = {
            run_pair_key(run)
            for run in primary_runs
            if run["model_id"] == PRIMARY_ORDER[0]
            and abs(float(run["supervision_ratio"]) - ratio) < 1e-12
        }
        for model_id in BASELINE_ORDER:
            current = {
                run_pair_key(run)
                for run in baseline_runs
                if run["model_id"] == model_id
                and abs(float(run["supervision_ratio"]) - ratio) < 1e-12
            }
            if current != reference:
                raise ValueError(
                    f"baseline seed pairs at r={ratio:.1f} do not match the primary archive: "
                    f"{model_id}"
                )


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def fmt(mean: float, sd: float, *, scale: float = 1.0, digits: int = 3) -> str:
    return f"{mean * scale:.{digits}f} \\textpm{{}} {sd * scale:.{digits}f}"


def stats(
    runs: Sequence[Mapping[str, Any]], model_id: str, ratio: float, metric: str
) -> Tuple[float, float]:
    return mean_sd(
        float(run[metric])
        for run in runs
        if run["model_id"] == model_id
        and abs(float(run["supervision_ratio"]) - ratio) < 1e-12
    )


def model_source(
    model_id: str,
    primary_runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> tuple[str, Sequence[Mapping[str, Any]]]:
    if model_id in BASELINE_ORDER:
        return BASELINE_LABELS[model_id], baseline_runs
    return PRIMARY_LABELS[model_id], primary_runs


def table_full_supervision(
    primary_runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for model_id in (*BASELINE_ORDER, *PRIMARY_ORDER):
        label, source = model_source(model_id, primary_runs, baseline_runs)
        mae = stats(source, model_id, 1.0, "test_mae")
        rmse = stats(source, model_id, 1.0, "test_rmse")
        r2 = stats(source, model_id, 1.0, "test_r2")
        rows.append(
            f"{label} & {fmt(*mae, scale=100)} & {fmt(*rmse, scale=100)} & "
            f"{fmt(*r2, digits=4)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{Same-protocol full-supervision comparison.}",
            r"\label{tab:all_models_full_supervision}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & MAE (\%) & RMSE (\%) & R-squared \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def paired_endpoint_change(
    runs: Sequence[Mapping[str, Any]], model_id: str
) -> Tuple[float, float]:
    dense = {
        run_pair_key(run): float(run["test_mae"])
        for run in runs
        if run["model_id"] == model_id
        and abs(float(run["supervision_ratio"]) - 1.0) < 1e-12
    }
    sparse = {
        run_pair_key(run): float(run["test_mae"])
        for run in runs
        if run["model_id"] == model_id
        and abs(float(run["supervision_ratio"]) - 0.3) < 1e-12
    }
    if dense.keys() != sparse.keys():
        raise ValueError(f"unpaired endpoint runs for {model_id}")
    return mean_sd(sparse[key] - dense[key] for key in sorted(dense))


def table_endpoint_robustness(
    primary_runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for model_id in (*BASELINE_ORDER, *PRIMARY_ORDER):
        label, source = model_source(model_id, primary_runs, baseline_runs)
        dense = stats(source, model_id, 1.0, "test_mae")
        sparse = stats(source, model_id, 0.3, "test_mae")
        change = paired_endpoint_change(source, model_id)
        rows.append(
            f"{label} & {fmt(*dense, scale=100)} & {fmt(*sparse, scale=100)} & "
            f"{fmt(*change, scale=100)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{Same-protocol endpoint sensitivity to label budget.}",
            r"\label{tab:all_models_endpoint_budget}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & \multicolumn{2}{c}{MAE (\%)} & Change (pp) \\",
            r"\cmidrule(lr){2-3}",
            r" & r = 1.0 & r = 0.3 & 0.3--1.0 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_sparse_trajectory(
    primary_runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for model_id in (*BASELINE_ORDER, *PRIMARY_ORDER):
        label, source = model_source(model_id, primary_runs, baseline_runs)
        mono = stats(
            source, model_id, 0.3, "monotonicity_violation_rate_percent"
        )
        upward = stats(
            source, model_id, 0.3, "mean_cumulative_upward_excess_per_battery"
        )
        roughness = stats(
            source, model_id, 0.3, "mean_absolute_second_difference"
        )
        rows.append(
            f"{label} & {fmt(*mono, digits=3)} & {fmt(*upward, scale=100, digits=4)} & "
            f"{fmt(*roughness, scale=100, digits=4)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{Same-protocol trajectory metrics at 30\% label retention.}",
            r"\label{tab:all_models_trajectory_sparse}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & Mono. viol. (\%) & Upward (pp) & Roughness (pp) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def generate(primary_path: Path, baseline_path: Path, output_dir: Path) -> List[Path]:
    primary_raw = primary_path.read_bytes()
    baseline_raw = baseline_path.read_bytes()
    primary = json.loads(primary_raw.decode("utf-8"))
    baseline = json.loads(baseline_raw.decode("utf-8"))
    if primary.get("pipeline_revision") != PRIMARY_REVISION:
        raise ValueError("primary archive has a mismatched pipeline revision")
    primary_runs = validate_primary_summary(primary, expected_repeats=EXPECTED_REPEATS)
    baseline_runs = validate_baseline_summary(baseline)
    validate_cross_archive_pairing(primary_runs, baseline_runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_all_models_full_supervision.tex": table_full_supervision(
            primary_runs, baseline_runs
        ),
        "table_all_models_endpoint_budget.tex": table_endpoint_robustness(
            primary_runs, baseline_runs
        ),
        "table_all_models_trajectory_r0p3.tex": table_sparse_trajectory(
            primary_runs, baseline_runs
        ),
    }
    written = []
    for name, content in tables.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_source": str(primary_path.resolve()),
        "primary_sha256": hashlib.sha256(primary_raw).hexdigest(),
        "baseline_source": str(baseline_path.resolve()),
        "baseline_sha256": hashlib.sha256(baseline_raw).hexdigest(),
        "formal_primary_runs": len(primary_runs),
        "formal_baseline_runs": len(baseline_runs),
        "files": list(tables),
    }
    manifest_path = output_dir / "baseline_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_summary", type=Path)
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = generate(
        args.primary_summary.resolve(),
        args.baseline_summary.resolve(),
        args.output_dir.resolve(),
    )
    print("Generated same-protocol comparison assets:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
