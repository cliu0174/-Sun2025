"""Generate auditable Springer result tables from the formal paper archive.

The script intentionally refuses incomplete or unpaired experiment matrices.
All displayed point-error values are converted from SOH fractions to percentage
points, while R2 and the already-percent monotonicity rate retain their native
units.  Smoke runs may coexist in ``summary.json`` but are never accepted as
evidence or included in the generated tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PIPELINE_REVISION = "v3_leakage_free"
MODEL_ORDER = ("single_no_pi", "single_pi", "multi_no_pi", "multi_pi")
MODEL_LABELS = {
    "single_no_pi": "CNN--LSTM",
    "single_pi": "PI--CNN--LSTM",
    "multi_no_pi": "MS--CNN--LSTM",
    "multi_pi": "PI--MSCL",
}
RATIOS = (1.0, 0.7, 0.5, 0.3)
RUN_METRICS = (
    "test_mae",
    "test_rmse",
    "test_r2",
    "monotonicity_violation_rate_percent",
    "mean_cumulative_upward_excess_per_battery",
    "mean_absolute_second_difference",
)


def ratio_tag(ratio: float) -> str:
    return f"r{ratio:.1f}".replace(".", "p")


def run_pair_key(run: Mapping[str, Any]) -> Tuple[int, int, int]:
    return (
        int(run["split_seed"]),
        int(run["mask_seed"]),
        int(run["training_seed"]),
    )


def formal_runs(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(run) for run in summary.get("runs", []) if not run.get("smoke", False)]


def validate_summary(
    summary: Mapping[str, Any], *, expected_repeats: int = 3
) -> List[Dict[str, Any]]:
    if summary.get("pipeline_revision") != PIPELINE_REVISION:
        raise ValueError(
            f"expected pipeline_revision={PIPELINE_REVISION!r}, "
            f"got {summary.get('pipeline_revision')!r}"
        )
    runs = formal_runs(summary)
    expected_total = len(MODEL_ORDER) * len(RATIOS) * expected_repeats
    if len(runs) != expected_total:
        raise ValueError(
            f"formal matrix is incomplete: expected {expected_total} runs, got {len(runs)}"
        )

    by_cell: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    seen_run_ids = set()
    for run in runs:
        run_id = run.get("run_id")
        if not run_id or run_id in seen_run_ids:
            raise ValueError(f"missing or duplicate run_id: {run_id!r}")
        seen_run_ids.add(run_id)
        if run.get("pipeline_revision") != PIPELINE_REVISION:
            raise ValueError(f"run {run_id} has a mismatched pipeline revision")
        model_id = run.get("model_id")
        ratio = float(run.get("supervision_ratio"))
        if model_id not in MODEL_ORDER or ratio not in RATIOS:
            raise ValueError(f"unexpected matrix cell in run {run_id}: {model_id}, {ratio}")
        missing = [metric for metric in RUN_METRICS if metric not in run]
        if missing:
            raise ValueError(f"run {run_id} is missing metrics: {missing}")
        by_cell[(model_id, ratio)].append(run)

    for model_id in MODEL_ORDER:
        for ratio in RATIOS:
            cell = by_cell[(model_id, ratio)]
            if len(cell) != expected_repeats:
                raise ValueError(
                    f"{model_id}/{ratio_tag(ratio)} has {len(cell)} formal runs; "
                    f"expected {expected_repeats}"
                )

    for ratio in RATIOS:
        reference = {run_pair_key(run) for run in by_cell[(MODEL_ORDER[0], ratio)]}
        for model_id in MODEL_ORDER[1:]:
            current = {run_pair_key(run) for run in by_cell[(model_id, ratio)]}
            if current != reference:
                raise ValueError(
                    f"unpaired seeds at {ratio_tag(ratio)}: {model_id} differs from "
                    f"{MODEL_ORDER[0]}"
                )
    return runs


def cell_runs(
    runs: Iterable[Mapping[str, Any]], model_id: str, ratio: float
) -> List[Mapping[str, Any]]:
    return sorted(
        (
            run
            for run in runs
            if run["model_id"] == model_id
            and abs(float(run["supervision_ratio"]) - ratio) < 1e-12
        ),
        key=run_pair_key,
    )


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def metric_stats(
    runs: Iterable[Mapping[str, Any]], model_id: str, ratio: float, metric: str
) -> Tuple[float, float]:
    values = [float(run[metric]) for run in cell_runs(runs, model_id, ratio)]
    return mean_sd(values)


def fmt(mean: float, sd: float, *, scale: float = 1.0, digits: int = 3) -> str:
    return f"{mean * scale:.{digits}f} \\textpm{{}} {sd * scale:.{digits}f}"


def table_full_supervision(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for model_id in MODEL_ORDER:
        mae = metric_stats(runs, model_id, 1.0, "test_mae")
        rmse = metric_stats(runs, model_id, 1.0, "test_rmse")
        r2 = metric_stats(runs, model_id, 1.0, "test_r2")
        rows.append(
            f"{MODEL_LABELS[model_id]} & {fmt(*mae, scale=100)} & "
            f"{fmt(*rmse, scale=100)} & {fmt(*r2, digits=4)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{Full-supervision performance on unseen HUST cells.}",
            r"\label{tab:full_supervision}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.5pt}",
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


def table_label_budget(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for model_id in MODEL_ORDER:
        cells = [
            fmt(*metric_stats(runs, model_id, ratio, "test_mae"), scale=100)
            for ratio in RATIOS
        ]
        rows.append(f"{MODEL_LABELS[model_id]} & " + " & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{MAE under decreasing training-label budgets.}",
            r"\label{tab:label_budget_mae}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & r = 1.0 & r = 0.7 & r = 0.5 & r = 0.3 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_label_budget_r2(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for model_id in MODEL_ORDER:
        cells = [
            fmt(*metric_stats(runs, model_id, ratio, "test_r2"), digits=4)
            for ratio in RATIOS
        ]
        rows.append(f"{MODEL_LABELS[model_id]} & " + " & ".join(cells) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{R-squared under decreasing training-label budgets.}",
            r"\label{tab:label_budget_r2}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & r = 1.0 & r = 0.7 & r = 0.5 & r = 0.3 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_trajectory(runs: List[Dict[str, Any]], ratio: float = 0.3) -> str:
    rows = []
    for model_id in MODEL_ORDER:
        mono = metric_stats(
            runs, model_id, ratio, "monotonicity_violation_rate_percent"
        )
        upward = metric_stats(
            runs, model_id, ratio, "mean_cumulative_upward_excess_per_battery"
        )
        roughness = metric_stats(
            runs, model_id, ratio, "mean_absolute_second_difference"
        )
        rows.append(
            f"{MODEL_LABELS[model_id]} & {fmt(*mono)} & "
            f"{fmt(*upward, scale=100, digits=4)} & "
            f"{fmt(*roughness, scale=100, digits=4)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            rf"\caption{{Trajectory metrics at label-retention ratio r = {ratio:.1f}.}}",
            r"\label{tab:trajectory_metrics_sparse}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.5pt}",
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


def paired_factorial_effects(
    runs: List[Dict[str, Any]], ratio: float
) -> Dict[str, Tuple[float, float]]:
    by_model = {
        model_id: {
            run_pair_key(run): float(run["test_mae"])
            for run in cell_runs(runs, model_id, ratio)
        }
        for model_id in MODEL_ORDER
    }
    keys = sorted(by_model[MODEL_ORDER[0]])
    arch, physics, interaction = [], [], []
    for key in keys:
        q00 = by_model["single_no_pi"][key]
        q01 = by_model["single_pi"][key]
        q10 = by_model["multi_no_pi"][key]
        q11 = by_model["multi_pi"][key]
        arch.append(0.5 * ((q00 - q10) + (q01 - q11)))
        physics.append(0.5 * ((q00 - q01) + (q10 - q11)))
        interaction.append((q10 - q11) - (q00 - q01))
    return {
        "architecture": mean_sd(arch),
        "physics": mean_sd(physics),
        "interaction": mean_sd(interaction),
    }


def table_factorial(runs: List[Dict[str, Any]]) -> str:
    rows = []
    for ratio in RATIOS:
        effects = paired_factorial_effects(runs, ratio)
        rows.append(
            f"{ratio:.1f} & {fmt(*effects['architecture'], scale=100)} & "
            f"{fmt(*effects['physics'], scale=100)} & "
            f"{fmt(*effects['interaction'], scale=100)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\caption{Paired factorial effects on MAE.}",
            r"\label{tab:factorial_effects}",
            r"\centering",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"r & MS effect (pp) & PI effect (pp) & Interaction (pp) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def generate(
    summary_path: Path, output_dir: Path, *, expected_repeats: int = 3
) -> List[Path]:
    raw = summary_path.read_bytes()
    summary = json.loads(raw.decode("utf-8"))
    runs = validate_summary(summary, expected_repeats=expected_repeats)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table_full_supervision.tex": table_full_supervision(runs),
        "table_label_budget.tex": table_label_budget(runs),
        "table_label_budget_r2.tex": table_label_budget_r2(runs),
        "table_trajectory_r0p3.tex": table_trajectory(runs),
        "table_factorial_effects.tex": table_factorial(runs),
    }
    written = []
    for name, content in tables.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(summary_path.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "pipeline_revision": PIPELINE_REVISION,
        "expected_repeats": expected_repeats,
        "formal_run_count": len(runs),
        "models": list(MODEL_ORDER),
        "ratios": list(RATIOS),
        "units": {
            "mae_rmse": "SOH percentage points",
            "r2": "dimensionless",
            "monotonicity_violation_rate": "percent of eligible adjacent pairs",
            "upward_excess_roughness": "SOH percentage points",
        },
        "files": list(tables),
    }
    manifest_path = output_dir / "results_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = generate(
        args.summary.resolve(),
        args.output_dir.resolve(),
        expected_repeats=args.expected_repeats,
    )
    print("Generated Springer result assets:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
