#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "This script requires `torch` to load .pt result files. "
        "Run it in the experiment environment."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "This script requires `matplotlib`. Install it in the experiment environment and rerun."
    ) from exc


SUMMARY_FILES = {
    "task": "eval_metrics_summary.pt",
    "other": "eval_other_metrics_summary.pt",
    "atom": "eval_atom_metrics_summary.pt",
}

RAW_FILES = {
    "task": "eval_metrics.pt",
    "other": "eval_other_metrics.pt",
    "atom": "eval_atom_metrics.pt",
}

TASK_METRICS = ["rmsd", "bind_ratio", "ss_ratio", "valid", "novel", "diversity"]
OTHER_METRICS = ["sample_CA_dist", "aar"]
ATOM_METRICS = [
    "all_atom_rmsd",
    "sidechain_rmsd",
    "chi1_mae_deg",
    "chi2_mae_deg",
    "chi3_mae_deg",
    "chi4_mae_deg",
    "rotamer_recovery",
    "clash_score",
    "bond_length_dev",
    "bond_angle_dev_deg",
]

PRIMARY_VARIANCE_METRICS = [
    "rmsd",
    "bind_ratio",
    "sample_CA_dist",
    "all_atom_rmsd",
    "sidechain_rmsd",
    "rotamer_recovery",
]

VARIANT_ORDER = ["pepsage", "pep_glad","discrete token","exp8_r6_gaussian_v4_rebalance", "exp8_r6_gaussian_v4_fix","sigma1_z005"]
VARIANT_DISPLAY = {
    "pepsage": "pepsage",
    "sigma1_z005":"sigma1_z005",
    "pep_glad": "pep_glad",
    "sigma1_z015": "sigma1_z015",
    "discrete token": "discrete token",
    "exp8_r6_gaussian_v4_fix": "StructToken (gaussian)",
    "exp8_r6_gaussian_v4_rebalance":"StructToken (gaussian)_no_refine",
    "other": "Other",
}
COLOR_MAP = {
    "pepsage": "#4E79A7",
    "pep_glad": "#59A14F",
    "sigma1_z005": "#B07AA1",
    "sigma1_z015": "#0B044E",
    "discrete token": "#E15759",
    "exp8_r6_gaussian_v4_fix": "#F28E2B",
    "exp8_r6_gaussian_v4_rebalance": "#9D0BB1",
    "other": "#7F7F7F",
}


def infer_variant(name: str) -> str:
    lower = name.lower()
    if "discrete token" in lower:
        return "discrete token"
    if "sigma1_z005" in lower:
        return "sigma1_z005"
    if "sigma1_z015" in lower:
        return "sigma1_z015"
    if "pepsage" in lower:
        return "pepsage"
    if "pep_glad" in lower:
        return "pep_glad"
    if "exp8_r6_gaussian_v4_rebalance" in lower:
        return "exp8_r6_gaussian_v4_rebalance"
    if "exp8_r6_gaussian_v4_fix" in lower:
        return "exp8_r6_gaussian_v4_fix"

    return "other"


def short_label(run_name: str) -> str:
    variant = infer_variant(run_name)
    return VARIANT_DISPLAY.get(variant, run_name)


def metric_direction(metric: str) -> str:
    lower_is_better = {
        "rmsd",
        "sample_CA_dist",
        "aar",
        "all_atom_rmsd",
        "sidechain_rmsd",
        "chi1_mae_deg",
        "chi2_mae_deg",
        "chi3_mae_deg",
        "chi4_mae_deg",
        "clash_score",
        "bond_length_dev",
        "bond_angle_dev_deg",
    }
    return "lower" if metric in lower_is_better else "higher"


def default_log_root() -> Path:
    exp2_root = Path(__file__).resolve().parents[1]
    for candidate in (exp2_root / "log", exp2_root / "logs"):
        if candidate.exists():
            return candidate
    return exp2_root / "log"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_numeric_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        value = list(value)
    else:
        value = [value]
    numbers = []
    for item in value:
        try:
            item = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(item):
            numbers.append(item)
    return numbers


def summarize_values(metric: str, values: list[float]) -> dict[str, float | int | str]:
    if not values:
        return {
            "metric": metric,
            "direction": metric_direction(metric),
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "var": math.nan,
            "min": math.nan,
            "max": math.nan,
            "best": math.nan,
            "worst": math.nan,
        }
    direction = metric_direction(metric)
    return {
        "metric": metric,
        "direction": direction,
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "var": float(np.var(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "best": float(np.min(values) if direction == "lower" else np.max(values)),
        "worst": float(np.max(values) if direction == "lower" else np.min(values)),
    }


def discover_run_dirs(log_root: Path, explicit_run_dirs: list[str] | None) -> list[Path]:
    if explicit_run_dirs:
        return [Path(item).resolve() for item in explicit_run_dirs]

    run_dirs = []
    for summary_name in SUMMARY_FILES.values():
        for path in log_root.rglob(summary_name):
            run_dirs.append(path.parent)
    deduped = sorted(set(run_dirs))
    return deduped


def parse_summary_file(run_name: str, variant: str, kind: str, path: Path) -> tuple[list[dict], list[dict]]:
    obj = load_pt(path)
    aggregate_rows = []
    per_target_rows = []

    aggregate = obj.get("aggregate", {}) if isinstance(obj, dict) else {}
    for metric, stats in aggregate.items():
        aggregate_rows.append(
            {
                "run_name": run_name,
                "variant": variant,
                "kind": kind,
                "metric": metric,
                "direction": metric_direction(metric),
                "num_targets": stats.get("num_targets", 0),
                "mean_of_means": stats.get("mean_of_means", math.nan),
                "std_of_means": stats.get("std_of_means", math.nan),
                "mean_of_stds": stats.get("mean_of_stds", math.nan),
                "mean_of_vars": stats.get("mean_of_vars", math.nan),
            }
        )

    per_target = obj.get("per_target", {}) if isinstance(obj, dict) else {}
    for target_id, metric_map in per_target.items():
        for metric, stats in metric_map.items():
            per_target_rows.append(
                {
                    "run_name": run_name,
                    "variant": variant,
                    "kind": kind,
                    "target_id": target_id,
                    "metric": metric,
                    "direction": metric_direction(metric),
                    "count": stats.get("count", 0),
                    "mean": stats.get("mean", math.nan),
                    "std": stats.get("std", math.nan),
                    "var": stats.get("var", math.nan),
                    "min": stats.get("min", math.nan),
                    "max": stats.get("max", math.nan),
                }
            )

    return aggregate_rows, per_target_rows


def parse_raw_file(run_name: str, variant: str, kind: str, path: Path) -> list[dict]:
    obj = load_pt(path)
    rows = []
    if not isinstance(obj, dict):
        return rows

    for target_id, metric_map in obj.items():
        if not isinstance(metric_map, dict):
            continue
        for metric, value in metric_map.items():
            values = to_numeric_list(value)
            stats = summarize_values(metric, values)
            rows.append(
                {
                    "run_name": run_name,
                    "variant": variant,
                    "kind": kind,
                    "target_id": target_id,
                    **stats,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], field_order: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = field_order or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grouped_rows(rows: list[dict], metrics: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["metric"] in metrics:
            grouped[row["metric"]].append(row)
    return grouped


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else 99,
            row["run_name"],
        ),
    )


def plot_aggregate_bars(path: Path, title: str, rows: list[dict], metrics: list[str], value_key: str, error_key: str | None = None) -> None:
    metric_rows = grouped_rows(rows, metrics)
    available = [metric for metric in metrics if metric_rows.get(metric)]
    if not available:
        return

    cols = 2
    rows_n = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 4.5 * rows_n), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles: dict[str, Any] = {}

    for ax, metric in zip(axes_flat, available):
        items = sort_rows(metric_rows[metric])
        labels = [short_label(item["run_name"]) for item in items]
        values = [item.get(value_key, math.nan) for item in items]
        errors = [item.get(error_key, math.nan) for item in items] if error_key else None
        colors = [COLOR_MAP.get(item["variant"], COLOR_MAP["other"]) for item in items]
        x = np.arange(len(items))
        bars = ax.bar(x, values, color=colors, alpha=0.85)
        for bar, item in zip(bars, items):
            variant = item["variant"]
            if variant not in legend_handles:
                legend_handles[variant] = bar
        if errors:
            clean_errors = [0.0 if not math.isfinite(err) else err for err in errors]
            ax.errorbar(x, values, yerr=clean_errors, fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(value_key)

    for ax in axes_flat[len(available):]:
        ax.axis("off")

    if legend_handles:
        ordered = [(v, VARIANT_DISPLAY.get(v, v)) for v in VARIANT_ORDER if v in legend_handles]
        ordered += [(v, VARIANT_DISPLAY.get(v, v)) for v in legend_handles if v not in VARIANT_ORDER]
        fig.legend(
            [legend_handles[v] for v, _ in ordered],
            [lbl for _, lbl in ordered],
            loc="upper center", ncol=min(4, len(ordered)),
        )
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_target_boxes(path: Path, title: str, rows: list[dict], metrics: list[str], value_key: str) -> None:
    metric_rows = grouped_rows(rows, metrics)
    available = [metric for metric in metrics if metric_rows.get(metric)]
    if not available:
        return

    cols = 2
    rows_n = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7 * cols, 4.5 * rows_n), squeeze=False)
    axes_flat = axes.flatten()

    legend_handles: dict[str, Any] = {}

    for ax, metric in zip(axes_flat, available):
        items = sort_rows(metric_rows[metric])
        by_run: dict[str, list[float]] = defaultdict(list)
        by_variant: dict[str, str] = {}
        for item in items:
            value = item.get(value_key, math.nan)
            if math.isfinite(value):
                by_run[item["run_name"]].append(value)
                by_variant[item["run_name"]] = item["variant"]

        labels_raw = list(by_run.keys())
        labels = [short_label(name) for name in labels_raw]
        data = [by_run[label] for label in labels_raw]
        if not data:
            ax.axis("off")
            continue
        box = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, raw_name in zip(box["boxes"], labels_raw):
            variant = by_variant[raw_name]
            color = COLOR_MAP.get(variant, COLOR_MAP["other"])
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            if variant not in legend_handles:
                legend_handles[variant] = patch
        ax.set_title(metric)
        ax.set_ylabel(value_key)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    for ax in axes_flat[len(available):]:
        ax.axis("off")

    if legend_handles:
        ordered = [(v, VARIANT_DISPLAY.get(v, v)) for v in VARIANT_ORDER if v in legend_handles]
        ordered += [(v, VARIANT_DISPLAY.get(v, v)) for v in legend_handles if v not in VARIANT_ORDER]
        fig.legend(
            [legend_handles[v] for v, _ in ordered],
            [lbl for _, lbl in ordered],
            loc="upper center", ncol=min(4, len(ordered)),
        )
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_best_metric_table(path: Path, rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["metric"]].append(row)

    best_rows = []
    for metric, items in grouped.items():
        direction = metric_direction(metric)
        filtered = [item for item in items if math.isfinite(item["mean_of_means"])]
        if not filtered:
            continue
        best = min(filtered, key=lambda item: item["mean_of_means"]) if direction == "lower" else max(
            filtered, key=lambda item: item["mean_of_means"]
        )
        best_rows.append(best)

    write_csv(
        path,
        sorted(best_rows, key=lambda row: row["metric"]),
        field_order=[
            "metric",
            "direction",
            "run_name",
            "variant",
            "kind",
            "num_targets",
            "mean_of_means",
            "std_of_means",
            "mean_of_stds",
            "mean_of_vars",
        ],
    )


def write_variant_comparison_table(path: Path, rows: list[dict]) -> None:
    by_metric: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        variant = row["variant"]
        metric = row["metric"]
        by_metric[metric][variant] = row

    all_variants = []
    for v in VARIANT_ORDER:
        if any(v in variants for variants in by_metric.values()):
            all_variants.append(v)
    for variants in by_metric.values():
        for v in variants:
            if v not in all_variants:
                all_variants.append(v)

    header = ["metric", "direction", "kind"]
    for v in all_variants:
        display = VARIANT_DISPLAY.get(v, v)
        header.extend([f"{display}_mean", f"{display}_std"])
    header.append("best_variant")

    out_rows: list[dict] = []
    for metric in sorted(by_metric.keys()):
        variants = by_metric[metric]
        direction = metric_direction(metric)
        kind = next((r["kind"] for r in variants.values()), "")
        row_dict: dict[str, Any] = {"metric": metric, "direction": direction, "kind": kind}

        best_variant = None
        best_val = None
        for v in all_variants:
            display = VARIANT_DISPLAY.get(v, v)
            if v in variants:
                val = variants[v]["mean_of_means"]
                std = variants[v]["std_of_means"]
                row_dict[f"{display}_mean"] = f"{val:.4f}" if math.isfinite(val) else ""
                row_dict[f"{display}_std"] = f"{std:.4f}" if math.isfinite(std) else ""
                if math.isfinite(val):
                    if best_val is None:
                        best_val, best_variant = val, v
                    elif direction == "lower" and val < best_val:
                        best_val, best_variant = val, v
                    elif direction == "higher" and val > best_val:
                        best_val, best_variant = val, v
            else:
                row_dict[f"{display}_mean"] = ""
                row_dict[f"{display}_std"] = ""
        row_dict["best_variant"] = VARIANT_DISPLAY.get(best_variant, best_variant or "")
        out_rows.append(row_dict)

    write_csv(path, out_rows, field_order=header)


def build_analysis(run_dirs: list[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

    aggregate_rows: list[dict] = []
    per_target_rows: list[dict] = []
    raw_rows: list[dict] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        variant = infer_variant(run_name)

        for kind, filename in SUMMARY_FILES.items():
            path = run_dir / filename
            if path.exists():
                agg, per_target = parse_summary_file(run_name, variant, kind, path)
                aggregate_rows.extend(agg)
                per_target_rows.extend(per_target)

        for kind, filename in RAW_FILES.items():
            path = run_dir / filename
            if path.exists():
                raw_rows.extend(parse_raw_file(run_name, variant, kind, path))

    aggregate_rows = sort_rows(aggregate_rows)
    per_target_rows = sort_rows(per_target_rows)
    raw_rows = sort_rows(raw_rows)

    write_csv(
        out_dir / "aggregate_summary.csv",
        aggregate_rows,
        field_order=[
            "run_name",
            "variant",
            "kind",
            "metric",
            "direction",
            "num_targets",
            "mean_of_means",
            "std_of_means",
            "mean_of_stds",
            "mean_of_vars",
        ],
    )
    write_csv(
        out_dir / "per_target_summary.csv",
        per_target_rows,
        field_order=[
            "run_name",
            "variant",
            "kind",
            "target_id",
            "metric",
            "direction",
            "count",
            "mean",
            "std",
            "var",
            "min",
            "max",
        ],
    )
    write_csv(
        out_dir / "raw_sample_summary.csv",
        raw_rows,
        field_order=[
            "run_name",
            "variant",
            "kind",
            "target_id",
            "metric",
            "direction",
            "count",
            "mean",
            "std",
            "var",
            "min",
            "max",
            "best",
            "worst",
        ],
    )

    write_best_metric_table(out_dir / "best_run_by_metric.csv", aggregate_rows)
    write_variant_comparison_table(out_dir / "variant_comparison.csv", aggregate_rows)

    task_aggregate = [row for row in aggregate_rows if row["kind"] == "task"]
    other_aggregate = [row for row in aggregate_rows if row["kind"] == "other"]
    atom_aggregate = [row for row in aggregate_rows if row["kind"] == "atom"]

    task_per_target = [row for row in per_target_rows if row["kind"] == "task"]
    other_per_target = [row for row in per_target_rows if row["kind"] == "other"]
    atom_per_target = [row for row in per_target_rows if row["kind"] == "atom"]

    raw_variance_rows = [row for row in raw_rows if row["metric"] in PRIMARY_VARIANCE_METRICS]

    plot_aggregate_bars(
        plot_dir / "01_task_aggregate.png",
        "Task Metrics (aggregate mean_of_means)",
        task_aggregate,
        TASK_METRICS,
        value_key="mean_of_means",
        error_key="std_of_means",
    )
    plot_aggregate_bars(
        plot_dir / "02_other_aggregate.png",
        "Other Metrics (aggregate mean_of_means)",
        other_aggregate,
        OTHER_METRICS,
        value_key="mean_of_means",
        error_key="std_of_means",
    )
    plot_aggregate_bars(
        plot_dir / "03_atom_aggregate.png",
        "Atom Metrics (aggregate mean_of_means)",
        atom_aggregate,
        ATOM_METRICS,
        value_key="mean_of_means",
        error_key="std_of_means",
    )

    plot_per_target_boxes(
        plot_dir / "04_task_per_target_box.png",
        "Task Metrics Across Targets",
        task_per_target,
        TASK_METRICS,
        value_key="mean",
    )
    plot_per_target_boxes(
        plot_dir / "05_other_per_target_box.png",
        "Other Metrics Across Targets",
        other_per_target,
        OTHER_METRICS,
        value_key="mean",
    )
    plot_per_target_boxes(
        plot_dir / "06_atom_per_target_box.png",
        "Atom Metrics Across Targets",
        atom_per_target,
        ATOM_METRICS,
        value_key="mean",
    )
    plot_per_target_boxes(
        plot_dir / "07_sample_variance_box.png",
        "Sample Variance Across Targets",
        raw_variance_rows,
        PRIMARY_VARIANCE_METRICS,
        value_key="std",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze eval .pt files and compare across structtoken (exp4), soft, and soft_gate (exp3) variants."
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Run directory containing eval_*.pt files. Repeat this argument to compare multiple runs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=str(default_log_root()),
        help="Root directory used to auto-discover run directories when --run_dir is not provided.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "eval_analysis_outputs"),
        help="Directory to save CSV summaries and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    run_dirs = discover_run_dirs(log_root, args.run_dir)
    if not run_dirs:
        raise SystemExit("No run directories with eval summary files were found.")
    out_dir = Path(args.out_dir).resolve()
    build_analysis(run_dirs, out_dir)
    print(f"Evaluation analysis complete. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
