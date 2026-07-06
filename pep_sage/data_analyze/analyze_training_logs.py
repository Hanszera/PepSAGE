#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

try:
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.util import tensor_util
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "This script requires the `tensorboard` package. "
        "Install it in the experiment environment and rerun."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "This script requires `matplotlib`. Install it in the experiment environment and rerun."
    ) from exc


EVENT_GLOB = "events.out.tfevents*"
VARIANT_ORDER = ["pepsage", "v3", "v4","sigma1_z005","sigma1_z015"]
COLOR_MAP = {
    "pepsage": "#4E79A7",
    "v3": "#59A14F",
    "v4": "#B07AA1",
    "sigma1_z005": "#E15759",
    "sigma1_z015": "#F28E2B",
    "sigma1_z020":"#F0F322",
    "other": "#7F7F7F",
}

# ── Directly comparable across all variants ──
SHARED_TRAIN_METRICS = [
    "train/loss",
    "train/trans_loss",
    "train/rot_loss",
    "train/bb_atom_loss",
]

SHARED_VAL_METRICS = [
    "val/trans_error",
    "val/rots_error",
    "val/aars_error",
    "val/recon_loss",
]

# ── Sequence / AA prediction (functionally equivalent) ──
SEQ_AA_METRICS = [
    "train/seqs_loss",          # exp3: K=20 discrete BFN
    "train/aa_aux_loss",        # exp4: auxiliary CE
]

# ── Sidechain quality: exp3-only ──
EXP3_SIDECHAIN_METRICS = [
    "val/chi1_error",
    "val/chi2_error",
    "val/chi3_error",
    "val/chi4_error",
    "val/incorrect_portion",
    "train/angle_loss",
    "train/torsion_loss",
]

# ── Token quality: exp4-only ──
EXP4_TOKEN_METRICS = [
    "val/token_acc_dim0",
    "val/token_acc_dim1",
    "val/token_acc_dim2",
    "val/token_acc_dim3",
    "val/token_acc_dim4",
    "val/token_acc_dim5",
    "val/token_acc_joint",
    "val/struct_rmsd",
    "train/token_bfn_loss",
    "train/structure_decode_loss",
]

# ── Reconstruction losses (shared + variant-specific) ──
RECON_METRICS = [
    "val/recon_loss",
    "val/recon_loss_trans",
    "val/recon_loss_rots",
    "val/recon_loss_bb_atom",

    "val/recon_loss_seqs",
    "val/recon_loss_torsion",
    "val/recon_loss_angle",

    "val/recon_loss_token",
    "val/recon_loss_aa",
    "val/recon_loss_struct",
]

EFFICIENCY_METRICS = [
    "train/step_time_ms",
    "train/max_memory_mb",
    "val/sample_runtime_s",
]

DEFAULT_EXPORT_METRICS = (
    SHARED_TRAIN_METRICS + SHARED_VAL_METRICS + SEQ_AA_METRICS
    + EXP3_SIDECHAIN_METRICS + EXP4_TOKEN_METRICS
    + RECON_METRICS + EFFICIENCY_METRICS
)


@dataclass
class ScalarPoint:
    step: int
    wall_time: float
    value: float


def infer_variant(name: str) -> str:
    lower = name.lower()
    if "v1" in lower:
        return "v1"
    if "v2" in lower:
        return "v2"
    if "v3" in lower:
        return "v3"
    if "v4_rebalance" in lower:
        return "v4"
    if "sigma1_z005" in lower:
        return "sigma1_z005"
    if "sigma1_z015" in lower:
        return "sigma1_z015"
    if "pepsage":
        return "pepsage"
    return "other"


def default_log_root() -> Path:
    exp3_root = Path(__file__).resolve().parents[1]
    for candidate in (exp3_root / "log", exp3_root / "logs"):
        if candidate.exists():
            return candidate
    return exp3_root / "log"


def smooth_series(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def metric_direction(tag: str) -> str:
    lower_keywords = ("loss", "error", "runtime", "memory", "incorrect", "rmsd")
    higher_keywords = ("accuracy", "acc", "valid", "bind_ratio", "ss_ratio", "novel", "diversity")
    if any(keyword in tag for keyword in higher_keywords):
        return "higher"
    if any(keyword in tag for keyword in lower_keywords):
        return "lower"
    return "lower"


def finite_indices(values: list[float]) -> list[int]:
    return [idx for idx, value in enumerate(values) if math.isfinite(value)]


def summarize_points(tag: str, points: list[ScalarPoint], tail_n: int) -> dict[str, float | int | str]:
    values = [point.value for point in points]
    steps = [point.step for point in points]
    tail = values[-tail_n:] if values else []
    finite_tail = [value for value in tail if math.isfinite(value)]
    valid_indices = finite_indices(values)
    direction = metric_direction(tag)

    if valid_indices:
        if direction == "lower":
            best_idx = min(valid_indices, key=values.__getitem__)
        else:
            best_idx = max(valid_indices, key=values.__getitem__)
        min_value = min(values[idx] for idx in valid_indices)
        max_value = max(values[idx] for idx in valid_indices)
    else:
        best_idx = None
        min_value = math.nan
        max_value = math.nan

    return {
        "metric": tag,
        "direction": direction,
        "count": len(points),
        "first_step": steps[0],
        "last_step": steps[-1],
        "first_value": values[0],
        "last_value": values[-1],
        "best_value": values[best_idx] if best_idx is not None else math.nan,
        "best_step": steps[best_idx] if best_idx is not None else math.nan,
        "mean_last_n": mean(finite_tail) if finite_tail else math.nan,
        "std_last_n": pstdev(finite_tail) if len(finite_tail) > 1 else 0.0 if len(finite_tail) == 1 else math.nan,
        "min_value": min_value,
        "max_value": max_value,
    }


def scalar_points_from_event(event_path: Path) -> dict[str, list[ScalarPoint]]:
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    tags = ea.Tags()
    series: dict[str, list[ScalarPoint]] = {}

    for tag in tags.get("scalars", []):
        series[tag] = [
            ScalarPoint(step=int(item.step), wall_time=float(item.wall_time), value=float(item.value))
            for item in ea.Scalars(tag)
        ]

    for tag in tags.get("tensors", []):
        if tag in series:
            continue
        points = []
        for item in ea.Tensors(tag):
            value = tensor_util.make_ndarray(item.tensor_proto)
            if getattr(value, "shape", ()) != ():
                continue
            points.append(
                ScalarPoint(
                    step=int(item.step),
                    wall_time=float(item.wall_time),
                    value=float(value.item()),
                )
            )
        if points:
            series[tag] = points

    return series


def discover_runs(log_root: Path) -> list[dict]:
    runs = []
    for event_path in sorted(log_root.rglob(EVENT_GLOB)):
        run_dir = event_path.parent.parent if event_path.parent.name.startswith("version_") else event_path.parent
        run_name = run_dir.name
        variant = infer_variant(run_name)
        runs.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "event_path": event_path,
                "variant": variant,
            }
        )
    runs.sort(key=lambda item: (VARIANT_ORDER.index(item["variant"]) if item["variant"] in VARIANT_ORDER else 99, item["run_name"]))
    return runs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_all_series_csv(out_path: Path, all_series: dict[str, dict[str, list[ScalarPoint]]]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_name", "variant", "metric", "step", "wall_time", "value"])
        writer.writeheader()
        for run_name, metric_map in all_series.items():
            variant = infer_variant(run_name)
            for metric, points in sorted(metric_map.items()):
                for point in points:
                    writer.writerow(
                        {
                            "run_name": run_name,
                            "variant": variant,
                            "metric": metric,
                            "step": point.step,
                            "wall_time": point.wall_time,
                            "value": point.value,
                        }
                    )


def write_summary_csv(out_path: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        return
    fieldnames = [
        "run_name",
        "variant",
        "metric",
        "direction",
        "count",
        "first_step",
        "last_step",
        "first_value",
        "last_value",
        "best_value",
        "best_step",
        "mean_last_n",
        "std_last_n",
        "min_value",
        "max_value",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_metric_group(
    out_path: Path,
    title: str,
    metrics: list[str],
    run_data: dict[str, dict[str, list[ScalarPoint]]],
    smoothing_alpha: float,
    show_raw: bool,
) -> None:
    available_metrics = [metric for metric in metrics if any(metric in per_run for per_run in run_data.values())]
    if not available_metrics:
        return

    cols = 2
    rows = math.ceil(len(available_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, available_metrics):
        for run_name, per_run in run_data.items():
            if metric not in per_run:
                continue
            points = per_run[metric]
            steps = [point.step for point in points]
            values = [point.value for point in points]
            variant = infer_variant(run_name)
            color = COLOR_MAP.get(variant, COLOR_MAP["other"])
            label = run_name
            if show_raw:
                ax.plot(steps, values, color=color, alpha=0.25, linewidth=1.0)
            ax.plot(steps, smooth_series(values, smoothing_alpha), color=color, linewidth=2.0, label=label)
        ax.set_title(metric)
        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(available_metrics):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analysis(
    log_root: Path,
    out_dir: Path,
    smoothing_alpha: float,
    tail_n: int,
    show_raw: bool,
) -> None:
    runs = discover_runs(log_root)
    if not runs:
        raise SystemExit(f"No TensorBoard event files found under: {log_root}")

    ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

    all_series: dict[str, dict[str, list[ScalarPoint]]] = {}
    tag_index: dict[str, list[str]] = defaultdict(list)
    summary_rows: list[dict] = []
    main_summary_rows: list[dict] = []

    for run in runs:
        run_name = run["run_name"]
        series = scalar_points_from_event(run["event_path"])
        all_series[run_name] = series
        for tag, points in series.items():
            tag_index[tag].append(run_name)
            row = {
                "run_name": run_name,
                "variant": run["variant"],
                **summarize_points(tag, points, tail_n),
            }
            summary_rows.append(row)
            if tag in DEFAULT_EXPORT_METRICS:
                main_summary_rows.append(row)

    write_all_series_csv(out_dir / "all_series.csv", all_series)
    write_summary_csv(out_dir / "summary_all_metrics.csv", summary_rows)
    write_summary_csv(out_dir / "summary_main_metrics.csv", main_summary_rows)

    with (out_dir / "available_tags.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs": [
                    {
                        "run_name": run["run_name"],
                        "variant": run["variant"],
                        "event_path": str(run["event_path"]),
                    }
                    for run in runs
                ],
                "tags": {tag: sorted(run_names) for tag, run_names in sorted(tag_index.items())},
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    plot_metric_group(
        plot_dir / "01_shared_train_loss.png",
        "Shared Training Losses (all variants)",
        SHARED_TRAIN_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "02_shared_val_metrics.png",
        "Shared Validation Metrics (all variants)",
        SHARED_VAL_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "03_seq_aa_comparison.png",
        "Sequence / AA Prediction (seqs_loss=exp3, aa_aux_loss=exp4)",
        SEQ_AA_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "04_exp3_sidechain.png",
        "Sidechain Quality — exp3 only (soft / soft_gate)",
        EXP3_SIDECHAIN_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "05_exp4_token_quality.png",
        "Token Quality — exp4 only (structtoken)",
        EXP4_TOKEN_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "06_recon_losses.png",
        "Reconstruction Losses (shared + variant-specific)",
        RECON_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )
    plot_metric_group(
        plot_dir / "07_efficiency.png",
        "Efficiency Comparison",
        EFFICIENCY_METRICS,
        all_series,
        smoothing_alpha,
        show_raw,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TensorBoard metrics across structtoken (exp4), soft, and soft_gate (exp3) variants."
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=str(default_log_root()),
        help="Root directory containing TensorBoard run folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "analysis_outputs"),
        help="Directory to save CSV summaries and plots.",
    )
    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.2,
        help="EMA smoothing coefficient for plotted curves.",
    )
    parser.add_argument(
        "--tail_n",
        type=int,
        default=5,
        help="Number of final points used for mean/std tail summary.",
    )
    parser.add_argument(
        "--hide_raw",
        action="store_true",
        help="Hide raw unsmoothed curves and draw only smoothed curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    build_analysis(
        log_root=log_root,
        out_dir=out_dir,
        smoothing_alpha=args.smoothing_alpha,
        tail_n=args.tail_n,
        show_raw=not args.hide_raw,
    )
    print(f"Analysis complete. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
