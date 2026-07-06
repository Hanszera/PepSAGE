#!/usr/bin/env python
"""Analyze Side-Chain Packing results across variants.

Loads test output .pt trajectory files from SC packing runs,
computes chi angle MAE, Correct%, and atom-level RMSD metrics,
and generates comparison tables and plots.

Supports both pepsage (angles-based) and StructToken (atom-coords-based) outputs.

Usage:
    python analyze_sc_packing.py --run_dir /path/to/test_outputs_sc_packing_v1 --run_dir /path/to/test_outputs_sc_packing_v2
    python analyze_sc_packing.py --log_root /path/to/logs
"""
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
except ImportError as exc:
    raise SystemExit("This script requires `torch`.") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("This script requires `matplotlib`.") from exc


VARIANT_ORDER = ["pepsage", "pep_glad", "exp8_r6_gaussian_sc", "pepsage_h200"]
VARIANT_DISPLAY = {
    "pepsage": "pepsage",
    "pep_glad": "PepGLAD",
    "exp8_r6_gaussian_sc": "StructToken_sc",
    "pepsage_h200": "pepsage-H200",
    "other": "Other",
}
COLOR_MAP = {
    "pepsage": "#4E79A7",
    "pep_glad": "#59A14F",
    "exp8_r6_gaussian_sc": "#B07AA1",
    "pepsage_h200": "#E15759",
    "other": "#7F7F7F",
}

ANGLE_NAMES = ["psi", "chi1", "chi2", "chi3", "chi4"]
CHI_INDICES = [1, 2, 3, 4]
CORRECT_THRESHOLD_DEG = 20.0

ANGLE_METRICS = [
    "chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg",
    "chi1_correct", "chi2_correct", "chi3_correct", "chi4_correct",
    "overall_correct",
    "psi_mae_deg",
]
ATOM_METRICS = ["all_atom_rmsd", "sidechain_rmsd", "clash_score", "bond_length_dev", "bond_angle_dev_deg"]
ALL_METRICS = ANGLE_METRICS + ATOM_METRICS
PRIMARY_PLOT_METRICS = [
    "chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg",
    "overall_correct",
    "all_atom_rmsd", "sidechain_rmsd",
]


def infer_variant(name: str) -> str:
    lower = name.lower()
    if "pepsage_h200" in lower:
        return "pepsage_h200"
    if "pepsage" in lower or "exp2" in lower:
        return "pepsage"
    if "pep_glad" in lower:
        return "pep_glad"
    if "exp8" in lower or "structtoken" in lower or "gaussian" in lower:
        return "exp8_r6_gaussian_sc"
    return "other"


def short_label(run_name: str) -> str:
    variant = infer_variant(run_name)
    return VARIANT_DISPLAY.get(variant, run_name[:30])


def metric_direction(metric: str) -> str:
    higher_is_better = {"chi1_correct", "chi2_correct", "chi3_correct",
                        "chi4_correct", "overall_correct"}
    return "higher" if metric in higher_is_better else "lower"


def default_log_root() -> Path:
    exp_root = Path(__file__).resolve().parents[1]
    for candidate in (exp_root / "log", exp_root / "logs"):
        if candidate.exists():
            return candidate
    return exp_root / "log"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def circular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(a - b)
    return torch.min(2 * math.pi - diff, diff)


def compute_angle_metrics(
    pred_angles: torch.Tensor,
    gt_angles: torch.Tensor,
    angle_mask: torch.Tensor,
    gen_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute chi angle MAE and Correct% from torsion angles.

    Args:
        pred_angles: (B, L, 5) predicted torsion angles in radians
        gt_angles:   (B, L, 5) ground-truth torsion angles in radians
        angle_mask:  (B, L, 5) valid angle mask
        gen_mask:    (B, L) generate mask
    """
    valid = gen_mask.unsqueeze(-1).bool() & angle_mask.bool()
    metrics: dict[str, float] = {}

    all_correct_count = 0
    all_valid_count = 0

    for idx in range(5):
        name = ANGLE_NAMES[idx]
        mask_i = valid[:, :, idx]
        if mask_i.sum() == 0:
            metrics[f"{name}_mae_deg"] = float("nan")
            if idx in CHI_INDICES:
                metrics[f"{name}_correct"] = float("nan")
            continue

        dist = circular_distance(pred_angles[:, :, idx], gt_angles[:, :, idx])
        mae_deg = (dist[mask_i] * 180.0 / math.pi).mean().item()
        metrics[f"{name}_mae_deg"] = mae_deg

        if idx in CHI_INDICES:
            correct = (dist[mask_i] * 180.0 / math.pi < CORRECT_THRESHOLD_DEG).float()
            metrics[f"{name}_correct"] = correct.mean().item()
            all_correct_count += correct.sum().item()
            all_valid_count += mask_i.sum().item()

    if all_valid_count > 0:
        metrics["overall_correct"] = all_correct_count / all_valid_count
    else:
        metrics["overall_correct"] = float("nan")

    return metrics


def compute_atom_metrics(
    pred_pos: torch.Tensor,
    gt_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    gen_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute all-atom RMSD and sidechain RMSD.

    Args:
        pred_pos:  (B, L, A, 3) predicted atom positions
        gt_pos:    (B, L, A, 3) ground-truth atom positions
        atom_mask: (B, L, A) atom existence mask
        gen_mask:  (B, L) generate mask
    """
    metrics: dict[str, float] = {}
    valid = atom_mask.bool() & gen_mask.bool().unsqueeze(-1)

    sq_diff = ((pred_pos - gt_pos) ** 2).sum(dim=-1)

    if valid.any():
        metrics["all_atom_rmsd"] = sq_diff[valid].mean().sqrt().item()
    else:
        metrics["all_atom_rmsd"] = float("nan")

    sc_valid = valid.clone()
    sc_valid[:, :, :4] = False
    if sc_valid.any():
        metrics["sidechain_rmsd"] = sq_diff[sc_valid].mean().sqrt().item()
    else:
        metrics["sidechain_rmsd"] = float("nan")

    return metrics


def process_trajectory_file(path: Path) -> dict[str, Any] | None:
    """Load a test trajectory .pt and compute SC packing metrics."""
    traj = load_pt(path)
    if not isinstance(traj, list) or len(traj) == 0:
        return None

    final = traj[-1]
    if not isinstance(final, dict) or "batch" not in final:
        return None

    batch = final["batch"]
    gen_mask = batch["generate_mask"]
    target_id = batch.get("id", [path.stem])[0]

    result: dict[str, Any] = {"target_id": target_id, "file": str(path)}
    all_metrics: dict[str, list[float]] = defaultdict(list)

    has_angles = "angles" in final and "torsion_angle" in batch
    has_atoms = ("pos_heavyatom" in final or "refined_pos_heavyatom" in final)

    if has_angles:
        pred_angles = final["angles"]
        gt_angles = batch["torsion_angle"]
        if "torsion_angle_mask" in batch:
            angle_mask = batch["torsion_angle_mask"]
        else:
            angle_mask = torch.ones_like(gt_angles).bool()

        num_samples = pred_angles.shape[0]
        for s in range(num_samples):
            m = compute_angle_metrics(
                pred_angles[s:s+1], gt_angles[s:s+1],
                angle_mask[s:s+1], gen_mask[s:s+1],
            )
            for k, v in m.items():
                if math.isfinite(v):
                    all_metrics[k].append(v)

    if has_atoms:
        pred_pos = final.get("pos_heavyatom", final.get("refined_pos_heavyatom"))
        gt_pos = batch["pos_heavyatom"]
        atom_mask = batch["mask_heavyatom"]

        num_samples = pred_pos.shape[0]
        for s in range(num_samples):
            m = compute_atom_metrics(
                pred_pos[s:s+1], gt_pos[s:s+1],
                atom_mask[s:s+1], gen_mask[s:s+1],
            )
            for k, v in m.items():
                if math.isfinite(v):
                    all_metrics[k].append(v)

    per_target: dict[str, dict] = {}
    for metric, values in all_metrics.items():
        if values:
            arr = np.array(values)
            per_target[metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": len(values),
            }

    result["metrics"] = per_target
    return result


def discover_run_dirs(log_root: Path, explicit_run_dirs: list[str] | None) -> list[Path]:
    if explicit_run_dirs:
        return [Path(d).resolve() for d in explicit_run_dirs]

    run_dirs = set()
    for summary in log_root.rglob("eval_atom_metrics_summary.pt"):
        run_dirs.add(summary.parent)
    for summary in log_root.rglob("eval_atom_metrics_sc_summary.pt"):
        run_dirs.add(summary.parent)
    for pt_file in log_root.rglob("*sc_packing*/*.pt"):
        run_dirs.add(pt_file.parent)
    return sorted(run_dirs)


def load_from_eval_summary(run_dir: Path) -> list[dict]:
    """Load results from eval_atom_metrics_summary.pt (fallback when no trajectory files)."""
    candidates = [
        "eval_atom_metrics_summary.pt",
        "eval_atom_metrics_sc_summary.pt",
    ]
    summary_path = None
    for name in candidates:
        p = run_dir / name
        if p.exists():
            summary_path = p
            break
    if summary_path is None:
        return []

    data = load_pt(summary_path)
    if not isinstance(data, dict) or "per_target" not in data:
        return []

    per_target_data = data["per_target"]
    results = []
    for target_id, metric_dict in per_target_data.items():
        per_target: dict[str, dict] = {}
        for metric_name, stats in metric_dict.items():
            if not isinstance(stats, dict) or stats.get("count", 0) == 0:
                continue
            mapped_name = _map_eval_metric_name(metric_name)
            if mapped_name is None:
                continue
            per_target[mapped_name] = {
                "mean": stats["mean"],
                "std": stats.get("std", 0.0),
                "min": stats.get("min", stats["mean"]),
                "max": stats.get("max", stats["mean"]),
                "count": stats["count"],
            }
        if per_target:
            results.append({"target_id": target_id, "file": str(summary_path), "metrics": per_target})

    return results


def _map_eval_metric_name(name: str) -> str | None:
    """Map evaluate_atom_metrics metric names to analyze_sc_packing names."""
    direct = {
        "all_atom_rmsd": "all_atom_rmsd",
        "sidechain_rmsd": "sidechain_rmsd",
        "rotamer_recovery": "overall_correct",
        "clash_score": "clash_score",
        "bond_length_dev": "bond_length_dev",
        "bond_angle_dev_deg": "bond_angle_dev_deg",
        "chi1_mae_deg": "chi1_mae_deg",
        "chi2_mae_deg": "chi2_mae_deg",
        "chi3_mae_deg": "chi3_mae_deg",
        "chi4_mae_deg": "chi4_mae_deg",
    }
    if name in direct:
        return direct[name]
    if "_all_atom_rmsd" in name:
        return None
    return None


def load_run_results(run_dir: Path) -> list[dict]:
    results = []
    pt_files = sorted(run_dir.glob("*.pt"))
    if not pt_files:
        print(f"  Warning: no .pt files in {run_dir}")
        return results

    for pt_file in pt_files:
        try:
            r = process_trajectory_file(pt_file)
            if r and r["metrics"]:
                results.append(r)
        except Exception as e:
            print(f"  Warning: failed to process {pt_file.name}: {e}")

    if not results:
        results = load_from_eval_summary(run_dir)
        if results:
            print(f"  Loaded {len(results)} targets from eval summary files")

    return results


def flatten_to_rows(
    run_name: str,
    variant: str,
    results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Convert results to per-target and aggregate rows."""
    per_target_rows = []
    metric_values: dict[str, list[float]] = defaultdict(list)

    for r in results:
        target_id = r["target_id"]
        for metric, stats in r["metrics"].items():
            per_target_rows.append({
                "run_name": run_name,
                "variant": variant,
                "target_id": target_id,
                "metric": metric,
                "direction": metric_direction(metric),
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
            })
            metric_values[metric].append(stats["mean"])

    aggregate_rows = []
    for metric, means in metric_values.items():
        arr = np.array(means)
        aggregate_rows.append({
            "run_name": run_name,
            "variant": variant,
            "metric": metric,
            "direction": metric_direction(metric),
            "num_targets": len(means),
            "mean_of_means": float(arr.mean()),
            "std_of_means": float(arr.std()),
            "median_of_means": float(np.median(arr)),
        })

    return per_target_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict], field_order: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = field_order or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_by_variant(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99,
            r["run_name"],
        ),
    )


def write_variant_comparison(path: Path, all_aggregate: list[dict]) -> None:
    by_metric: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in all_aggregate:
        by_metric[row["metric"]][row["variant"]] = row

    all_variants = []
    for v in VARIANT_ORDER:
        if any(v in variants for variants in by_metric.values()):
            all_variants.append(v)
    for variants in by_metric.values():
        for v in variants:
            if v not in all_variants:
                all_variants.append(v)

    header = ["metric", "direction"]
    for v in all_variants:
        display = VARIANT_DISPLAY.get(v, v)
        header.extend([f"{display}_mean", f"{display}_std"])
    header.append("best_variant")

    out_rows = []
    for metric in ALL_METRICS:
        variants = by_metric.get(metric, {})
        if not variants:
            continue
        direction = metric_direction(metric)
        row_dict: dict[str, Any] = {"metric": metric, "direction": direction}
        best_variant, best_val = None, None
        for v in all_variants:
            display = VARIANT_DISPLAY.get(v, v)
            if v in variants:
                val = variants[v]["mean_of_means"]
                std = variants[v]["std_of_means"]
                row_dict[f"{display}_mean"] = f"{val:.4f}"
                row_dict[f"{display}_std"] = f"{std:.4f}"
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


def plot_chi_mae_bars(path: Path, all_aggregate: list[dict]) -> None:
    """Bar chart: chi1-4 MAE per variant."""
    chi_metrics = ["chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)

    for ax, metric in zip(axes.flatten(), chi_metrics):
        items = [r for r in all_aggregate if r["metric"] == metric]
        items = sort_by_variant(items)
        if not items:
            ax.axis("off")
            continue
        labels = [short_label(r["run_name"]) for r in items]
        means = [r["mean_of_means"] for r in items]
        stds = [r["std_of_means"] for r in items]
        colors = [COLOR_MAP.get(r["variant"], COLOR_MAP["other"]) for r in items]
        x = np.arange(len(items))
        ax.bar(x, means, color=colors, alpha=0.85)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1)
        ax.set_title(metric.replace("_mae_deg", " MAE (°)"))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("MAE (°)")

    fig.suptitle("Chi Angle MAE — Side-Chain Packing", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_correct_bars(path: Path, all_aggregate: list[dict]) -> None:
    """Bar chart: per-chi correct% and overall correct%."""
    correct_metrics = ["chi1_correct", "chi2_correct", "chi3_correct", "chi4_correct", "overall_correct"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), squeeze=False)

    for ax, metric in zip(axes.flatten(), correct_metrics):
        items = [r for r in all_aggregate if r["metric"] == metric]
        items = sort_by_variant(items)
        if not items:
            ax.axis("off")
            continue
        labels = [short_label(r["run_name"]) for r in items]
        means = [r["mean_of_means"] * 100 for r in items]
        stds = [r["std_of_means"] * 100 for r in items]
        colors = [COLOR_MAP.get(r["variant"], COLOR_MAP["other"]) for r in items]
        x = np.arange(len(items))
        ax.bar(x, means, color=colors, alpha=0.85)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1)
        display = metric.replace("_correct", "").replace("overall", "Overall")
        ax.set_title(f"{display} Correct%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Correct %")
        ax.set_ylim(0, 100)

    fig.suptitle("Correct Rate (<20°) — Side-Chain Packing", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rmsd_bars(path: Path, all_aggregate: list[dict]) -> None:
    """Bar chart: all-atom and sidechain RMSD."""
    rmsd_metrics = ["all_atom_rmsd", "sidechain_rmsd"]
    items_any = [r for r in all_aggregate if r["metric"] in rmsd_metrics]
    if not items_any:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, metric in zip(axes.flatten(), rmsd_metrics):
        items = [r for r in all_aggregate if r["metric"] == metric]
        items = sort_by_variant(items)
        if not items:
            ax.axis("off")
            continue
        labels = [short_label(r["run_name"]) for r in items]
        means = [r["mean_of_means"] for r in items]
        stds = [r["std_of_means"] for r in items]
        colors = [COLOR_MAP.get(r["variant"], COLOR_MAP["other"]) for r in items]
        x = np.arange(len(items))
        ax.bar(x, means, color=colors, alpha=0.85)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=4, linewidth=1)
        display = metric.replace("_", " ").title()
        ax.set_title(display)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("RMSD (Å)")

    fig.suptitle("Atom-Level RMSD — Side-Chain Packing", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_target_boxes(path: Path, all_per_target: list[dict]) -> None:
    """Box plots: chi MAE distribution across targets."""
    chi_metrics = ["chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)

    for ax, metric in zip(axes.flatten(), chi_metrics):
        items = [r for r in all_per_target if r["metric"] == metric]
        by_run: dict[str, list[float]] = defaultdict(list)
        by_variant: dict[str, str] = {}
        for r in items:
            val = r.get("mean", float("nan"))
            if math.isfinite(val):
                by_run[r["run_name"]].append(val)
                by_variant[r["run_name"]] = r["variant"]

        run_names = sorted(
            by_run.keys(),
            key=lambda n: (
                VARIANT_ORDER.index(by_variant[n]) if by_variant[n] in VARIANT_ORDER else 99, n
            ),
        )
        if not run_names:
            ax.axis("off")
            continue
        labels = [short_label(n) for n in run_names]
        data = [by_run[n] for n in run_names]
        box = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, name in zip(box["boxes"], run_names):
            color = COLOR_MAP.get(by_variant[name], COLOR_MAP["other"])
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(metric.replace("_mae_deg", " MAE (°)"))
        ax.set_ylabel("MAE (°)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Chi Angle MAE Distribution Across Targets", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pepflow_style_table(path: Path, all_aggregate: list[dict]) -> None:
    """Generate a PepFlow-Table-3-style comparison image."""
    metrics_order = [
        ("chi1_mae_deg", "χ₁ MAE°↓"),
        ("chi2_mae_deg", "χ₂ MAE°↓"),
        ("chi3_mae_deg", "χ₃ MAE°↓"),
        ("chi4_mae_deg", "χ₄ MAE°↓"),
        ("overall_correct", "Correct%↑"),
    ]

    by_metric: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in all_aggregate:
        by_metric[row["metric"]][row["variant"]] = row

    variants_present = []
    for v in VARIANT_ORDER:
        if any(v in variants for variants in by_metric.values()):
            variants_present.append(v)
    for variants in by_metric.values():
        for v in variants:
            if v not in variants_present:
                variants_present.append(v)

    if not variants_present:
        return

    col_labels = [VARIANT_DISPLAY.get(v, v) for v in variants_present]
    row_labels = [display for _, display in metrics_order]

    cell_text = []
    for metric_key, _ in metrics_order:
        row_data = []
        variants_data = by_metric.get(metric_key, {})
        direction = metric_direction(metric_key)
        values_for_best = {}
        for v in variants_present:
            if v in variants_data:
                val = variants_data[v]["mean_of_means"]
                values_for_best[v] = val
            else:
                values_for_best[v] = float("nan")

        if direction == "lower":
            best_v = min(
                (v for v in values_for_best if math.isfinite(values_for_best[v])),
                key=lambda v: values_for_best[v], default=None
            )
        else:
            best_v = max(
                (v for v in values_for_best if math.isfinite(values_for_best[v])),
                key=lambda v: values_for_best[v], default=None
            )

        for v in variants_present:
            if v in variants_data:
                val = variants_data[v]["mean_of_means"]
                if metric_key == "overall_correct":
                    text = f"{val*100:.1f}"
                else:
                    text = f"{val:.2f}"
                if v == best_v:
                    text = f"**{text}**"
            else:
                text = "—"
            row_data.append(text)
        cell_text.append(row_data)

    fig, ax = plt.subplots(figsize=(2.5 * len(col_labels) + 2, 0.6 * len(row_labels) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif col == -1:
            cell.set_facecolor("#D6E4F0")
            cell.set_text_props(weight="bold")
        text = cell.get_text().get_text()
        if text.startswith("**") and text.endswith("**"):
            cell.get_text().set_text(text[2:-2])
            cell.get_text().set_weight("bold")
            cell.set_facecolor("#E2EFDA")

    fig.suptitle("Side-Chain Packing Comparison (PepFlow Table 3 style)", fontsize=13, y=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analysis(run_dirs: list[Path], out_dir: Path) -> None:
    ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

    all_aggregate: list[dict] = []
    all_per_target: list[dict] = []

    for run_dir in run_dirs:
        run_name = run_dir.parent.name if run_dir.name.startswith("test_outputs") else run_dir.name
        variant = infer_variant(run_name)
        print(f"Loading {run_name} (variant={variant}) from {run_dir}")

        results = load_run_results(run_dir)
        if not results:
            print(f"  Warning: no valid results in {run_dir}")
            continue

        print(f"  Loaded {len(results)} targets")
        per_target, aggregate = flatten_to_rows(run_name, variant, results)
        all_per_target.extend(per_target)
        all_aggregate.extend(aggregate)

    if not all_aggregate:
        raise SystemExit("No SC packing results found in any run directory.")

    all_aggregate = sort_by_variant(all_aggregate)
    all_per_target = sort_by_variant(all_per_target)

    write_csv(
        out_dir / "sc_packing_aggregate.csv",
        all_aggregate,
        field_order=[
            "run_name", "variant", "metric", "direction",
            "num_targets", "mean_of_means", "std_of_means", "median_of_means",
        ],
    )
    write_csv(
        out_dir / "sc_packing_per_target.csv",
        all_per_target,
        field_order=[
            "run_name", "variant", "target_id", "metric", "direction",
            "mean", "std", "count",
        ],
    )
    write_variant_comparison(out_dir / "sc_packing_variant_comparison.csv", all_aggregate)

    plot_chi_mae_bars(plot_dir / "01_chi_mae_bars.png", all_aggregate)
    plot_correct_bars(plot_dir / "02_correct_bars.png", all_aggregate)
    plot_rmsd_bars(plot_dir / "03_rmsd_bars.png", all_aggregate)
    plot_per_target_boxes(plot_dir / "04_chi_mae_boxes.png", all_per_target)
    plot_pepflow_style_table(plot_dir / "05_pepflow_table3_style.png", all_aggregate)

    print(f"\n=== Side-Chain Packing Summary ===")
    for row in all_aggregate:
        label = short_label(row["run_name"])
        metric = row["metric"]
        val = row["mean_of_means"]
        std = row["std_of_means"]
        if "correct" in metric:
            print(f"  {label:20s}  {metric:20s}: {val*100:.1f}% ± {std*100:.1f}%")
        else:
            print(f"  {label:20s}  {metric:20s}: {val:.2f} ± {std:.2f}")
    print(f"\nResults saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Side-Chain Packing results across variants."
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Test output directory containing trajectory .pt files. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=str(default_log_root()),
        help="Root directory to auto-discover SC packing result dirs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "sc_packing_outputs"),
        help="Directory to save CSV summaries and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    run_dirs = discover_run_dirs(log_root, args.run_dir)
    if not run_dirs:
        raise SystemExit(
            "No SC packing result directories found.\n"
            "Use --run_dir to specify test output directories explicitly,\n"
            "or ensure directories containing '*sc_packing*' exist under --log_root."
        )
    print(f"Found {len(run_dirs)} run directories:")
    for d in run_dirs:
        print(f"  {d}")
    out_dir = Path(args.out_dir).resolve()
    build_analysis(run_dirs, out_dir)


if __name__ == "__main__":
    main()
