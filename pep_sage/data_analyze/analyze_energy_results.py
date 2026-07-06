#!/usr/bin/env python
"""Analyze Rosetta energy results across variants.

Loads rosetta_results_*.pt files, aggregates stability and binding energy
per target and across targets, and generates comparison plots.

Usage:
    python analyze_energy_results.py --run_dir /path/to/run1 --run_dir /path/to/run2
    python analyze_energy_results.py --log_root /path/to/logs
"""
from __future__ import annotations

import argparse
import csv
import math
import re
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


VARIANT_ORDER = ["pepsage", "pep_glad", "exp8_r6_gaussian_v4_rebalance", "pepsage_h200"]
VARIANT_DISPLAY = {
    "pepsage": "pepsage",
    "pep_glad": "pep_glad",
    "exp8_r6_gaussian_v4_rebalance": "StructToken (gaussian)",
    "pepsage_h200": "pepsage_h200",
    "other": "Other",
}
COLOR_MAP = {
    "pepsage": "#4E79A7",
    "pep_glad": "#59A14F",
    "exp8_r6_gaussian_v4_rebalance": "#B07AA1",
    "pepsage_h200": "#E15759",
    "other": "#7F7F7F",
}

ENERGY_METRICS = ["stab", "bind"]
ENERGY_DIRECTION = {"stab": "lower", "bind": "lower"}
ENERGY_DISPLAY = {
    "stab": "Stability Energy (REU)",
    "bind": "Binding Energy dG (REU)",
}


def infer_variant(name: str) -> str:
    lower = name.lower()
    if "pepsage_h200" in lower:
        return "pepsage_h200"
    if "pepsage" in lower:
        return "pepsage"
    if "pep_glad" in lower:
        return "pep_glad"
    if "exp8_r6_gaussian_v4_rebalance" in lower:
        return "exp8_r6_gaussian_v4_rebalance"
    return "other"


def short_label(run_name: str) -> str:
    variant = infer_variant(run_name)
    return VARIANT_DISPLAY.get(variant, run_name)


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


def extract_target_id(pdb_path: str) -> str:
    """Extract target ID from PDB path.

    Expected path structure: .../generated_pep/{target_folder}/{sample}.pdb
    """
    parts = Path(pdb_path).parts
    for i, part in enumerate(parts):
        if part in ("generated_pep", "generated_pep_packsc") and i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 2:
        return parts[-2]
    return Path(pdb_path).stem


def discover_run_dirs(log_root: Path, explicit_run_dirs: list[str] | None) -> list[Path]:
    if explicit_run_dirs:
        return [Path(item).resolve() for item in explicit_run_dirs]

    run_dirs = set()
    for path in log_root.rglob("rosetta_results_merged.pt"):
        run_dirs.add(path.parent)
    for path in log_root.rglob("rosetta_results_0.pt"):
        run_dirs.add(path.parent)
    return sorted(run_dirs)


def load_energy_results(run_dir: Path) -> list[dict]:
    """Load rosetta results from a run directory.

    Tries merged file first, then individual rank files.
    """
    merged = run_dir / "rosetta_results_merged.pt"
    if merged.exists():
        results = load_pt(merged)
        if isinstance(results, list):
            return results

    all_results = []
    for pt_file in sorted(run_dir.glob("rosetta_results_*.pt")):
        if "merged" in pt_file.name or "_sc" in pt_file.name:
            continue
        data = load_pt(pt_file)
        if isinstance(data, list):
            all_results.extend(data)
    return all_results


def aggregate_by_target(results: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Group results by target, collecting stab/bind values."""
    by_target: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in results:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        target_id = extract_target_id(name)
        for metric in ENERGY_METRICS:
            val = entry.get(metric, float("nan"))
            if isinstance(val, (int, float)) and math.isfinite(val):
                by_target[target_id][metric].append(val)
    return by_target


def compute_per_target_stats(
    by_target: dict[str, dict[str, list[float]]],
) -> list[dict]:
    rows = []
    for target_id, metric_map in sorted(by_target.items()):
        for metric in ENERGY_METRICS:
            values = metric_map.get(metric, [])
            if not values:
                continue
            arr = np.array(values)
            rows.append({
                "target_id": target_id,
                "metric": metric,
                "direction": ENERGY_DIRECTION[metric],
                "count": len(values),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "median": float(np.median(arr)),
            })
    return rows


def compute_aggregate_stats(
    per_target: list[dict],
) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in per_target:
        grouped[row["metric"]].append(row["mean"])

    rows = []
    for metric in ENERGY_METRICS:
        means = grouped.get(metric, [])
        if not means:
            continue
        arr = np.array(means)
        rows.append({
            "metric": metric,
            "direction": ENERGY_DIRECTION[metric],
            "num_targets": len(means),
            "mean_of_means": float(arr.mean()),
            "std_of_means": float(arr.std()),
            "median_of_means": float(np.median(arr)),
            "min_of_means": float(arr.min()),
            "max_of_means": float(arr.max()),
        })
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


def sort_by_variant(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            VARIANT_ORDER.index(r["variant"]) if r["variant"] in VARIANT_ORDER else 99,
            r["run_name"],
        ),
    )


def plot_aggregate_bars(
    path: Path,
    all_aggregate: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, ENERGY_METRICS):
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
        ax.set_title(ENERGY_DISPLAY.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Energy (REU)")

    handles = []
    labels_legend = []
    for v in VARIANT_ORDER:
        if any(r["variant"] == v for r in all_aggregate):
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLOR_MAP.get(v, COLOR_MAP["other"]), alpha=0.85))
            labels_legend.append(VARIANT_DISPLAY.get(v, v))
    for r in all_aggregate:
        if r["variant"] not in VARIANT_ORDER and r["variant"] not in [v for v, _ in zip(VARIANT_ORDER, labels_legend)]:
            v = r["variant"]
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLOR_MAP.get(v, COLOR_MAP["other"]), alpha=0.85))
            labels_legend.append(VARIANT_DISPLAY.get(v, v))

    if handles:
        fig.legend(handles, labels_legend, loc="upper center", ncol=min(4, len(handles)))
    fig.suptitle("Rosetta Energy Comparison (aggregate)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_target_boxes(
    path: Path,
    all_per_target: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, ENERGY_METRICS):
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
                VARIANT_ORDER.index(by_variant[n]) if by_variant[n] in VARIANT_ORDER else 99,
                n,
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
        ax.set_title(ENERGY_DISPLAY.get(metric, metric))
        ax.set_ylabel("Energy (REU)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Rosetta Energy Distribution Across Targets", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_target_scatter(
    path: Path,
    all_per_target: list[dict],
) -> None:
    """Scatter plot: stab vs bind per target, one color per variant."""
    by_run: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    by_variant: dict[str, str] = {}
    for r in all_per_target:
        by_run[r["run_name"]][r["target_id"]][r["metric"]] = r["mean"]
        by_variant[r["run_name"]] = r["variant"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for run_name in sorted(
        by_run.keys(),
        key=lambda n: VARIANT_ORDER.index(by_variant[n]) if by_variant[n] in VARIANT_ORDER else 99,
    ):
        targets = by_run[run_name]
        stabs, binds = [], []
        for target_id, metrics in targets.items():
            s = metrics.get("stab", float("nan"))
            b = metrics.get("bind", float("nan"))
            if math.isfinite(s) and math.isfinite(b):
                stabs.append(s)
                binds.append(b)
        if stabs:
            variant = by_variant[run_name]
            color = COLOR_MAP.get(variant, COLOR_MAP["other"])
            label = short_label(run_name)
            ax.scatter(stabs, binds, c=color, label=label, alpha=0.5, s=15, edgecolors="none")

    ax.set_xlabel("Stability Energy (REU)")
    ax.set_ylabel("Binding Energy dG (REU)")
    ax.set_title("Stability vs Binding Energy (per target)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_binding_histogram(
    path: Path,
    all_per_target: list[dict],
) -> None:
    """Overlaid histograms of binding energy across variants."""
    fig, ax = plt.subplots(figsize=(8, 5))
    by_run: dict[str, list[float]] = defaultdict(list)
    by_variant: dict[str, str] = {}
    for r in all_per_target:
        if r["metric"] == "bind":
            val = r.get("mean", float("nan"))
            if math.isfinite(val):
                by_run[r["run_name"]].append(val)
                by_variant[r["run_name"]] = r["variant"]

    for run_name in sorted(
        by_run.keys(),
        key=lambda n: VARIANT_ORDER.index(by_variant[n]) if by_variant[n] in VARIANT_ORDER else 99,
    ):
        values = by_run[run_name]
        variant = by_variant[run_name]
        color = COLOR_MAP.get(variant, COLOR_MAP["other"])
        label = short_label(run_name)
        ax.hist(values, bins=30, alpha=0.5, color=color, label=label, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Binding Energy dG (REU)")
    ax.set_ylabel("Number of Targets")
    ax.set_title("Distribution of Binding Energy Across Targets")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        header.extend([f"{display}_mean", f"{display}_std", f"{display}_median"])
    header.append("best_variant")

    out_rows = []
    for metric in ENERGY_METRICS:
        variants = by_metric.get(metric, {})
        direction = ENERGY_DIRECTION[metric]
        row_dict: dict[str, Any] = {"metric": metric, "direction": direction}

        best_variant = None
        best_val = None
        for v in all_variants:
            display = VARIANT_DISPLAY.get(v, v)
            if v in variants:
                val = variants[v]["mean_of_means"]
                std = variants[v]["std_of_means"]
                med = variants[v]["median_of_means"]
                row_dict[f"{display}_mean"] = f"{val:.2f}"
                row_dict[f"{display}_std"] = f"{std:.2f}"
                row_dict[f"{display}_median"] = f"{med:.2f}"
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
                row_dict[f"{display}_median"] = ""
        row_dict["best_variant"] = VARIANT_DISPLAY.get(best_variant, best_variant or "")
        out_rows.append(row_dict)

    write_csv(path, out_rows, field_order=header)


def is_gt_entry(entry: dict) -> bool:
    """Check if a rosetta result entry is from a ground-truth PDB (gt.pdb)."""
    name = entry.get("name", "")
    return Path(name).name == "gt.pdb"


def extract_gt_from_results(results: list[dict]) -> dict[str, dict[str, float]]:
    """Extract ground-truth energies from rosetta results (gt.pdb entries).

    Returns {target_id: {"stab": ..., "bind": ...}}.
    """
    gt: dict[str, dict[str, float]] = {}
    for entry in results:
        if not isinstance(entry, dict) or not is_gt_entry(entry):
            continue
        target_id = extract_target_id(entry.get("name", ""))
        stab = entry.get("stab", float("nan"))
        bind = entry.get("bind", float("nan"))
        if isinstance(stab, (int, float)) and math.isfinite(stab) and \
           isinstance(bind, (int, float)) and math.isfinite(bind):
            gt[target_id] = {"stab": float(stab), "bind": float(bind)}
    return gt


def collect_gt_from_run_dirs(run_dirs: list[Path]) -> dict[str, dict[str, float]]:
    """Scan all run directories for gt.pdb entries in rosetta results."""
    gt: dict[str, dict[str, float]] = {}
    for run_dir in run_dirs:
        results = load_energy_results(run_dir)
        found = extract_gt_from_results(results)
        if found:
            gt.update(found)
    return gt


def compute_affinity_stability(
    all_raw: list[dict],
    gt: dict[str, dict[str, float]],
) -> list[dict]:
    """Compute Affinity % and Stability % per run (PepFlow definition).

    Affinity %  = percentage of designed peptides with bind < gt_bind
    Stability % = percentage of designed peptides with stab < gt_stab
    Counted across all samples globally.
    """
    by_run: dict[str, list[dict]] = defaultdict(list)
    run_variant: dict[str, str] = {}
    for row in all_raw:
        if Path(row.get("pdb_path", "")).name == "gt.pdb":
            continue
        run_name = row["run_name"]
        run_variant[run_name] = row["variant"]
        if math.isfinite(row.get("stab", float("nan"))) and math.isfinite(row.get("bind", float("nan"))):
            by_run[run_name].append(row)

    results = []
    for run_name, samples in by_run.items():
        total = 0
        affinity_hits = 0
        stability_hits = 0
        matched_targets = set()

        for s in samples:
            target_id = s["target_id"]
            if target_id not in gt:
                continue
            gt_bind = gt[target_id].get("bind")
            gt_stab = gt[target_id].get("stab")
            if gt_bind is None or gt_stab is None:
                continue

            total += 1
            matched_targets.add(target_id)
            if s["bind"] < gt_bind:
                affinity_hits += 1
            if s["stab"] < gt_stab:
                stability_hits += 1

        if total > 0:
            results.append({
                "run_name": run_name,
                "variant": run_variant[run_name],
                "num_targets": len(matched_targets),
                "num_samples": total,
                "affinity_pct": float(affinity_hits / total * 100),
                "stability_pct": float(stability_hits / total * 100),
            })

    return sort_by_variant(results)


def plot_affinity_stability(path: Path, aff_stab: list[dict]) -> None:
    if not aff_stab:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

    for ax, metric_key, title in zip(
        axes.flatten(),
        ["affinity_pct", "stability_pct"],
        ["Affinity % ↑", "Stability % ↑"],
    ):
        labels = [short_label(r["run_name"]) for r in aff_stab]
        vals = [r[metric_key] for r in aff_stab]
        colors = [COLOR_MAP.get(r["variant"], COLOR_MAP["other"]) for r in aff_stab]
        x = np.arange(len(aff_stab))

        bars = ax.bar(x, vals, color=colors, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, min(max(vals) + 15, 105))
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Affinity & Stability vs Ground Truth", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_analysis(run_dirs: list[Path], out_dir: Path, gt_path: Path | None = None) -> None:
    ensure_dir(out_dir)
    plot_dir = out_dir / "plots"
    ensure_dir(plot_dir)

    all_aggregate: list[dict] = []
    all_per_target: list[dict] = []
    all_raw: list[dict] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        variant = infer_variant(run_name)
        results = load_energy_results(run_dir)
        if not results:
            print(f"Warning: no rosetta results in {run_dir}")
            continue

        valid = [r for r in results if isinstance(r, dict) and math.isfinite(r.get("stab", float("nan")))]
        print(f"Loaded {run_name}: {len(results)} total, {len(valid)} valid")

        for r in results:
            if not isinstance(r, dict):
                continue
            raw_row = {
                "run_name": run_name,
                "variant": variant,
                "pdb_path": r.get("name", ""),
                "target_id": extract_target_id(r.get("name", "")),
                "stab": r.get("stab", float("nan")),
                "bind": r.get("bind", float("nan")),
            }
            all_raw.append(raw_row)

        by_target = aggregate_by_target(results)
        per_target = compute_per_target_stats(by_target)
        for row in per_target:
            row["run_name"] = run_name
            row["variant"] = variant
        all_per_target.extend(per_target)

        aggregate = compute_aggregate_stats(per_target)
        for row in aggregate:
            row["run_name"] = run_name
            row["variant"] = variant
        all_aggregate.extend(aggregate)

    if not all_aggregate:
        raise SystemExit("No energy results found in any run directory.")

    write_csv(
        out_dir / "energy_aggregate.csv",
        sort_by_variant(all_aggregate),
        field_order=[
            "run_name", "variant", "metric", "direction",
            "num_targets", "mean_of_means", "std_of_means",
            "median_of_means", "min_of_means", "max_of_means",
        ],
    )
    write_csv(
        out_dir / "energy_per_target.csv",
        sort_by_variant(all_per_target),
        field_order=[
            "run_name", "variant", "target_id", "metric", "direction",
            "count", "mean", "std", "min", "max", "median",
        ],
    )
    write_csv(
        out_dir / "energy_raw.csv",
        all_raw,
        field_order=["run_name", "variant", "target_id", "pdb_path", "stab", "bind"],
    )
    write_variant_comparison(out_dir / "energy_variant_comparison.csv", all_aggregate)

    plot_aggregate_bars(plot_dir / "01_energy_aggregate.png", all_aggregate)
    plot_per_target_boxes(plot_dir / "02_energy_per_target_box.png", all_per_target)
    plot_per_target_scatter(plot_dir / "03_energy_stab_vs_bind.png", all_per_target)
    plot_binding_histogram(plot_dir / "04_binding_histogram.png", all_per_target)

    # ── Affinity % / Stability % ──────────────────────────────────────
    # Auto-extract gt.pdb entries from all rosetta results as ground truth
    if gt_path is not None and gt_path.exists():
        print(f"\nLoading ground-truth energies from: {gt_path}")
        gt_results = load_energy_results(gt_path.parent)
        gt = extract_gt_from_results(gt_results)
    else:
        print("\nAuto-detecting ground-truth (gt.pdb) entries from rosetta results...")
        gt = collect_gt_from_run_dirs(run_dirs)

    if gt:
        print(f"  Ground-truth targets found: {len(gt)}")
        gt_stabs = [v["stab"] for v in gt.values() if "stab" in v]
        gt_binds = [v["bind"] for v in gt.values() if "bind" in v]
        if gt_stabs:
            print(f"  GT stab:  mean={np.mean(gt_stabs):.2f}  median={np.median(gt_stabs):.2f}  std={np.std(gt_stabs):.2f}")
        if gt_binds:
            print(f"  GT bind:  mean={np.mean(gt_binds):.2f}  median={np.median(gt_binds):.2f}  std={np.std(gt_binds):.2f}")
        aff_stab = compute_affinity_stability(all_raw, gt)
        if aff_stab:
            write_csv(
                out_dir / "affinity_stability.csv",
                aff_stab,
                field_order=[
                    "run_name", "variant", "num_targets", "num_samples",
                    "affinity_pct", "stability_pct",
                ],
            )
            plot_affinity_stability(plot_dir / "05_affinity_stability.png", aff_stab)

            print(f"\n=== Affinity & Stability ===")
            for row in aff_stab:
                label = short_label(row["run_name"])
                print(
                    f"  {label:30s}  "
                    f"Affinity={row['affinity_pct']:.2f}%  "
                    f"Stability={row['stability_pct']:.2f}%  "
                    f"({row['num_samples']} samples, {row['num_targets']} targets)"
                )
    else:
        print("  No gt.pdb entries found in rosetta results, skipping Affinity/Stability %")

    print(f"\n=== Energy Analysis Summary ===")
    for row in sort_by_variant(all_aggregate):
        label = short_label(row["run_name"])
        metric = row["metric"]
        print(f"  {label:30s}  {metric}: mean={row['mean_of_means']:.2f} ± {row['std_of_means']:.2f}  median={row['median_of_means']:.2f}")
    print(f"\nResults saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Rosetta energy results across variants."
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        default=[],
        help="Run directory containing rosetta_results_*.pt. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=str(default_log_root()),
        help="Root directory to auto-discover rosetta result files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "energy_analysis_outputs"),
        help="Directory to save CSV summaries and plots.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to ground-truth rosetta_results_merged.pt. "
             "Auto-detected from log_root/pepsage/ if not specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).resolve()
    run_dirs = discover_run_dirs(log_root, args.run_dir)
    if not run_dirs:
        raise SystemExit("No run directories with rosetta result files found.")
    out_dir = Path(args.out_dir).resolve()
    gt_path = Path(args.gt_path).resolve() if args.gt_path else None
    build_analysis(run_dirs, out_dir, gt_path=gt_path)


if __name__ == "__main__":
    main()
