#!/usr/bin/env python
"""Target-level paired bootstrap (Table 16) + training-seed stability (Table 17).

Two subcommands, two DIFFERENT statistical units -- do not mix them:

  significance : Table 16. Unit = TARGET. For each metric and each
                 (PepSAGE vs baseline) pair, computes the paired difference,
                 bootstrap 95% CI, two-sided bootstrap p-value, effect size
                 (Cohen's d_z), and per-target win rate. Each target contributes
                 one scalar = its per-target mean over samples (already stored in
                 per_target[tid][metric]['mean']). Targets are paired across
                 methods by PDB_chain prefix (the `_batchid_NN` suffix is stripped),
                 verified to match 1:1 across methods.

  seeds        : Table 17. Unit = TRAINING SEED. Each seed run is collapsed to one
                 global scalar per metric (mean over its per-target means), then
                 reported as mean +/- std across seeds. A method with a single seed
                 reports '-' for std (no fabricated variance). This is the ONLY place
                 a +/- belongs for seed variance; per-target CIs go in Table 16.

Usage:
    # Table 16 (ready now -- all baselines single-run):
    python data_analyze/bootstrap_significance.py significance \
        --pepsage   logs/exp8_r6_gaussian_v4_fix \
        --pepsage    logs/pepsage \
        --pepglad   logs/pep_glad \
        --discrete  "logs/Discrete token" \
        --out data_analyze/significance_outputs --n-boot 10000 --seed 1234

    # Table 17 (after >=3 PepSAGE seeds are trained + evaluated):
    python data_analyze/bootstrap_significance.py seeds \
        --method PepSAGE  logs/pepsage_seed1 logs/pepsage_seed2 logs/pepsage_seed3 \
        --method pepsage   logs/pepsage \
        --method PepGLAD  logs/pep_glad \
        --method Discrete "logs/Discrete token" \
        --out data_analyze/significance_outputs
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires `torch` to load .pt result files. "
        "Run it in the experiment environment."
    ) from exc


SUMMARY_FILES = {
    "task": "eval_metrics_summary.pt",
    "other": "eval_other_metrics_summary.pt",
    "atom": "eval_atom_metrics_summary.pt",
}

# metric -> ("low"=lower is better, "high"=higher is better). Used only for
# win-rate / direction reporting; CI-crosses-zero significance is direction free.
METRIC_DIRECTION = {
    # task-level
    "rmsd": "low",
    "bind_ratio": "high",
    "ss_ratio": "high",
    "valid": "high",
    "novel": "high",
    "diversity": "high",
    # other
    "aar": "high",
    "sample_CA_dist": "low",
    # atom-level
    "all_atom_rmsd": "low",
    "sidechain_rmsd": "low",
    "rotamer_recovery": "high",
    "clash_score": "low",
    "bond_length_dev": "low",
    "bond_angle_dev_deg": "low",
    "chi1_mae_deg": "low",
    "chi2_mae_deg": "low",
    "chi3_mae_deg": "low",
    "chi4_mae_deg": "low",
}

# Metrics promoted to the main-paper Table 1 / Table 2 narrative.
PRIMARY_METRICS = ["rmsd", "all_atom_rmsd", "sidechain_rmsd"]

# Metrics reported in Supplementary Table 17 (training-seed stability).
TABLE17_METRICS = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bind_ratio", "diversity"]
TABLE17_LABELS = {
    "rmsd": "Backbone RMSD",
    "all_atom_rmsd": "AA-RMSD",
    "sidechain_rmsd": "SC-RMSD",
    "bind_ratio": "Contact recovery",
    "diversity": "Diversity",
}


def strip_batchid(tid: str) -> str:
    return re.sub(r"_batchid_\d+$", "", tid)


def load_per_target(method_dir: Path) -> dict[str, dict[str, float]]:
    """Return {target_prefix: {metric: per_target_mean}} merged across summary files."""
    merged: dict[str, dict[str, float]] = {}
    for key, fname in SUMMARY_FILES.items():
        fpath = method_dir / fname
        if not fpath.exists():
            print(f"  [warn] missing {fpath}")
            continue
        obj = torch.load(fpath, map_location="cpu", weights_only=False)
        per_target = obj["per_target"]
        for tid, metrics in per_target.items():
            prefix = strip_batchid(tid)
            slot = merged.setdefault(prefix, {})
            for mname, stats in metrics.items():
                if not isinstance(stats, dict):
                    continue
                if stats.get("count", 0) == 0:
                    continue
                val = stats.get("mean")
                if val is None or not np.isfinite(val):
                    continue
                slot[mname] = float(val)
    return merged


def paired_arrays(a: dict[str, dict[str, float]],
                  b: dict[str, dict[str, float]],
                  metric: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return aligned (a_vals, b_vals, target_ids) for targets where both have metric."""
    common = sorted(set(a) & set(b))
    av, bv, tids = [], [], []
    for tid in common:
        if metric in a[tid] and metric in b[tid]:
            av.append(a[tid][metric])
            bv.append(b[tid][metric])
            tids.append(tid)
    return np.asarray(av), np.asarray(bv), tids


def bootstrap_paired(diff: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Bootstrap over targets. diff = pepsage - baseline (per target).

    Returns (mean_diff, ci_low, ci_high, p_two_sided, cohen_dz, boot_means).
    """
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    mean_diff = float(diff.mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    # two-sided bootstrap p-value: proportion of resampled means on the
    # opposite side of zero from the observed mean, doubled.
    if mean_diff >= 0:
        p = 2.0 * float(np.mean(boot_means <= 0.0))
    else:
        p = 2.0 * float(np.mean(boot_means >= 0.0))
    p = min(1.0, p)
    sd = diff.std(ddof=1)
    cohen_dz = mean_diff / sd if sd > 0 else float("nan")
    return mean_diff, float(ci_low), float(ci_high), p, float(cohen_dz), boot_means


def win_rate(diff: np.ndarray, direction: str) -> float:
    """Fraction of targets where PepSAGE is better than baseline."""
    if direction == "low":
        wins = (diff < 0).sum()  # pepsage lower = better
    else:
        wins = (diff > 0).sum()  # pepsage higher = better
    return float(wins) / diff.shape[0]


def seed_global_metric(per_target: dict[str, dict[str, float]], metric: str) -> float:
    """Collapse one seed's run to a single global scalar for a metric.

    = unweighted mean over that seed's per-target means. Returns nan if absent.
    """
    vals = [d[metric] for d in per_target.values() if metric in d and np.isfinite(d[metric])]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def run_seed_stability(args, rng):
    """Table 17: training-seed mean +/- std.

    Each --seed-dir is one fully-trained-and-evaluated seed of a method.
    Pass one or more --seed-dirs per method via repeated --method blocks:
        --method PepSAGE dirA dirB dirC
        --method Discrete dirX
    A method with a single seed reports '-' for std (no fabricated variance).
    """
    methods: dict[str, list[Path]] = {}
    for block in args.method:
        if len(block) < 2:
            raise SystemExit(f"--method needs a name and >=1 dir, got {block}")
        name, *dirs = block
        methods[name] = [Path(d) for d in dirs]

    rows = []
    for name, dirs in methods.items():
        seed_runs = [load_per_target(d) for d in dirs]
        n_seeds = len(seed_runs)
        for metric in TABLE17_METRICS:
            per_seed = [seed_global_metric(run, metric) for run in seed_runs]
            per_seed = [v for v in per_seed if np.isfinite(v)]
            if not per_seed:
                continue
            mean = float(np.mean(per_seed))
            if len(per_seed) >= 2:
                std = float(np.std(per_seed, ddof=1))
                std_str = f"{std:.4f}"
            else:
                std = float("nan")
                std_str = "-"  # single seed: no fabricated std
            rows.append({
                "method": name,
                "metric": metric,
                "label": TABLE17_LABELS.get(metric, metric),
                "n_seeds": len(per_seed),
                "mean": round(mean, 4),
                "std": std_str,
                "per_seed_values": ";".join(f"{v:.4f}" for v in per_seed),
            })

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "table17_seed_stability.csv"
    fields = ["method", "metric", "label", "n_seeds", "mean", "std", "per_seed_values"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {csv_path}")

    print("\n" + "=" * 80)
    print("TABLE 17: TRAINING-SEED STABILITY (mean +/- std across seeds)")
    print("=" * 80)
    print(f"{'method':<14}{'metric':<18}{'#seeds':>7}{'mean':>10}{'std':>10}")
    for r in rows:
        print(f"{r['method']:<14}{r['label']:<18}{r['n_seeds']:>7}{r['mean']:>10.4f}{r['std']:>10}")
    print("\nNotes:")
    print(" - Statistical unit = training seed; std = std across seeds (ddof=1).")
    print(" - '-' std means a single seed was provided; do NOT fabricate variance.")
    print(" - Per-target CIs belong in Table 16, NOT here (different statistical unit).")


def run_significance(args, rng):
    print("Loading PepSAGE:", args.pepsage)
    pepsage = load_per_target(args.pepsage)
    print(f"  PepSAGE targets: {len(pepsage)}")

    baselines = {}
    for name, path in [("pepsage", args.pepsage), ("PepGLAD", args.pepglad),
                       ("Discrete", args.discrete)]:
        if path is not None:
            print("Loading", name, ":", path)
            baselines[name] = load_per_target(path)
            print(f"  {name} targets: {len(baselines[name])}")

    # Decide metric list.
    all_metrics = set()
    for d in pepsage.values():
        all_metrics.update(d.keys())
    metrics = args.metrics or [m for m in METRIC_DIRECTION if m in all_metrics]

    rows = []
    for name, bdata in baselines.items():
        for metric in metrics:
            a, b, tids = paired_arrays(pepsage, bdata, metric)
            if a.shape[0] == 0:
                continue
            diff = a - b  # pepsage - baseline
            direction = METRIC_DIRECTION.get(metric, "low")
            mean_diff, lo, hi, p, dz, _ = bootstrap_paired(diff, args.n_boot, rng)
            sig = not (lo <= 0.0 <= hi)
            wr = win_rate(diff, direction)
            rows.append({
                "metric": metric,
                "comparison": f"PepSAGE vs {name}",
                "direction": "lower_better" if direction == "low" else "higher_better",
                "n_targets": a.shape[0],
                "pepsage_mean": round(float(a.mean()), 4),
                "baseline_mean": round(float(b.mean()), 4),
                "paired_diff": round(mean_diff, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "p_value": round(p, 5),
                "cohen_dz": round(dz, 4),
                "win_rate": round(wr, 4),
                "significant_95ci": sig,
                "primary": metric in PRIMARY_METRICS,
            })

    # Write CSV.
    csv_path = args.out / "table16_significance.csv"
    fields = ["metric", "comparison", "direction", "n_targets",
              "pepsage_mean", "baseline_mean", "paired_diff",
              "ci_low", "ci_high", "p_value", "cohen_dz", "win_rate",
              "significant_95ci", "primary"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path}")

    # Console summary: primary metrics first.
    print("\n" + "=" * 100)
    print("PRIMARY METRICS (main-paper Table 1/2 narrative)")
    print("=" * 100)
    hdr = f"{'metric':<16}{'comparison':<22}{'pepsage':>9}{'base':>9}{'diff':>9}{'95% CI':>22}{'p':>9}{'win%':>7}  sig"
    print(hdr)
    for r in rows:
        if not r["primary"]:
            continue
        ci = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
        print(f"{r['metric']:<16}{r['comparison']:<22}{r['pepsage_mean']:>9.3f}"
              f"{r['baseline_mean']:>9.3f}{r['paired_diff']:>+9.3f}{ci:>22}"
              f"{r['p_value']:>9.4f}{r['win_rate']*100:>6.1f}%  {'YES' if r['significant_95ci'] else 'no'}")

    print("\nFull results (all metrics) written to CSV. Notes:")
    print(" - Statistical unit = target (per-target mean over samples), not individual samples.")
    print(" - paired_diff = PepSAGE - baseline; for lower-better metrics negative = PepSAGE better.")
    print(" - significant_95ci = True iff the 95% bootstrap CI excludes 0.")
    print(" - win_rate = fraction of targets where PepSAGE beats the baseline (direction-aware).")


def main():
    ap = argparse.ArgumentParser(
        description="Target-level paired bootstrap (Table 16) and "
                    "training-seed stability (Table 17).")
    sub = ap.add_subparsers(dest="mode", required=True)

    # --- Table 16: target-level paired bootstrap ---
    p16 = sub.add_parser("significance", help="Table 16: paired bootstrap vs baselines")
    p16.add_argument("--pepsage", required=True, type=Path)
    p16.add_argument("--pepsage", type=Path, default=None)
    p16.add_argument("--pepglad", type=Path, default=None)
    p16.add_argument("--discrete", type=Path, default=None)
    p16.add_argument("--out", type=Path, default=Path("data_analyze/significance_outputs"))
    p16.add_argument("--n-boot", type=int, default=10000)
    p16.add_argument("--seed", type=int, default=1234)
    p16.add_argument("--metrics", nargs="*", default=None,
                     help="Subset of metrics; default = all known metrics present.")

    # --- Table 17: training-seed stability ---
    p17 = sub.add_parser("seeds", help="Table 17: training-seed mean +/- std")
    p17.add_argument("--method", action="append", nargs="+", required=True, metavar=("NAME", "DIR"),
                     help="Method name followed by one or more seed result dirs. "
                          "Repeat --method per method. Single dir -> std reported as '-'.")
    p17.add_argument("--out", type=Path, default=Path("data_analyze/significance_outputs"))
    p17.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()
    rng = np.random.default_rng(getattr(args, "seed", 1234))
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "significance":
        run_significance(args, rng)
    elif args.mode == "seeds":
        run_seed_stability(args, rng)


if __name__ == "__main__":
    main()
