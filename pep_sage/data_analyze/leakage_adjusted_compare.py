#!/usr/bin/env python3
"""Compare PepSAGE vs PepFlow on the FULL 187 set vs the LEAKAGE-FREE subset.

PepFlow's public model1 was trained on 9849 complexes; 33 of our 187 test
targets (PDB_chain exact match) appear in that training set -> train/test
leakage. On those targets PepFlow memorizes the native structure (e.g. 1aze_B
RMSD 0.746, std 0.018), deflating its mean. The fair comparison restricts to
the 154 targets PepFlow never trained on.

Reads the long-format per_target_summary.csv (one row per
run/target/metric) and the leakage id list, no torch/numpy required.

Outputs, for each metric:
  - PepSAGE mean / PepFlow mean on FULL (187) and CLEAN (154)
  - paired difference + 95% bootstrap CI + win rate on the CLEAN subset
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

PRIMARY = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bind_ratio", "diversity",
           "bond_length_dev", "bond_angle_dev_deg", "clash_score", "novel"]
LOWER_BETTER = {"rmsd", "all_atom_rmsd", "sidechain_rmsd", "bond_length_dev",
                "bond_angle_dev_deg", "clash_score", "sample_CA_dist"}


def strip_batch(tid: str) -> str:
    return re.sub(r"_batchid_\d+$", "", tid)


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def bootstrap_ci(diffs, n_boot, rng):
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    md = sum(diffs) / n
    lo, hi = pct(means, 0.025), pct(means, 0.975)
    if md >= 0:
        p = 2 * sum(1 for x in means if x <= 0) / n_boot
    else:
        p = 2 * sum(1 for x in means if x >= 0) / n_boot
    return md, lo, hi, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path,
                    help="per_target_summary.csv (long format)")
    ap.add_argument("--leak", required=True, type=Path,
                    help="text file of leaked target ids (one PDB_chain per line)")
    ap.add_argument("--pepsage-run", default="StructToken (gaussian)",
                    help="run_name for PepSAGE in the csv")
    ap.add_argument("--pepflow-run", default="pepflow")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    leak = {l.strip() for l in args.leak.read_text().splitlines() if l.strip()}
    print(f"Leaked ids loaded: {len(leak)}")

    # data[run][metric][target_prefix] = mean
    data = defaultdict(lambda: defaultdict(dict))
    runs_seen = set()
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            run = row["run_name"]
            runs_seen.add(run)
            metric = row["metric"]
            tgt = strip_batch(row["target_id"])
            try:
                data[run][metric][tgt] = float(row["mean"])
            except (ValueError, TypeError):
                pass

    ps_run, pf_run = args.pepsage_run, args.pepflow_run
    if ps_run not in runs_seen or pf_run not in runs_seen:
        print("\nAvailable run_name values:")
        for r in sorted(runs_seen):
            print("  ", repr(r))
        raise SystemExit(f"\nrun_name {ps_run!r} or {pf_run!r} not found; pick from above with --pepsage-run/--pepflow-run")

    rng = random.Random(args.seed)

    print("\n" + "=" * 110)
    print(f"{'metric':<18}{'set':<8}{'n':>5}{'PepSAGE':>10}{'PepFlow':>10}{'diff(PS-PF)':>13}"
          f"{'95% CI':>22}{'PS win%':>9}")
    print("=" * 110)

    for metric in PRIMARY:
        ps = data[ps_run].get(metric, {})
        pf = data[pf_run].get(metric, {})
        common = sorted(set(ps) & set(pf))
        if not common:
            continue
        clean = [t for t in common if t not in leak]
        for label, ids in [("FULL", common), ("CLEAN", clean)]:
            a = [ps[t] for t in ids]
            b = [pf[t] for t in ids]
            ps_mean = sum(a) / len(a)
            pf_mean = sum(b) / len(b)
            diffs = [ps[t] - pf[t] for t in ids]
            md, lo, hi, p = bootstrap_ci(diffs, args.n_boot, rng)
            better_low = metric in LOWER_BETTER
            win = sum(1 for d in diffs if (d < 0) == better_low) / len(diffs)
            ci = f"[{lo:+.3f},{hi:+.3f}]"
            tag = ""
            if label == "CLEAN":
                # is PepSAGE better on clean set?
                ps_better = (ps_mean < pf_mean) if better_low else (ps_mean > pf_mean)
                tag = "  <-PS better" if ps_better else ""
            print(f"{metric if label=='FULL' else '':<18}{label:<8}{len(ids):>5}"
                  f"{ps_mean:>10.3f}{pf_mean:>10.3f}{md:>+13.3f}{ci:>22}{win*100:>8.1f}%{tag}")
        print("-" * 110)

    print("\nNotes:")
    print(" - lower-better metrics: negative diff = PepSAGE better.")
    print(" - FULL = all 187 shared targets; CLEAN = excludes leaked targets.")
    print(" - Compare FULL vs CLEAN PepFlow mean: the jump quantifies the leakage benefit.")


if __name__ == "__main__":
    main()
