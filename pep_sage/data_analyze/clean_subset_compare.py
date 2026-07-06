#!/usr/bin/env python3
"""Leakage-free comparison of ALL methods on the clean subset.

Removes any test target that leaked into EITHER PepSAGE's or PepFlow's
training set (union = 67 of 187 -> 120 clean targets). Reports, per metric,
each method's clean-subset mean and PepSAGE's clean-subset mean, plus a
paired bootstrap (PepSAGE vs each method) on the clean subset.
"""
from __future__ import annotations
import argparse, csv, random, re
from collections import defaultdict
from pathlib import Path

PRIMARY = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bind_ratio",
           "diversity", "novel", "bond_length_dev", "bond_angle_dev_deg", "clash_score"]
LOWER_BETTER = {"rmsd", "all_atom_rmsd", "sidechain_rmsd", "bond_length_dev",
                "bond_angle_dev_deg", "clash_score", "sample_CA_dist"}


def strip_batch(t): return re.sub(r"_batchid_\d+$", "", t)


def pct(s, q):
    if not s: return float("nan")
    k = (len(s) - 1) * q; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def boot(diffs, nb, rng):
    n = len(diffs); ms = []
    for _ in range(nb):
        ms.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    ms.sort(); md = sum(diffs) / n
    lo, hi = pct(ms, 0.025), pct(ms, 0.975)
    p = (2 * sum(1 for x in ms if x <= 0) / nb) if md >= 0 else (2 * sum(1 for x in ms if x >= 0) / nb)
    return md, lo, hi, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--leak-union", required=True, type=Path)
    ap.add_argument("--pepsage-run", default="exp8_r6_gaussian_v4_fix")
    ap.add_argument("--methods", nargs="+",
                    default=["pepflow", "pep_glad", "pepsage", "Discrete token"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    leak = {l.strip() for l in args.leak_union.read_text().splitlines() if l.strip()}
    data = defaultdict(lambda: defaultdict(dict))
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            try: data[row["run_name"]][row["metric"]][strip_batch(row["target_id"])] = float(row["mean"])
            except (ValueError, TypeError): pass

    rng = random.Random(args.seed)
    ps = data[args.pepsage_run]

    print(f"Clean subset = all shared targets MINUS {len(leak)} leaked (union of PepSAGE+PepFlow leakage)\n")
    for metric in PRIMARY:
        psm = ps.get(metric, {})
        if not psm: continue
        better_low = metric in LOWER_BETTER
        arrow = "(lower better)" if better_low else "(higher better)"
        print(f"=== {metric} {arrow} ===")
        print(f"{'method':<18}{'n_clean':>8}{'method_mean':>13}{'PepSAGE':>10}{'diff':>10}{'95% CI':>20}{'PS win%':>9}  verdict")
        for m in args.methods:
            mm = data[m].get(metric, {})
            if not mm: continue
            common = sorted((set(psm) & set(mm)) - leak)
            if not common: continue
            a = [psm[t] for t in common]; b = [mm[t] for t in common]
            psmean = sum(a)/len(a); mmean = sum(b)/len(b)
            diffs = [psm[t]-mm[t] for t in common]
            md, lo, hi, p = boot(diffs, args.n_boot, rng)
            win = sum(1 for d in diffs if (d < 0) == better_low)/len(diffs)
            ps_better = (psmean < mmean) if better_low else (psmean > mmean)
            sig = not (lo <= 0 <= hi)
            verdict = ("PS wins" if ps_better else "PS loses") + (" (sig)" if sig else " (ns)")
            ci = f"[{lo:+.3f},{hi:+.3f}]"
            print(f"{m:<18}{len(common):>8}{mmean:>13.3f}{psmean:>10.3f}{md:>+10.3f}{ci:>20}{win*100:>8.1f}%  {verdict}")
        print()


if __name__ == "__main__":
    main()
