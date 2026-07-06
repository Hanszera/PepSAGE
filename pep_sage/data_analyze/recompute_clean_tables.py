#!/usr/bin/env python3
"""Recompute all leakage-affected tables on the CLEAN subset.

Background (see AAAI_PAPER_GUIDE.md section 20): the pepsage benchmark split has
ID-level train/test overlap and PepFlow's public weights were trained on part of
our test set. Union leakage = 67 of 187 targets -> clean subset = 120 targets.
Every table that scores generation quality on the test set must be recomputed on
the clean subset. This script does that and writes one CSV per table into
data_analyze/.

Data sources (all per-target, 187 targets, verified 2026-06-12; see guide 20.6):
  - standard 6-pt method dirs in exp8/logs/<method>/eval_{metrics,atom_metrics,other_metrics}_summary.pt
  - temperature sweep in exp8_temp/logs/{05,08,10,15,20}/ (task+other only)
  - sc_detail per-sample in exp8/data_analyze/sc_analysis_outputs/1/sc_detail_*.pt

No torch dependency: .pt files are PyTorch zip archives; we unpickle the
data.pkl member with a stub that turns torch objects into placeholders. All
metrics we need are plain python floats/lists, so this is sufficient.

Tables produced (🔴 in the guide):
  table01_overall.csv          all-atom generation (rmsd/aa/sc/contact/diversity/novel...)
  table02_geometry.csv         local geometry (bond_length/bond_angle/clash/...)
  table03_sigma.csv            sigma1_z ablation
  table04_temperature.csv      temperature sweep (Fig.2 source)
  table10_12_sidechain.csv     side-chain detail (sc_rmsd by length, joint chi acc)
  table05_decoder.csv          decoder intermediate vs final
  table16_significance.csv     paired bootstrap PepSAGE vs each baseline (clean)
  table17_seed.csv             training-seed stability (clean subset)

Usage:
  python data_analyze/recompute_clean_tables.py \
      --leak-union /tmp/leak_union.txt
  (leak-union defaults to data_analyze/leak_union.txt if present)
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- pt loading
class _Stub:
    def __init__(self, *a, **k):
        pass
    def __setstate__(self, s):
        self.s = s


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch"):
            return _Stub
        return super().find_class(module, name)
    def persistent_load(self, pid):
        return _Stub()


def load_pt(path: Path):
    with zipfile.ZipFile(path) as z:
        member = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        with z.open(member) as f:
            return _Unpickler(f).load()


def strip_batch(tid: str) -> str:
    return re.sub(r"_batchid_\d+$", "", tid)


# ---------------------------------------------------------------- stats
def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs, ddof=1):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def paired_bootstrap(diffs, n_boot, rng):
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    md = sum(diffs) / n
    lo, hi = percentile(boot, 0.025), percentile(boot, 0.975)
    if md >= 0:
        p = 2 * sum(1 for x in boot if x <= 0) / n_boot
    else:
        p = 2 * sum(1 for x in boot if x >= 0) / n_boot
    return md, lo, hi, min(1.0, p)


# ---------------------------------------------------------------- config
LOWER_BETTER = {
    "rmsd", "all_atom_rmsd", "sidechain_rmsd", "bond_length_dev",
    "bond_angle_dev_deg", "clash_score", "sample_CA_dist",
    "chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg",
    "short_all_atom_rmsd", "medium_all_atom_rmsd", "long_all_atom_rmsd",
    "sc_rmsd_bb_aligned", "short_sc_sc_rmsd", "medium_sc_sc_rmsd", "long_sc_sc_rmsd",
}
# Note: aar = amino-acid recovery rate -> higher is better (NOT in LOWER_BETTER,
# despite the legacy variant_comparison.csv mislabeling it "lower").
# gt_CA_dist / sample_CA_dist are descriptive geometry references, not quality
# scores; sample_CA_dist is treated lower-better only for directional display.

# metric -> which summary file it lives in
TASK_METRICS = ["rmsd", "bind_ratio", "ss_ratio", "valid", "novel", "diversity"]
ATOM_METRICS = ["all_atom_rmsd", "sidechain_rmsd", "rotamer_recovery", "clash_score",
                "bond_length_dev", "bond_angle_dev_deg", "short_all_atom_rmsd",
                "chi1_mae_deg", "chi2_mae_deg", "chi3_mae_deg", "chi4_mae_deg"]
OTHER_METRICS = ["gt_CA_dist", "sample_CA_dist", "aar"]

TABLE1_METRICS = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bind_ratio", "diversity", "novel", "aar"]
TABLE2_METRICS = ["bond_length_dev", "bond_angle_dev_deg", "clash_score", "rotamer_recovery"]

LOGS = None       # set in main
TEMP_LOGS = None
SC_DIR = None

METHODS = {
    "PepSAGE": "pep_sage",
    "PepFlow": "pepflow180",
    "PepGLAD": "pep_glad",
    "pepsage": "pepsage",
    "Discrete": "Discrete token",
}


def direction_label(metric):
    return "lower_better" if metric in LOWER_BETTER else "higher_better"


# ---------------------------------------------------------------- loaders
_summary_cache = {}


def method_per_target(run_dir: Path):
    """Return {metric: {target_prefix: mean}} merged across the 3 summary files."""
    key = str(run_dir)
    if key in _summary_cache:
        return _summary_cache[key]
    merged = defaultdict(dict)
    for fn in ["eval_metrics_summary.pt", "eval_atom_metrics_summary.pt",
               "eval_other_metrics_summary.pt"]:
        path = run_dir / fn
        if not path.exists():
            continue
        per_target = load_pt(path)["per_target"]
        for tid, metrics in per_target.items():
            pref = strip_batch(tid)
            for m, stat in metrics.items():
                if isinstance(stat, dict) and stat.get("mean") is not None:
                    v = stat["mean"]
                    if isinstance(v, float) and math.isfinite(v):
                        merged[m][pref] = v
    _summary_cache[key] = merged
    return merged


def sc_per_target(path: Path):
    """Side-chain per-target values from per_sample; chi joint uses count ratios."""
    res = load_pt(path)
    per_sample = res.get("per_sample", {})
    out = defaultdict(dict)  # metric -> {target_prefix: value}

    def finite(xs):
        return [float(x) for x in xs if x is not None and math.isfinite(float(x))]

    for tid, tm in per_sample.items():
        pref = strip_batch(tid)
        # length-bucketed sc rmsd + overall + per-sample-averaged simple metrics
        for m in ["sc_rmsd_bb_aligned", "short_sc_sc_rmsd", "medium_sc_sc_rmsd",
                  "long_sc_sc_rmsd", "sc_rmsd_bb_aligned_all_legacy"]:
            vals = finite(tm.get(m, []))
            if vals:
                out[m][pref] = sum(vals) / len(vals)
        # weighted joint chi accuracy: sum(correct)/sum(valid) over samples
        for depth in [1, 2, 3, 4]:
            num = finite(tm.get(f"chi_joint_depth{depth}_correct_count", []))
            den = finite(tm.get(f"chi_joint_depth{depth}_valid_count", []))
            size = min(len(num), len(den))
            if size:
                d = sum(den[:size])
                if d > 0:
                    out[f"chi_joint_acc_depth{depth}_weighted"][pref] = sum(num[:size]) / d
        # sequence match coverage
        num = finite(tm.get("sequence_match_residue_count", []))
        den = finite(tm.get("matched_residue_count", []))
        size = min(len(num), len(den))
        if size and sum(den[:size]) > 0:
            out["sequence_match_coverage"][pref] = sum(num[:size]) / sum(den[:size])
    return out


# ---------------------------------------------------------------- table writers
def clean_filter(target_set, leak):
    return sorted(t for t in target_set if t not in leak)


def write_method_table(out_csv, metrics, leak, methods=("PepSAGE", "PepFlow", "PepGLAD", "pepsage")):
    """One row per metric: each method's clean-subset mean over its targets."""
    data = {name: method_per_target(LOGS / METHODS[name]) for name in methods}
    rows = []
    for metric in metrics:
        row = {"metric": metric, "direction": direction_label(metric)}
        # common clean target set across all methods that have this metric
        present = [n for n in methods if metric in data[n]]
        if not present:
            continue
        common = set.intersection(*[set(data[n][metric]) for n in present])
        common = clean_filter(common, leak)
        row["n_clean"] = len(common)
        for n in methods:
            if metric in data[n]:
                row[n] = round(mean([data[n][metric][t] for t in common if t in data[n][metric]]), 4)
            else:
                row[n] = ""
        rows.append(row)
    fields = ["metric", "direction", "n_clean"] + list(methods)
    _dump(out_csv, fields, rows)
    return rows


def write_table16(out_csv, leak, n_boot, seed):
    """Paired bootstrap PepSAGE vs each baseline on clean subset."""
    rng = random.Random(seed)
    ps = method_per_target(LOGS / METHODS["PepSAGE"])
    metrics = TABLE1_METRICS + TABLE2_METRICS
    rows = []
    for base in ["PepFlow", "PepGLAD", "pepsage", "Discrete"]:
        bd = method_per_target(LOGS / METHODS[base])
        for metric in metrics:
            if metric not in ps or metric not in bd:
                continue
            common = clean_filter(set(ps[metric]) & set(bd[metric]), leak)
            if not common:
                continue
            diffs = [ps[metric][t] - bd[metric][t] for t in common]
            md, lo, hi, p = paired_bootstrap(diffs, n_boot, rng)
            lower = metric in LOWER_BETTER
            win = sum(1 for d in diffs if (d < 0) == lower) / len(diffs)
            ps_mean = mean([ps[metric][t] for t in common])
            b_mean = mean([bd[metric][t] for t in common])
            sig = not (lo <= 0 <= hi)
            ps_better = (ps_mean < b_mean) if lower else (ps_mean > b_mean)
            rows.append({
                "metric": metric, "comparison": f"PepSAGE vs {base}",
                "direction": direction_label(metric), "n_clean": len(common),
                "pepsage_mean": round(ps_mean, 4), "baseline_mean": round(b_mean, 4),
                "paired_diff": round(md, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "p_value": round(p, 5), "win_rate": round(win, 4),
                "significant_95ci": sig, "pepsage_better": ps_better,
            })
    fields = ["metric", "comparison", "direction", "n_clean", "pepsage_mean",
              "baseline_mean", "paired_diff", "ci_low", "ci_high", "p_value",
              "win_rate", "significant_95ci", "pepsage_better"]
    _dump(out_csv, fields, rows)
    return rows


def write_table17(out_csv, leak):
    """Training-seed stability on clean subset (PepSAGE seeds; baselines single)."""
    seed_dirs = {
        "PepSAGE": ["exp8_r6_gaussian_v4_fix", "seed4321"],
        "PepFlow": ["pepflow"], "PepGLAD": ["pep_glad"],
        "pepsage": ["pepsage"], "Discrete": ["Discrete token"],
    }
    metrics = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bind_ratio", "diversity", "novel"]
    rows = []
    for method, dirs in seed_dirs.items():
        per_seed = [method_per_target(LOGS / d) for d in dirs]
        for metric in metrics:
            seed_means = []
            for spt in per_seed:
                if metric in spt:
                    common = clean_filter(set(spt[metric]), leak)
                    seed_means.append(mean([spt[metric][t] for t in common]))
            seed_means = [x for x in seed_means if math.isfinite(x)]
            if not seed_means:
                continue
            m = mean(seed_means)
            s = std(seed_means) if len(seed_means) >= 2 else None
            rows.append({
                "method": method, "metric": metric, "direction": direction_label(metric),
                "n_seeds": len(seed_means), "mean": round(m, 4),
                "std": round(s, 4) if s is not None else "-",
                "per_seed_values": ";".join(f"{x:.4f}" for x in seed_means),
            })
    fields = ["method", "metric", "direction", "n_seeds", "mean", "std", "per_seed_values"]
    _dump(out_csv, fields, rows)
    return rows


def write_temperature(out_csv, leak):
    """Temperature sweep: per-temperature clean mean of task+other+atom metrics.

    Atom metrics (clash/bond/all-atom) added 2026-06-12 so Fig.2 can show whether
    high-temperature diversity comes at the cost of geometry validity.
    """
    temps = {"0.5": "05", "0.8": "08", "1.0": "10", "1.5": "15", "2.0": "20"}
    geom = ["all_atom_rmsd", "sidechain_rmsd", "clash_score",
            "bond_length_dev", "bond_angle_dev_deg", "rotamer_recovery"]
    metrics = TASK_METRICS + OTHER_METRICS + geom
    rows = []
    for tval, d in temps.items():
        run = TEMP_LOGS / d
        merged = defaultdict(dict)
        for fn in ["eval_metrics_summary.pt", "eval_other_metrics_summary.pt",
                   "eval_atom_metrics_summary.pt"]:
            path = run / fn
            if not path.exists():
                continue
            for tid, ms in load_pt(path)["per_target"].items():
                pref = strip_batch(tid)
                for m, stat in ms.items():
                    if isinstance(stat, dict) and isinstance(stat.get("mean"), float) and math.isfinite(stat["mean"]):
                        merged[m][pref] = stat["mean"]
        row = {"temperature": tval}
        for m in metrics:
            if m in merged:
                common = clean_filter(set(merged[m]), leak)
                row[m] = round(mean([merged[m][t] for t in common]), 4)
                row["n_clean"] = len(common)
            else:
                row[m] = ""
        rows.append(row)
    fields = ["temperature", "n_clean"] + metrics
    _dump(out_csv, fields, rows)
    return rows


def write_sidechain(out_csv, leak):
    """Tables 10-12: side-chain detail per method on clean subset."""
    files = {
        "PepSAGE": "sc_detail_pepsage40.pt",
        "pepsage": "sc_detail_pepsage40.pt",
        "PepGLAD": "sc_detail_pepglad40.pt",
        "Direct": "sc_detail_Direct40.pt",
    }
    metrics = ["sc_rmsd_bb_aligned", "short_sc_sc_rmsd", "medium_sc_sc_rmsd", "long_sc_sc_rmsd",
               "chi_joint_acc_depth1_weighted", "chi_joint_acc_depth2_weighted",
               "chi_joint_acc_depth3_weighted", "chi_joint_acc_depth4_weighted",
               "sequence_match_coverage"]
    data = {}
    for name, fn in files.items():
        path = SC_DIR / fn
        if path.exists():
            data[name] = sc_per_target(path)
    rows = []
    for metric in metrics:
        row = {"metric": metric, "direction": direction_label(metric)}
        for name in files:
            if name in data and metric in data[name]:
                common = clean_filter(set(data[name][metric]), leak)
                row[name] = round(mean([data[name][metric][t] for t in common]), 4)
                row[f"{name}_n"] = len(common)
            else:
                row[name] = ""
                row[f"{name}_n"] = 0
        rows.append(row)
    fields = ["metric", "direction"]
    for name in files:
        fields += [name, f"{name}_n"]
    _dump(out_csv, fields, rows)
    return rows


def write_decoder(out_csv, leak):
    """Table 13: decoder intermediate (v4_rebalance) vs final (v4_fix)."""
    inter = method_per_target(LOGS / "exp8_r6_gaussian_v4_rebalance_2026-05-17-10_32_12")
    final = method_per_target(LOGS / METHODS["PepSAGE"])
    metrics = ["rmsd", "all_atom_rmsd", "sidechain_rmsd", "bond_length_dev",
               "bond_angle_dev_deg", "clash_score"]
    rows = []
    for metric in metrics:
        if metric not in inter or metric not in final:
            continue
        common = clean_filter(set(inter[metric]) & set(final[metric]), leak)
        if not common:
            continue
        im = mean([inter[metric][t] for t in common])
        fm = mean([final[metric][t] for t in common])
        note = ("backbone shared (decoder only refines all-atom/side-chain) "
                "-> rmsd identical by design" if metric == "rmsd" else "")
        rows.append({
            "metric": metric, "direction": direction_label(metric), "n_clean": len(common),
            "decoder_intermediate": round(im, 4), "pepsage_final": round(fm, 4),
            "change": round(fm - im, 4), "note": note,
        })
    fields = ["metric", "direction", "n_clean", "decoder_intermediate", "pepsage_final", "change", "note"]
    _dump(out_csv, fields, rows)
    return rows


def _dump(path: Path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):>3} rows -> {path.name}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent          # exp8_ex/data_analyze
    aa = here.parents[1]                            # aa/
    ap.add_argument("--logs", type=Path, default=here.parent / "output",)
    ap.add_argument("--temp-logs", type=Path, default=here.parent / "output" / "temp",)
    ap.add_argument("--sc-dir", type=Path,
                    default=aa / "PepSAGE" / "pep_sage" / "output",)
    ap.add_argument("--leak-union", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=here)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    global LOGS, TEMP_LOGS, SC_DIR
    LOGS, TEMP_LOGS, SC_DIR = args.logs, args.temp_logs, args.sc_dir

    leak_path = args.leak_union or (here / "leak_union.txt")
    if not leak_path.exists():
        raise SystemExit(
            f"Leak-union file not found: {leak_path}\n"
            "Generate it with the snippet in guide 20.5 (intersect test with "
            "pepsage_train ∪ pepflow_train) and pass via --leak-union.")
    leak = {l.strip() for l in leak_path.read_text().splitlines() if l.strip()}
    print(f"Leak-union targets: {len(leak)}  (clean subset = shared - leak)\n")

    out = args.out_dir
    print("Recomputing clean-subset tables:")
    write_method_table(out / "table01_overall.csv", TABLE1_METRICS, leak)
    write_method_table(out / "table02_geometry.csv", TABLE2_METRICS, leak)
    # sigma ablation: PepSAGE main + two sigma variants
    _write_sigma(out / "table03_sigma.csv", leak)
    write_temperature(out / "table04_temperature.csv", leak)
    write_sidechain(out / "table10_12_sidechain.csv", leak)
    write_decoder(out / "table05_decoder.csv", leak)
    write_table16(out / "table16_significance.csv", leak, args.n_boot, args.seed)
    write_table17(out / "table17_seed.csv", leak)
    print("\nDone. CSVs in", out)


def _write_sigma(out_csv, leak):
    variants = {
        "sigma1_z=0.05": "exp8_r6_gaussian_sigma1_z005_2026-05-19-20_58_52",
        "sigma1_z=0.10": "pep_sage",
        "sigma1_z=0.15": "exp8_r6_gaussian_sigma1_z015_2026-05-19-21_02_52",
    }
    metrics = TABLE1_METRICS + TABLE2_METRICS
    data = {name: method_per_target(LOGS / d) for name, d in variants.items()}
    rows = []
    for metric in metrics:
        row = {"metric": metric, "direction": direction_label(metric)}
        present = [n for n in variants if metric in data[n]]
        if not present:
            continue
        common = set.intersection(*[set(data[n][metric]) for n in present])
        common = clean_filter(common, leak)
        row["n_clean"] = len(common)
        for n in variants:
            if metric in data[n]:
                row[n] = round(mean([data[n][metric][t] for t in common if t in data[n][metric]]), 4)
            else:
                row[n] = ""
        rows.append(row)
    fields = ["metric", "direction", "n_clean"] + list(variants)
    _dump(out_csv, fields, rows)


if __name__ == "__main__":
    main()
