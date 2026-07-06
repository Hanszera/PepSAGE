"""
Analyze side-chain detail evaluation files produced by evaluate_sidechain_detail.py.

Default input:
    data_analyze/sc_analysis_outputs/sc_detail_*.pt

Outputs:
    sc_detail_summary.csv
    sc_detail_paired_comparisons.csv
    sc_detail_tables.md

Example:
    python data_analyze/analyze_sc_detail_results.py

    python data_analyze/analyze_sc_detail_results.py \
        --input_dir data_analyze/sc_analysis_outputs \
        --num_bootstrap 10000 \
        --seed 2026
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


METHOD_ORDER = [
    "PepSAGE",
    "PepFlow",
    "PepGLAD",
    "pepsage",
]

FILE_METHOD_NAMES = {
    "sc_detail_pepsega": "PepSAGE",
    "sc_detail_pepflow_180": "PepFlow",
    "sc_detail_pepgald": "PepGLAD",
    "sc_detail_pepsage": "pepsage",
}

METRICS = {
    "sc_rmsd_bb_aligned": {
        "label": "BB-aligned SC-RMSD",
        "direction": "lower",
        "digits": 3,
    },
    "short_sc_sc_rmsd": {
        "label": "Short-chain SC-RMSD",
        "direction": "lower",
        "digits": 3,
    },
    "medium_sc_sc_rmsd": {
        "label": "Medium-chain SC-RMSD",
        "direction": "lower",
        "digits": 3,
    },
    "long_sc_sc_rmsd": {
        "label": "Long-chain SC-RMSD",
        "direction": "lower",
        "digits": 3,
    },
    "chi_joint_acc_depth1_weighted": {
        "label": "Joint chi1 accuracy",
        "direction": "higher",
        "digits": 3,
    },
    "chi_joint_acc_depth2_weighted": {
        "label": "Joint chi1&chi2 accuracy",
        "direction": "higher",
        "digits": 3,
    },
    "chi_joint_acc_depth3_weighted": {
        "label": "Joint chi1&chi2&chi3 accuracy",
        "direction": "higher",
        "digits": 3,
    },
    "chi_joint_acc_depth4_weighted": {
        "label": "Joint chi1&chi2&chi3&chi4 accuracy",
        "direction": "higher",
        "digits": 3,
    },
    "sequence_match_coverage": {
        "label": "Sequence-match coverage",
        "direction": "higher",
        "digits": 3,
    },
    "bond_length_dev": {
        "label": "Bond-length deviation",
        "direction": "lower",
        "digits": 3,
    },
    "clash_score": {
        "label": "Clash score",
        "direction": "lower",
        "digits": 3,
    },
}

TABLE10_METRICS = ["sc_rmsd_bb_aligned"]
TABLE11_METRICS = [
    "short_sc_sc_rmsd",
    "medium_sc_sc_rmsd",
    "long_sc_sc_rmsd",
]
TABLE12_METRICS = [
    "chi_joint_acc_depth1_weighted",
    "chi_joint_acc_depth2_weighted",
    "chi_joint_acc_depth3_weighted",
    "chi_joint_acc_depth4_weighted",
]


def load_result(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_method_name(path: Path, result: dict) -> str:
    stem = path.stem.lower()
    if stem in FILE_METHOD_NAMES:
        return FILE_METHOD_NAMES[stem]

    run_name = str(result.get("run_name", path.stem)).lower()
    if "pepsage" in run_name:
        return "pepsage-40"
    if "pepglad" in run_name:
        return "PepGLAD-40"
    if "direct" in run_name:
        return "PepSAGE-Direct-40"
    if "pepsage" in run_name and "64" in run_name:
        return "PepSAGE-64"
    if "pepsage" in run_name and "40" in run_name:
        return "PepSAGE-40"
    return result.get("run_name", path.stem)


def get_per_sample(result: dict) -> dict:
    if "per_sample" in result:
        return result["per_sample"]

    # Older files may store raw lists directly under per_target.
    per_target = result.get("per_target", {})
    for target_metrics in per_target.values():
        if target_metrics:
            first_value = next(iter(target_metrics.values()))
            if isinstance(first_value, list):
                return per_target
            break
    return {}


def finite_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def target_metric_values(result: dict, metric: str) -> dict[str, float]:
    per_sample = get_per_sample(result)
    values = {}

    if metric.startswith("chi_joint_acc_depth") and metric.endswith("_weighted"):
        depth = metric.removeprefix("chi_joint_acc_depth").removesuffix("_weighted")
        numerator_name = f"chi_joint_depth{depth}_correct_count"
        denominator_name = f"chi_joint_depth{depth}_valid_count"
        for target, target_metrics in per_sample.items():
            numerators = finite_array(target_metrics.get(numerator_name, []))
            denominators = finite_array(target_metrics.get(denominator_name, []))
            size = min(numerators.size, denominators.size)
            if size == 0:
                continue
            denominator = denominators[:size].sum()
            if denominator > 0:
                values[target] = float(numerators[:size].sum() / denominator)
        return values

    if metric == "sequence_match_coverage":
        for target, target_metrics in per_sample.items():
            numerators = finite_array(
                target_metrics.get("sequence_match_residue_count", [])
            )
            denominators = finite_array(
                target_metrics.get("matched_residue_count", [])
            )
            size = min(numerators.size, denominators.size)
            if size == 0:
                continue
            denominator = denominators[:size].sum()
            if denominator > 0:
                values[target] = float(numerators[:size].sum() / denominator)
        return values

    for target, target_metrics in per_sample.items():
        metric_values = finite_array(target_metrics.get(metric, []))
        if metric_values.size:
            values[target] = float(metric_values.mean())
    return values


def stored_summary(result: dict, metric: str) -> dict | None:
    summary = result.get("paper_aggregate", {}).get(metric)
    if not summary:
        return None
    return {
        "estimate": float(summary["estimate"]),
        "ci_low": float(summary["ci_low"]),
        "ci_high": float(summary["ci_high"]),
        "num_targets": int(summary["num_targets"]),
        "numerator": summary.get("numerator"),
        "denominator": summary.get("denominator"),
    }


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    num_bootstrap: int,
    confidence: float,
) -> tuple[float, float]:
    if values.size == 1 or num_bootstrap == 0:
        return float(values.mean()), float(values.mean())
    indices = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def derive_summary(
    result: dict,
    metric: str,
    rng: np.random.Generator,
    num_bootstrap: int,
    confidence: float,
) -> dict | None:
    target_values = target_metric_values(result, metric)
    if not target_values:
        return None
    array = np.asarray(list(target_values.values()), dtype=float)
    ci_low, ci_high = bootstrap_mean(
        array, rng, num_bootstrap, confidence
    )
    return {
        "estimate": float(array.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "num_targets": int(array.size),
        "numerator": None,
        "denominator": None,
    }


def paired_bootstrap(
    baseline_values: dict[str, float],
    method_values: dict[str, float],
    direction: str,
    rng: np.random.Generator,
    num_bootstrap: int,
    confidence: float,
) -> dict | None:
    common_targets = sorted(set(baseline_values) & set(method_values))
    if not common_targets:
        return None

    baseline = np.asarray([baseline_values[t] for t in common_targets])
    method = np.asarray([method_values[t] for t in common_targets])
    difference = method - baseline

    if num_bootstrap > 0 and difference.size > 1:
        indices = rng.integers(
            0, difference.size, size=(num_bootstrap, difference.size)
        )
        sampled_differences = difference[indices].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        ci_low = float(np.quantile(sampled_differences, alpha))
        ci_high = float(np.quantile(sampled_differences, 1.0 - alpha))
    else:
        ci_low = ci_high = float(difference.mean())

    if direction == "lower":
        wins = method < baseline
    else:
        wins = method > baseline

    baseline_mean = float(baseline.mean())
    method_mean = float(method.mean())
    relative_change = (
        (method_mean - baseline_mean) / abs(baseline_mean) * 100.0
        if baseline_mean != 0 else np.nan
    )
    return {
        "num_targets": len(common_targets),
        "baseline_mean": baseline_mean,
        "method_mean": method_mean,
        "mean_difference": float(difference.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "relative_change_percent": float(relative_change),
        "win_rate": float(wins.mean()),
    }


def format_value(summary: dict | None, digits: int) -> str:
    if not summary:
        return "N/A"
    return (
        f"{summary['estimate']:.{digits}f} "
        f"[{summary['ci_low']:.{digits}f}, "
        f"{summary['ci_high']:.{digits}f}]"
    )


def write_summary_csv(path: Path, summaries: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "metric", "label", "direction", "estimate",
            "ci_low", "ci_high", "num_targets", "numerator", "denominator",
        ])
        for method in METHOD_ORDER:
            for metric, spec in METRICS.items():
                summary = summaries.get(method, {}).get(metric)
                if not summary:
                    continue
                writer.writerow([
                    method, metric, spec["label"], spec["direction"],
                    summary["estimate"], summary["ci_low"], summary["ci_high"],
                    summary["num_targets"], summary.get("numerator"),
                    summary.get("denominator"),
                ])


def write_paired_csv(path: Path, comparisons: list[dict]) -> None:
    fieldnames = [
        "comparison", "metric", "label", "direction", "num_targets",
        "baseline_mean", "method_mean", "mean_difference", "ci_low",
        "ci_high", "relative_change_percent", "win_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)


def markdown_table(
    title: str,
    metrics: list[str],
    methods: list[str],
    summaries: dict,
) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| Metric | " + " | ".join(methods) + " |")
    lines.append("|---|" + "|".join(["---:" for _ in methods]) + "|")
    for metric in metrics:
        spec = METRICS[metric]
        cells = [
            format_value(
                summaries.get(method, {}).get(metric),
                spec["digits"],
            )
            for method in methods
        ]
        lines.append(f"| {spec['label']} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_markdown(path: Path, summaries: dict, comparisons: list[dict]) -> None:
    main_methods = ["pepsage-40", "PepGLAD-40", "PepSAGE-40"]
    lines = [
        "# Side-chain Detail Analysis",
        "",
        "Values are estimates with target-level bootstrap 95% confidence intervals.",
        "The main comparison uses 40 samples per target for every method.",
        "",
    ]
    lines += markdown_table(
        "Table 10: BB-aligned side-chain RMSD",
        TABLE10_METRICS,
        main_methods,
        summaries,
    )
    lines += markdown_table(
        "Table 11: Side-chain complexity",
        TABLE11_METRICS,
        main_methods,
        summaries,
    )
    lines += markdown_table(
        "Table 12: Sequence-matched joint chi accuracy",
        TABLE12_METRICS,
        main_methods,
        summaries,
    )
    lines += markdown_table(
        "Sequence-match coverage",
        ["sequence_match_coverage"],
        main_methods,
        summaries,
    )
    lines += markdown_table(
        "PepSAGE sample-count sensitivity",
        list(METRICS),
        ["PepSAGE-40", "PepSAGE-64"],
        summaries,
    )
    lines += markdown_table(
        "Table 13: Direct vs standardized decoder output",
        [
            "sc_rmsd_bb_aligned",
            "bond_length_dev",
            "clash_score",
            "chi_joint_acc_depth1_weighted",
        ],
        ["PepSAGE-Direct-40", "PepSAGE-40"],
        summaries,
    )
    direct_has_geometry = all(
        metric in summaries.get("PepSAGE-Direct-40", {})
        for metric in ("bond_length_dev", "clash_score")
    )
    standardized_has_geometry = all(
        metric in summaries.get("PepSAGE-40", {})
        for metric in ("bond_length_dev", "clash_score")
    )
    lines += [
        "Table 13 uses sequence-matched residues for bond-length deviation and "
        "the predicted sequence topology for clash scoring.",
        "",
    ]
    if not (direct_has_geometry and standardized_has_geometry):
        lines += [
            "> Warning: bond-length deviation and/or clash score is missing. "
            "Re-run `evaluate_sidechain_detail.py` for both Direct and "
            "standardized outputs with the updated metric implementation.",
            "",
        ]

    lines += [
        "## Paired target comparisons",
        "",
        "| Comparison | Metric | n | Baseline | Method | Difference | 95% CI | Relative change | Win rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['comparison']} | {item['label']} | "
            f"{item['num_targets']} | {item['baseline_mean']:.4f} | "
            f"{item['method_mean']:.4f} | {item['mean_difference']:.4f} | "
            f"[{item['ci_low']:.4f}, {item['ci_high']:.4f}] | "
            f"{item['relative_change_percent']:+.1f}% | "
            f"{item['win_rate']:.3f} |"
        )
    lines.append("")
    lines += [
        "Interpretation:",
        "",
        "- For RMSD metrics, negative differences favor PepSAGE.",
        "- For joint-chi accuracy and sequence coverage, positive differences favor PepSAGE.",
        "- A paired-difference confidence interval excluding zero provides evidence of a consistent target-level difference.",
        "- PepSAGE-40 should be used for the fair main comparison; PepSAGE-64 is a sensitivity analysis.",
        "- PepSAGE-Direct-40 vs PepSAGE-40 isolates the decoder output choice; negative RMSD differences and positive accuracy differences favor the standardized output.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent / "sc_analysis_outputs"
    parser.add_argument("--input_dir", type=Path, default=default_dir)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--num_bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"input directory not found: {args.input_dir}")
    if args.num_bootstrap < 0:
        parser.error("--num_bootstrap must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")

    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("sc_detail_*.pt"))
    if not paths:
        parser.error(f"no sc_detail_*.pt files found in {args.input_dir}")

    results = {}
    source_paths = {}
    for path in paths:
        result = load_result(path)
        method = infer_method_name(path, result)
        if method in results:
            raise RuntimeError(
                f"duplicate method label {method}: {source_paths[method]} and {path}"
            )
        results[method] = result
        source_paths[method] = path
        print(f"Loaded {method}: {path.name}")

    rng = np.random.default_rng(args.seed)
    summaries = {}
    for method, result in results.items():
        summaries[method] = {}
        for metric in METRICS:
            summary = stored_summary(result, metric)
            if summary is None:
                summary = derive_summary(
                    result,
                    metric,
                    rng,
                    args.num_bootstrap,
                    args.confidence,
                )
            if summary is not None:
                summaries[method][metric] = summary

    comparisons = []
    comparison_specs = [
        ("PepSAGE vs PepFlow", "PepFlow", "PepSAGE"),
    ]
    for comparison_name, baseline_name, method_name in comparison_specs:
        if baseline_name not in results or method_name not in results:
            continue
        for metric, spec in METRICS.items():
            paired = paired_bootstrap(
                target_metric_values(results[baseline_name], metric),
                target_metric_values(results[method_name], metric),
                spec["direction"],
                rng,
                args.num_bootstrap,
                args.confidence,
            )
            if paired is None:
                continue
            comparisons.append({
                "comparison": comparison_name,
                "metric": metric,
                "label": spec["label"],
                "direction": spec["direction"],
                **paired,
            })

    summary_path = output_dir / "sc_detail_summary.csv"
    paired_path = output_dir / "sc_detail_paired_comparisons.csv"
    markdown_path = output_dir / "sc_detail_tables.md"
    write_summary_csv(summary_path, summaries)
    write_paired_csv(paired_path, comparisons)
    write_markdown(markdown_path, summaries, comparisons)

    print("\nMain comparison (40 samples per target)")
    for metric in TABLE10_METRICS + TABLE11_METRICS + TABLE12_METRICS:
        spec = METRICS[metric]
        cells = []
        for method in ["pepsage-40", "PepGLAD-40", "PepSAGE-40"]:
            cells.append(
                f"{method}={format_value(summaries.get(method, {}).get(metric), spec['digits'])}"
            )
        print(f"  {spec['label']}: " + "; ".join(cells))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {paired_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
