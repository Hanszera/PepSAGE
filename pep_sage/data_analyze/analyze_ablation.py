#!/usr/bin/env python3
"""
Ablation experiment analysis for StructToken (exp8).

Reads training logs, extracts sigma1_z sweep + discrete-vs-continuous comparison,
produces ablation table (CSV + LaTeX) and visualizations.

Usage:
    python analyze_ablation.py

Outputs (in ablation_outputs/):
    ablation_table.csv              Raw table
    ablation_table_latex.txt        LaTeX-formatted table
    ablation_bar_chart.pdf/.png     Bar chart comparison
    ablation_training_curves.pdf/.png  Training curve comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ─── Configuration ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "analysis_outputs"
OUTPUT_DIR = Path(__file__).parent / "ablation_outputs"

# How many steps ≈ 50 epochs (batch_size=16, dataset ~10360 samples)
STEPS_PER_EPOCH = 648
FAIR_COMPARE_EPOCHS = 50
FAIR_COMPARE_STEPS = STEPS_PER_EPOCH * FAIR_COMPARE_EPOCHS  # ~32400

# v4 fully-trained reference (sigma1_z=0.1, 180 epochs)
V4_VARIANT = "v4"

# Metrics for ablation table and bar chart
TABLE_METRICS = [
    ("val/trans_error",     "Trans. Err. ↓",   "lower"),
    ("val/struct_rmsd",     "Struct. RMSD ↓",   "lower"),
    ("val/token_acc_joint", "Token Acc. ↑",     "higher"),
    ("val/aars_error",      "AAR Err. ↓",       "lower"),
    ("val/recon_loss",      "Recon. Loss ↓",    "lower"),
]

# Metrics for training curve plots
CURVE_METRICS = [
    ("val/trans_error",     "Translation Error",      "lower"),
    ("val/struct_rmsd",     "Structure RMSD",         "lower"),
    ("val/token_acc_joint", "Token Accuracy (joint)",  "higher"),
    ("val/aars_error",      "AAR Error",              "lower"),
    ("val/recon_loss",      "Reconstruction Loss",    "lower"),
]

# Color palette (colorblind-friendly, from ColorBrewer)
COLORS = {
    "discrete":  "#7570b3",   # purple
    "sigma_0.05": "#1b9e77",  # teal
    "sigma_0.10": "#d95f02",  # orange
    "sigma_0.15": "#e7298a",  # pink
    "sigma_0.20": "#66a61e",  # green
    "v4_ref":    "#999999",   # gray
}


# ─── Variant Detection ────────────────────────────────────────────────────

def detect_variants(summary_df):
    """Auto-detect ablation variants from the summary data.

    Returns list of dicts: {key, col, display, color, marker, sigma_val}
    """
    all_variants = set(summary_df["variant"].unique())
    all_run_names = set(summary_df["run_name"].unique())
    variants = []

    # 1. Discrete Token baseline (always first row)
    if "Discrete token" in all_run_names:
        variants.append({
            "key": "Discrete token", "col": "run_name",
            "display": "Discrete Token",
            "color": COLORS["discrete"], "marker": "s",
            "sigma_val": None,
        })

    # 2. Collect sigma1_z sweep variants
    sigma_map = OrderedDict()

    for v in sorted(all_variants):
        if v.startswith("sigma1_z"):
            val_str = v.replace("sigma1_z", "").replace("_", ".")
            # sigma1_z005 → "005" → 0.05
            if "." not in val_str:
                sigma_val = int(val_str) / 100
            else:
                sigma_val = float(val_str)
            sigma_map[sigma_val] = ("variant", v)

    # v3 is sigma1_z=0.10 at ~50 epochs
    if "v3" in all_variants:
        sigma_map[0.10] = ("variant", "v3")

    # Sort by sigma and assign colors
    for sigma_val, (col, key) in sorted(sigma_map.items()):
        is_default = (sigma_val == 0.10)
        color_key = f"sigma_{sigma_val:.2f}"
        variants.append({
            "key": key, "col": col,
            "display": f"σ₁_z={sigma_val:.2f}" + (" (default)" if is_default else ""),
            "color": COLORS.get(color_key, "#333333"), "marker": "D" if is_default else "o",
            "sigma_val": sigma_val,
        })

    return variants


def get_v4_reference(summary_df):
    """Get v4 (180ep) values as reference lines."""
    ref = {}
    sub = summary_df[summary_df["variant"] == V4_VARIANT]
    for metric_name, _, _ in TABLE_METRICS:
        row = sub[sub["metric"] == metric_name]
        if len(row) > 0:
            ref[metric_name] = row.iloc[0]["mean_last_n"]
    return ref


# ─── Table Building ───────────────────────────────────────────────────────

def get_metric_value(summary_df, variant_info, metric_name):
    key, col = variant_info["key"], variant_info["col"]
    sub = summary_df[(summary_df[col] == key) & (summary_df["metric"] == metric_name)]
    if len(sub) == 0:
        return np.nan
    return sub.iloc[0]["mean_last_n"]


def build_ablation_table(summary_df, variants, v4_ref):
    """Build the ablation table DataFrame."""
    rows = []
    for vinfo in variants:
        row = {"Variant": vinfo["display"]}
        for metric_name, col_label, _ in TABLE_METRICS:
            row[col_label] = get_metric_value(summary_df, vinfo, metric_name)
        rows.append(row)

    # Append v4 (180ep) as reference row
    ref_row = {"Variant": "σ₁_z=0.10 (180ep, ref.)"}
    for metric_name, col_label, _ in TABLE_METRICS:
        ref_row[col_label] = v4_ref.get(metric_name, np.nan)
    rows.append(ref_row)

    return pd.DataFrame(rows)


def format_latex(table_df):
    """Format table as LaTeX with best values bolded."""
    cols = table_df.columns.tolist()
    metric_cols = cols[1:]
    n = len(metric_cols)

    lines = [
        f"\\begin{{tabular}}{{l{'c' * n}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]

    # Find best per column (excluding the v4 reference row)
    n_data = len(table_df) - 1  # last row is reference
    for _, row in table_df.iterrows():
        cells = [row["Variant"].replace("σ₁_z", "$\\sigma_1^z$")
                                .replace("(default)", "(\\textit{default})")]
        for col_label in metric_cols:
            val = row[col_label]
            if pd.isna(val):
                cells.append("--")
                continue

            direction = "higher" if "↑" in col_label else "lower"
            data_vals = table_df[col_label].iloc[:n_data].dropna()
            if direction == "lower":
                is_best = (val == data_vals.min())
            else:
                is_best = (val == data_vals.max())

            if abs(val) < 1:
                fmt = f"{val:.4f}"
            elif abs(val) < 100:
                fmt = f"{val:.2f}"
            else:
                fmt = f"{val:.1f}"

            if is_best and row.name < n_data:
                fmt = f"\\textbf{{{fmt}}}"

            cells.append(fmt)
        lines.append(" & ".join(cells) + " \\\\")

        # Add midrule before reference row
        if row.name == n_data - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ─── Bar Chart ────────────────────────────────────────────────────────────

def plot_bar_chart(table_df, variants, v4_ref, output_stem):
    """One subplot per metric, bars for each variant."""
    metric_cols = [c for c in table_df.columns if c != "Variant"]
    n_metrics = len(metric_cols)

    # Exclude the v4 reference row from bars
    n_data = len(table_df) - 1
    bar_df = table_df.iloc[:n_data]
    bar_colors = [v["color"] for v in variants]

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.0 * n_metrics, 4.0))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_data)

    for ax, (col_label, (metric_name, _, direction)) in zip(axes, zip(metric_cols, TABLE_METRICS)):
        vals = bar_df[col_label].values.astype(float)

        bars = ax.bar(x, vals, color=bar_colors, width=0.62,
                      edgecolor="white", linewidth=0.8)

        # Highlight best
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if valid:
            fn = min if direction == "lower" else max
            best_i = fn(valid, key=lambda t: t[1])[0]
            bars[best_i].set_edgecolor("black")
            bars[best_i].set_linewidth(2.0)

        # v4 reference line
        if metric_name in v4_ref:
            ax.axhline(v4_ref[metric_name], color=COLORS["v4_ref"],
                       linestyle="--", linewidth=1.2, alpha=0.8, label="180ep ref.")

        # Value labels
        for i, v in enumerate(vals):
            if not np.isnan(v):
                fmt = f"{v:.3f}" if v < 1 else f"{v:.1f}"
                ax.text(i, v, fmt, ha="center", va="bottom", fontsize=7.5)

        ax.set_title(col_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_df["Variant"], rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        # Adjust y range for readability
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)

    fig.suptitle("Ablation: Validation Metrics Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{output_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_stem}.pdf/png")


# ─── Training Curves ──────────────────────────────────────────────────────

def plot_training_curves(series_df, variants, v4_ref, output_stem,
                         max_step=None, title_suffix=""):
    """Val metric curves over training steps."""
    n_metrics = len(CURVE_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes_flat = axes.flatten()

    for ax_idx, (metric_name, title, direction) in enumerate(CURVE_METRICS):
        ax = axes_flat[ax_idx]

        for vinfo in variants:
            key, col = vinfo["key"], vinfo["col"]
            sub = series_df[
                (series_df[col] == key) & (series_df["metric"] == metric_name)
            ].sort_values("step")

            if len(sub) == 0:
                continue
            if max_step:
                sub = sub[sub["step"] <= max_step]
            if len(sub) == 0:
                continue

            ax.plot(sub["step"], sub["value"],
                    label=vinfo["display"],
                    color=vinfo["color"],
                    marker=vinfo["marker"],
                    markersize=5, linewidth=1.8, alpha=0.85)

        # v4 fully-trained reference
        if metric_name in v4_ref:
            ax.axhline(v4_ref[metric_name], color=COLORS["v4_ref"],
                       linestyle="--", linewidth=1.2, alpha=0.6)
            ax.text(ax.get_xlim()[1] * 0.98, v4_ref[metric_name],
                    " 180ep", va="bottom" if direction == "lower" else "top",
                    ha="right", fontsize=7, color=COLORS["v4_ref"])

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        arrow = "↓" if direction == "lower" else "↑"
        ax.set_ylabel(f"({arrow} better)")

    # Hide unused axes
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(variants), 5),
               bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=True)

    fig.suptitle(f"Ablation: Validation Curves{title_suffix}", fontsize=13)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{output_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_stem}.pdf/png")


# ─── Normalized Radar / Heatmap (bonus) ──────────────────────────────────

def plot_heatmap(table_df, variants, output_stem):
    """Heatmap normalized to v4 reference (last row). Green=better, red=worse."""
    metric_cols = [c for c in table_df.columns if c != "Variant"]
    n_data = len(table_df) - 1  # exclude reference row
    ref_row = table_df.iloc[-1]

    # Build normalized matrix: values relative to reference
    norm_data = np.zeros((n_data, len(metric_cols)))
    for j, (col_label, (_, _, direction)) in enumerate(zip(metric_cols, TABLE_METRICS)):
        ref_val = ref_row[col_label]
        if pd.isna(ref_val) or ref_val == 0:
            norm_data[:, j] = 0
            continue
        for i in range(n_data):
            val = table_df.iloc[i][col_label]
            if pd.isna(val):
                norm_data[i, j] = 0
            else:
                ratio = (val - ref_val) / abs(ref_val)
                if direction == "lower":
                    norm_data[i, j] = -ratio  # positive = better for lower
                else:
                    norm_data[i, j] = ratio    # positive = better for higher

    fig, ax = plt.subplots(figsize=(len(metric_cols) * 1.6 + 2, n_data * 0.7 + 1.5))
    vmax = max(abs(norm_data.min()), abs(norm_data.max()), 0.1)
    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_data))
    ax.set_yticklabels(table_df["Variant"].iloc[:n_data], fontsize=9)

    # Annotate cells with raw values
    for i in range(n_data):
        for j, col_label in enumerate(metric_cols):
            val = table_df.iloc[i][col_label]
            if pd.isna(val):
                txt = "--"
            elif abs(val) < 1:
                txt = f"{val:.4f}"
            else:
                txt = f"{val:.2f}"
            pct = norm_data[i, j] * 100
            sign = "+" if pct > 0 else ""
            ax.text(j, i, f"{txt}\n({sign}{pct:.1f}%)",
                    ha="center", va="center", fontsize=7.5,
                    color="black" if abs(norm_data[i, j]) < vmax * 0.6 else "white")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Relative to 180ep reference\n(green = better)", fontsize=8)

    ax.set_title("Ablation: Normalized to σ₁_z=0.10 (180ep)", fontsize=12)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{output_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_stem}.pdf/png")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ablation Analysis for StructToken (exp8)")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    summary_df = pd.read_csv(DATA_DIR / "summary_main_metrics.csv")
    series_df = pd.read_csv(DATA_DIR / "all_series.csv")
    print(f"  Summary: {len(summary_df)} rows")
    print(f"  Series:  {len(series_df)} rows")

    # Detect variants
    print("\n[2/5] Detecting ablation variants...")
    variants = detect_variants(summary_df)
    for v in variants:
        sigma = f"  (σ₁_z={v['sigma_val']:.2f})" if v["sigma_val"] else ""
        print(f"  • {v['display']:35s}{sigma}")

    v4_ref = get_v4_reference(summary_df)
    print(f"\n  v4 (180ep) reference:")
    for metric_name, col_label, _ in TABLE_METRICS:
        val = v4_ref.get(metric_name, float("nan"))
        print(f"    {col_label}: {val:.4f}")

    # Build table
    print("\n[3/5] Building ablation table...")
    table = build_ablation_table(summary_df, variants, v4_ref)

    csv_path = OUTPUT_DIR / "ablation_table.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    latex_path = OUTPUT_DIR / "ablation_table_latex.txt"
    latex_str = format_latex(table)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  Saved: {latex_path}")

    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Plots
    print("\n[4/5] Generating visualizations...")
    plot_bar_chart(table, variants, v4_ref,
                   str(OUTPUT_DIR / "ablation_bar_chart"))

    plot_heatmap(table, variants,
                 str(OUTPUT_DIR / "ablation_heatmap"))

    print("\n[5/5] Generating training curve plots...")
    plot_training_curves(series_df, variants, v4_ref,
                         str(OUTPUT_DIR / "ablation_curves_50ep"),
                         max_step=FAIR_COMPARE_STEPS,
                         title_suffix=f" (first {FAIR_COMPARE_EPOCHS} epochs)")

    plot_training_curves(series_df, variants, v4_ref,
                         str(OUTPUT_DIR / "ablation_curves_full"),
                         title_suffix=" (all available epochs)")

    print("\n" + "=" * 60)
    print("All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
