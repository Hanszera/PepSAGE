#!/usr/bin/env python3
"""Fig.2: quality-diversity frontier on the CLEAN 120-target subset.

X = diversity (higher = more diverse), Y = backbone RMSD (lower = better quality).
The ideal region is the lower-right (high diversity, low RMSD).

- PepSAGE temperature sweep (T=0.5..2.0) is drawn as a connected curve -> shows
  the operating-point frontier our model can traverse by changing sampling temp.
- Each baseline (PepFlow, PepGLAD, pepsage, Discrete) is a single point at its
  default setting.

Reads the recomputed clean-subset CSVs in this directory:
  table01_overall.csv     (baseline + PepSAGE T=1.0 points)
  table08_temperature.csv (PepSAGE temperature sweep)

Outputs: fig2_quality_diversity.png / .pdf
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read_csv(name):
    with open(HERE / name, newline="") as f:
        return list(csv.DictReader(f))


def main():
    overall = {r["metric"]: r for r in read_csv("table01_overall.csv")}
    temp = read_csv("table08_temperature.csv")

    # baseline points: (diversity, rmsd)
    div_row = overall["diversity"]
    rmsd_row = overall["rmsd"]
    baselines = {
        "PepFlow":  ("#d62728", "s"),
        "PepGLAD":  ("#9467bd", "^"),
        "pepsage":   ("#8c564b", "D"),
        "Discrete": ("#7f7f7f", "v"),
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # --- PepSAGE temperature frontier ---
    temps = sorted(temp, key=lambda r: float(r["temperature"]))
    xs = [float(r["diversity"]) for r in temps]
    ys = [float(r["rmsd"]) for r in temps]
    labels = [r["temperature"] for r in temps]
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=2.2, ms=8, zorder=5,
            label="PepSAGE (temperature 0.5–2.0)", markerfacecolor="white",
            markeredgecolor="#1f77b4", markeredgewidth=2)
    for x, y, t in zip(xs, ys, labels):
        ax.annotate(f"T={t}", (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color="#1f77b4")

    # --- baseline points ---
    for name, (color, marker) in baselines.items():
        x = float(div_row[name]); y = float(rmsd_row[name])
        ax.scatter([x], [y], s=130, c=color, marker=marker, zorder=6,
                   edgecolors="black", linewidths=0.6, label=name)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(9, -4),
                    fontsize=9, fontweight="bold")

    # ideal-direction arrow
    ax.annotate("", xy=(0.62, 2.1), xytext=(0.42, 2.1),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(0.52, 1.95, "better\n(diverse + accurate)", color="green",
            fontsize=8, ha="center", va="top")

    ax.set_xlabel("Diversity  (higher → more diverse)", fontsize=11)
    ax.set_ylabel("Backbone RMSD (Å)  (lower → more accurate)", fontsize=11)
    ax.set_title("Quality–Diversity Frontier (clean 120-target subset)", fontsize=12)
    ax.invert_yaxis()  # lower RMSD on top = better
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = HERE / f"fig2_quality_diversity.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out.name)


if __name__ == "__main__":
    main()
