"""Bar charts comparing models with BOTH metric formulas (target-weighted vs
T-PatchGNN's variable-averaged), so reviewers can see the gap directly.

Reads phase5_consolidated_v2.csv produced by aggregate_phase5_v2.py.

Outputs:
  aggregate/phase5_bars_v2_{activity,ushcn}.{pdf,png}
  aggregate/phase5_bars_v2_combined.{pdf,png}
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real")
CSV  = ROOT / "aggregate" / "phase5_consolidated_v2.csv"
OUT  = ROOT / "aggregate"

PAPER = {
    "activity": {"mse": 2.66, "mae": 3.15, "mse_unit": r"MSE $\times 10^{-3}$", "mae_unit": r"MAE $\times 10^{-2}$",
                 "ms": 1e-3, "as": 1e-2},
    "ushcn":    {"mse": 5.00, "mae": 3.08, "mse_unit": r"MSE $\times 10^{-1}$", "mae_unit": r"MAE $\times 10^{-1}$",
                 "ms": 1e-1, "as": 1e-1},
}

LABEL = {
    ("S5", "default"):       ("S5",                "#4c72b0"),
    ("RoMAE", "default"):    ("RoMAE",             "#dd8452"),
    ("Mamba-MV", "replace"): ("Mamba-MV (replace)","#55a868"),
    ("Mamba-MV", "learned"): ("Mamba-MV (learned)","#c44e52"),
    ("Mamba-MV", "concat"):  ("Mamba-MV (concat)", "#8172b3"),
}
ORDER = [("S5","default"), ("RoMAE","default"),
         ("Mamba-MV","replace"), ("Mamba-MV","learned"), ("Mamba-MV","concat")]


def load_rows():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "model":   r["model"],
                "variant": r["variant"],
                "dataset": r["dataset"],
                # ours (target-weighted)
                "mse":  float(r["mse_mean"]),  "mse_std":  float(r["mse_std"]),
                "mae":  float(r["mae_mean"]),  "mae_std":  float(r["mae_std"]),
                # T-PatchGNN (variable-averaged)
                "mse_tpg": float(r["mse_tpg_mean"]),  "mse_tpg_std": float(r["mse_tpg_std"]),
                "mae_tpg": float(r["mae_tpg_mean"]),  "mae_tpg_std": float(r["mae_tpg_std"]),
                "n":     int(r["n_seeds"]),
            })
    return rows


def make_dataset_panel(ax, ds, rows_ds, by_key, metric, std_metric, scale, ylabel, paper_value, title):
    """Plot bars for one (dataset, metric_kind) combo with grouped bars (ours vs TPG)."""
    labels, colors = [], []
    means_ours, stds_ours, means_tpg, stds_tpg = [], [], [], []
    for k in ORDER:
        if k not in by_key:
            continue
        name, color = LABEL[k]
        r = by_key[k]
        labels.append(name); colors.append(color)
        means_ours.append(r[metric] / scale);   stds_ours.append(r[std_metric] / scale)
        means_tpg.append(r[metric + "_tpg"] / scale);   stds_tpg.append(r[std_metric + "_tpg"] / scale)

    x = np.arange(len(labels))
    width = 0.38
    bars_ours = ax.bar(x - width/2, means_ours, width, yerr=stds_ours, capsize=3,
                        color=colors, edgecolor="black", linewidth=0.5,
                        alpha=0.55, label="ours (target-weighted)")
    bars_tpg = ax.bar(x + width/2, means_tpg,  width, yerr=stds_tpg,  capsize=3,
                        color=colors, edgecolor="black", linewidth=1.0, hatch="//",
                        label="TPG (variable-averaged)")
    # Reference line: T-PatchGNN paper
    ax.axhline(paper_value, color="black", linestyle="--", linewidth=1.0,
               label=f"T-PatchGNN paper ({paper_value:.2f})")
    # Numeric labels above bars
    for xi, m, s in zip(x - width/2, means_ours, stds_ours):
        ax.text(xi, m + s + 0.02 * max(means_ours+means_tpg), f"{m:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    for xi, m, s in zip(x + width/2, means_tpg, stds_tpg):
        ax.text(xi, m + s + 0.02 * max(means_ours+means_tpg), f"{m:.2f}",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def make_per_dataset_plot(ds, rows_ds, save_dir):
    cfg = PAPER[ds]
    by_key = {(r["model"], r["variant"]): r for r in rows_ds}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    make_dataset_panel(axes[0], ds, rows_ds, by_key,
                        metric="mse", std_metric="mse_std",
                        scale=cfg["ms"], ylabel=cfg["mse_unit"],
                        paper_value=cfg["mse"],
                        title=f"{ds.capitalize()} — MSE (both formulas)")
    make_dataset_panel(axes[1], ds, rows_ds, by_key,
                        metric="mae", std_metric="mae_std",
                        scale=cfg["as"], ylabel=cfg["mae_unit"],
                        paper_value=cfg["mae"],
                        title=f"{ds.capitalize()} — MAE (both formulas)")
    fig.suptitle(f"Phase-5 IMTS — {ds.capitalize()} — solid=ours (target-weighted), hatched=TPG (variable-avg)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / f"phase5_bars_v2_{ds}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}{{,.png}}")


def make_combined_plot(rows, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for j, ds in enumerate(("activity", "ushcn")):
        cfg = PAPER[ds]
        rows_ds = [r for r in rows if r["dataset"] == ds]
        by_key = {(r["model"], r["variant"]): r for r in rows_ds}
        make_dataset_panel(axes[j, 0], ds, rows_ds, by_key,
                            metric="mse", std_metric="mse_std",
                            scale=cfg["ms"], ylabel=cfg["mse_unit"],
                            paper_value=cfg["mse"],
                            title=f"{ds.capitalize()} — MSE")
        make_dataset_panel(axes[j, 1], ds, rows_ds, by_key,
                            metric="mae", std_metric="mae_std",
                            scale=cfg["as"], ylabel=cfg["mae_unit"],
                            paper_value=cfg["mae"],
                            title=f"{ds.capitalize()} — MAE")
    fig.suptitle("Phase-5 real IMTS — both metric formulas (solid bars = ours; hatched bars = T-PatchGNN's variable-averaged)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / "phase5_bars_v2_combined.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}{{,.png}}")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} aggregate rows.")
    save_dir = OUT
    for ds in ("activity", "ushcn"):
        rows_ds = [r for r in rows if r["dataset"] == ds]
        if rows_ds:
            make_per_dataset_plot(ds, rows_ds, save_dir)
    make_combined_plot(rows, save_dir)


if __name__ == "__main__":
    main()
