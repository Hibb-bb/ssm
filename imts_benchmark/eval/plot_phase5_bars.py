"""Bar charts comparing all models on Activity / USHCN, with T-PatchGNN reference line.

Reads phase5_consolidated.csv (produced by inline aggregator in conversation),
writes one PDF + PNG per dataset showing MSE & MAE bars side-by-side.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real")
CSV  = ROOT / "aggregate" / "phase5_consolidated.csv"
OUT  = ROOT / "aggregate"

PAPER = {
    "activity": {"mse": 2.66, "mae": 3.15, "mse_unit": r"MSE $\times 10^{-3}$", "mae_unit": r"MAE $\times 10^{-2}$"},
    "ushcn":    {"mse": 5.00, "mae": 3.08, "mse_unit": r"MSE $\times 10^{-1}$", "mae_unit": r"MAE $\times 10^{-1}$"},
}
WARPFORMER = {
    "activity": {"mse": 2.79, "mae": 3.39},
    "ushcn":    {"mse": 5.25, "mae": 3.23},
}

# Display name + color per model row
LABEL = {
    ("S5", "default"):       ("S5",                "#4c72b0"),
    ("RoMAE", "default"):    ("RoMAE",             "#dd8452"),
    ("Mamba-MV", "replace"): ("Mamba-MV (replace)","#55a868"),
    ("Mamba-MV", "learned"): ("Mamba-MV (learned)","#c44e52"),
    ("Mamba-MV", "concat"):  ("Mamba-MV (concat)", "#8172b3"),
}
ORDER_KEYS = [("S5","default"), ("RoMAE","default"),
              ("Mamba-MV","replace"), ("Mamba-MV","learned"), ("Mamba-MV","concat")]


def load_rows():
    out = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            out.append({
                "model":      r["model"],
                "variant":    r["variant"],
                "dataset":    r["dataset"],
                "mse":        float(r["mse_scaled"]),
                "mse_std":    float(r["mse_std_scaled"]),
                "mae":        float(r["mae_scaled"]),
                "mae_std":    float(r["mae_std_scaled"]),
                "n":          int(r["n"]),
            })
    return out


def make_bar_plot(ds: str, rows: list[dict], save_dir: Path):
    """One figure per dataset with two side-by-side bar groups (MSE | MAE)."""
    cfg = PAPER[ds]
    rows_ds = [r for r in rows if r["dataset"] == ds]
    by_key = {(r["model"], r["variant"]): r for r in rows_ds}

    labels, colors, mse_means, mse_stds, mae_means, mae_stds = [], [], [], [], [], []
    for k in ORDER_KEYS:
        if k not in by_key:
            continue
        name, color = LABEL[k]
        r = by_key[k]
        labels.append(name)
        colors.append(color)
        mse_means.append(r["mse"]);  mse_stds.append(r["mse_std"])
        mae_means.append(r["mae"]);  mae_stds.append(r["mae_std"])

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # -------- MSE panel --------
    ax = axes[0]
    bars = ax.bar(x, mse_means, yerr=mse_stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.6)
    # Reference lines: T-PatchGNN (paper) and Warpformer
    ax.axhline(cfg["mse"],            color="black", linestyle="--", linewidth=1.2,
               label=f"T-PatchGNN paper ({cfg['mse']:.2f})")
    ax.axhline(WARPFORMER[ds]["mse"], color="0.5",   linestyle=":",  linewidth=1.0,
               label=f"Warpformer ({WARPFORMER[ds]['mse']:.2f})")
    # numeric label above each bar
    for xi, m, s in zip(x, mse_means, mse_stds):
        ax.text(xi, m + s + (0.02 * max(mse_means)), f"{m:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(cfg["mse_unit"])
    ax.set_title(f"{ds.capitalize()} — Test MSE")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    # -------- MAE panel --------
    ax = axes[1]
    bars = ax.bar(x, mae_means, yerr=mae_stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(cfg["mae"],            color="black", linestyle="--", linewidth=1.2,
               label=f"T-PatchGNN paper ({cfg['mae']:.2f})")
    ax.axhline(WARPFORMER[ds]["mae"], color="0.5",   linestyle=":",  linewidth=1.0,
               label=f"Warpformer ({WARPFORMER[ds]['mae']:.2f})")
    for xi, m, s in zip(x, mae_means, mae_stds):
        ax.text(xi, m + s + (0.02 * max(mae_means)), f"{m:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(cfg["mae_unit"])
    ax.set_title(f"{ds.capitalize()} — Test MAE")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Phase-5 real IMTS benchmark — {ds.capitalize()} (5 seeds, patience=10)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / f"phase5_bars_{ds}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  .png")


def make_combined_plot(rows: list[dict], save_dir: Path):
    """Side-by-side: 2 datasets x 2 metrics = 4 panels in one figure."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, ds in enumerate(("activity", "ushcn")):
        cfg = PAPER[ds]
        rows_ds = [r for r in rows if r["dataset"] == ds]
        by_key = {(r["model"], r["variant"]): r for r in rows_ds}
        labels, colors, mse_means, mse_stds, mae_means, mae_stds = [], [], [], [], [], []
        for k in ORDER_KEYS:
            if k not in by_key: continue
            name, color = LABEL[k]
            r = by_key[k]
            labels.append(name); colors.append(color)
            mse_means.append(r["mse"]);  mse_stds.append(r["mse_std"])
            mae_means.append(r["mae"]);  mae_stds.append(r["mae_std"])
        x = np.arange(len(labels))
        # MSE
        ax = axes[j, 0]
        ax.bar(x, mse_means, yerr=mse_stds, color=colors, capsize=4,
               edgecolor="black", linewidth=0.6)
        ax.axhline(cfg["mse"], color="black", linestyle="--", linewidth=1.2,
                   label=f"T-PatchGNN ({cfg['mse']:.2f})")
        ax.axhline(WARPFORMER[ds]["mse"], color="0.5", linestyle=":", linewidth=1.0,
                   label=f"Warpformer ({WARPFORMER[ds]['mse']:.2f})")
        for xi, m, s in zip(x, mse_means, mse_stds):
            ax.text(xi, m + s + 0.02 * max(mse_means), f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(cfg["mse_unit"])
        ax.set_title(f"{ds.capitalize()} — MSE")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", alpha=0.3)
        # MAE
        ax = axes[j, 1]
        ax.bar(x, mae_means, yerr=mae_stds, color=colors, capsize=4,
               edgecolor="black", linewidth=0.6)
        ax.axhline(cfg["mae"], color="black", linestyle="--", linewidth=1.2,
                   label=f"T-PatchGNN ({cfg['mae']:.2f})")
        ax.axhline(WARPFORMER[ds]["mae"], color="0.5", linestyle=":", linewidth=1.0,
                   label=f"Warpformer ({WARPFORMER[ds]['mae']:.2f})")
        for xi, m, s in zip(x, mae_means, mae_stds):
            ax.text(xi, m + s + 0.02 * max(mae_means), f"{m:.2f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(cfg["mae_unit"])
        ax.set_title(f"{ds.capitalize()} — MAE")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Phase-5 real IMTS benchmark (5 seeds, patience=10)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf = save_dir / "phase5_bars_combined.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}  +  .png")


def main():
    rows = load_rows()
    save_dir = OUT
    for ds in ("activity", "ushcn"):
        make_bar_plot(ds, rows, save_dir)
    make_combined_plot(rows, save_dir)


if __name__ == "__main__":
    main()
