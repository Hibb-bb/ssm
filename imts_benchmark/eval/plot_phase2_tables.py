"""Render publication-quality table images (PNG) for slide decks.

Tables produced:
  - phase2_table_dt_mode_ablation.png  — Mamba-MV 3 dt_modes x 4 regimes (MSE, R2)
  - phase2_table_headline.png          — best-per-cell 4 models x 4 regimes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")
REGIME_LABELS = {
    "multisin_regular":    "Regular\n(frac=1.0)",
    "multisin_low_irreg":  "Low irreg\n(frac=0.8)",
    "multisin_med_irreg":  "Med irreg\n(frac=0.3)",
    "multisin_high_irreg": "High irreg\n(frac=0.0)",
}


def fmt(mean, std, digits=4):
    return f"{mean:.{digits}f}\n± {std:.{digits}f}"


def plot_dt_mode_ablation(df: pd.DataFrame, out: Path) -> None:
    """Mamba-MV only: 3 dt_mode variants x 4 regimes, MSE and R2 side by side."""
    sub = df[df["model"] == "mamba_mv"].copy()
    variants = [("learned", "Learned Δ\n(vanilla Mamba)"),
                ("replace", "Replace: Δ = Δtrue\n(our main)"),
                ("additive", "Additive: Δ = learned + Δtrue")]

    # Build cell data for a single combined table: rows = variants (3),
    # columns = 4 regimes * 2 metrics (MSE, R2)
    col_headers = []
    for rg in REGIMES:
        col_headers.append(REGIME_LABELS[rg] + "\nMSE ↓")
        col_headers.append("R² ↑")

    cell_text = []
    cell_colors = []
    best_mse_per_regime = {}
    best_r2_per_regime = {}
    for rg in REGIMES:
        rs = sub[sub["regime"] == rg]
        if not rs.empty:
            best_mse_per_regime[rg] = rs["mse_mean"].min()
            best_r2_per_regime[rg] = rs["r2_mean"].max()

    for v, _ in variants:
        row = []
        row_colors = []
        for rg in REGIMES:
            r = sub[(sub["regime"] == rg) & (sub["variant"] == v)]
            if r.empty:
                row.append("—"); row.append("—")
                row_colors.extend(["white", "white"])
                continue
            mse_m = float(r["mse_mean"].iloc[0]); mse_s = float(r["mse_std"].iloc[0])
            r2_m  = float(r["r2_mean"].iloc[0]);  r2_s  = float(r["r2_std"].iloc[0])
            row.append(fmt(mse_m, mse_s))
            row.append(fmt(r2_m,  r2_s,  digits=3))
            # Highlight best per regime in light green.
            is_best_mse = abs(mse_m - best_mse_per_regime[rg]) < 1e-9
            is_best_r2  = abs(r2_m  - best_r2_per_regime[rg])  < 1e-9
            row_colors.append("#c8e6c9" if is_best_mse else "white")
            row_colors.append("#c8e6c9" if is_best_r2  else "white")
        cell_text.append(row)
        cell_colors.append(row_colors)

    row_labels = [label for _, label in variants]

    # Figure sizing: 4 regimes × 2 cols = 8 data cols
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.axis("off")
    t = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=cell_colors,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1.0, 2.3)

    # Bold best-per-regime cells (both MSE + R2 columns).
    for ci, color in enumerate(cell_colors[0]):
        pass  # colors already applied

    # Bold the variant-row-labels
    for ri in range(len(row_labels)):
        cell = t[(ri + 1, -1)]
        cell.get_text().set_fontweight("bold")

    # Bold column headers
    for ci in range(len(col_headers)):
        cell = t[(0, ci)]
        cell.get_text().set_fontweight("bold")
        cell.set_facecolor("#e3f2fd")

    plt.suptitle("Mamba-MV: dt_mode ablation  (mean ± std over 5 seeds,  green = best per regime)",
                 y=0.98, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_headline_table(df: pd.DataFrame, out: Path, note: str = "") -> None:
    """4 models x 4 regimes. Mamba shows its best variant. Best per cell highlighted."""
    MODELS = ["mamba_mv", "romae", "mtan", "s5"]
    MODEL_LABELS = {
        "mamba_mv": "Mamba-MV\n(ours, best dt_mode)",
        "romae": "RoMAE",
        "mtan": "mTAN",
        "s5": "S5",
    }

    col_headers = [REGIME_LABELS[rg] + "\nMSE ↓" for rg in REGIMES] + \
                  [REGIME_LABELS[rg] + "\nR² ↑" for rg in REGIMES]

    best_per_regime_mse = {rg: df[df["regime"] == rg]["mse_mean"].min() for rg in REGIMES}
    best_per_regime_r2  = {rg: df[df["regime"] == rg]["r2_mean"].max() for rg in REGIMES}

    cell_text = []; cell_colors = []
    for model in MODELS:
        mse_row, r2_row = [], []
        mse_colors, r2_colors = [], []
        for rg in REGIMES:
            rs = df[(df["model"] == model) & (df["regime"] == rg)]
            if rs.empty:
                mse_row.append("—"); r2_row.append("—")
                mse_colors.append("white"); r2_colors.append("white"); continue
            if model == "mamba_mv":
                best_var = rs.loc[rs["mse_mean"].idxmin()]
            else:
                best_var = rs.iloc[0]
            mse_m = float(best_var["mse_mean"]); mse_s = float(best_var["mse_std"])
            r2_m  = float(best_var["r2_mean"]);  r2_s  = float(best_var["r2_std"])
            mse_row.append(fmt(mse_m, mse_s))
            r2_row.append(fmt(r2_m, r2_s, digits=3))
            mse_colors.append("#c8e6c9" if abs(mse_m - best_per_regime_mse[rg]) < 1e-9 else "white")
            r2_colors.append("#c8e6c9"  if abs(r2_m  - best_per_regime_r2[rg])  < 1e-9 else "white")
        cell_text.append(mse_row + r2_row)
        cell_colors.append(mse_colors + r2_colors)

    row_labels = [MODEL_LABELS[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(17, 4.0))
    ax.axis("off")
    t = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_headers,
                 cellColours=cell_colors, cellLoc="center", rowLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.3)

    for ri in range(len(row_labels)):
        c = t[(ri + 1, -1)]; c.get_text().set_fontweight("bold")
        if row_labels[ri].startswith("Mamba"):
            c.set_facecolor("#fff3e0")

    for ci in range(len(col_headers)):
        c = t[(0, ci)]; c.get_text().set_fontweight("bold"); c.set_facecolor("#e3f2fd")

    title = "Phase 2: 4 models × 4 irregularity regimes  (mean ± std over 5 seeds,  green = best per regime)"
    if note:
        title += f"\n{note}"
    plt.suptitle(title, y=0.99, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    plot_dt_mode_ablation(df, out_dir / "phase2_table_dt_mode_ablation.png")
    plot_headline_table(df,  out_dir / "phase2_table_headline.png", note=args.note)


if __name__ == "__main__":
    main()
