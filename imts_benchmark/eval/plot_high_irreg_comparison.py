"""Focused model comparison on multisin_high_irreg only, for Phase 3 and
Phase 4-2 side-by-side. Keeps one regime to cut through visual noise.

Lineup (5 "models" per phase):
  - S5 (default)
  - RoMAE (default)
  - Mamba-MV  dt_mode=learned  (HPO-tuned)
  - Mamba-MV  dt_mode=replace  (HPO-tuned)
  - Mamba-MV  dt_mode=concat   (HPO-tuned)

Outputs (into --out_dir, default imts_benchmark/docs/):
  high_irreg_comparison_mse.png          side-by-side P3 / P4-2, MSE bars
  high_irreg_comparison_all_metrics.png  2x4 grid: MSE, MAE, R², Pearson × 2 phases
  high_irreg_comparison_table.png        publication table with best-cell highlight
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIME = "multisin_high_irreg"

# Row spec: (row_label, model, variant)
ROW_SPEC = [
    ("S5",                          "s5",       "default"),
    ("RoMAE",                       "romae",    "default"),
    ("Mamba-MV (learned, HPO)",     "mamba_mv", "hpo_tuned_learned"),
    ("Mamba-MV (replace, HPO)",     "mamba_mv", "hpo_tuned_replace"),
    ("Mamba-MV (concat, HPO)",      "mamba_mv", "hpo_tuned_concat"),
]
COLORS = {
    "S5":                          "#ff7f0e",
    "RoMAE":                       "#1f77b4",
    "Mamba-MV (learned, HPO)":     "#f1a7a7",
    "Mamba-MV (replace, HPO)":     "#d62728",
    "Mamba-MV (concat, HPO)":      "#8b0000",
}
PHASE_TITLES = {
    "phase3":   "Phase 3 — Async Dense",
    "phase4_2": "Phase 4-2 — Gap on random variate",
}
METRIC_SPECS = [
    ("mse",     "mse_mean",     "mse_std",     "Test MSE ↓",     False),
    ("mae",     "mae_mean",     "mae_std",     "Test MAE ↓",     False),
    ("r2",      "r2_mean",      "r2_std",      "Test R² ↑",      True),
    ("pearson", "pearson_mean", "pearson_std", "Test Pearson ↑", True),
]


def get_row(df: pd.DataFrame, model: str, variant: str) -> pd.Series | None:
    sub = df[(df["model"] == model) & (df["regime"] == REGIME) & (df["variant"] == variant)]
    return sub.iloc[0] if not sub.empty else None


def _collect(df: pd.DataFrame, mcol: str, scol: str) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for _, model, variant in ROW_SPEC:
        r = get_row(df, model, variant)
        if r is None:
            means.append(np.nan); stds.append(np.nan)
        else:
            means.append(float(r[mcol])); stds.append(float(r[scol]))
    return means, stds


def _bar_panel(ax, df: pd.DataFrame, phase: str, mcol: str, scol: str, ylabel: str,
               higher_is_better: bool) -> None:
    means, stds = _collect(df, mcol, scol)
    x = np.arange(len(ROW_SPEC))
    labels = [lbl for lbl, _, _ in ROW_SPEC]
    for i, lbl in enumerate(labels):
        ax.bar(x[i], means[i], yerr=stds[i], capsize=3,
               color=COLORS[lbl], edgecolor="black", linewidth=0.4, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace(" (", "\n(") for lbl in labels],
                       rotation=0, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    # Highlight winner with a star above its bar.
    arr = np.array(means)
    if np.isfinite(arr).any():
        best_idx = int(np.nanargmax(arr) if higher_is_better else np.nanargmin(arr))
        yerr_val = stds[best_idx] if np.isfinite(stds[best_idx]) else 0.0
        star_y = means[best_idx] + yerr_val
        pad = 0.03 * (np.nanmax(arr) - np.nanmin(arr) + 1e-9)
        ax.annotate("★", xy=(best_idx, star_y + pad),
                    ha="center", va="bottom", fontsize=14, color="#1b5e20")
    ax.set_title(PHASE_TITLES[phase], fontsize=11)


def plot_mse_side_by_side(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, phase in zip(axes, ("phase3", "phase4_2")):
        _bar_panel(ax, dfs[phase], phase, "mse_mean", "mse_std",
                   "Test MSE (lower = better)", higher_is_better=False)
    fig.suptitle(f"Model comparison on {REGIME} — Test MSE (5 seeds, mean ± std; "
                 f"★ = best)", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_all_metrics(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=False)
    for row_i, phase in enumerate(("phase3", "phase4_2")):
        for col_i, (_, mcol, scol, ylabel, higher) in enumerate(METRIC_SPECS):
            ax = axes[row_i, col_i]
            _bar_panel(ax, dfs[phase], phase, mcol, scol, ylabel, higher)
    fig.suptitle(f"Model comparison on {REGIME} — all metrics "
                 f"(5 seeds, mean ± std; ★ = best per panel)",
                 y=1.00, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _fmt(mean, std, digits=4):
    if not np.isfinite(mean):
        return "—"
    return f"{mean:.{digits}f}\n± {std:.{digits}f}"


def plot_table(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    phases = ("phase3", "phase4_2")
    col_headers = []
    for phase in phases:
        col_headers.extend([f"{PHASE_TITLES[phase].split(' — ')[0]}\nMSE ↓",
                            f"MAE ↓", f"R² ↑", f"Pearson ↑"])

    # Best per (phase, metric).
    best_by_col: list[float] = []
    for phase in phases:
        for _, mcol, _, _, higher in METRIC_SPECS:
            vals = []
            for _, model, variant in ROW_SPEC:
                r = get_row(dfs[phase], model, variant)
                if r is not None:
                    vals.append(float(r[mcol]))
            best_by_col.append((max(vals) if higher else min(vals)) if vals else np.nan)

    cell_text, cell_colors = [], []
    for label, model, variant in ROW_SPEC:
        row, cols = [], []
        col_idx = 0
        for phase in phases:
            r = get_row(dfs[phase], model, variant)
            for _, mcol, scol, _, _ in METRIC_SPECS:
                if r is None:
                    row.append("—"); cols.append("white"); col_idx += 1; continue
                m = float(r[mcol]); s = float(r[scol])
                digits = 5 if mcol == "mse_mean" else 3
                row.append(_fmt(m, s, digits=digits))
                is_best = np.isfinite(best_by_col[col_idx]) and abs(m - best_by_col[col_idx]) < 1e-9
                cols.append("#c8e6c9" if is_best else "white")
                col_idx += 1
        cell_text.append(row); cell_colors.append(cols)

    row_labels = [lbl for lbl, _, _ in ROW_SPEC]
    fig, ax = plt.subplots(figsize=(18, 4.6))
    ax.axis("off")
    t = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_headers,
                 cellColours=cell_colors, cellLoc="center", rowLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 2.5)
    for ri, lbl in enumerate(row_labels):
        c = t[(ri + 1, -1)]
        c.get_text().set_fontweight("bold")
        if lbl.startswith("Mamba"):
            c.set_facecolor("#fff3e0")
    for ci in range(len(col_headers)):
        c = t[(0, ci)]; c.get_text().set_fontweight("bold"); c.set_facecolor("#e3f2fd")
    # Visual separator between P3 and P4-2 groups (cols 0-3 vs 4-7).
    plt.suptitle(f"Model comparison on {REGIME}  (mean ± std over 5 seeds, green = best per column)",
                 y=1.01, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3_csv", default="imts_benchmark/docs/phase3_results/phase3_summary_wide.csv")
    ap.add_argument("--phase4_2_csv", default="imts_benchmark/docs/phase4_2_results/phase4_2_summary_wide.csv")
    ap.add_argument("--out_dir", default="imts_benchmark/docs")
    args = ap.parse_args()

    dfs = {
        "phase3":   pd.read_csv(args.phase3_csv),
        "phase4_2": pd.read_csv(args.phase4_2_csv),
    }
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    plot_mse_side_by_side(dfs, out_dir / "high_irreg_comparison_mse.png")
    plot_all_metrics(dfs,     out_dir / "high_irreg_comparison_all_metrics.png")
    plot_table(dfs,           out_dir / "high_irreg_comparison_table.png")
    print("Done.")


if __name__ == "__main__":
    main()
