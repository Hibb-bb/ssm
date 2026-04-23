"""Plot Phase 2 headline metrics (MSE, MAE, R2, Pearson) per model per regime.

For Mamba-MV we pick the dt_mode with lowest test MSE on each regime (reflects
the 'best variant' reporting convention used in the paper tables). Bars are
mean over 5 seeds with std error bars.

Also produces:
  - per-variate grouped bars (mse_v0, v1, v2 target-weighted) showing v3
    advantage for Mamba-MV
  - training-time bar plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")
REGIME_LABELS = {
    "multisin_regular":    "regular (100%)",
    "multisin_low_irreg":  "low irreg (80%)",
    "multisin_med_irreg":  "med irreg (30%)",
    "multisin_high_irreg": "high irreg (0%)",
}
MODELS = ("mamba_mv", "romae", "mtan", "s5")
MODEL_LABELS = {"mamba_mv": "Mamba-MV (best dt_mode)", "romae": "RoMAE",
                "mtan": "mTAN", "s5": "S5"}
COLORS = {"mamba_mv": "#d62728", "romae": "#1f77b4",
          "mtan": "#2ca02c", "s5": "#ff7f0e"}


def pick_best_variant(df: pd.DataFrame, model: str, regime: str, key: str = "mse_mean") -> pd.Series:
    sub = df[(df["model"] == model) & (df["regime"] == regime)]
    if sub.empty:
        return None
    # For Mamba, pick variant with lowest mse. For others, default is the only variant.
    if model == "mamba_mv":
        return sub.loc[sub[key].idxmin()]
    return sub.iloc[0]


def plot_metric_bars(df: pd.DataFrame, out_dir: Path) -> None:
    metric_specs = [
        ("mse",     "mse_mean",     "mse_std",     "Test MSE (lower = better)",     False),
        ("mae",     "mae_mean",     "mae_std",     "Test MAE (lower = better)",     False),
        ("r2",      "r2_mean",      "r2_std",      "Test R² (higher = better)", True),
        ("pearson", "pearson_mean", "pearson_std", "Test Pearson (higher = better)", True),
    ]
    x = np.arange(len(REGIMES))
    width = 0.2
    for name, mean_col, std_col, ylabel, higher_better in metric_specs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, model in enumerate(MODELS):
            means, stds = [], []
            for regime in REGIMES:
                # For R^2 and Pearson, pick variant with best mean (highest).
                # For MSE/MAE, pick lowest.
                key = mean_col  # but pick_best_variant uses mse_mean default
                row = pick_best_variant(df, model, regime, key="mse_mean")
                if row is None:
                    means.append(np.nan); stds.append(np.nan); continue
                means.append(row[mean_col])
                stds.append(row[std_col])
            ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
                   label=MODEL_LABELS[model], color=COLORS[model], edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Phase 2 — {ylabel.split(' (')[0]} by regime (n=5 seeds, mean ± std)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"phase2_metric_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_per_variate(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: for each regime, show v0/v1/v2 target-weighted MSE for each model.
    Highlights Mamba-MV's v3 (=v2) advantage from cross-variate alignment.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)
    width = 0.18
    variate_labels = ("v1", "v2", "v3")
    for ax, regime in zip(axes, REGIMES):
        x = np.arange(3)
        for i, model in enumerate(MODELS):
            row = pick_best_variant(df, model, regime)
            if row is None:
                continue
            tw = [row.get(f"mse_v{d}_tw", np.nan) for d in range(3)]
            ax.bar(x + (i - 1.5) * width, tw, width,
                   label=MODEL_LABELS[model], color=COLORS[model], edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(variate_labels)
        ax.set_title(REGIME_LABELS[regime], fontsize=10)
        ax.set_ylabel("target-weighted MSE" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Phase 2 — Per-variate target-weighted MSE (v3 = convex mix of v1, v2)", y=1.02)
    fig.tight_layout()
    out = out_dir / "phase2_per_variate_mse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_training_time(df: pd.DataFrame, out_dir: Path) -> None:
    x = np.arange(len(REGIMES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(MODELS):
        times = []
        for regime in REGIMES:
            row = pick_best_variant(df, model, regime)
            times.append(row["wall_fit_sec_mean"] if row is not None else np.nan)
        ax.bar(x + (i - 1.5) * width, times, width,
               label=MODEL_LABELS[model], color=COLORS[model], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
    ax.set_ylabel("Training time (seconds, mean over 5 seeds)")
    ax.set_title("Phase 2 — Training wall-clock time (H100 80GB)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "phase2_training_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_dt_mode_ablation(df: pd.DataFrame, out_dir: Path) -> None:
    """Mamba-only: 3 dt_mode variants × 4 regimes on MSE."""
    sub = df[df["model"] == "mamba_mv"].copy()
    variants = ("learned", "replace", "additive")
    x = np.arange(len(REGIMES))
    width = 0.25
    colors_v = {"learned": "#7f7f7f", "replace": "#d62728", "additive": "#9467bd"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, v in enumerate(variants):
        means, stds = [], []
        for regime in REGIMES:
            r = sub[(sub["regime"] == regime) & (sub["variant"] == v)]
            if r.empty:
                means.append(np.nan); stds.append(np.nan); continue
            means.append(float(r["mse_mean"].iloc[0]))
            stds.append(float(r["mse_std"].iloc[0]))
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=f"dt_mode={v}", color=colors_v[v], edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
    ax.set_ylabel("Test MSE")
    ax.set_title("Phase 2 — Mamba-MV dt_mode ablation (n=5 seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "phase2_dt_mode_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting global metrics...")
    plot_metric_bars(df, out_dir)
    print("Plotting per-variate MSE...")
    plot_per_variate(df, out_dir)
    print("Plotting training time...")
    plot_training_time(df, out_dir)
    print("Plotting dt_mode ablation (Mamba only)...")
    plot_dt_mode_ablation(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
