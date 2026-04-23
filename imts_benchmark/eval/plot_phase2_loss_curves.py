"""Plot train/val MSE curves per (model, regime).

Reads Lightning's CSVLogger output at
  <results_root>/<model>/<regime>/<variant>/seed<S>/lightning_logs/version_*/metrics.csv

Aggregation strategy:
  - For each run, bucket train/mse by epoch and take the mean (train/mse is
    logged every step, ~8 steps/epoch).
  - val/mse is already logged once per epoch (on_validation_epoch_end).
  - Across seeds, compute mean ± std per epoch, truncated to min epoch count.

Output: one PNG per regime, 4 subplots (one per model) showing train (solid)
and val (dashed) curves. For Mamba-MV we pick the dt_mode with lowest test
MSE per regime (matching metric-bar convention).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")
REGIME_LABELS = {
    "multisin_regular": "regular (100%)",
    "multisin_low_irreg": "low irreg (80%)",
    "multisin_med_irreg": "med irreg (30%)",
    "multisin_high_irreg": "high irreg (0%)",
}
MODELS = ("mamba_mv", "romae", "mtan", "s5")
MODEL_LABELS = {"mamba_mv": "Mamba-MV", "romae": "RoMAE", "mtan": "mTAN", "s5": "S5"}
COLORS = {"mamba_mv": "#d62728", "romae": "#1f77b4",
          "mtan": "#2ca02c", "s5": "#ff7f0e"}


def load_run_metrics(metrics_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_per_epoch, val_per_epoch) as 1D arrays aligned by epoch index."""
    df = pd.read_csv(metrics_csv)
    # train/mse: mean over steps within each epoch
    tr = df[df["train/mse"].notna()]
    tr_by_epoch = tr.groupby("epoch")["train/mse"].mean().sort_index()
    vl = df[df["val/mse"].notna()]
    vl_by_epoch = vl.groupby("epoch")["val/mse"].mean().sort_index()
    return tr_by_epoch.to_numpy(), vl_by_epoch.to_numpy()


def aggregate_model_regime(results_root: Path, model: str, regime: str, variant: str) -> dict:
    seed_dirs = sorted((results_root / model / regime / variant).glob("seed*"))
    tr_runs, vl_runs = [], []
    for sd in seed_dirs:
        csvs = sorted((sd / "lightning_logs").glob("version_*/metrics.csv"))
        if not csvs:
            continue
        try:
            tr, vl = load_run_metrics(csvs[-1])
            tr_runs.append(tr)
            vl_runs.append(vl)
        except Exception as e:
            print(f"  skip {sd}: {e}")
    if not tr_runs:
        return None
    # Truncate to min length across seeds.
    min_tr = min(len(a) for a in tr_runs)
    min_vl = min(len(a) for a in vl_runs)
    tr_mat = np.stack([a[:min_tr] for a in tr_runs], axis=0)
    vl_mat = np.stack([a[:min_vl] for a in vl_runs], axis=0)
    return dict(
        n_seeds=len(tr_runs),
        tr_mean=tr_mat.mean(axis=0), tr_std=tr_mat.std(axis=0),
        vl_mean=vl_mat.mean(axis=0), vl_std=vl_mat.std(axis=0),
    )


def pick_best_mamba_variant(summary_csv: Path, regime: str) -> str:
    df = pd.read_csv(summary_csv)
    sub = df[(df["model"] == "mamba_mv") & (df["regime"] == regime)]
    return sub.loc[sub["mse_mean"].idxmin(), "variant"]


def variant_for(model: str, regime: str, summary_csv: Path) -> str:
    if model == "mamba_mv":
        return pick_best_mamba_variant(summary_csv, regime)
    return "default"


def plot_regime_panel(results_root: Path, summary_csv: Path, regime: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)
    for ax, model in zip(axes, MODELS):
        variant = variant_for(model, regime, summary_csv)
        agg = aggregate_model_regime(results_root, model, regime, variant)
        if agg is None:
            ax.set_title(f"{MODEL_LABELS[model]} (no data)", fontsize=10)
            continue
        ep_tr = np.arange(len(agg["tr_mean"]))
        ep_vl = np.arange(len(agg["vl_mean"]))
        color = COLORS[model]
        ax.plot(ep_tr, agg["tr_mean"], color=color, label="train", linewidth=1.5)
        ax.fill_between(ep_tr, agg["tr_mean"] - agg["tr_std"], agg["tr_mean"] + agg["tr_std"],
                        alpha=0.25, color=color, linewidth=0)
        ax.plot(ep_vl, agg["vl_mean"], color=color, linestyle="--", label="val", linewidth=1.5)
        ax.fill_between(ep_vl, agg["vl_mean"] - agg["vl_std"], agg["vl_mean"] + agg["vl_std"],
                        alpha=0.15, color=color, linewidth=0)
        variant_tag = f" ({variant})" if model == "mamba_mv" else ""
        ax.set_title(f"{MODEL_LABELS[model]}{variant_tag}  n_seeds={agg['n_seeds']}", fontsize=10)
        ax.set_xlabel("epoch")
        if ax is axes[0]:
            ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Phase 2 — Train/Val MSE curves — {REGIME_LABELS[regime]}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--summary_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        out = out_dir / f"phase2_loss_curves_{regime}.png"
        plot_regime_panel(results_root, Path(args.summary_csv), regime, out)


if __name__ == "__main__":
    main()
