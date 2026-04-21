"""Plot per-epoch train/val MSE curves across all 40 runs.

One figure per metric (train, val). Each figure has 2 subplots (regimes).
Each (model, variant) gets a line colored by model, with a shaded band
showing mean ± std across seeds at each epoch index.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_ROOT = Path(
    "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1"
)

COLORS = {
    ("mamba_mv", "learned"):  "#1f77b4",
    ("mamba_mv", "additive"): "#17becf",
    ("mamba_mv", "replace"):  "#2ca02c",
    ("romae",    "default"):  "#d62728",
}


def load_per_epoch_series(path: Path, metric: str):
    """Reduce each epoch to a single value (mean of the metric in that epoch).
    Lightning may log `train/mse` per step; we average per epoch."""
    by_epoch = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get(metric) and r.get("epoch") != "":
                try:
                    by_epoch[int(r["epoch"])].append(float(r[metric]))
                except ValueError:
                    pass
    if not by_epoch:
        return None
    epochs = sorted(by_epoch)
    vals = [float(np.mean(by_epoch[e])) for e in epochs]
    return np.array(epochs), np.array(vals)


def gather(metric: str):
    """Return nested dict: regime -> (model, variant) -> list of (epochs, vals) across seeds."""
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(LOG_ROOT.glob("*/*/*/seed*/lightning_logs/version_*/metrics.csv")):
        model, regime, variant = p.parts[-7], p.parts[-6], p.parts[-5]
        curve = load_per_epoch_series(p, metric)
        if curve is None:
            continue
        out[regime][(model, variant)].append(curve)
    return out


def pad_to_common_length(curves):
    """Align per-seed curves to a common length: up to the max epoch any seed
    reached. Pad shorter seeds with NaN (so nanmean/std ignore them)."""
    max_len = max(len(c[1]) for c in curves)
    stacked = np.full((len(curves), max_len), np.nan)
    for i, (_, vals) in enumerate(curves):
        stacked[i, :len(vals)] = vals
    return np.arange(max_len), stacked


def plot_metric(metric: str, out_path: Path, log_y: bool = True):
    data = gather(metric)
    regimes = ["sparse_independent", "sparse_dependent"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, regime in zip(axes, regimes):
        cells = data.get(regime, {})
        for (model, variant), curves in sorted(cells.items()):
            if not curves:
                continue
            epochs, stacked = pad_to_common_length(curves)
            mean = np.nanmean(stacked, axis=0)
            std  = np.nanstd(stacked, axis=0)
            color = COLORS.get((model, variant), "gray")
            label = f"{model} / {variant}"
            ax.plot(epochs, mean, color=color, lw=1.4, label=label)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=color, alpha=0.18, linewidth=0)
        ax.set_title(regime)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale("log")

    axes[0].set_ylabel(f"{metric}")
    axes[0].legend(fontsize=8, loc="upper right", frameon=True)

    fig.suptitle(
        f"{metric} across 5 seeds per (model, variant). "
        f"Line = seed mean, band = ±1 std.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    outdir = LOG_ROOT / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_metric("val/mse",   outdir / "curve_val_mse.png",   log_y=True)
    plot_metric("train/mse", outdir / "curve_train_mse.png", log_y=True)


if __name__ == "__main__":
    main()
