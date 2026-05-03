"""Plot per-cell training/validation loss + accuracy curves from CSVLogger output.

Walks `<root>/<dataset>/<dt_mode>/seed<S>/final/csv/version_*/metrics.csv` files
written by `mamba_pretrain.train_cls` (with CSVLogger). For each (dataset, dt_mode)
group, produces a 2x2 figure: [train/loss, val/loss, train/acc, val/acc] with
3 seeds overlaid.

Usage:
    python -m imts_benchmark.eval.plot_uea_cls_curves \
        --root output/log/uea_cls_pretrain_70d_repo/final \
        --out_dir output/log/uea_cls_pretrain_70d_repo/curves
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SEED_COLORS = {27: "tab:blue", 42: "tab:orange", 1024: "tab:green",
               123: "tab:red", 0: "tab:purple"}


def load_metrics_csv(cell_dir: Path) -> pd.DataFrame | None:
    """Find metrics.csv under <cell_dir>/csv/version_*/ and load it."""
    candidates = list(cell_dir.glob("csv/version_*/metrics.csv"))
    if not candidates:
        return None
    # Multiple version_* dirs can exist if a cell was rerun; take the latest.
    candidates.sort(key=lambda p: p.parent.name)
    df = pd.read_csv(candidates[-1])
    return df


def collapse_to_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """CSVLogger writes one row per logged step; collapse to per-epoch by
    grouping on `epoch` and taking the mean (training metrics) and last
    (validation metrics, which are logged once per epoch).
    """
    if "epoch" not in df.columns:
        return df
    # Separate train (logged per step) and val (logged at val epoch end) metrics.
    train_cols = [c for c in df.columns if c.startswith("train/")]
    val_cols = [c for c in df.columns if c.startswith("val/")]
    out = []
    for ep, g in df.groupby("epoch"):
        row = {"epoch": ep}
        for c in train_cols:
            v = g[c].dropna()
            if len(v):
                row[c] = float(v.mean())
        for c in val_cols:
            v = g[c].dropna()
            if len(v):
                row[c] = float(v.iloc[-1])
        out.append(row)
    return pd.DataFrame(out).sort_values("epoch").reset_index(drop=True)


def collect(root: Path) -> dict:
    """Return {(ds, dt): {seed: per_epoch_df}}."""
    rows = {}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ds = ds_dir.name
        for dt_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            dt = dt_dir.name
            for seed_dir in sorted(p for p in dt_dir.iterdir() if p.is_dir()):
                m = seed_dir.name  # "seedXX"
                if not m.startswith("seed"):
                    continue
                seed = int(m[4:])
                # cell layout: <seed_dir>/<run_name>/csv/version_*/metrics.csv
                # run_name is "final" by default
                for run_dir in seed_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    df = load_metrics_csv(run_dir)
                    if df is None or df.empty:
                        continue
                    pe = collapse_to_per_epoch(df)
                    rows.setdefault((ds, dt), {})[seed] = pe
    return rows


def plot_group(ds: str, dt: str, seed_to_df: dict, out_path: Path) -> None:
    """One 2x2 figure per (ds, dt): train/loss, val/loss, train/acc, val/acc."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("train/loss", axes[0, 0], "Train loss"),
        ("val/loss",   axes[0, 1], "Val loss"),
        ("train/acc",  axes[1, 0], "Train acc"),
        ("val/acc",    axes[1, 1], "Val acc"),
    ]
    for seed, df in sorted(seed_to_df.items()):
        c = SEED_COLORS.get(seed, "gray")
        for col, ax, title in panels:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], color=c, label=f"seed {seed}",
                        alpha=0.85, linewidth=1.5)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    for col, ax, title in panels:
        ax.set_xlabel("epoch")
        if "loss" in col:
            ax.set_yscale("log")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    n_seeds = len(seed_to_df)
    final_epochs = [int(df["epoch"].max()) for df in seed_to_df.values()]
    fig.suptitle(f"{ds}  /  dt_mode={dt}  ({n_seeds} seeds, "
                 f"final epoch {min(final_epochs)}–{max(final_epochs)})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, required=True,
                   help="Final-eval tree with per-(ds,dt,seed) cells.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to write per-(ds,dt) plots.")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(root)
    if not rows:
        print(f"No metrics.csv found under {root}")
        print(f"Expected layout: {root}/<ds>/<dt>/seed<S>/<run_name>/csv/version_0/metrics.csv")
        return

    print(f"Found {len(rows)} (ds, dt) groups with curves:")
    for (ds, dt), seeds in sorted(rows.items()):
        out_path = out_dir / f"{ds}__{dt}.png"
        plot_group(ds, dt, seeds, out_path)
        n = len(seeds)
        epochs_str = ", ".join(f"seed{s}={int(d['epoch'].max())}ep"
                               for s, d in sorted(seeds.items()))
        print(f"  {ds:24s} {dt:8s}  {n} seeds  ({epochs_str})  -> {out_path.name}")
    print(f"\nWrote {len(rows)} figures to {out_dir}")


if __name__ == "__main__":
    main()
