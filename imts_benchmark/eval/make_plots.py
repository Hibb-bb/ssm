"""Generate headline plots for Mamba-MV vs RoMAE comparison.

Reads:  imts_benchmark_v1/summary.csv  + per-seed per_sample.jsonl files.
Writes: imts_benchmark_v1/plots/*.png
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("mse_mean","mse_std","mae_mean","mae_std","r2_mean","r2_std",
                      "pearson_mean","pearson_std","wall_fit_sec_mean"):
                r[k] = float(r[k]) if r[k] else float("nan")
            r["n_seeds"] = int(r["n_seeds"])
            rows.append(r)
    return rows


def gapped_mse_per_sample(jsonl_path: Path):
    vals = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            k = "n_obs_v" if "n_obs_v0" in r else "n_pred_v"
            ns = [r[f"{k}0"], r[f"{k}1"], r[f"{k}2"]]
            ms = [r["mse_v0"], r["mse_v1"], r["mse_v2"]]
            g = int(np.argmin(ns))
            vals.append(ms[g])
    return np.array(vals)


def gather_persample(root: Path):
    by_cell = defaultdict(list)  # (model, regime, variant) -> list of (seed, np.array)
    for p in sorted(root.glob("*/*/*/seed*/per_sample.jsonl")):
        parts = p.parts
        model, regime, variant = parts[-5], parts[-4], parts[-3]
        seed = int(parts[-2].removeprefix("seed"))
        by_cell[(model, regime, variant)].append((seed, gapped_mse_per_sample(p)))
    return by_cell


def plot_overall_bars(summary, outdir: Path):
    regimes = ["sparse_independent", "sparse_dependent"]
    mamba_variants = ["learned", "additive", "replace"]
    metrics = [("mse_mean","mse_std","MSE"), ("mae_mean","mae_std","MAE"),
               ("r2_mean","r2_std","R2"), ("pearson_mean","pearson_std","Pearson")]
    for mkey, skey, label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, regime in zip(axes, regimes):
            labels, means, stds, colors = [], [], [], []
            for v in mamba_variants:
                row = next((r for r in summary if r["model"]=="mamba_mv"
                            and r["regime"]==regime and r["variant"]==v), None)
                if row:
                    labels.append(f"Mamba\n{v}")
                    means.append(row[mkey]); stds.append(row[skey])
                    colors.append("#1f77b4")
            row = next((r for r in summary if r["model"]=="romae"
                        and r["regime"]==regime), None)
            if row:
                labels.append("RoMAE")
                means.append(row[mkey]); stds.append(row[skey])
                colors.append("#d62728")
            xs = np.arange(len(labels))
            ax.bar(xs, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
            ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(regime)
            ax.axhline(0, color="gray", lw=0.5)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(label)
        fig.suptitle(f"Overall test {label} (mean +/- std across 5 seeds, N=200 samples)")
        fig.tight_layout()
        out = outdir / f"bar_{label.lower()}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")


def plot_gapped_bars(per_cell, outdir: Path):
    regimes = ["sparse_independent", "sparse_dependent"]
    mamba_variants = ["learned", "additive", "replace"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        labels, means, stds, colors = [], [], [], []
        for v in mamba_variants:
            seeds = per_cell.get(("mamba_mv", regime, v), [])
            if seeds:
                per_seed_means = [x.mean() for _, x in seeds]
                labels.append(f"Mamba\n{v}")
                means.append(np.mean(per_seed_means))
                stds.append(np.std(per_seed_means))
                colors.append("#1f77b4")
        seeds = per_cell.get(("romae", regime, "default"), [])
        if seeds:
            per_seed_means = [x.mean() for _, x in seeds]
            labels.append("RoMAE")
            means.append(np.mean(per_seed_means))
            stds.append(np.std(per_seed_means))
            colors.append("#d62728")
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(regime)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Gapped-variate MSE")
    fig.suptitle("Gapped-variate MSE (variate with fewest obs per sample)")
    fig.tight_layout()
    out = outdir / "bar_gapped_mse.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def plot_paired_scatter(per_cell, outdir: Path, mamba_variant="learned"):
    for regime in ["sparse_independent", "sparse_dependent"]:
        m = per_cell.get(("mamba_mv", regime, mamba_variant), [])
        r = per_cell.get(("romae", regime, "default"), [])
        if not m or not r:
            continue
        # average per-sample MSE across seeds (items aligned by index -> same test split)
        m_avg = np.mean([x for _, x in m], axis=0)
        r_avg = np.mean([x for _, x in r], axis=0)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(m_avg, r_avg, s=8, alpha=0.5)
        lim = max(m_avg.max(), r_avg.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.7, label="y=x")
        ax.set_xlabel(f"Mamba-MV ({mamba_variant}) gapped-variate MSE")
        ax.set_ylabel("RoMAE gapped-variate MSE")
        ax.set_title(f"Paired per-sample MSE  —  {regime}\n(N={len(m_avg)}; points above y=x = Mamba wins)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        out = outdir / f"scatter_{regime}_{mamba_variant}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")


def plot_overall_paired_scatter(root: Path, outdir: Path, mamba_variant="learned"):
    for regime in ["sparse_independent", "sparse_dependent"]:
        m_paths = sorted((root/"mamba_mv"/regime/mamba_variant).glob("seed*/per_sample.jsonl"))
        r_paths = sorted((root/"romae"/regime/"default").glob("seed*/per_sample.jsonl"))
        if not m_paths or not r_paths:
            continue
        def load_mse(paths):
            arrs = []
            for p in paths:
                v = []
                with open(p) as f:
                    for line in f: v.append(json.loads(line)["mse"])
                arrs.append(np.array(v))
            return np.mean(arrs, axis=0)
        m_avg = load_mse(m_paths); r_avg = load_mse(r_paths)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(m_avg, r_avg, s=8, alpha=0.5)
        lim = max(m_avg.max(), r_avg.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.7, label="y=x")
        ax.set_xlabel(f"Mamba-MV ({mamba_variant}) overall MSE")
        ax.set_ylabel("RoMAE overall MSE")
        ax.set_title(f"Paired per-sample overall MSE  —  {regime}")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        out = outdir / f"scatter_overall_{regime}_{mamba_variant}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    root = Path(args.results_root)
    outdir = Path(args.outdir) if args.outdir else root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(root / "summary.csv")
    per_cell = gather_persample(root)

    plot_overall_bars(summary, outdir)
    plot_gapped_bars(per_cell, outdir)
    plot_paired_scatter(per_cell, outdir, "learned")
    plot_overall_paired_scatter(root, outdir, "learned")
    print(f"\nAll plots -> {outdir}")


if __name__ == "__main__":
    main()
