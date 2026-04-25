"""Phase-general plotting for Phase 3 / 4-1 / 4-2 (current active lineup:
Mamba-MV {learned, replace, concat}, S5 paper-native, RoMAE paper-native).

Post-HPO (2026-04-24): the Mamba-MV dt_mode ablation prefers `hpo_tuned_{mode}`
rows when present (fair per-mode HPs), falling back to pre-HPO `{mode}` rows.
Headline "best dt_mode" plots also benefit — pick_best_variant picks min MSE
across all available variants including HPO-tuned.

Phase 2 had a different lineup (4 models incl. mTAN, 3 dt_modes incl. additive);
it is handled by the dedicated `plot_phase2_*.py` scripts. This script produces
the same-family outputs for the current 3-model × 3-dt-mode regime and, for
Phase 4, an extra bar on the gapped-variate's `out_gap` target-weighted MSE.

Outputs (into --out_dir):
  <phase>_metric_{mse,mae,r2,pearson}.png   headline bars (3 models × 4 regimes)
  <phase>_per_variate_mse.png               per-variate tw MSE (4-panel by regime)
  <phase>_dt_mode_ablation.png              Mamba learned vs replace
  <phase>_training_time.png                 mean wall-clock
  <phase>_table_headline.png                publication table
  <phase>_table_dt_mode_ablation.png        Mamba dt_mode table
  <phase>_loss_curves_<regime>.png          train/val MSE per regime
  (Phase 4 only)
  <phase>_gap_out_<regime>.png              out-gap tw MSE bars by variate
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
    "multisin_regular":    "regular (100%)",
    "multisin_low_irreg":  "low irreg (80%)",
    "multisin_med_irreg":  "med irreg (30%)",
    "multisin_high_irreg": "high irreg (0%)",
}
REGIME_TABLE_LABELS = {
    "multisin_regular":    "Regular\n(frac=1.0)",
    "multisin_low_irreg":  "Low irreg\n(frac=0.8)",
    "multisin_med_irreg":  "Med irreg\n(frac=0.3)",
    "multisin_high_irreg": "High irreg\n(frac=0.0)",
}
MODELS = ("mamba_mv", "s5", "romae")
MODEL_LABELS = {"mamba_mv": "Mamba-MV (best dt_mode)", "s5": "S5", "romae": "RoMAE"}
COLORS = {"mamba_mv": "#d62728", "s5": "#ff7f0e", "romae": "#1f77b4"}
DT_MODES = ("learned", "replace", "concat")
DT_COLORS = {"learned": "#7f7f7f", "replace": "#d62728", "concat": "#2ca02c"}


def _mamba_row(sub: pd.DataFrame, regime: str, mode: str) -> pd.Series | None:
    """Return the best-available Mamba row for (regime, dt_mode).
    Prefer `hpo_tuned_{mode}` if present (fair per-mode tuned HPs), else `{mode}`.
    Returns None if neither is in the frame (e.g. `concat` on pre-HPO regimes).
    """
    for vname in (f"hpo_tuned_{mode}", mode):
        r = sub[(sub["regime"] == regime) & (sub["variant"] == vname)]
        if not r.empty:
            return r.iloc[0]
    return None


def _phase_title(phase: str) -> str:
    return {
        "phase3":   "Phase 3 — Async Dense",
        "phase4_1": "Phase 4-1 — Gap on v3 (fixed)",
        "phase4_2": "Phase 4-2 — Gap on random variate",
    }.get(phase, phase)


def pick_best_variant(df: pd.DataFrame, model: str, regime: str) -> pd.Series | None:
    sub = df[(df["model"] == model) & (df["regime"] == regime)]
    if sub.empty:
        return None
    if model == "mamba_mv":
        return sub.loc[sub["mse_mean"].idxmin()]
    return sub.iloc[0]


def plot_metric_bars(df: pd.DataFrame, phase: str, out_dir: Path) -> None:
    specs = [
        ("mse",     "mse_mean",     "mse_std",     "Test MSE (lower = better)"),
        ("mae",     "mae_mean",     "mae_std",     "Test MAE (lower = better)"),
        ("r2",      "r2_mean",      "r2_std",      "Test R² (higher = better)"),
        ("pearson", "pearson_mean", "pearson_std", "Test Pearson (higher = better)"),
    ]
    x = np.arange(len(REGIMES))
    width = 0.25
    for name, mcol, scol, ylabel in specs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, model in enumerate(MODELS):
            means, stds = [], []
            for regime in REGIMES:
                row = pick_best_variant(df, model, regime)
                if row is None:
                    means.append(np.nan); stds.append(np.nan); continue
                means.append(row[mcol]); stds.append(row[scol])
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
                   label=MODEL_LABELS[model], color=COLORS[model],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{_phase_title(phase)} — {ylabel.split(' (')[0]} by regime (n=5 seeds, mean ± std)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"{phase}_metric_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_per_variate(df: pd.DataFrame, phase: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)
    width = 0.25
    variate_labels = ("v1", "v2", "v3")
    for ax, regime in zip(axes, REGIMES):
        x = np.arange(3)
        for i, model in enumerate(MODELS):
            row = pick_best_variant(df, model, regime)
            if row is None:
                continue
            tw = [row.get(f"mse_v{d}_tw", np.nan) for d in range(3)]
            ax.bar(x + (i - 1) * width, tw, width,
                   label=MODEL_LABELS[model], color=COLORS[model],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(variate_labels)
        ax.set_title(REGIME_LABELS[regime], fontsize=10)
        ax.set_ylabel("target-weighted MSE" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{_phase_title(phase)} — Per-variate target-weighted MSE (v3 = convex mix of v1, v2)", y=1.02)
    fig.tight_layout()
    out = out_dir / f"{phase}_per_variate_mse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_training_time(df: pd.DataFrame, phase: str, out_dir: Path) -> None:
    x = np.arange(len(REGIMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(MODELS):
        times = []
        for regime in REGIMES:
            row = pick_best_variant(df, model, regime)
            times.append(row.get("wall_fit_sec_mean", np.nan) if row is not None else np.nan)
        ax.bar(x + (i - 1) * width, times, width,
               label=MODEL_LABELS[model], color=COLORS[model],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
    ax.set_ylabel("Training time (seconds, mean over 5 seeds)")
    ax.set_title(f"{_phase_title(phase)} — Training wall-clock time")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"{phase}_training_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_dt_mode_ablation(df: pd.DataFrame, phase: str, out_dir: Path) -> None:
    sub = df[df["model"] == "mamba_mv"].copy()
    x = np.arange(len(REGIMES))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    n = len(DT_MODES)
    for i, v in enumerate(DT_MODES):
        means, stds = [], []
        for regime in REGIMES:
            row = _mamba_row(sub, regime, v)
            if row is None:
                means.append(np.nan); stds.append(np.nan); continue
            means.append(float(row["mse_mean"])); stds.append(float(row["mse_std"]))
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=f"dt_mode={v}", color=DT_COLORS[v],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES])
    ax.set_ylabel("Test MSE (lower = better)")
    ax.set_title(f"{_phase_title(phase)} — Mamba-MV dt_mode ablation "
                 f"(HPO-tuned where available, n=5 seeds)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"{phase}_dt_mode_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_gap_out(df: pd.DataFrame, phase: str, out_dir: Path) -> None:
    """Phase 4 only: target-weighted MSE on the gapped variate restricted to
    out-of-gap forecast positions, per regime. For phase4_1 gapped_variate=2
    always; for phase4_2 all three variates appear. Plot one bar per model
    × variate-where-data-exists per regime.
    """
    gap_variates = []
    for d in range(3):
        col = f"mse_v{d}_out_gap_tw"
        if col in df.columns and df[col].notna().any():
            gap_variates.append(d)
    if not gap_variates:
        print("  no out_gap fields present; skipping gap plot")
        return

    width = 0.25
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=False)
    variate_labels = {0: "v1", 1: "v2", 2: "v3"}
    for ax, regime in zip(axes, REGIMES):
        x = np.arange(len(gap_variates))
        for i, model in enumerate(MODELS):
            row = pick_best_variant(df, model, regime)
            if row is None:
                continue
            vals = [row.get(f"mse_v{d}_out_gap_tw", np.nan) for d in gap_variates]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=MODEL_LABELS[model], color=COLORS[model],
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([variate_labels[d] for d in gap_variates])
        ax.set_title(REGIME_LABELS[regime], fontsize=10)
        ax.set_ylabel("out-gap tw MSE" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    subtitle = ("gapped variate(s): " +
                ", ".join(variate_labels[d] for d in gap_variates))
    fig.suptitle(f"{_phase_title(phase)} — Target-weighted MSE on gapped variate, out-of-gap positions\n({subtitle})", y=1.04)
    fig.tight_layout()
    out = out_dir / f"{phase}_gap_out_mse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------- tables ---------------------------------------------------

def _fmt(mean, std, digits=4):
    return f"{mean:.{digits}f}\n± {std:.{digits}f}"


def plot_dt_mode_table(df: pd.DataFrame, phase: str, out: Path) -> None:
    sub = df[df["model"] == "mamba_mv"].copy()
    variants = [("learned", "Learned Δ\n(vanilla Mamba)"),
                ("replace", "Replace: Δ = Δtrue\n(ZOH at physical Δt)"),
                ("concat",  "Concat: Δtrue → dt_proj\n(ours, selection-preserving)")]
    col_headers = []
    for rg in REGIMES:
        col_headers.append(REGIME_TABLE_LABELS[rg] + "\nMSE ↓")
        col_headers.append("R² ↑")
    # Best across all three variants for highlighting.
    per_regime_mode_mse = {rg: [] for rg in REGIMES}
    per_regime_mode_r2  = {rg: [] for rg in REGIMES}
    for v, _ in variants:
        for rg in REGIMES:
            row = _mamba_row(sub, rg, v)
            if row is None:
                continue
            per_regime_mode_mse[rg].append(float(row["mse_mean"]))
            per_regime_mode_r2[rg].append(float(row["r2_mean"]))
    best_mse = {rg: (min(v) if v else np.nan) for rg, v in per_regime_mode_mse.items()}
    best_r2  = {rg: (max(v) if v else np.nan) for rg, v in per_regime_mode_r2.items()}
    cell_text, cell_colors = [], []
    for v, _ in variants:
        row, cols = [], []
        for rg in REGIMES:
            r = _mamba_row(sub, rg, v)
            if r is None:
                row.append("—"); row.append("—")
                cols.extend(["white", "white"]); continue
            mse_m = float(r["mse_mean"]); mse_s = float(r["mse_std"])
            r2_m  = float(r["r2_mean"]);  r2_s  = float(r["r2_std"])
            row.append(_fmt(mse_m, mse_s))
            row.append(_fmt(r2_m,  r2_s,  digits=3))
            cols.append("#c8e6c9" if abs(mse_m - best_mse[rg]) < 1e-9 else "white")
            cols.append("#c8e6c9" if abs(r2_m  - best_r2[rg])  < 1e-9 else "white")
        cell_text.append(row); cell_colors.append(cols)
    row_labels = [label for _, label in variants]
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.axis("off")
    t = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_headers,
                 cellColours=cell_colors, cellLoc="center", rowLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1.0, 2.3)
    for ri in range(len(row_labels)):
        t[(ri + 1, -1)].get_text().set_fontweight("bold")
    for ci in range(len(col_headers)):
        c = t[(0, ci)]; c.get_text().set_fontweight("bold"); c.set_facecolor("#e3f2fd")
    plt.suptitle(f"{_phase_title(phase)} — Mamba-MV dt_mode ablation "
                 f"(HPO-tuned where available; mean ± std, 5 seeds, green = best)",
                 y=0.98, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_headline_table(df: pd.DataFrame, phase: str, out: Path, note: str = "") -> None:
    MODEL_TABLE_LABELS = {
        "mamba_mv": "Mamba-MV\n(ours, best dt_mode)",
        "s5": "S5",
        "romae": "RoMAE",
    }
    col_headers = [REGIME_TABLE_LABELS[rg] + "\nMSE ↓" for rg in REGIMES] + \
                  [REGIME_TABLE_LABELS[rg] + "\nR² ↑" for rg in REGIMES]
    best_mse = {rg: df[df["regime"] == rg]["mse_mean"].min() for rg in REGIMES}
    best_r2  = {rg: df[df["regime"] == rg]["r2_mean"].max()  for rg in REGIMES}
    cell_text, cell_colors = [], []
    for model in MODELS:
        mse_row, r2_row, mc, rc = [], [], [], []
        for rg in REGIMES:
            rs = df[(df["model"] == model) & (df["regime"] == rg)]
            if rs.empty:
                mse_row.append("—"); r2_row.append("—"); mc.append("white"); rc.append("white"); continue
            best = rs.loc[rs["mse_mean"].idxmin()] if model == "mamba_mv" else rs.iloc[0]
            mse_m = float(best["mse_mean"]); mse_s = float(best["mse_std"])
            r2_m  = float(best["r2_mean"]);  r2_s  = float(best["r2_std"])
            mse_row.append(_fmt(mse_m, mse_s)); r2_row.append(_fmt(r2_m, r2_s, digits=3))
            mc.append("#c8e6c9" if abs(mse_m - best_mse[rg]) < 1e-9 else "white")
            rc.append("#c8e6c9" if abs(r2_m  - best_r2[rg])  < 1e-9 else "white")
        cell_text.append(mse_row + r2_row); cell_colors.append(mc + rc)
    row_labels = [MODEL_TABLE_LABELS[m] for m in MODELS]
    fig, ax = plt.subplots(figsize=(17, 3.4))
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
    title = f"{_phase_title(phase)}: 3 models × 4 irregularity regimes (mean ± std over 5 seeds, green = best)"
    if note:
        title += f"\n{note}"
    plt.suptitle(title, y=0.99, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------- loss curves ---------------------------------------------

def load_run_metrics(metrics_csv: Path):
    df = pd.read_csv(metrics_csv)
    cols = df.columns
    if "train/mse" not in cols or "val/mse" not in cols:
        return None, None
    tr = df[df["train/mse"].notna()].groupby("epoch")["train/mse"].mean().sort_index()
    vl = df[df["val/mse"].notna()].groupby("epoch")["val/mse"].mean().sort_index()
    return tr.to_numpy(), vl.to_numpy()


def aggregate_model_regime(results_root: Path, model: str, regime: str, variant: str) -> dict | None:
    seed_dirs = sorted((results_root / model / regime / variant).glob("seed*"))
    tr_runs, vl_runs = [], []
    for sd in seed_dirs:
        csvs = sorted((sd / "lightning_logs").glob("version_*/metrics.csv"))
        if not csvs:
            continue
        try:
            tr, vl = load_run_metrics(csvs[-1])
            if tr is None:
                continue
            tr_runs.append(tr); vl_runs.append(vl)
        except Exception as e:
            print(f"  skip {sd}: {e}")
    if not tr_runs:
        return None
    min_tr = min(len(a) for a in tr_runs)
    min_vl = min(len(a) for a in vl_runs)
    tr_mat = np.stack([a[:min_tr] for a in tr_runs], axis=0)
    vl_mat = np.stack([a[:min_vl] for a in vl_runs], axis=0)
    return dict(n_seeds=len(tr_runs),
                tr_mean=tr_mat.mean(axis=0), tr_std=tr_mat.std(axis=0),
                vl_mean=vl_mat.mean(axis=0), vl_std=vl_mat.std(axis=0))


def best_mamba_variant(summary_csv: Path, regime: str) -> str:
    df = pd.read_csv(summary_csv)
    sub = df[(df["model"] == "mamba_mv") & (df["regime"] == regime)]
    return sub.loc[sub["mse_mean"].idxmin(), "variant"]


def plot_loss_curves(results_root: Path, summary_csv: Path, phase: str, out_dir: Path) -> None:
    for regime in REGIMES:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 4.2), sharey=False)
        for ax, model in zip(axes, MODELS):
            variant = best_mamba_variant(summary_csv, regime) if model == "mamba_mv" else "default"
            agg = aggregate_model_regime(results_root, model, regime, variant)
            if agg is None:
                ax.set_title(f"{MODEL_LABELS[model]} (no data)", fontsize=10); continue
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
            ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"{_phase_title(phase)} — Train/Val MSE curves — {REGIME_LABELS[regime]}", y=1.02)
        fig.tight_layout()
        out = out_dir / f"{phase}_loss_curves_{regime}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}")


# ----------------- driver --------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("phase3", "phase4_1", "phase4_2"))
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    df = pd.read_csv(args.summary_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting metric bars...")
    plot_metric_bars(df, args.phase, out_dir)
    print("Plotting per-variate MSE...")
    plot_per_variate(df, args.phase, out_dir)
    print("Plotting training time...")
    plot_training_time(df, args.phase, out_dir)
    print("Plotting dt_mode ablation (Mamba)...")
    plot_dt_mode_ablation(df, args.phase, out_dir)
    if args.phase.startswith("phase4"):
        print("Plotting gap out-of-gap MSE (Phase 4)...")
        plot_gap_out(df, args.phase, out_dir)
    print("Rendering headline table...")
    plot_headline_table(df, args.phase, out_dir / f"{args.phase}_table_headline.png", note=args.note)
    print("Rendering dt_mode table...")
    plot_dt_mode_table(df, args.phase, out_dir / f"{args.phase}_table_dt_mode_ablation.png")
    print("Plotting loss curves...")
    plot_loss_curves(Path(args.results_root), Path(args.summary_csv), args.phase, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
