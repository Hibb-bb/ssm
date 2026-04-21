"""Qualitative forecast plots: Mamba-MV vs RoMAE on a few test samples.

For each chosen test item, draws one figure with V=3 subplots (one per variate).
Each subplot shows:
  - observed history points                  (black circles)
  - ground-truth forecast points             (green squares)
  - Mamba-MV predictions                      (blue x)
  - RoMAE predictions                         (red +)
  - vertical line at history=7.0
  - shaded band over the long gap interval (if derivable from observation gaps)

Usage:
    python -m imts_benchmark.eval.plot_forecast_examples \
        --regime sparse_dependent --dt_mode learned --seed 1 \
        --n_samples 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_data.multivariate_datamodule import (
    MultivariateSinusoidalDataModule,
)
from imts_benchmark.mamba_mv.multivariate_forecaster import MultivariateMambaForecaster
from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster


LOG_ROOT = Path(
    "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1"
)


def _resolve_best_ckpt(ckpt_dir: Path) -> Path:
    """Return the newest checkpoint file matching best*.ckpt in ckpt_dir.
    Lightning appends -v1/-v2/... when a file already exists; we take the
    most-recent by mtime so we always use the latest training run."""
    candidates = sorted(ckpt_dir.glob("best*.ckpt"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no best*.ckpt found in {ckpt_dir}")
    return candidates[-1]


def load_mamba(regime: str, dt_mode: str, seed: int, device: str):
    ckpt_path = _resolve_best_ckpt(
        LOG_ROOT / "mamba_mv" / regime / dt_mode / f"seed{seed}" / "checkpoints"
    )
    print(f"  mamba ckpt: {ckpt_path.name}")
    model = MultivariateMambaForecaster.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model.eval().to(device)
    return model


def load_romae(regime: str, seed: int, device: str):
    ckpt_path = _resolve_best_ckpt(
        LOG_ROOT / "romae" / regime / "default" / f"seed{seed}" / "checkpoints"
    )
    print(f"  romae ckpt: {ckpt_path.name}")
    model = RoMAEForecaster.load_from_checkpoint(
        str(ckpt_path), map_location=device
    )
    model.eval().to(device)
    return model


@torch.no_grad()
def predict_mamba(model, batch_pv, device):
    """Return preds [B, V, L] aligned with batch_pv['values']."""
    batch_pv = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_pv.items()
    }
    preds = model(batch_pv)
    return preds.cpu().numpy()


@torch.no_grad()
def predict_romae_per_variate(model, batch_ft, device):
    """Returns a list of dicts (per sample) with per-variate (ts, pred) arrays
    aligned to the flat-token forecast positions."""
    batch_ft_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_ft.items()
    }
    logits, _ = model.forward(batch_ft_dev)  # [B, pred_max, 1]

    B = batch_ft["values"].shape[0]
    results = []
    for i in range(B):
        n_true = int(batch_ft["pred_real_count"][i].item())
        real_mask = (batch_ft["pred_mask"][i] & batch_ft["pad_mask"][i]).numpy()
        ts_i = batch_ft["timestamps"][i].numpy()
        var_i = batch_ft["variate_id"][i].numpy()
        ts_pred = ts_i[real_mask]
        var_pred = var_i[real_mask]
        y_pred = logits[i, :n_true, 0].cpu().float().numpy()
        assert ts_pred.shape[0] == n_true == y_pred.shape[0]
        per_var = {}
        for d in range(model.n_vars):
            sel = var_pred == d
            order = np.argsort(ts_pred[sel])
            per_var[d] = dict(ts=ts_pred[sel][order], y=y_pred[sel][order])
        results.append(per_var)
    return results


def pick_diverse_indices(dataset, n_samples: int, history: float = 7.0):
    """Pick test samples: one where variate 0 is gapped, one v1, one v2, plus one with
    smallest gap-variate n_obs. Falls back to evenly spaced indices."""
    V = len(dataset[0]["values_per_var"])
    indices_by_gapvar = {d: [] for d in range(V)}
    for i in range(len(dataset)):
        counts = [len(dataset[i]["values_per_var"][d]) for d in range(V)]
        g = int(np.argmin(counts))
        indices_by_gapvar[g].append(i)
    chosen = []
    for d in range(V):
        if indices_by_gapvar[d]:
            chosen.append(indices_by_gapvar[d][0])
    while len(chosen) < n_samples and len(chosen) < len(dataset):
        i = (len(chosen) * len(dataset)) // n_samples
        if i not in chosen:
            chosen.append(i)
    return chosen[:n_samples]


def gap_interval_for_variate(ts_d, history=7.0, t_max=10.0, min_gap=0.8):
    """Infer the forbidden interval for variate d from the observed ts_d.
    Returns (lo, hi) or None. Searches for the largest gap in [0, history).
    """
    if len(ts_d) < 2:
        return None
    ts_sorted = np.sort(ts_d[ts_d < history])
    if len(ts_sorted) < 2:
        return None
    diffs = np.diff(ts_sorted)
    k = int(np.argmax(diffs))
    if diffs[k] < min_gap:
        return None
    return (float(ts_sorted[k]), float(ts_sorted[k + 1]))


def plot_one_sample(item_id, sample_pv, mamba_pred_bvl, romae_pv, out_path,
                    history=7.0, t_max=10.0):
    V = len(sample_pv["values_per_var"])
    fig, axes = plt.subplots(V, 1, figsize=(10, 2.3 * V), sharex=True)
    if V == 1:
        axes = [axes]

    for d in range(V):
        ax = axes[d]
        vals = sample_pv["values_per_var"][d]
        ts = sample_pv["timestamps_per_var"][d]
        n = len(vals)

        # Split history / forecast
        is_hist = ts < history
        is_fcst = ts >= history
        hist_ts, hist_y = ts[is_hist], vals[is_hist]
        fcst_ts, fcst_y_true = ts[is_fcst], vals[is_fcst]

        ax.scatter(hist_ts, hist_y, s=28, facecolors="none", edgecolors="black",
                   label="observed history" if d == 0 else None, zorder=3)
        ax.scatter(fcst_ts, fcst_y_true, s=34, marker="s",
                   facecolors="none", edgecolors="tab:green",
                   label="true forecast" if d == 0 else None, zorder=3)

        # Mamba preds for this variate: mamba_pred_bvl[d, :n] aligned with ts
        # (same as dataset's per-variate ordering because collate preserves it)
        mamba_pred_full = mamba_pred_bvl[d, :n]
        mamba_pred_fcst = mamba_pred_full[is_fcst]
        order = np.argsort(fcst_ts)
        ax.plot(fcst_ts[order], mamba_pred_fcst[order],
                "x-", color="tab:blue", lw=1.2, ms=6,
                label="Mamba-MV pred" if d == 0 else None, zorder=2)

        # RoMAE preds
        romae_d = romae_pv[d]
        ax.plot(romae_d["ts"], romae_d["y"],
                "+-", color="tab:red", lw=1.2, ms=8,
                label="RoMAE pred" if d == 0 else None, zorder=2)

        ax.axvline(history, color="gray", lw=0.8, ls="--")

        gap = gap_interval_for_variate(ts, history=history, t_max=t_max)
        if gap is not None:
            ax.axvspan(gap[0], gap[1], alpha=0.15, color="orange",
                       label="inferred gap" if d == 0 else None)

        ax.set_ylabel(f"variate {d}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("t")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.35),
                   ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"{item_id}    (history up to t={history})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="sparse_dependent",
                    choices=["sparse_independent", "sparse_dependent"])
    ap.add_argument("--dt_mode", default="learned",
                    choices=["learned", "additive", "replace"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--data_root",
                    default="/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/"
                            "moirai/uni2ts_hongyu/ssm_dk/data")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    outdir = Path(args.outdir) if args.outdir else (LOG_ROOT / "plots" / "examples")
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-variate datamodule for Mamba and raw sample access
    dm_pv = MultivariateSinusoidalDataModule(
        data_root=args.data_root, regime=args.regime, format="per_variate",
        train_batch_size=1, val_batch_size=1, num_workers=0,
    )
    dm_pv.setup("test")
    test_ds_pv = dm_pv.test_ds

    dm_ft = MultivariateSinusoidalDataModule(
        data_root=args.data_root, regime=args.regime, format="flat_tokens",
        train_batch_size=1, val_batch_size=1, num_workers=0,
    )
    dm_ft.setup("test")
    test_ds_ft = dm_ft.test_ds

    indices = pick_diverse_indices(test_ds_pv, args.n_samples)
    print(f"chosen test indices: {indices}")

    # Load both models
    print("loading Mamba-MV...")
    mamba = load_mamba(args.regime, args.dt_mode, args.seed, device)
    print("loading RoMAE...")
    romae = load_romae(args.regime, args.seed, device)

    from imts_benchmark.shared_data.multivariate_datamodule import (
        collate_per_variate, collate_flat_tokens,
    )

    for idx in indices:
        item_pv = test_ds_pv[idx]
        item_ft = test_ds_ft[idx]
        item_id = item_pv["item_id"]

        batch_pv = collate_per_variate([item_pv])
        batch_ft = collate_flat_tokens([item_ft])

        mamba_preds_bvl = predict_mamba(mamba, batch_pv, device)[0]   # [V, L]
        romae_results = predict_romae_per_variate(romae, batch_ft, device)[0]

        out = outdir / f"forecast_{args.regime}_{item_id}.png"
        plot_one_sample(item_id, item_pv, mamba_preds_bvl, romae_results,
                        out, history=item_pv["history"])

    print(f"\n{len(indices)} example plot(s) -> {outdir}")


if __name__ == "__main__":
    main()
