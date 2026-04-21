"""Previous-experiment-style qualitative comparison plots for Mamba-MV vs RoMAE.

Mirrors the layout of
  ssm_model/ssm/mamba_experiments/forecaster/plot_mamba_comparison.py
but adapted to the multivariate (V=3) setting.

Produces:
  - imts_benchmark_by_frequency.png : rows = 4 test samples at 25/50/75/100 percentile
    of dominant frequency (averaged across variates), cols = (Mamba-MV, RoMAE).
    Each cell stacks the 3 variates vertically. Gray context, black truth,
    red prediction, cornflower |error| fill. Title: {model} | R², MSE.
  - imts_benchmark_distributions.png : 3 panels — R² scatter (Mamba vs RoMAE),
    R² histogram, MSE histogram — over all 200 test samples.

Usage:
  python -m imts_benchmark.eval.plot_comparison_v2 \
      --regime sparse_dependent --dt_mode learned --seed 1

Files are written to
  {LOG_ROOT}/plots/comparison_v2/<regime>/
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
    collate_per_variate, collate_flat_tokens,
)
from imts_benchmark.mamba_mv.multivariate_forecaster import MultivariateMambaForecaster
from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster
from imts_benchmark.eval.plot_forecast_examples import (
    _resolve_best_ckpt, LOG_ROOT,
)


def load_mamba(regime, dt_mode, seed, device):
    ckpt = _resolve_best_ckpt(
        LOG_ROOT / "mamba_mv" / regime / dt_mode / f"seed{seed}" / "checkpoints"
    )
    print(f"  mamba ckpt: {ckpt.name}")
    m = MultivariateMambaForecaster.load_from_checkpoint(str(ckpt), map_location=device)
    return m.eval().to(device)


def load_romae(regime, seed, device):
    ckpt = _resolve_best_ckpt(
        LOG_ROOT / "romae" / regime / "default" / f"seed{seed}" / "checkpoints"
    )
    print(f"  romae ckpt: {ckpt.name}")
    m = RoMAEForecaster.load_from_checkpoint(str(ckpt), map_location=device)
    return m.eval().to(device)


def estimate_frequency(times, values):
    if len(times) < 4:
        return 0.0
    order = np.argsort(times)
    t = times[order]; v = values[order]
    n_interp = 256
    t_uniform = np.linspace(t.min(), t.max(), n_interp)
    v_uniform = np.interp(t_uniform, t, v)
    v_uniform -= v_uniform.mean()
    dt = (t.max() - t.min()) / (n_interp - 1)
    fft_vals = np.abs(np.fft.rfft(v_uniform))
    freqs = np.fft.rfftfreq(n_interp, d=dt)
    fft_vals[0] = 0
    return float(freqs[np.argmax(fft_vals)])


@torch.no_grad()
def predict_mamba_per_sample(mamba, dm_pv, device):
    """Returns per-sample dict: {item_id, per_var: [(ts, y_true, y_pred)]}.
    Runs the whole test set in one pass."""
    mamba.eval()
    out_list = []
    for batch in dm_pv.test_dataloader():
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        preds = mamba(batch)        # [B, V, L]
        values = batch["values"]
        timestamps = batch["timestamps"]
        pred_mask = batch["pred_mask"]
        B, V, L = values.shape
        for i in range(B):
            per_var = []
            for d in range(V):
                m = pred_mask[i, d]
                ts_d = timestamps[i, d, m].detach().cpu().numpy()
                yt = values[i, d, m].detach().cpu().numpy()
                yp = preds[i, d, m].detach().cpu().numpy()
                # also grab context (all valid obs before history)
                valid = batch["valid_mask"][i, d]
                ts_all = timestamps[i, d, valid].detach().cpu().numpy()
                y_all = values[i, d, valid].detach().cpu().numpy()
                ctx_mask = ts_all < float(batch["history"][i].item())
                per_var.append(dict(
                    pred_times=ts_d, pred_true=yt, pred_hat=yp,
                    ctx_times=ts_all[ctx_mask], ctx_vals=y_all[ctx_mask],
                ))
            out_list.append(dict(item_id=batch["item_id"][i], per_var=per_var))
    return out_list


@torch.no_grad()
def predict_romae_per_sample(romae, dm_ft, device, V=3):
    romae.eval()
    out_list = []
    for batch in dm_ft.test_dataloader():
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits, _ = romae(batch_dev)   # [B, pred_max, 1]
        B = batch["values"].shape[0]
        for i in range(B):
            n_true = int(batch["pred_real_count"][i].item())
            real_mask = (batch["pred_mask"][i] & batch["pad_mask"][i]).numpy()
            ts_i = batch["timestamps"][i].numpy()
            var_i = batch["variate_id"][i].numpy()
            vals_i = batch["values"][i].numpy()

            ts_pred_all = ts_i[real_mask]
            var_pred_all = var_i[real_mask]
            y_pred_all = logits[i, :n_true, 0].cpu().float().numpy()
            assert ts_pred_all.shape[0] == n_true == y_pred_all.shape[0]

            # Context = real tokens with ts < history (historical observations)
            hist = float(batch["history"][i].item())
            real_tok = batch["pad_mask"][i].numpy()
            ctx_all = real_tok & (ts_i < hist)

            per_var = []
            for d in range(V):
                sel = var_pred_all == d
                order = np.argsort(ts_pred_all[sel])
                ts_d = ts_pred_all[sel][order]
                yp = y_pred_all[sel][order]

                real_d = real_tok & (var_i == d)
                ts_r = ts_i[real_d]; y_r = vals_i[real_d]
                pred_region = ts_r >= hist
                order_t = np.argsort(ts_r[pred_region])
                yt = y_r[pred_region][order_t]

                ctx_mask = ctx_all & (var_i == d)
                per_var.append(dict(
                    pred_times=ts_d, pred_true=yt, pred_hat=yp,
                    ctx_times=ts_i[ctx_mask], ctx_vals=vals_i[ctx_mask],
                ))
            out_list.append(dict(item_id=batch["item_id"][i], per_var=per_var))
    return out_list


def compute_metrics(per_var):
    y_t = np.concatenate([pv["pred_true"] for pv in per_var])
    y_p = np.concatenate([pv["pred_hat"]  for pv in per_var])
    if len(y_t) == 0:
        return dict(r2=0.0, mse=0.0)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    mse = float(np.mean((y_t - y_p) ** 2))
    return dict(r2=r2, mse=mse)


def plot_by_frequency(mamba_res, romae_res, outdir: Path, regime: str,
                      mamba_label="Mamba-MV (learned)", romae_label="RoMAE",
                      n_plots=4):
    outdir.mkdir(parents=True, exist_ok=True)

    # Frequency per sample = mean of per-variate estimated dominant freq (from full ts)
    sample_freqs = []
    for res in mamba_res:
        fs = []
        for pv in res["per_var"]:
            all_t = np.concatenate([pv["ctx_times"], pv["pred_times"]])
            all_v = np.concatenate([pv["ctx_vals"], pv["pred_true"]])
            fs.append(estimate_frequency(all_t, all_v))
        sample_freqs.append(float(np.mean(fs)))
    sample_freqs = np.array(sample_freqs)

    sort_idx = np.argsort(sample_freqs)
    n_total = len(sample_freqs)
    percentiles = (0.25, 0.5, 0.75, 1.0)[:n_plots]
    selected = [int(round(p * (n_total - 1))) for p in percentiles]

    V = 3
    fig, axes = plt.subplots(
        len(selected) * V, 2,
        figsize=(14, 2.6 * len(selected) * V),
        squeeze=False,
    )

    for row_i, sel_idx in enumerate(selected):
        sample_i = sort_idx[sel_idx]
        freq = sample_freqs[sample_i]
        metrics_m = compute_metrics(mamba_res[sample_i]["per_var"])
        metrics_r = compute_metrics(romae_res[sample_i]["per_var"])
        for d in range(V):
            mv = mamba_res[sample_i]["per_var"][d]
            rv = romae_res[sample_i]["per_var"][d]

            for col, (label, pv, met) in enumerate(
                [(mamba_label, mv, metrics_m), (romae_label, rv, metrics_r)]
            ):
                ax = axes[row_i * V + d, col]
                ax.plot(pv["ctx_times"], pv["ctx_vals"], "o-", color="gray",
                        markersize=3, alpha=0.5, label="Context", linewidth=0.8)
                ax.plot(pv["pred_times"], pv["pred_true"], "o", color="black",
                        markersize=3, label="True", zorder=5)
                ax.plot(pv["pred_times"], pv["pred_hat"], "s-", color="red",
                        markersize=3, label="Predicted", zorder=5,
                        linewidth=0.8, alpha=0.8)
                if pv["pred_times"].size > 0:
                    ax.fill_between(pv["pred_times"], pv["pred_true"],
                                    pv["pred_hat"], alpha=0.12,
                                    color="cornflowerblue", label="|Error|")
                title_prefix = f"{label}  variate {d}"
                if d == 0:
                    title_prefix += (
                        f"  |  R²={met['r2']:.3f}, MSE={met['mse']:.4f}  "
                        f"(f̄={freq:.2f})"
                    )
                ax.set_title(title_prefix, fontsize=9)
                ax.grid(alpha=0.3)
                if row_i * V + d == 0:
                    ax.legend(fontsize=7, loc="upper left")
                ax.set_ylabel(f"v{d}", fontsize=8)
            # xlabel on bottom-most row of the sample
        axes[row_i * V + V - 1, 0].set_xlabel("time")
        axes[row_i * V + V - 1, 1].set_xlabel("time")

    fig.suptitle(
        f"{regime} — rows: test samples sorted by dominant frequency "
        f"(25/50/75/100 percentile, low→high).  "
        f"Each sample: 3 variates stacked.",
        fontsize=12, y=1.002,
    )
    fig.tight_layout()
    path = outdir / "imts_benchmark_by_frequency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_distributions(mamba_res, romae_res, outdir: Path, regime: str,
                       mamba_label="Mamba-MV (learned)", romae_label="RoMAE"):
    outdir.mkdir(parents=True, exist_ok=True)
    r2_m = np.array([compute_metrics(r["per_var"])["r2"]  for r in mamba_res])
    r2_r = np.array([compute_metrics(r["per_var"])["r2"]  for r in romae_res])
    ms_m = np.array([compute_metrics(r["per_var"])["mse"] for r in mamba_res])
    ms_r = np.array([compute_metrics(r["per_var"])["mse"] for r in romae_res])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(r2_m, r2_r, s=12, alpha=0.4, color="tab:blue")
    lo = min(r2_m.min(), r2_r.min()) - 0.1
    hi = max(r2_m.max(), r2_r.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.set_xlabel(f"R² ({mamba_label})")
    ax.set_ylabel(f"R² ({romae_label})")
    ax.set_title("Per-sample R² scatter", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(r2_m, bins=40, alpha=0.55, color="tab:blue",
            label=f"{mamba_label} (mean={r2_m.mean():.3f})")
    ax.hist(r2_r, bins=40, alpha=0.55, color="tab:red",
            label=f"{romae_label} (mean={r2_r.mean():.3f})")
    ax.axvline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("R²"); ax.set_ylabel("Count")
    ax.set_title("R² distribution", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(ms_m, bins=40, alpha=0.55, color="tab:blue",
            label=f"{mamba_label} (mean={ms_m.mean():.4f})")
    ax.hist(ms_r, bins=40, alpha=0.55, color="tab:red",
            label=f"{romae_label} (mean={ms_r.mean():.4f})")
    ax.set_xlabel("MSE"); ax.set_ylabel("Count")
    ax.set_title("MSE distribution", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{regime}", fontsize=12)
    fig.tight_layout()
    path = outdir / "imts_benchmark_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="sparse_dependent",
                    choices=["sparse_independent", "sparse_dependent"])
    ap.add_argument("--dt_mode", default="learned",
                    choices=["learned", "additive", "replace"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_root",
                    default="/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/"
                            "moirai/uni2ts_hongyu/ssm_dk/data")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    dm_pv = MultivariateSinusoidalDataModule(
        data_root=args.data_root, regime=args.regime, format="per_variate",
        train_batch_size=32, val_batch_size=32, num_workers=0,
    )
    dm_pv.setup("test")
    dm_ft = MultivariateSinusoidalDataModule(
        data_root=args.data_root, regime=args.regime, format="flat_tokens",
        train_batch_size=32, val_batch_size=32, num_workers=0,
    )
    dm_ft.setup("test")

    print("loading Mamba-MV...")
    mamba = load_mamba(args.regime, args.dt_mode, args.seed, device)
    print("loading RoMAE...")
    romae = load_romae(args.regime, args.seed, device)

    print("running Mamba-MV inference on test set...")
    mamba_res = predict_mamba_per_sample(mamba, dm_pv, device)
    print(f"  {len(mamba_res)} samples")

    print("running RoMAE inference on test set...")
    romae_res = predict_romae_per_sample(romae, dm_ft, device)
    print(f"  {len(romae_res)} samples")

    # Align by item_id (both datamodules iterate in the same order already,
    # but assert for safety).
    m_ids = [r["item_id"] for r in mamba_res]
    r_ids = [r["item_id"] for r in romae_res]
    assert m_ids == r_ids, "item_id order mismatch between Mamba and RoMAE test sets"

    outdir = (
        Path(args.outdir) if args.outdir
        else LOG_ROOT / "plots" / "comparison_v2" / args.regime
    )
    plot_by_frequency(mamba_res, romae_res, outdir, args.regime,
                      mamba_label=f"Mamba-MV ({args.dt_mode})",
                      romae_label="RoMAE")
    plot_distributions(mamba_res, romae_res, outdir, args.regime,
                       mamba_label=f"Mamba-MV ({args.dt_mode})",
                       romae_label="RoMAE")
    print(f"\nAll plots -> {outdir}")


if __name__ == "__main__":
    main()
