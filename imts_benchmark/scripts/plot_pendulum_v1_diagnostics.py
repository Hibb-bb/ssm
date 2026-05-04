"""Generate two diagnostic plots for v1 Pendulum:
  1. Training/validation loss curves across the HPO landscape (training_curves.png)
  2. Per-sample (sin, cos) prediction vs ground truth on test samples
     (predictions.png), with the noisy input images shown above.

Output: /u/seojininus/ssm/imts_benchmark/docs/figures/pendulum_v1_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/u/seojininus/ssm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb

from imts_benchmark.shared_data.pendulum_datamodule import PendulumDataModule
from imts_benchmark.mamba_pretrain.pendulum_forecaster import PendulumMambaForecaster

OUT_DIR = Path("/u/seojininus/ssm/imts_benchmark/docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. Training curves (from wandb)
# =====================================================================
def fetch_curve(api, name_pattern):
    runs = list(api.runs("magicslabnorthwestern/TSKing",
                         filters={"display_name": {"$regex": name_pattern}},
                         per_page=5))
    if not runs:
        return None
    r = runs[0]
    by_ep = {}
    for row in r.scan_history():
        e = row.get("epoch")
        if e is None: continue
        d = by_ep.setdefault(e, {})
        for k in ("train/nll", "val/nll", "val/mse", "train/mse"):
            if k in row and row[k] is not None: d[k] = row[k]
    return by_ep


def plot_training_curves():
    print("[curves] fetching from wandb…")
    api = wandb.Api()
    cells = [
        # (label, run name regex, color)
        ("BEST: replace lr=2e-3 B=16",  "p20_pendulum_pt_replace_lr-2e-3_b-16_seed1_a1",
         "tab:blue"),
        ("learned lr=2e-3 B=16 (neg control)",  "p20_pendulum_pt_learned_lr-2e-3_b-16_seed1_a1",
         "tab:orange"),
        ("low-LR: replace lr=5e-4 B=16",  "pendulum_pt_hpo_replace_lr-5e-4_b-16_seed1",
         "tab:green"),
        ("big-batch: replace lr=2e-3 B=64",  "pendulum_pt_hpo_replace_lr-2e-3_b-64_seed1",
         "tab:red"),
        ("COLLAPSE: replace lr=1e-2 B=32",  "pendulum_pt_hpo_replace_lr-1e-2_b-32_seed1",
         "tab:purple"),
        ("concat lr=1e-2 B=32",  "p20_pendulum_pt_concat_lr-1e-2_b-32_seed1_a1",
         "tab:brown"),
    ]
    cell_data = []
    for label, pat, color in cells:
        by_ep = fetch_curve(api, pat)
        if by_ep is None:
            print(f"  ({label}) no run found"); continue
        eps = sorted(by_ep.keys())
        cell_data.append((label, color, eps, by_ep))
        print(f"  ({label}) {len(eps)} epochs, last {eps[-1]}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Pendulum v1 training dynamics — `mamba_pretrain` SMALL across HPO landscape\n"
                 "All curves converge to val/MSE ~9-10 ×10⁻³ floor. Train-val NLL gap grows = overfitting.",
                 fontsize=12, y=0.99)

    for label, color, eps, by_ep in cell_data:
        train_nll = [by_ep[e].get("train/nll") for e in eps]
        val_nll = [by_ep[e].get("val/nll") for e in eps]
        train_mse = [by_ep[e].get("train/mse") for e in eps]
        val_mse = [by_ep[e].get("val/mse") for e in eps]
        # filter Nones
        def keep(xs, ys):
            f = [(e, y) for e, y in zip(xs, ys) if y is not None]
            return [e for e, y in f], [y for e, y in f]

        e_tn, v_tn = keep(eps, train_nll)
        e_vn, v_vn = keep(eps, val_nll)
        e_tm, v_tm = keep(eps, train_mse)
        e_vm, v_vm = keep(eps, val_mse)

        axes[0, 0].plot(e_tn, v_tn, color=color, alpha=0.85, lw=1.4, label=label)
        axes[0, 1].plot(e_vn, v_vn, color=color, alpha=0.85, lw=1.4, label=label)
        axes[1, 0].plot(e_tm, v_tm, color=color, alpha=0.85, lw=1.4, label=label)
        axes[1, 1].plot(e_vm, v_vm, color=color, alpha=0.85, lw=1.4, label=label)

    axes[0, 0].set_title("train / NLL  (lower is better)")
    axes[0, 1].set_title("val / NLL  (lower is better)")
    axes[1, 0].set_title("train / MSE  (sin/cos)")
    axes[1, 1].set_title("val / MSE  (sin/cos) — the metric we report on test")
    for ax in axes.flat:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.set_xlim(0, 100)
    axes[1, 0].set_ylim(0, 0.05)
    axes[1, 1].set_ylim(0, 0.05)
    # Reference lines for baselines on val/MSE panel
    axes[1, 1].axhline(0.00332, color="k", ls=":", lw=1.0, alpha=0.7)
    axes[1, 1].text(2, 0.00332+0.0005, "RoMAE 3.32×10⁻³", fontsize=8)
    axes[1, 1].axhline(0.00341, color="dimgray", ls=":", lw=1.0, alpha=0.7)
    axes[1, 1].text(2, 0.00341-0.0014, "S5 3.41×10⁻³", fontsize=8)
    axes[1, 1].axhline(0.00668, color="gray", ls=":", lw=1.0, alpha=0.7)
    axes[1, 1].text(2, 0.00668+0.0003, "S5-drop (no Δt) 6.68×10⁻³", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "pendulum_v1_training_curves.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[curves] wrote {out}")


# =====================================================================
# 2. Predictions vs ground truth (from best.ckpt + test data)
# =====================================================================
def load_model(ckpt_path):
    """Load checkpoint, handling new dropout hparam that v1 ckpts don't have."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp.setdefault("dropout", 0.0)  # new in v2; default for v1 ckpts
    # also clear hparams the new model doesn't accept (warmup_pct/etc may be old names)
    hp.pop("num_warmup_steps", None)
    hp.pop("num_training_steps", None)
    model = PendulumMambaForecaster(**hp)
    # load state dict into the (potentially-new-shape) model
    sd = ckpt["state_dict"]
    # missing keys should only be the new dropout buffers (Identity has no params)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  load_state_dict missing keys (probably new dropout submodules): {len(missing)}")
    if unexpected:
        print(f"  load_state_dict unexpected: {unexpected[:3]}…")
    model.eval()
    return model


def plot_predictions(ckpt_path, label, n_samples=4):
    print(f"[pred] {label} — loading {ckpt_path}")
    model = load_model(ckpt_path)
    dm = PendulumDataModule(
        data_root="/u/seojininus/ssm/pendulum_data",
        regime="pendulum",
        train_batch_size=4, val_batch_size=4, num_workers=0,
    )
    dm.setup()
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))

    with torch.no_grad():
        out = model.forward(batch)
    mu = out["mu"][:, 0].numpy()           # [B, T, 2]
    target = batch["target"][:, 0].numpy()  # [B, T, 2]
    timestamps = batch["timestamps"][:, 0].numpy()  # [B, T]
    images = batch["images"][:, 0].numpy()  # [B, T, 576]
    if "logvar" in out:
        logvar = out["logvar"][:, 0].numpy()
        var = np.where(logvar > 0, logvar, np.exp(logvar))  # rough approx for elu+1
        # better: use the same elu+1 the model uses
        l = logvar
        var = np.where(l > 0, l + 1.0, np.exp(l)) + 1e-6
        std = np.sqrt(var)
    else:
        std = None

    n = min(n_samples, mu.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(4*n, 6),
                             gridspec_kw={"height_ratios": [1, 2.2]})
    if n == 1: axes = axes.reshape(2, 1)
    fig.suptitle(f"{label} — predictions on test samples (mamba_pretrain SMALL v1, Gaussian head)",
                 fontsize=12)

    for i in range(n):
        T = timestamps.shape[1]
        ts = timestamps[i]
        sin_true, cos_true = target[i, :, 0], target[i, :, 1]
        sin_pred, cos_pred = mu[i, :, 0], mu[i, :, 1]

        # Top row: a few of the noisy input images at evenly spaced obs
        row1_ax = axes[0, i]
        n_show = 6
        idx_show = np.linspace(0, T-1, n_show, dtype=int)
        montage = np.hstack([images[i, k].reshape(24, 24) for k in idx_show])
        row1_ax.imshow(montage, cmap="gray", aspect="equal", vmin=0, vmax=1)
        row1_ax.set_xticks([12 + 24*k for k in range(n_show)])
        row1_ax.set_xticklabels([f"t={int(ts[k])}" for k in idx_show], fontsize=7)
        row1_ax.set_yticks([])
        row1_ax.set_title(f"sample {i}: noisy input images at 6 of 50 obs", fontsize=9)

        # Bottom row: sin/cos predicted vs true
        row2_ax = axes[1, i]
        row2_ax.plot(ts, sin_true, "b-", lw=1.2, alpha=0.9, label="sin(θ) true")
        row2_ax.plot(ts, cos_true, "orange", lw=1.2, alpha=0.9, label="cos(θ) true")
        row2_ax.plot(ts, sin_pred, "b--", lw=1.2, alpha=0.9, label="sin(θ) pred")
        row2_ax.plot(ts, cos_pred, color="orange", ls="--", lw=1.2, alpha=0.9, label="cos(θ) pred")
        if std is not None:
            row2_ax.fill_between(ts, sin_pred - std[i, :, 0], sin_pred + std[i, :, 0],
                                 color="b", alpha=0.15, lw=0)
            row2_ax.fill_between(ts, cos_pred - std[i, :, 1], cos_pred + std[i, :, 1],
                                 color="orange", alpha=0.15, lw=0)
        row2_ax.scatter(ts, sin_true, c="b", s=8, alpha=0.4)
        row2_ax.scatter(ts, cos_true, c="orange", s=8, alpha=0.4)
        # MSE for this sample
        sample_mse = float(((mu[i] - target[i])**2).mean())
        row2_ax.set_title(f"per-sample MSE = {sample_mse:.4f} (={sample_mse*1000:.2f}×10⁻³)", fontsize=9)
        row2_ax.set_xlabel("time")
        row2_ax.set_ylabel("sin / cos value")
        row2_ax.set_ylim(-1.4, 1.4)
        row2_ax.grid(alpha=0.3)
        if i == 0:
            row2_ax.legend(loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("/", "-")
    out = OUT_DIR / f"pendulum_v1_predictions_{safe_label}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[pred] wrote {out}")


if __name__ == "__main__":
    plot_training_curves()
    cells = [
        ("/projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20/"
         "replace_lr-2e-3_b-16/seed1/checkpoints/best.ckpt", "replace (true Δt)"),
        ("/projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20/"
         "learned_lr-2e-3_b-16/seed1/checkpoints/best.ckpt", "learned (no Δt, neg control)"),
    ]
    for ckpt, label in cells:
        if Path(ckpt).exists():
            plot_predictions(ckpt, label, n_samples=4)
        else:
            print(f"[pred] missing ckpt: {ckpt}")
    print("\n[done] figures in /u/seojininus/ssm/imts_benchmark/docs/figures/")
