"""Cleaner v1 diagnostic plots — bigger, more readable, one figure per concept."""
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


def fetch_curve(api, name_pattern):
    runs = list(api.runs("magicslabnorthwestern/TSKing",
                         filters={"display_name": {"$regex": name_pattern}}, per_page=5))
    if not runs: return None
    by_ep = {}
    for row in runs[0].scan_history():
        e = row.get("epoch")
        if e is None: continue
        d = by_ep.setdefault(e, {})
        for k in ("train/nll", "val/nll", "val/mse", "train/mse"):
            if k in row and row[k] is not None: d[k] = row[k]
    return by_ep


def plot_one_curve_pair():
    """Single 2-panel figure: train vs val MSE for the BEST cell, big and readable."""
    api = wandb.Api()
    by_ep = fetch_curve(api, "p20_pendulum_pt_replace_lr-2e-3_b-16_seed1_a1")
    eps = sorted(by_ep.keys())
    train_mse = [(e, by_ep[e].get("train/mse")) for e in eps if by_ep[e].get("train/mse") is not None]
    val_mse = [(e, by_ep[e].get("val/mse")) for e in eps if by_ep[e].get("val/mse") is not None]
    train_nll = [(e, by_ep[e].get("train/nll")) for e in eps if by_ep[e].get("train/nll") is not None]
    val_nll = [(e, by_ep[e].get("val/nll")) for e in eps if by_ep[e].get("val/nll") is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # MSE panel
    ax1.plot(*zip(*train_mse), color="tab:blue", lw=2.0, label="train MSE")
    ax1.plot(*zip(*val_mse), color="tab:red", lw=2.0, label="validation MSE")
    ax1.axhline(0.00332, color="green", ls="--", lw=1.5, label="RoMAE 3.32×10⁻³ (best baseline)")
    ax1.axhline(0.00341, color="darkgreen", ls=":", lw=1.5, label="S5 3.41×10⁻³")
    ax1.axhline(0.00668, color="gray", ls=":", lw=1.5, label="S5-drop (no Δt) 6.68×10⁻³")
    best_val = min(y for _, y in val_mse)
    ax1.axhline(best_val, color="black", ls="-", lw=0.8, alpha=0.5)
    ax1.text(40, best_val + 0.001, f"our val/MSE floor ≈ {best_val*1000:.1f}×10⁻³", fontsize=11)
    ax1.set_xlabel("epoch", fontsize=12)
    ax1.set_ylabel("MSE on (sin θ, cos θ)", fontsize=12)
    ax1.set_title("MSE — train keeps dropping, val plateaus 3× above baselines",
                  fontsize=13, weight="bold")
    ax1.set_ylim(0, 0.06)
    ax1.set_xlim(0, 50)
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(alpha=0.3)

    # NLL panel — clearer overfit signal
    ax2.plot(*zip(*train_nll), color="tab:blue", lw=2.0, label="train NLL")
    ax2.plot(*zip(*val_nll), color="tab:red", lw=2.0, label="validation NLL")
    # Shade the gap
    e_t, v_t = zip(*train_nll); e_v, v_v = zip(*val_nll)
    common = sorted(set(e_t) & set(e_v))
    t_dict = dict(train_nll); v_dict = dict(val_nll)
    ax2.fill_between(common, [t_dict[e] for e in common], [v_dict[e] for e in common],
                     color="orange", alpha=0.25, label="train-val gap (overfit)")
    ax2.set_xlabel("epoch", fontsize=12)
    ax2.set_ylabel("Gaussian NLL (lower = better fit)", fontsize=12)
    ax2.set_title("NLL — train-val gap GROWS = clear overfitting", fontsize=13, weight="bold")
    ax2.set_xlim(0, 50)
    ax2.legend(loc="upper right", fontsize=11)
    ax2.grid(alpha=0.3)

    fig.suptitle("Training dynamics — BEST cell: replace, lr=2e-3, B=16  (mamba_pretrain SMALL v1)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "pendulum_v1_curves_best_cell.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def plot_curves_landscape():
    """The 'all cells hit the same floor' figure — clearer."""
    api = wandb.Api()
    cells = [
        ("BEST: replace lr=2e-3 B=16", "p20_pendulum_pt_replace_lr-2e-3_b-16_seed1_a1", "tab:blue"),
        ("learned lr=2e-3 B=16", "p20_pendulum_pt_learned_lr-2e-3_b-16_seed1_a1", "tab:orange"),
        ("low LR: replace lr=5e-4", "pendulum_pt_hpo_replace_lr-5e-4_b-16_seed1", "tab:green"),
        ("big batch: replace B=64", "pendulum_pt_hpo_replace_lr-2e-3_b-64_seed1", "tab:red"),
        ("HIGH LR (collapsed): replace lr=1e-2 B=32", "pendulum_pt_hpo_replace_lr-1e-2_b-32_seed1", "tab:purple"),
    ]
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, pat, color in cells:
        by_ep = fetch_curve(api, pat)
        if by_ep is None: continue
        eps = sorted(by_ep.keys())
        v_mse = [(e, by_ep[e].get("val/mse")) for e in eps if by_ep[e].get("val/mse") is not None]
        if not v_mse: continue
        ax.plot(*zip(*v_mse), color=color, lw=2.0, alpha=0.85, label=label)
    ax.axhline(0.00332, color="green", ls="--", lw=1.8, label="RoMAE-tiny 3.32×10⁻³")
    ax.axhline(0.00341, color="darkgreen", ls=":", lw=1.5, label="S5 3.41×10⁻³")
    ax.axhline(0.00668, color="gray", ls=":", lw=1.5, label="S5-drop (no Δt) 6.68×10⁻³")
    ax.axhline(0.00509, color="brown", ls=":", lw=1.5, label="RKN-Δt 5.09×10⁻³")
    ax.set_xlabel("epoch", fontsize=13)
    ax.set_ylabel("validation MSE on (sin, cos)", fontsize=13)
    ax.set_title("ALL HPO cells converge to ~10×10⁻³ floor regardless of LR or batch size\n"
                 "(LR sweep spans 5e-4 → 1e-2; batch spans 16 → 64; floor is unmoved)",
                 fontsize=13, weight="bold")
    ax.set_ylim(0, 0.05)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "pendulum_v1_curves_landscape.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp.setdefault("dropout", 0.0)
    hp.pop("num_warmup_steps", None); hp.pop("num_training_steps", None)
    model = PendulumMambaForecaster(**hp)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model


def plot_predictions_one_per_row(ckpt_path, label, n_samples=4):
    """One sample per ROW — much more readable."""
    model = load_model(ckpt_path)
    dm = PendulumDataModule(data_root="/u/seojininus/ssm/pendulum_data", regime="pendulum",
                            train_batch_size=4, val_batch_size=4, num_workers=0)
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    with torch.no_grad():
        out = model.forward(batch)
    mu = out["mu"][:, 0].numpy()           # [B, T, 2]
    target = batch["target"][:, 0].numpy()  # [B, T, 2]
    timestamps = batch["timestamps"][:, 0].numpy()  # [B, T]
    images = batch["images"][:, 0].numpy()  # [B, T, 576]

    n = min(n_samples, mu.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.5*n),
                             gridspec_kw={"width_ratios": [1.0, 2.5]})
    if n == 1: axes = axes.reshape(1, 2)
    fig.suptitle(f"{label} — predictions on 4 test samples", fontsize=15, weight="bold", y=1.005)

    for i in range(n):
        T = timestamps.shape[1]
        ts = timestamps[i]
        sin_true, cos_true = target[i, :, 0], target[i, :, 1]
        sin_pred, cos_pred = mu[i, :, 0], mu[i, :, 1]
        sample_mse = float(((mu[i] - target[i])**2).mean())
        sample_mse_x10_3 = sample_mse * 1000
        # noise estimate: std of pixels per image, averaged across the sample
        per_img_std = images[i].std(axis=1)  # [T]
        avg_pixel_std = per_img_std.mean()

        # Left: 8 evenly-spaced input images
        n_show = 8
        idx_show = np.linspace(0, T-1, n_show, dtype=int)
        montage = np.hstack([images[i, k].reshape(24, 24) for k in idx_show])
        axes[i, 0].imshow(montage, cmap="gray", aspect="equal", vmin=0, vmax=1)
        axes[i, 0].set_xticks([12 + 24*k for k in range(n_show)])
        axes[i, 0].set_xticklabels([f"t={int(ts[k])}" for k in idx_show], fontsize=8)
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title(f"sample {i}: 8 of 50 input images   (avg pixel std = {avg_pixel_std:.3f})",
                             fontsize=10)

        # Right: predicted vs true sin/cos
        ax = axes[i, 1]
        ax.plot(ts, sin_true, "b-", lw=2.0, label="sin(θ) TRUE")
        ax.plot(ts, sin_pred, "b--", lw=2.0, alpha=0.7, label="sin(θ) PRED")
        ax.plot(ts, cos_true, color="darkorange", lw=2.0, label="cos(θ) TRUE")
        ax.plot(ts, cos_pred, color="darkorange", ls="--", lw=2.0, alpha=0.7, label="cos(θ) PRED")
        ax.scatter(ts, sin_true, c="b", s=20, alpha=0.5, zorder=5)
        ax.scatter(ts, cos_true, c="darkorange", s=20, alpha=0.5, zorder=5)
        # Diagnostic colored band
        if sample_mse_x10_3 < 1.0:
            tag = " ✓ tracks well (clean signal)"
            tag_color = "green"
        elif sample_mse_x10_3 < 5.0:
            tag = " ~ partial track"
            tag_color = "darkorange"
        else:
            tag = " ✗ FLAT prediction (model gave up — noise dominates)"
            tag_color = "red"
        ax.set_title(f"per-sample MSE = {sample_mse_x10_3:.2f} ×10⁻³  →  {tag}",
                     fontsize=11, color=tag_color, weight="bold")
        ax.set_xlabel("time", fontsize=10)
        ax.set_ylabel("sin / cos value", fontsize=10)
        ax.set_ylim(-1.4, 1.4); ax.set_xlim(-1, 100)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9, ncol=2)

    plt.tight_layout()
    safe = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out = OUT_DIR / f"pendulum_v1_predictions_v2_{safe}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_one_curve_pair()
    plot_curves_landscape()
    for ckpt, label in [
        ("/projects/bfrf/seojininus/ssm/output/log/pendulum/mamba_pretrain_p20/"
         "replace_lr-2e-3_b-16/seed1/checkpoints/best.ckpt", "replace true Δt"),
    ]:
        plot_predictions_one_per_row(ckpt, label, n_samples=4)
    print("[done] new figures in /u/seojininus/ssm/imts_benchmark/docs/figures/")
