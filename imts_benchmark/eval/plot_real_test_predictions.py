"""Plot test-set predictions for real IMTS datasets (Activity / USHCN / PhysioNet).

For each chosen sample:
  - per-variate subplots
  - history observations: gray dots
  - ground-truth future:  black squares
  - model prediction:     colored x

Usage (single model, single dataset):
    python -m imts_benchmark.eval.plot_real_test_predictions \
        --model s5 \
        --regime activity \
        --ckpt /path/to/best.ckpt \
        --out_dir /path/to/output \
        --n_samples 4

Or pass --all to find checkpoints automatically across all available models
in LOG_ROOT and produce one PDF per (model, dataset) combination.
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

_THIS = Path(__file__).resolve().parent
_SSM_DK = _THIS.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_data.multivariate_datamodule import (
    MultivariateSinusoidalDataModule,
)


LOG_ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real")
DATA_ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/tpatchgnn_data")


def load_mamba_mv(ckpt: Path):
    from imts_benchmark.mamba_mv.multivariate_forecaster import MultivariateMambaForecaster
    return MultivariateMambaForecaster.load_from_checkpoint(str(ckpt), map_location="cpu")


def load_s5(ckpt: Path):
    from imts_benchmark.s5_forecaster.s5_forecaster import S5Forecaster
    return S5Forecaster.load_from_checkpoint(str(ckpt), map_location="cpu")


def load_romae(ckpt: Path):
    from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster
    return RoMAEForecaster.load_from_checkpoint(str(ckpt), map_location="cpu")


LOADERS = {"mamba_mv": load_mamba_mv, "s5": load_s5, "romae": load_romae}
FORMATS = {"mamba_mv": "per_variate", "s5": "per_variate", "romae": "flat_tokens"}
COLORS  = {"mamba_mv": "#c44e52", "s5": "#4c72b0", "romae": "#dd8452"}


def extract_per_variate_arrays(batch: dict) -> list[list[dict]]:
    """For per_variate batches, return [[{ts_ctx, val_ctx, ts_pred, val_true_pred}]]."""
    values = batch["values"].cpu().numpy()
    timestamps = batch["timestamps"].cpu().numpy()
    valid_mask = batch["valid_mask"].cpu().numpy()
    pred_mask = batch["pred_mask"].cpu().numpy()
    B, V, _ = values.shape
    out = []
    for i in range(B):
        per_var = []
        for d in range(V):
            valid = valid_mask[i, d]
            pmask = pred_mask[i, d]
            ctx = valid & ~pmask
            per_var.append({
                "ts_ctx":        timestamps[i, d, ctx].tolist(),
                "val_ctx":       values[i, d, ctx].tolist(),
                "ts_pred":       timestamps[i, d, pmask].tolist(),
                "val_true_pred": values[i, d, pmask].tolist(),
            })
        out.append(per_var)
    return out


@torch.no_grad()
def predict_per_variate_model(model, batch_pv, device: str):
    """preds [B, V, L]."""
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_pv.items()}
    return model(batch_dev).cpu().numpy()


@torch.no_grad()
def predict_romae(model, batch_ft, batch_pv, device: str, n_vars: int):
    """Returns: list[B] of list[V] of {ts_pred, val_pred} aligned to per-variate format."""
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_ft.items()}
    logits, _ = model.forward(batch_dev)
    logits = logits.cpu().numpy()  # [B, pred_max, 1]

    values_ft = batch_ft["values"].cpu().numpy()
    pred_mask = batch_ft["pred_mask"].cpu().numpy()
    pad_mask = batch_ft["pad_mask"].cpu().numpy()
    variate_id = batch_ft["variate_id"].cpu().numpy()
    timestamps = batch_ft["timestamps"].cpu().numpy()
    pred_real_count = batch_ft["pred_real_count"].cpu().numpy()

    B = values_ft.shape[0]
    out = []
    for i in range(B):
        n_true = int(pred_real_count[i])
        real_mask = pred_mask[i] & pad_mask[i]
        variates_flat = variate_id[i][real_mask]
        ts_flat = timestamps[i][real_mask]
        y_pred_flat = logits[i, :n_true, 0]
        per_var = []
        for d in range(n_vars):
            m_d = variates_flat == d
            per_var.append({
                "ts_pred":  ts_flat[m_d].tolist(),
                "val_pred": y_pred_flat[m_d].tolist(),
            })
        out.append(per_var)
    return out


def make_one_sample_figure(model_name: str, regime: str, sample_idx: int,
                           per_var_arrays: list[dict], pred_per_var: list[dict],
                           history: float, n_plot_vars: int, sample_meta: str = ""):
    """Make a figure with n_plot_vars subplots, showing history + true + pred per variate."""
    n_v_actual = min(len(per_var_arrays), n_plot_vars)
    fig, axes = plt.subplots(n_v_actual, 1, figsize=(8, 1.6 * n_v_actual + 0.4),
                             sharex=True)
    if n_v_actual == 1:
        axes = [axes]
    color = COLORS[model_name]
    label = {"mamba_mv": "Mamba-MV", "s5": "S5", "romae": "RoMAE"}[model_name]

    for d, ax in enumerate(axes):
        pv = per_var_arrays[d]
        # observed history
        if pv["ts_ctx"]:
            ax.scatter(pv["ts_ctx"], pv["val_ctx"], color="0.4", s=14, alpha=0.7,
                       label="history" if d == 0 else None, zorder=2)
        # true future
        if pv["ts_pred"]:
            ax.scatter(pv["ts_pred"], pv["val_true_pred"], color="black",
                       marker="s", s=22, alpha=0.85,
                       label="true" if d == 0 else None, zorder=3)
        # predicted future
        pp = pred_per_var[d]
        if pp.get("ts_pred") and pp.get("val_pred"):
            ax.scatter(pp["ts_pred"], pp["val_pred"], color=color,
                       marker="x", s=26, alpha=0.9,
                       label=label if d == 0 else None, zorder=4)
        # history boundary
        ax.axvline(history, color="orange", linestyle=":", alpha=0.6, linewidth=1)
        ax.set_ylabel(f"v{d}", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(f"timestamp", fontsize=9)
    axes[0].set_title(f"{label} on {regime} — sample {sample_idx} {sample_meta}",
                      fontsize=10)
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    plt.tight_layout()
    return fig


def run_one_model_one_dataset(model_name: str, regime: str, ckpt: Path,
                              out_dir: Path, n_samples: int = 4,
                              n_plot_vars: int = 6, device: str = "cpu"):
    """Load model, run on test, and save N plots into out_dir/{model}/{regime}/."""
    print(f"\n[{model_name}/{regime}] loading {ckpt.name}")
    model = LOADERS[model_name](ckpt).to(device).eval()
    n_vars = int(getattr(model, "n_vars", None) or model.hparams.n_vars)

    # Datamodule for context arrays (per_variate format always)
    dm_pv = MultivariateSinusoidalDataModule(
        data_root=str(DATA_ROOT), regime=regime, format="per_variate",
        train_batch_size=n_samples, val_batch_size=n_samples, num_workers=0,
    )
    dm_pv.setup("test")
    pv_loader = dm_pv.test_dataloader()
    pv_batch = next(iter(pv_loader))
    per_var_arrays = extract_per_variate_arrays(pv_batch)

    # Predictions
    if FORMATS[model_name] == "per_variate":
        preds = predict_per_variate_model(model, pv_batch, device)  # [B,V,L]
        pred_mask = pv_batch["pred_mask"].cpu().numpy()
        timestamps = pv_batch["timestamps"].cpu().numpy()
        B = preds.shape[0]
        pred_per_sample = []
        for i in range(B):
            per_var = []
            for d in range(n_vars):
                pm = pred_mask[i, d]
                per_var.append({
                    "ts_pred":  timestamps[i, d, pm].tolist(),
                    "val_pred": preds[i, d, pm].tolist(),
                })
            pred_per_sample.append(per_var)
    else:  # romae
        dm_ft = MultivariateSinusoidalDataModule(
            data_root=str(DATA_ROOT), regime=regime, format="flat_tokens",
            train_batch_size=n_samples, val_batch_size=n_samples, num_workers=0,
        )
        dm_ft.setup("test")
        ft_batch = next(iter(dm_ft.test_dataloader()))
        pred_per_sample = predict_romae(model, ft_batch, pv_batch, device, n_vars)

    history = float(pv_batch["history"][0].item())
    save_dir = out_dir / model_name / regime
    save_dir.mkdir(parents=True, exist_ok=True)

    n_actual = min(n_samples, len(per_var_arrays))
    for i in range(n_actual):
        # Compute per-sample MSE on the predicted positions for the title
        true_concat = np.concatenate([np.asarray(per_var_arrays[i][d]["val_true_pred"]) for d in range(n_vars)])
        pred_concat = np.concatenate([np.asarray(pred_per_sample[i][d]["val_pred"])     for d in range(n_vars)])
        mse_i = float(np.mean((true_concat - pred_concat) ** 2)) if true_concat.size else float("nan")

        fig = make_one_sample_figure(
            model_name, regime, i, per_var_arrays[i], pred_per_sample[i],
            history, n_plot_vars=n_plot_vars,
            sample_meta=f"  (n_obs={true_concat.size}, MSE={mse_i:.4f})"
        )
        fname = save_dir / f"sample_{i:02d}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fname}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def find_checkpoints_for_dataset(model_name: str, regime: str, seed: int = 1) -> list[Path]:
    """Return matching best.ckpt under LOG_ROOT for this (model, regime, seed)."""
    cands = []
    for sub in (LOG_ROOT / f"{model_name}_p10" / regime).glob("*"):
        ck = sub / f"seed{seed}" / "checkpoints" / "best.ckpt"
        if ck.exists():
            cands.append(ck)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   choices=("mamba_mv", "s5", "romae"), default=None)
    ap.add_argument("--regime",  choices=("activity", "ushcn", "physionet"), default=None)
    ap.add_argument("--ckpt",    type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed",    type=int, default=1)
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--n_plot_vars", type=int, default=6,
                    help="for activity (V=12) we only show first N variates per sample")
    ap.add_argument("--device",  type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--all",     action="store_true",
                    help="auto-find checkpoints for s5+romae+mamba_mv on activity+ushcn at --seed")
    args = ap.parse_args()

    if args.all:
        for model_name in ("s5", "romae", "mamba_mv"):
            for regime in ("activity", "ushcn"):
                cands = find_checkpoints_for_dataset(model_name, regime, args.seed)
                if not cands:
                    print(f"[skip] no checkpoint for {model_name}/{regime}/seed{args.seed}")
                    continue
                for ck in cands:
                    # Use sub-folder name (the variant) as a discriminator in out_dir
                    variant = ck.parent.parent.parent.name
                    sub_out = args.out_dir / variant
                    run_one_model_one_dataset(
                        model_name, regime, ck, sub_out,
                        n_samples=args.n_samples, n_plot_vars=args.n_plot_vars,
                        device=args.device,
                    )
    else:
        assert args.model and args.regime and args.ckpt, \
            "Without --all, must pass --model --regime --ckpt"
        run_one_model_one_dataset(
            args.model, args.regime, args.ckpt, args.out_dir,
            n_samples=args.n_samples, n_plot_vars=args.n_plot_vars,
            device=args.device,
        )


if __name__ == "__main__":
    main()
