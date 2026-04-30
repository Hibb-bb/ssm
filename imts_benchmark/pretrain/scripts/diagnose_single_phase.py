"""Diagnose single_phase Stage A: LOTSA stalled learning + synth in-dist plots.

Run::

    cd /home/ubuntu/hongyu/ssm
    /home/ubuntu/envs/mamba/bin/python -m imts_benchmark.pretrain.scripts.diagnose_single_phase

Loads the best step-26K checkpoint from the single_phase run, builds an
in-distribution snapshot dataloader (same config + a different seed) so
no batch overlaps training, then:

  (1) Per-source histogram of effective context length, n_obs/sample,
      and asinh-z target value range.  Checks whether LOTSA is being
      starved of usable windows (the leading hypothesis for why
      val/synth_lotsa_degraded_r2 plateaus near 0.20 from step 6K).

  (2) For each source (chronos2_synth, kernelsynth, lotsa_degraded),
      pick K well-formed windows and plot context (asinh-z) +
      ground-truth horizon + model prediction overlay.  Saved to
      runs/single_phase/<run>/diagnostics/.

  (3) Print per-source aggregate: overall MSE, MAE, R^2 in asinh-z space.

This does NOT touch wandb or the checkpoint; it is a read-only snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imts_benchmark.mamba_mv.multivariate_forecaster import (
    MultivariateMambaForecaster,
    MultivariateMambaSandwichForecaster,
)
from imts_benchmark.pretrain.datamodule import (
    PretrainDataModule, PretrainDataModuleArgs,
)


RUN_DIR = Path(
    "/home/ubuntu/hongyu/ssm/imts_benchmark/pretrain/runs/single_phase/"
    "single_synth30_lotsa70_regimeMix40_d384_warmup1000_multistep_80000_s42"
)
CKPT = RUN_DIR / "best-step00026000-mse0.9671.ckpt"
ABL = RUN_DIR / "ablation.json"
OUT = RUN_DIR / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

# Snapshot config: must match what training would see for in-distribution
# samples, just with a different seed so we don't replay training batches.
SNAPSHOT_BATCHES = 64           # ≈ 64*32 = 2048 windows total
SNAPSHOT_BATCH_SIZE = 32
PLOTS_PER_SOURCE = 6            # how many sample plots per source


def load_model_from_ckpt(ckpt_path: Path, abl_path: Path):
    """Reconstruct model + load state from .ckpt."""
    abl = json.loads(abl_path.read_text())

    # All builder args we need (matches train_pretrain._build_model).
    cls = (
        MultivariateMambaSandwichForecaster
        if abl.get("arch") == "sandwich"
        else MultivariateMambaForecaster
    )
    kwargs = dict(
        d_model=abl["d_model"],
        d_hidden=abl["d_hidden"],
        max_dim=abl["max_dim"],
        n_perv_layer=abl["n_perv_layer"],
        n_fusion_blocks=abl["n_fusion_blocks"],
        n_heads_varattn=abl.get("n_heads_varattn", 4),
        d_state=abl.get("d_state", 16),
        d_conv=abl.get("d_conv", 4),
        expand=abl.get("expand", 2),
        dt_mode=abl.get("dt_mode", "replace"),
        grid_K=abl["grid_K"],
        t_max=1.0,
        n_freq=abl.get("n_freq", 8),
        lr=abl["lr"],
        weight_decay=abl["weight_decay"],
        num_warmup_steps=abl["num_warmup_steps"],
        num_training_steps=abl["max_steps"],
        loss_type=abl["loss"],
        huber_delta=abl["huber_delta"],
        lr_schedule=abl.get("lr_schedule", "cosine"),
    )
    if abl["arch"] == "sandwich":
        kwargs["n_tail_grid_mamba"] = abl.get("n_tail_grid_mamba", 0)

    model = cls(**kwargs)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {ckpt_path.name}")
    print(f"       missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing[:3]:
        print(f"       sample missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"       sample unexpected: {unexpected[:3]}")
    model.eval()
    return model, abl


def build_snapshot(abl: dict):
    """Fresh in-distribution dataloader, seeded != training seed."""
    args = PretrainDataModuleArgs(
        stage_cfg_path=abl["stage_cfg"],
        sources_cfg_path=abl["sources_cfg"],
        batch_size=SNAPSHOT_BATCH_SIZE,
        num_workers=4,
        max_dim=abl["max_dim"],
        seed=abl["seed"] + 5000,    # well-separated from train (+0) and synth_val (+1000)
        persistent_workers=False,
    )
    dm = PretrainDataModule(args)
    dm.setup()
    dl = dm.train_dataloader()
    return dl


@torch.no_grad()
def collect_predictions(model, dl, n_batches: int):
    """Run model on ``n_batches`` and return list of dicts per sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    samples = []
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        batch_dev = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        preds = model.forward(batch_dev)        # [B, V, L]  asinh-z
        # Move back to cpu numpy
        preds_np = preds.detach().float().cpu().numpy()
        for k in ("values", "timestamps", "valid_mask", "pred_mask",
                  "history", "n_obs_per_var", "target_variate_mask",
                  "valid_variate_mask"):
            batch[k] = batch[k].cpu().numpy() if isinstance(batch[k], torch.Tensor) else batch[k]

        B = batch["values"].shape[0]
        for b in range(B):
            samples.append({
                "source":   batch["source_name"][b],
                "regime":   batch["regime"][b],
                "values":   batch["values"][b],          # [V, L]  asinh-z
                "timestamps": batch["timestamps"][b],    # [V, L]
                "valid":    batch["valid_mask"][b],
                "pred_mask": batch["pred_mask"][b],
                "preds":    preds_np[b],                  # [V, L]  asinh-z
                "history":  float(batch["history"][b]),
                "n_obs":    batch["n_obs_per_var"][b],
                "target":   batch["target_variate_mask"][b],
                "valid_var": batch["valid_variate_mask"][b],
            })
    print(f"[snapshot] collected {len(samples)} samples across {n_batches} batches")
    return samples


def per_source_stats(samples):
    """Aggregate per-source diagnostics."""
    by_src: dict[str, list] = {}
    for s in samples:
        by_src.setdefault(s["source"], []).append(s)

    print("\n=== Per-source aggregates ===")
    for src in sorted(by_src):
        ss = by_src[src]
        n_act_var = []
        ctx_obs = []
        pred_obs = []
        ctx_max_abs_z = []
        pred_max_abs_z = []
        regime_counts: dict[str, int] = {}
        # stream pred error
        sum_sq = sum_abs = n_pred = 0.0
        sum_y = sum_y2 = 0.0
        sum_yh = 0.0
        for s in ss:
            valid_var = s["valid_var"]
            n_act_var.append(int(valid_var.sum()))
            regime_counts[s["regime"]] = regime_counts.get(s["regime"], 0) + 1
            for v in range(s["values"].shape[0]):
                if not valid_var[v]:
                    continue
                ts = s["timestamps"][v]
                vl = s["valid"][v]
                vals = s["values"][v]
                ctx_mask = vl & (ts < s["history"])
                pm = s["pred_mask"][v]
                ctx_obs.append(int(ctx_mask.sum()))
                pred_obs.append(int(pm.sum()))
                if ctx_mask.any():
                    ctx_max_abs_z.append(float(np.abs(vals[ctx_mask]).max()))
                if pm.any():
                    pred_max_abs_z.append(float(np.abs(vals[pm]).max()))
                if pm.any():
                    yt = vals[pm]
                    yp = s["preds"][v][pm]
                    diff = yp - yt
                    sum_sq += float((diff * diff).sum())
                    sum_abs += float(np.abs(diff).sum())
                    n_pred += int(pm.sum())
                    sum_y  += float(yt.sum())
                    sum_y2 += float((yt * yt).sum())
                    sum_yh += float(yp.sum())
        if n_pred == 0:
            print(f"  {src}: no valid pred targets")
            continue
        mse = sum_sq / n_pred
        mae = sum_abs / n_pred
        var_y = (sum_y2 - sum_y * sum_y / n_pred) / max(1, n_pred - 1)
        ss_res = sum_sq
        ss_tot = (sum_y2 - sum_y * sum_y / n_pred)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        print(f"  {src}:")
        print(f"    samples={len(ss)}, regimes={regime_counts}")
        print(f"    active variates / sample: median={int(np.median(n_act_var))}, p10/p90 = {int(np.percentile(n_act_var,10))} / {int(np.percentile(n_act_var,90))}")
        print(f"    context obs / variate:   median={int(np.median(ctx_obs))}, p10/p90 = {int(np.percentile(ctx_obs,10))} / {int(np.percentile(ctx_obs,90))}")
        print(f"    horizon obs / variate:   median={int(np.median(pred_obs))}, p10/p90 = {int(np.percentile(pred_obs,10))} / {int(np.percentile(pred_obs,90))}")
        print(f"    max|asinh-z| in ctx:     median={np.median(ctx_max_abs_z):.3f}, p99={np.percentile(ctx_max_abs_z,99):.3f}")
        print(f"    max|asinh-z| in pred:    median={np.median(pred_max_abs_z):.3f}, p99={np.percentile(pred_max_abs_z,99):.3f}")
        print(f"    MSE={mse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}  Var(y)={var_y:.4f}  N_pred={n_pred}")


def plot_predictions_per_source(samples, k_per_source: int = PLOTS_PER_SOURCE):
    """For each source, plot k well-formed sample predictions."""
    by_src: dict[str, list] = {}
    for s in samples:
        by_src.setdefault(s["source"], []).append(s)

    for src in sorted(by_src):
        # filter to samples with at least one variate having both ctx and pred
        ok: list[dict] = []
        for s in by_src[src]:
            for v in range(s["values"].shape[0]):
                if not s["valid_var"][v]:
                    continue
                ctx = s["valid"][v] & (s["timestamps"][v] < s["history"])
                if int(ctx.sum()) < 5:
                    continue
                if int(s["pred_mask"][v].sum()) < 3:
                    continue
                ok.append({"sample": s, "v": v})
                break
            if len(ok) >= k_per_source:
                break
        if not ok:
            print(f"  {src}: no plottable samples")
            continue

        nrows = min(k_per_source, len(ok))
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.6 * nrows), squeeze=False)
        for i, item in enumerate(ok[:nrows]):
            s = item["sample"]
            v = item["v"]
            ts = s["timestamps"][v]
            vl = s["valid"][v]
            vals = s["values"][v]
            preds = s["preds"][v]
            ctx_mask = vl & (ts < s["history"])
            pm = s["pred_mask"][v]
            ax = axes[i, 0]
            # All valid points (ground truth in asinh-z space)
            ax.plot(ts[ctx_mask], vals[ctx_mask], "o-", color="tab:blue",
                    label="context (asinh-z)", markersize=4, linewidth=1)
            ax.plot(ts[pm], vals[pm], "o", color="black",
                    label="GT future", markersize=5)
            ax.plot(ts[pm], preds[pm], "x--", color="tab:red",
                    label="prediction", markersize=6, linewidth=1.2)
            ax.axvline(s["history"], color="gray", linestyle=":", linewidth=1)
            err = float(np.mean((preds[pm] - vals[pm]) ** 2))
            ax.set_title(f"{src} | regime={s['regime']} | v={v} | "
                         f"ctx={int(ctx_mask.sum())} pred={int(pm.sum())} "
                         f"| MSE={err:.3f} (asinh-z)", fontsize=9)
            ax.set_xlabel("t (normalized [0, 1])")
            ax.set_ylabel("value (asinh-z)")
            if i == 0:
                ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = OUT / f"single_phase_predictions_{src}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.relative_to(OUT.parent.parent)}")


def main() -> None:
    print("=" * 80)
    print("single_phase Stage A diagnosis  |  ckpt=best-step26000")
    print("=" * 80)

    model, abl = load_model_from_ckpt(CKPT, ABL)
    dl = build_snapshot(abl)
    samples = collect_predictions(model, dl, n_batches=SNAPSHOT_BATCHES)
    per_source_stats(samples)
    print("\n=== Per-source sample plots ===")
    plot_predictions_per_source(samples)
    print("\nAll outputs written under", OUT)


if __name__ == "__main__":
    main()
