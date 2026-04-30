"""Diagnose moirai_v2 single-phase run: ILINet regression + per-V buckets.

Builds an in-distribution snapshot from the exact same training mix
(seed=train_seed+5000), runs the step-16K best ckpt, then breaks
predictions down by:

  - source   (chronos2_synth / kernelsynth / lotsa_degraded)
  - V bucket (1, 2..8, 9..16, 17..20)

All metrics in asinh-z space (no sinh blow-up).  Goal: figure out
whether the model is learning lotsa_degraded at all (the metric in
the W&B run was bugged, see (a)), and whether stacked-univariate-MV
samples are getting "easier" R² than natively-MV ones.

Then evaluates on the actual ILINet test split (V=12, natively-MV)
to verify whether the §7.M-motivated regression is real or just a
val-side artifact.

Run from /home/ubuntu/hongyu/ssm.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

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
    "single_moirai_v2_synth30_lotsa70_regimeMix40_d384_cosine30000_warmup500_s42"
)
CKPT = RUN_DIR / "best-step00016000-mse0.9649.ckpt"
ABL = RUN_DIR / "ablation.json"

SNAPSHOT_BATCHES = 64       # ~2048 windows
SNAPSHOT_BATCH_SIZE = 32


def load_model_from_ckpt(ckpt_path: Path, abl_path: Path):
    abl = json.loads(abl_path.read_text())
    cls = (
        MultivariateMambaSandwichForecaster
        if abl.get("arch") == "sandwich"
        else MultivariateMambaForecaster
    )
    kwargs = dict(
        d_model=abl["d_model"], d_hidden=abl["d_hidden"],
        max_dim=abl["max_dim"], n_perv_layer=abl["n_perv_layer"],
        n_fusion_blocks=abl["n_fusion_blocks"],
        n_heads_varattn=abl.get("n_heads_varattn", 4),
        d_state=abl.get("d_state", 16), d_conv=abl.get("d_conv", 4),
        expand=abl.get("expand", 2),
        dt_mode=abl.get("dt_mode", "replace"),
        grid_K=abl["grid_K"], t_max=1.0,
        n_freq=abl.get("n_freq", 8),
        lr=abl["lr"], weight_decay=abl["weight_decay"],
        num_warmup_steps=abl["num_warmup_steps"],
        num_training_steps=abl["max_steps"],
        loss_type=abl["loss"], huber_delta=abl["huber_delta"],
        lr_schedule=abl.get("lr_schedule", "cosine"),
    )
    if abl.get("arch") == "sandwich":
        kwargs["n_tail_grid_mamba"] = abl.get("n_tail_grid_mamba", 0)
    model = cls(**kwargs)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    miss, unx = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt_path.name}  missing={len(miss)} unexpected={len(unx)}")
    model.eval()
    return model, abl


def build_snapshot(abl: dict):
    args = PretrainDataModuleArgs(
        stage_cfg_path=abl["stage_cfg"], sources_cfg_path=abl["sources_cfg"],
        batch_size=SNAPSHOT_BATCH_SIZE, num_workers=4,
        max_dim=abl["max_dim"], seed=abl["seed"] + 5000,
        persistent_workers=False,
    )
    dm = PretrainDataModule(args); dm.setup()
    return dm.train_dataloader()


@torch.no_grad()
def collect(model, dl, n_batches: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    samples = []
    for bi, batch in enumerate(dl):
        if bi >= n_batches: break
        bg = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
        preds = model.forward(bg).detach().float().cpu().numpy()
        for k in ("values", "pred_mask", "valid_variate_mask"):
            batch[k] = batch[k].cpu().numpy() if isinstance(batch[k], torch.Tensor) else batch[k]
        for b in range(batch["values"].shape[0]):
            samples.append({
                "source": batch["source_name"][b],
                "V_active": int(batch["valid_variate_mask"][b].sum()),
                "values_a": batch["values"][b],
                "preds_a":  preds[b],
                "pred_mask": batch["pred_mask"][b],
                "valid_var": batch["valid_variate_mask"][b],
            })
    print(f"[snap] {len(samples)} samples")
    return samples


def v_bucket(v: int) -> str:
    if v == 1: return "V=1"
    if v <= 8: return "V=2..8"
    if v <= 16: return "V=9..16"
    return "V=17..20"


def per_source_per_V(samples):
    """Aggregate per-source and per-V-bucket metrics in BOTH spaces.

    asinh-z space = where model trains; loss is here.
    z space       = sinh(asinh-z); IMM-TSF paper reports MSE here.

    For deployment-style raw-y metrics we'd need μ_d, σ_d which are
    per-(b, v) and discarded after the collator's standardization;
    z-space is the right comparable surface for benchmarks.
    """
    agg_a = defaultdict(lambda: {"sse": 0.0, "sae": 0.0, "n": 0, "sumy": 0.0, "sumy2": 0.0, "n_samp": 0})
    agg_z = defaultdict(lambda: {"sse": 0.0, "sae": 0.0, "n": 0, "sumy": 0.0, "sumy2": 0.0, "n_samp": 0})
    for s in samples:
        src = s["source"]
        bk = v_bucket(s["V_active"])
        for v in range(s["values_a"].shape[0]):
            if not s["valid_var"][v]: continue
            pm = s["pred_mask"][v]
            if not pm.any(): continue

            # asinh-z space (model + loss space)
            yt_a = s["values_a"][v][pm]
            yp_a = s["preds_a"][v][pm]
            # Clamp BEFORE sinh inversion (per §7.L) so a few outliers
            # don't blow up the metric.
            yp_a_clamp = np.clip(yp_a, -10.0, 10.0)
            yt_a_clamp = np.clip(yt_a, -10.0, 10.0)
            d_a = yp_a - yt_a

            # z space (Chronos-2's ẑ; IMM-TSF paper's reported MSE space)
            yt_z = np.sinh(yt_a_clamp)
            yp_z = np.sinh(yp_a_clamp)
            d_z = yp_z - yt_z

            for key in ((src, "ALL"), (src, bk)):
                a, z = agg_a[key], agg_z[key]
                a["sse"] += float((d_a * d_a).sum())
                a["sae"] += float(np.abs(d_a).sum())
                a["n"]   += int(pm.sum())
                a["sumy"] += float(yt_a.sum()); a["sumy2"] += float((yt_a * yt_a).sum())
                z["sse"] += float((d_z * d_z).sum())
                z["sae"] += float(np.abs(d_z).sum())
                z["n"]   += int(pm.sum())
                z["sumy"] += float(yt_z.sum()); z["sumy2"] += float((yt_z * yt_z).sum())
        for key in ((src, "ALL"), (src, bk)):
            agg_a[key]["n_samp"] += 1; agg_z[key]["n_samp"] += 1

    def _print(agg, label):
        print(f"\n=== {label} ===")
        print(f"{'source':<20s}{'V-bucket':<10s}{'#samp':>8s}{'#obs':>10s}{'MSE':>12s}{'MAE':>12s}{'R²':>12s}")
        print("-" * 84)
        for (src, bk) in sorted(agg, key=lambda x: (x[0], x[1] != "ALL", x[1])):
            a = agg[(src, bk)]
            if a["n"] == 0: continue
            mse = a["sse"] / a["n"]
            mae = a["sae"] / a["n"]
            ss_tot = a["sumy2"] - a["sumy"]**2 / a["n"]
            r2 = 1.0 - a["sse"] / max(ss_tot, 1e-12)
            print(f"{src[:20]:<20s}{bk:<10s}{a['n_samp']:>8d}{a['n']:>10d}{mse:>12.4f}{mae:>12.4f}{r2:>12.4f}")

    _print(agg_a, "asinh-z space (model/loss space — diagnostic only)")
    _print(agg_z, "z space (Chronos-2 ẑ; comparable to IMM-TSF paper)")


def evaluate_ilinet(model, abl):
    """Evaluate the model on the actual IMM-TSF ILINet test split.

    Use the val_imm_tsf loader to mirror what training itself logged."""
    from imts_benchmark.pretrain.val_imts import build_imts_val_loaders
    loaders = build_imts_val_loaders(
        data_root="/home/ubuntu/hongyu/ssm/imts_benchmark/data/imm_tsf_sparse",
        datasets_list=["ILINet"],
        split="test", subset_size=None,
        batch_size=32, num_workers=2, max_dim=abl["max_dim"],
        seed=12345, norm_mode="precomputed_per_record",
    )
    if not loaders:
        print("[ilinet] no loader built")
        return
    device = next(model.parameters()).device
    print("\n=== ILINet test split (real, V≈11) — z-space metrics (paper-comparable) ===")
    for ds_name, dl in zip(["ILINet"], loaders):
        # Track both spaces for completeness
        sse_a = sae_a = sumy_a = sumy2_a = 0.0
        sse_z = sae_z = sumy_z = sumy2_z = 0.0
        n_pred = 0
        n_b = 0
        with torch.no_grad():
            for batch in dl:
                bg = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in batch.items()}
                preds = model.forward(bg)
                pm = bg["pred_mask"]
                yt_a = bg["values"][pm].float()
                yp_a = preds[pm].float()
                d_a = (yp_a - yt_a)
                sse_a += float((d_a * d_a).sum())
                sae_a += float(d_a.abs().sum())
                sumy_a += float(yt_a.sum()); sumy2_a += float((yt_a * yt_a).sum())

                # Invert to z, with the same ±10 clamp as _log_imts_val
                yt_z = torch.sinh(yt_a.clamp(-10.0, 10.0))
                yp_z = torch.sinh(yp_a.clamp(-10.0, 10.0))
                d_z = yp_z - yt_z
                sse_z += float((d_z * d_z).sum())
                sae_z += float(d_z.abs().sum())
                sumy_z += float(yt_z.sum()); sumy2_z += float((yt_z * yt_z).sum())

                n_pred += int(pm.sum().item())
                n_b += 1
        if n_pred == 0:
            print(f"  {ds_name}: no preds"); continue

        for label, sse, sae, sumy, sumy2 in [
            ("asinh-z (model space)", sse_a, sae_a, sumy_a, sumy2_a),
            ("z space  (paper)     ", sse_z, sae_z, sumy_z, sumy2_z),
        ]:
            mse = sse / n_pred
            mae = sae / n_pred
            ss_tot = sumy2 - sumy**2 / n_pred
            r2 = 1.0 - sse / max(ss_tot, 1e-12)
            print(f"  {ds_name}  [{label}]  N={n_pred}  MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")


def main():
    model, abl = load_model_from_ckpt(CKPT, ABL)
    dl = build_snapshot(abl)
    samples = collect(model, dl, SNAPSHOT_BATCHES)
    per_source_per_V(samples)
    evaluate_ilinet(model, abl)


if __name__ == "__main__":
    main()
