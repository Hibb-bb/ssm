"""Load each (model, regime) checkpoint at seed 1, run forward on the test
set, and save per-sample per-variate predictions to JSONL.

Output path: <out_dir>/<model>/<regime>/<variant>/predictions.jsonl
Each line:
  {
    "item_id": "...",
    "history": 8.0,
    "regime": "...",
    "model": "...",
    "variates": [
      {"ts_ctx": [...], "val_ctx": [...],
       "ts_pred": [...], "val_true_pred": [...], "val_pred": [...]},
      ...3 variates...
    ]
  }

For Mamba/mTAN/S5 (per_variate format): preds come out as [B, V, L].
For RoMAE (flat_tokens): logits are [B, pred_max, 1]; we reconstruct per-sample
per-variate predictions by iterating real pred positions in positional order
(matching romae_forecaster.py test_step logic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_SSM_DK = _THIS.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_data.multivariate_datamodule import (
    MultivariateSinusoidalDataModule,
)


REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")

MODEL_INFO = {
    "mamba_mv": dict(format="per_variate", variants=("learned", "replace", "additive")),
    "romae":    dict(format="flat_tokens", variants=("default",)),
    "mtan":     dict(format="per_variate", variants=("default",)),
    "s5":       dict(format="per_variate", variants=("default",)),
}


def load_mamba_mv(ckpt_path: Path):
    from imts_benchmark.mamba_mv.multivariate_forecaster import MultivariateMambaForecaster
    m = MultivariateMambaForecaster.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    return m


def load_romae(ckpt_path: Path):
    from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster
    return RoMAEForecaster.load_from_checkpoint(str(ckpt_path), map_location="cpu")


def load_mtan(ckpt_path: Path):
    from imts_benchmark.mtan_forecaster.mtan_forecaster import MTANForecaster
    return MTANForecaster.load_from_checkpoint(str(ckpt_path), map_location="cpu")


def load_s5(ckpt_path: Path):
    from imts_benchmark.s5_forecaster.s5_forecaster import S5Forecaster
    return S5Forecaster.load_from_checkpoint(str(ckpt_path), map_location="cpu")


LOADERS = {
    "mamba_mv": load_mamba_mv,
    "romae":    load_romae,
    "mtan":     load_mtan,
    "s5":       load_s5,
}


def extract_per_variate(batch: dict) -> list[list[dict]]:
    """For per_variate batches, extract per-sample per-variate context/pred arrays.
    Returns [[{ts_ctx, val_ctx, ts_pred, val_true_pred}, ...V], ...B]."""
    values = batch["values"].cpu().numpy()          # [B, V, L]
    timestamps = batch["timestamps"].cpu().numpy()  # [B, V, L]
    valid_mask = batch["valid_mask"].cpu().numpy()
    pred_mask = batch["pred_mask"].cpu().numpy()
    B, V, _ = values.shape
    result = []
    for i in range(B):
        per_var = []
        for d in range(V):
            valid = valid_mask[i, d]
            pmask = pred_mask[i, d]
            ctx_mask = valid & ~pmask
            per_var.append(dict(
                ts_ctx=timestamps[i, d, ctx_mask].tolist(),
                val_ctx=values[i, d, ctx_mask].tolist(),
                ts_pred=timestamps[i, d, pmask].tolist(),
                val_true_pred=values[i, d, pmask].tolist(),
            ))
        result.append(per_var)
    return result


def run_per_variate_model(model, dataloader, device: str, model_key: str) -> list[dict]:
    """Run a per_variate-format model, writing per-sample per-variate predictions.

    mamba_mv / s5 return a single tensor [B, V, L]. mTAN returns a tuple
    (pred[B,T,V], time_steps[B,T], pred_mask_full[B,T,V], true_full[B,T,V]):
    predictions are indexed by sorted-flat time, so we reshape them back to
    per-variate via pred_mask_full.
    """
    import numpy as np
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            raw = model(batch_dev)
            per_sample_variates = extract_per_variate(batch)
            item_ids = batch["item_id"]
            hist = float(batch["history"][0].item())

            if model_key == "mtan":
                pred, time_steps, pred_mask_full, true_full = raw
                pred_np = pred.cpu().numpy()                  # [B, T, V]
                pmf = pred_mask_full.cpu().numpy()            # [B, T, V] bool
                ts = time_steps.cpu().numpy()                 # [B, T]
                B, T, V = pred_np.shape
                for i in range(B):
                    entry = {"item_id": item_ids[i], "history": hist, "variates": []}
                    for d in range(V):
                        pv = per_sample_variates[i][d]
                        mask_td = pmf[i, :, d]
                        # Sort by ts to preserve plotting order & align ts_pred/val_pred
                        sel_ts = ts[i, mask_td]
                        sel_pred = pred_np[i, mask_td, d]
                        # Override wrapper-computed ts_pred/val_true_pred with these
                        # so (ts_pred, val_pred, val_true_pred) share the same order.
                        pv["ts_pred"] = sel_ts.tolist()
                        pv["val_pred"] = sel_pred.tolist()
                        # val_true_pred from batch (time-major); also filter & sort by
                        # pred_mask (already time-sorted in extract_per_variate since
                        # per-variate batch preserves positional order).
                        entry["variates"].append(pv)
                    out.append(entry)
            else:
                preds = raw.cpu().numpy()                     # [B, V, L]
                pred_mask = batch["pred_mask"].cpu().numpy()
                B, V, L = preds.shape
                for i in range(B):
                    entry = {"item_id": item_ids[i], "history": hist, "variates": []}
                    for d in range(V):
                        pv = per_sample_variates[i][d]
                        pm = pred_mask[i, d]
                        pv["val_pred"] = preds[i, d, pm].tolist()
                        entry["variates"].append(pv)
                    out.append(entry)
    return out


def run_romae_model(model, per_var_dataloader, flat_dataloader, device: str) -> list[dict]:
    """RoMAE consumes flat_tokens, but for plotting we still want per-variate
    context + targets per sample. We run both loaders in parallel so the order
    matches (both are DataLoader(shuffle=False) -> stable order by item_id)."""
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for pv_batch, flat_batch in zip(per_var_dataloader, flat_dataloader):
            flat_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in flat_batch.items()}
            logits, _ = model.forward(flat_dev)  # logits: [B, pred_max, 1]
            logits = logits.cpu().numpy()
            values = flat_batch["values"].cpu().numpy()
            pred_mask = flat_batch["pred_mask"].cpu().numpy()
            pad_mask = flat_batch["pad_mask"].cpu().numpy()
            variate_id = flat_batch["variate_id"].cpu().numpy()
            timestamps = flat_batch["timestamps"].cpu().numpy()
            pred_real_count = flat_batch["pred_real_count"].cpu().numpy()
            B = values.shape[0]

            per_sample_variates = extract_per_variate(pv_batch)
            item_ids = pv_batch["item_id"]
            hist = float(pv_batch["history"][0].item())

            for i in range(B):
                n_true = int(pred_real_count[i])
                real_mask = pred_mask[i] & pad_mask[i]
                # Row-order of logits[i] corresponds to positional-order True
                # positions in mask[i] (which equals pred_mask[i] in forward).
                # We only use the first n_true rows (real, not padded dummies).
                variates_flat = variate_id[i][real_mask]
                ts_flat = timestamps[i][real_mask]
                vals_flat = values[i][real_mask]
                y_pred_flat = logits[i, :n_true, 0]

                entry = {
                    "item_id": item_ids[i],
                    "history": hist,
                    "variates": [],
                }
                for d in range(3):
                    m_d = variates_flat == d
                    pv = per_sample_variates[i][d]
                    # Use flat-derived ts_pred / val_true_pred to match the y_pred order:
                    pv["ts_pred"] = ts_flat[m_d].tolist()
                    pv["val_true_pred"] = vals_flat[m_d].tolist()
                    pv["val_pred"] = y_pred_flat[m_d].tolist()
                    entry["variates"].append(pv)
                out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--summary_csv", type=str, required=True,
                    help="For Mamba: used to pick best dt_mode per regime.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    summary = pd.read_csv(args.summary_csv)

    for regime in REGIMES:
        print(f"\n=== regime: {regime} ===")
        pv_dm = MultivariateSinusoidalDataModule(
            data_root=args.data_root, regime=regime, format="per_variate",
            train_batch_size=64, val_batch_size=64, num_workers=0,
        )
        pv_dm.setup("test")
        flat_dm = MultivariateSinusoidalDataModule(
            data_root=args.data_root, regime=regime, format="flat_tokens",
            train_batch_size=64, val_batch_size=64, num_workers=0,
        )
        flat_dm.setup("test")

        for model_key in ("mamba_mv", "romae", "mtan", "s5"):
            info = MODEL_INFO[model_key]
            if model_key == "mamba_mv":
                sub = summary[(summary["model"] == "mamba_mv") & (summary["regime"] == regime)]
                variant = sub.loc[sub["mse_mean"].idxmin(), "variant"]
            else:
                variant = "default"
            ckpt = results_root / model_key / regime / variant / f"seed{args.seed}" / "checkpoints" / "best.ckpt"
            if not ckpt.exists():
                print(f"  [skip] {model_key}/{regime}/{variant}: no checkpoint at {ckpt}")
                continue
            print(f"  [run]  {model_key}/{regime}/{variant} (seed={args.seed})")
            model = LOADERS[model_key](ckpt)

            out_file = out_dir / model_key / regime / variant / "predictions.jsonl"
            if out_file.exists() and out_file.stat().st_size > 0:
                print(f"    [cache] {out_file.name} already exists; skipping")
                del model
                if args.device.startswith("cuda"): torch.cuda.empty_cache()
                continue

            if info["format"] == "per_variate":
                loader = pv_dm.test_dataloader()
                preds = run_per_variate_model(model, loader, args.device, model_key)
            else:
                preds = run_romae_model(model, pv_dm.test_dataloader(), flat_dm.test_dataloader(), args.device)

            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                for entry in preds:
                    entry["model"] = model_key
                    entry["regime"] = regime
                    entry["variant"] = variant
                    f.write(json.dumps(entry) + "\n")
            print(f"    wrote {len(preds)} samples -> {out_file}")
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
