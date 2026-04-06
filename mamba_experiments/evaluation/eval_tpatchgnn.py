"""
Evaluate a trained MOIRAI checkpoint on T-PatchGNN benchmark test sets.
Computes MSE, RMSE, MAE using T-PatchGNN per-variable-then-average protocol.

Usage:
    python eval_tpatchgnn.py --dataset physionet --checkpoint /path/to/last.ckpt
"""

import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import datasets, numpy as np, torch

UNI2TS_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(UNI2TS_ROOT / "src"))

from uni2ts.model.moirai.pretrain_tpatchgnn import MoiraiPretrainTPatchGNN
from uni2ts.transform import (
    AddObservedMask, AddTimeIndex, AddVariateIndex,
    DefaultPatchSizeConstraints, DummyValueImputation, ExtendMask,
    FixedHorizonPrediction, FlatPackCollection, FlatPackFields,
    GetPatchSize, ImputeTimeSeries, PackFields, Patchify,
    SelectFields, SequencifyField,
)


def build_transform(model):
    return (
        GetPatchSize(min_time_patches=model.hparams.min_patches, target_field="target",
                     patch_sizes=model.module.patch_sizes,
                     patch_size_constraints=DefaultPatchSizeConstraints(), offset=True)
        + PackFields(output_field="target", fields=("target",), feat=False)
        + PackFields(output_field="past_feat_dynamic_real", fields=tuple(),
                     optional_fields=("past_feat_dynamic_real",), feat=False)
        + AddObservedMask(fields=("target",), optional_fields=("past_feat_dynamic_real",),
                          observed_mask_field="observed_mask", collection_type=dict)
        + ImputeTimeSeries(fields=("target",), optional_fields=("past_feat_dynamic_real",),
                           imputation_method=DummyValueImputation(value=0.0))
        + Patchify(max_patch_size=max(model.module.patch_sizes),
                   fields=("target", "observed_mask"), optional_fields=("past_feat_dynamic_real",))
        + AddVariateIndex(fields=("target",), optional_fields=("past_feat_dynamic_real",),
                          variate_id_field="variate_id", expected_ndim=3,
                          max_dim=model.hparams.max_dim, randomize=False, collection_type=dict)
        + AddTimeIndex(fields=("target",), optional_fields=("past_feat_dynamic_real",),
                       time_id_field="time_id", expected_ndim=3, collection_type=dict)
        + FixedHorizonPrediction(history_field="history", time_id_field="time_id",
                                 target_field="target",
                                 truncate_fields=("variate_id", "time_id", "observed_mask"),
                                 optional_truncate_fields=("past_feat_dynamic_real",),
                                 prediction_mask_field="prediction_mask", expected_ndim=3)
        + ExtendMask(fields=tuple(), optional_fields=("past_feat_dynamic_real",),
                     mask_field="prediction_mask", expected_ndim=3)
        + FlatPackCollection(field="variate_id", feat=False)
        + FlatPackCollection(field="time_id", feat=False)
        + FlatPackCollection(field="prediction_mask", feat=False)
        + FlatPackCollection(field="observed_mask", feat=True)
        + FlatPackFields(output_field="target", fields=("target",),
                         optional_fields=("past_feat_dynamic_real",), feat=True)
        + SequencifyField(field="patch_size", target_field="target")
        + SelectFields(fields=["target","observed_mask","time_id","variate_id","prediction_mask","patch_size"])
    )


def hf_row_to_data_entry(row):
    return {
        "target": np.array(row["target"], dtype=np.float32),
        "timestamp": np.array(row["timestamp"], dtype=np.float32),
        "past_feat_dynamic_real": np.array(row["past_feat_dynamic_real"], dtype=np.float32),
        "history": float(row["history"]),
        "freq": "irregular",
        "item_id": row["item_id"],
    }


def run_single_sample(model, transform, data_entry, device, num_samples=100):
    gt_target = np.array(data_entry["target"], dtype=np.float32).copy()
    gt_ts = np.array(data_entry["timestamp"], dtype=np.float32).copy()
    history = float(data_entry["history"])
    gt_forecast = gt_target[gt_ts >= history]
    if len(gt_forecast) == 0:
        return None, None

    transformed = transform(data_entry)
    seq_fields = ["target","observed_mask","time_id","variate_id","prediction_mask","patch_size"]
    batch = {}
    for f in seq_fields:
        batch[f] = torch.from_numpy(np.array(transformed[f])).unsqueeze(0).to(device)
    seq_len = batch["target"].shape[1]
    batch["sample_id"] = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        distr = model(**{f: batch[f] for f in seq_fields}, sample_id=batch["sample_id"])
        samples = distr.sample(torch.Size((num_samples,)))
        point_pred = torch.median(samples, dim=0).values

    pred_mask = batch["prediction_mask"][0].bool()
    preds = point_pred[0, pred_mask, 0].cpu().numpy()
    n = min(len(gt_forecast), len(preds))
    return preds[:n], gt_forecast[:n]


def evaluate_checkpoint(ckpt_path, dataset_name, data_root, num_samples, device):
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = MoiraiPretrainTPatchGNN.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    transform = build_transform(model)

    test_ds = datasets.load_from_disk(str(data_root / dataset_name / "test"))
    with open(data_root / dataset_name / "norm_stats.json") as f:
        ns = json.load(f)
    n_vars = ns["n_vars"]
    print(f"[INFO] {dataset_name}: n_vars={n_vars}, test_samples={len(test_ds)}")

    se, ae, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    skip = 0
    for i in range(len(test_ds)):
        row = test_ds[i]
        v = row["var_index"]
        try:
            p, t = run_single_sample(model, transform, hf_row_to_data_entry(row), device, num_samples)
        except Exception as e:
            if i < 5: print(f"[WARN] {i} failed: {e}")
            skip += 1; continue
        if p is None:
            skip += 1; continue
        se[v] += float(((p - t)**2).sum())
        ae[v] += float(np.abs(p - t).sum())
        cnt[v] += len(p)
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(test_ds)}")

    mse_l, mae_l = [], []
    for d in range(n_vars):
        if cnt[d] == 0: continue
        mse_l.append(se[d]/cnt[d]); mae_l.append(ae[d]/cnt[d])
    mse = float(np.mean(mse_l)) if mse_l else 0.0
    mae = float(np.mean(mae_l)) if mae_l else 0.0
    rmse = float(np.sqrt(mse))
    print(f"\n[RESULTS] {dataset_name}: MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f}")
    return {"mse": mse, "rmse": rmse, "mae": mae}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", required=True, choices=["physionet","activity","ushcn"])
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--checkpoint_dir", default=None)
    pa.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    pa.add_argument("--data_root", default=None)
    pa.add_argument("--num_samples", type=int, default=100)
    pa.add_argument("--device", default="cuda")
    args = pa.parse_args()
    dr = Path(args.data_root) if args.data_root else UNI2TS_ROOT / "tpatchgnn_data"

    if args.checkpoint:
        evaluate_checkpoint(Path(args.checkpoint), args.dataset, dr, args.num_samples, args.device)
    elif args.checkpoint_dir:
        rs = []
        for s in args.seeds:
            cs = list(Path(args.checkpoint_dir).glob(f"*seed{s}*/**/last.ckpt"))
            if not cs: print(f"[WARN] No ckpt seed {s}"); continue
            rs.append(evaluate_checkpoint(cs[0], args.dataset, dr, args.num_samples, args.device))
        if rs:
            for m in ["mse","rmse","mae"]:
                v = [r[m] for r in rs]
                print(f"  {m.upper()}: {np.mean(v):.6f} +/- {np.std(v):.6f}")
    else:
        pa.error("Specify --checkpoint or --checkpoint_dir")

if __name__ == "__main__":
    main()
