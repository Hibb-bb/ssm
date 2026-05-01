"""Convert t-PatchGNN's MIMIC-III-Ext-tPatchGNN data into our sparse-flat HF Arrow
layout, faithfully replicating the t-PatchGNN preprocessing pipeline.

Data source: https://physionet.org/content/mimic-iii-ext-tpatchgnn/1.0.0/
  - full_dataset.csv  : columns ID, Time (minutes), Value_1..Value_V, Mask_1..Mask_V
  - mimic.pt          : (optional) pre-built PyTorch cache produced by lib/mimic.py;
                        we don't need it — we build directly from the CSV.

Pipeline matches t-PatchGNN/lib/mimic.py + lib/parse_datasets.py exactly:
  1. Per-patient: (record_id, tt = Time/60 [hours], vals [N,V], mask [N,V]).
  2. Split: train_test_split(samples, test_size=0.2, random_state=42, shuffle=True)
            → train_test_split(seen, test_size=0.25, shuffle=False) (60/20/20).
  3. Compute data_min/data_max per variable on seen (train+val), excluding masked
     entries (replicates lib/parse_datasets.py:get_data_min_max).
  4. Compute time_max as max tt across seen split.
  5. For each sample, convert dense (tt, vals, mask) → per-variate sparse arrays
     concatenated (matches our shared_data/multivariate_datamodule.py format).
  6. Write HF Arrow {train,val,test}/ + norm_stats.json under {out_root}/mimic/.

Usage:
    python data_pipeline/import_mimic_tpatchgnn.py \
        --raw_csv /path/to/full_dataset.csv \
        --out_root /projects/bfrf/seojininus/ssm/tpatchgnn_data

After this script runs, our trainer can hit MIMIC with `--regime mimic --auto_meta`
identically to how it hits PhysioNet today. The trainer never sees t-PatchGNN's
exact code path; only the same input distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


HISTORY = 24.0  # hours — t-PatchGNN run_all.sh `--dataset mimic --history 24`
# pred_window: t-PatchGNN does NOT cap the predict window. Their collate
# (lib/physionet.py:variable_time_collate_fn) puts everything tt < history into
# observed and everything tt >= history into predict, with no upper bound. We
# match this by passing all rows through unchanged. time_max in norm_stats.json
# is then computed from the seen split's actual max(tt) (matches their
# get_data_min_max behavior).
N_VARS_DEFAULT = 96  # MIMIC-III-Ext-tPatchGNN has 96 variables; verified at runtime


def load_mimic_csv(csv_path: Path) -> tuple[list[dict], int]:
    """Load full_dataset.csv into per-patient samples.

    Returns:
        samples: list of dicts with keys 'record_id', 'tt' (np.ndarray [N]),
                 'vals' (np.ndarray [N, V]), 'mask' (np.ndarray [N, V])
        n_vars : int — number of value columns
    """
    print(f"[mimic] loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[mimic] csv shape: {df.shape}")
    print(f"[mimic] columns (first 10): {list(df.columns[:10])}")

    value_cols = sorted(c for c in df.columns if c.startswith("Value_"))
    mask_cols = sorted(c for c in df.columns if c.startswith("Mask_"))
    if len(value_cols) != len(mask_cols):
        raise ValueError(f"value/mask col count mismatch: {len(value_cols)} vs {len(mask_cols)}")
    n_vars = len(value_cols)
    print(f"[mimic] n_vars (Value_*/Mask_* columns) = {n_vars}")

    if "ID" not in df.columns or "Time" not in df.columns:
        raise ValueError(f"expected columns 'ID' and 'Time' in CSV, got: {list(df.columns[:8])}")

    samples = []
    for record_id, sub in df.groupby("ID", sort=True):
        sub = sub.sort_values("Time")
        tt = sub["Time"].to_numpy(dtype=np.float32) / 60.0  # minutes → hours
        vals = sub[value_cols].to_numpy(dtype=np.float32)
        mask = sub[mask_cols].to_numpy(dtype=np.float32)
        # NOTE: NO truncation. t-PatchGNN's collate has no upper bound on tt
        # (lib/physionet.py:variable_time_collate_fn). We pass through every
        # row in full_dataset.csv as-is, matching their preprocessing exactly.
        samples.append({
            "record_id": str(record_id),
            "tt": tt,
            "vals": vals,
            "mask": mask,
        })

    print(f"[mimic] built {len(samples)} per-patient samples")
    return samples, n_vars


def split_samples(samples: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """Replicate t-PatchGNN's split: 80/20 then 75/25 within seen → 60/20/20."""
    seen, test = train_test_split(samples, test_size=0.2, random_state=seed, shuffle=True)
    train, val = train_test_split(seen, test_size=0.25, shuffle=False)
    print(f"[mimic] split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def compute_seen_stats(seen: list[dict], n_vars: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Replicate get_data_min_max from lib/parse_datasets.py:
    masked min/max across seen split (train+val), per variable.
    """
    inf = np.float32("inf")
    data_min = np.full(n_vars, inf, dtype=np.float32)
    data_max = np.full(n_vars, -inf, dtype=np.float32)
    time_max = -inf

    for s in seen:
        vals = s["vals"]
        mask = s["mask"]
        tt = s["tt"]
        for d in range(n_vars):
            valid = mask[:, d] == 1
            if valid.any():
                col = vals[valid, d]
                data_min[d] = min(data_min[d], col.min())
                data_max[d] = max(data_max[d], col.max())
        if len(tt) > 0:
            time_max = max(time_max, float(tt.max()))

    print(f"[mimic] seen-split stats — time_max: {time_max:.3f}h")
    print(f"[mimic] data_min[:5]: {data_min[:5]}")
    print(f"[mimic] data_max[:5]: {data_max[:5]}")
    return data_min, data_max, time_max


def convert_sample_to_sparse_flat(s: dict, n_vars: int) -> dict | None:
    """Convert one (tt, vals, mask) sample to our sparse-flat row.

    Returns:
        {'item_id', 'target', 'timestamp', 'past_feat_dynamic_real', 'n_obs_per_var', 'history'}
        or None if the sample has zero observations (skip).
    """
    tt = s["tt"]
    vals = s["vals"]
    mask = s["mask"]

    target_chunks = []
    timestamp_chunks = []
    dt_chunks = []
    n_obs_per_var = np.zeros(n_vars, dtype=np.int32)

    total = 0
    for d in range(n_vars):
        valid = mask[:, d] == 1
        if not valid.any():
            continue  # this variate is empty for this patient — n_obs_per_var[d]=0
        tt_d = tt[valid].astype(np.float32)
        vals_d = vals[valid, d].astype(np.float32)
        n_obs_per_var[d] = len(tt_d)
        total += len(tt_d)
        target_chunks.append(vals_d)
        timestamp_chunks.append(tt_d)
        # per-variate Δt: 0 for first; differences for the rest. Does NOT leak across variates.
        dt_d = np.zeros_like(tt_d, dtype=np.float32)
        if len(tt_d) > 1:
            dt_d[1:] = np.diff(tt_d)
        dt_chunks.append(dt_d)

    if total == 0:
        return None

    return {
        "item_id": s["record_id"],
        "target": np.concatenate(target_chunks).tolist(),
        "timestamp": np.concatenate(timestamp_chunks).tolist(),
        "past_feat_dynamic_real": np.concatenate(dt_chunks).tolist(),
        "n_obs_per_var": n_obs_per_var.tolist(),
        "history": HISTORY,
    }


def write_split(rows: list[dict], out_dir: Path) -> int:
    """Write a list of sparse-flat rows as an HF Arrow dataset under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    features = hfds.Features({
        "item_id": hfds.Value("string"),
        "target": hfds.Sequence(hfds.Value("float32")),
        "timestamp": hfds.Sequence(hfds.Value("float32")),
        "past_feat_dynamic_real": hfds.Sequence(hfds.Value("float32")),
        "n_obs_per_var": hfds.Sequence(hfds.Value("int32")),
        "history": hfds.Value("float32"),
    })
    ds = hfds.Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(out_dir))
    return len(ds)


def main():
    p = argparse.ArgumentParser(description="Convert MIMIC-III-Ext-tPatchGNN to our sparse-flat HF Arrow layout")
    p.add_argument("--raw_csv", type=str, required=True,
                   help="Path to full_dataset.csv (from PhysioNet MIMIC-III-Ext-tPatchGNN)")
    p.add_argument("--out_root", type=str, default="tpatchgnn_data",
                   help="Output root; writes to {out_root}/mimic/{train,val,test}/ + norm_stats.json")
    p.add_argument("--seed", type=int, default=42, help="t-PatchGNN uses 42")
    args = p.parse_args()

    csv_path = Path(args.raw_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"raw csv not found: {csv_path}")

    out_dir = Path(args.out_root) / "mimic"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, n_vars = load_mimic_csv(csv_path)
    train, val, test = split_samples(samples, seed=args.seed)

    seen = train + val
    data_min, data_max, time_max = compute_seen_stats(seen, n_vars)

    if not np.isfinite(data_min).all() or not np.isfinite(data_max).all():
        raise RuntimeError("data_min or data_max contains inf — some variable has no observations in seen split")

    print(f"[mimic] converting samples to sparse-flat layout...")
    train_rows = [r for r in (convert_sample_to_sparse_flat(s, n_vars) for s in train) if r is not None]
    val_rows   = [r for r in (convert_sample_to_sparse_flat(s, n_vars) for s in val)   if r is not None]
    test_rows  = [r for r in (convert_sample_to_sparse_flat(s, n_vars) for s in test)  if r is not None]
    print(f"[mimic] non-empty rows — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)}")

    n_train = write_split(train_rows, out_dir / "train")
    n_val   = write_split(val_rows,   out_dir / "val")
    n_test  = write_split(test_rows,  out_dir / "test")
    print(f"[mimic] wrote HF Arrow datasets — train: {n_train}, val: {n_val}, test: {n_test}")

    norm_stats = {
        "data_min": [float(x) for x in data_min],
        "data_max": [float(x) for x in data_max],
        "time_max": float(time_max),
        "normalize_vals": True,
        "history": HISTORY,
        "n_vars": int(n_vars),
        "dataset": "mimic",
    }
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"[mimic] wrote {out_dir / 'norm_stats.json'}")
    print(f"[mimic] DONE — load with --regime mimic --auto_meta")


if __name__ == "__main__":
    main()
