"""Convert TIME-IMM (NeurIPS 2025) datasets into the sparse-flat Arrow layout
that imts_benchmark/shared_data/multivariate_datamodule.py expects.

Strategy: import the upstream IMM-TSF chunker (lib.parse_datasets) directly so
chunk boundaries, drop rules, per-feature normalization, and 60/20/20 splits are
byte-identical to the runs that produced paper Tables 3-11. Then convert each
chunk to a per-variate sparse-flat row.

Output layout per dataset:
    {out_root}/imm_<name>/
        train/  data-00000-of-00001.arrow + dataset_info.json + state.json
        val/    ...
        test/   ...
        norm_stats.json

Run on each dataset:
    python data_pipeline/import_timeimm.py --dataset FNSPID \
        --time_imm_root _time_imm_repo/data \
        --imm_tsf_root _imm_tsf_repo \
        --out_root time_imm_data

NB on normalization: IMM-TSF z-score-normalizes per feature column-wise *before*
chunking (parse_datasets.py:103-111). Values written to Arrow are therefore
already normalized; norm_stats.json sets normalize_vals=false to prevent the
trainer from double-normalizing. Reported MSE then lives in the same normalized
space as paper Tables 3-11.

Determinism: split_method="sample" is fully chronological per record; no RNG
participates in chunk selection or split assignment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import torch


DATASETS = {
    "GDELT":        {"V": 5,  "history": 14, "pred_window": 14, "stride": 14, "time_unit": "days"},
    "RepoHealth":   {"V": 10, "history": 31, "pred_window": 31, "stride": 31, "time_unit": "days"},
    "MIMIC":        {"V": 30, "history": 24, "pred_window": 24, "stride": 24, "time_unit": "hours"},
    "FNSPID":       {"V": 6,  "history": 31, "pred_window": 31, "stride": 31, "time_unit": "days"},
    "ClusterTrace": {"V": 11, "history": 12, "pred_window": 12, "stride": 12, "time_unit": "hours"},
    "StudentLife": {"V": 9,  "history": 31, "pred_window": 31, "stride": 31, "time_unit": "days"},
    "ILINet":       {"V": 11, "history": 36, "pred_window": 36, "stride": 4,  "time_unit": "weeks"},
    "CESNET":       {"V": 10, "history": 7,  "pred_window": 7,  "stride": 7,  "time_unit": "days"},
    "EPA-Air":      {"V": 4,  "history": 7,  "pred_window": 7,  "stride": 7,  "time_unit": "days"},
}


def regime_name(dataset: str) -> str:
    return "imm_" + dataset.lower().replace("-", "_")


def build_imm_args(dataset: str, time_imm_data_root: str) -> argparse.Namespace:
    cfg = DATASETS[dataset]
    args = argparse.Namespace(
        data_root=time_imm_data_root,
        dataset=dataset,
        device=torch.device("cpu"),
        history=cfg["history"],
        pred_window=cfg["pred_window"],
        stride=cfg["stride"],
        time_unit=cfg["time_unit"],
        unit_scale=None,
        batch_size=1,
        enable_text=False,
        use_text_embeddings=False,
        llm_model_fusion=None,
        llm_layers_fusion=None,
        max_length=1024,
        model="adapter",
        split_method="sample",
        npatch=None,
        patch_size=None,
        patch_stride=None,
        rec_ids=None,
    )
    return args


def chunk_to_sparse_flat(chunk, n_vars: int, item_id: str, history: float) -> dict:
    """Convert one IMM-TSF chunk -> sparse-flat Arrow row.

    chunk = (chunk_id, tt: [T], vals: [T, D], mask: [T, D], texts: list)
    where mask[i, d] == 1.0 iff variate d was observed at time tt[i].
    """
    _, tt, vals, mask, _ = chunk
    tt_np = tt.detach().cpu().numpy().astype(np.float32)
    vals_np = vals.detach().cpu().numpy().astype(np.float32)
    mask_np = mask.detach().cpu().numpy().astype(np.bool_)

    target_chunks = []
    timestamp_chunks = []
    delta_chunks = []
    n_obs_per_var = []

    for d in range(n_vars):
        idx = np.where(mask_np[:, d])[0]
        if idx.size == 0:
            n_obs_per_var.append(0)
            continue
        # Observations come out in chronological order because tt is already
        # sorted ascending in the upstream chunker (df was sorted by _ts_raw
        # before chunking; sub_tt = tt[idx] - st preserves order).
        ts_d = tt_np[idx]
        val_d = vals_np[idx, d]
        # delta_t per variate: first entry must be 0 so it doesn't leak across
        # variate boundaries (multivariate_datamodule.py:8-9).
        if ts_d.size == 1:
            dt_d = np.array([0.0], dtype=np.float32)
        else:
            dt_d = np.empty(ts_d.size, dtype=np.float32)
            dt_d[0] = 0.0
            dt_d[1:] = np.diff(ts_d)
        target_chunks.append(val_d)
        timestamp_chunks.append(ts_d)
        delta_chunks.append(dt_d)
        n_obs_per_var.append(int(ts_d.size))

    target = (
        np.concatenate(target_chunks).astype(np.float32)
        if target_chunks
        else np.zeros(0, dtype=np.float32)
    )
    timestamp = (
        np.concatenate(timestamp_chunks).astype(np.float32)
        if timestamp_chunks
        else np.zeros(0, dtype=np.float32)
    )
    deltat = (
        np.concatenate(delta_chunks).astype(np.float32)
        if delta_chunks
        else np.zeros(0, dtype=np.float32)
    )

    return dict(
        item_id=item_id,
        target=target.tolist(),
        timestamp=timestamp.tolist(),
        past_feat_dynamic_real=deltat.tolist(),
        n_obs_per_var=list(map(int, n_obs_per_var)),
        history=float(history),
    )


def write_split(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows for split at {out_dir}")
    features = datasets.Features(
        {
            "item_id": datasets.Value("string"),
            "target": datasets.Sequence(datasets.Value("float32")),
            "timestamp": datasets.Sequence(datasets.Value("float32")),
            "past_feat_dynamic_real": datasets.Sequence(datasets.Value("float32")),
            "n_obs_per_var": datasets.Sequence(datasets.Value("int32")),
            "history": datasets.Value("float32"),
        }
    )
    ds = datasets.Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(out_dir))


def run_dataset(args_cli: argparse.Namespace, dataset: str) -> None:
    cfg = DATASETS[dataset]
    history = float(cfg["history"])
    pred_window = float(cfg["pred_window"])
    n_vars = cfg["V"]

    sys.path.insert(0, str(Path(args_cli.imm_tsf_root).resolve()))
    from lib.parse_datasets import parse_datasets

    imm_args = build_imm_args(dataset, str(Path(args_cli.time_imm_root).resolve()))
    print(f"\n=== {dataset}: history={history} pred_window={pred_window} stride={cfg['stride']} unit={cfg['time_unit']} ===")
    data_obj = parse_datasets(imm_args, show_summary=False)

    ds = data_obj["ds"]
    train_idx = list(data_obj["train_dataloader"].dataset.indices)
    val_idx = list(data_obj["val_dataloader"].dataset.indices)
    test_idx = list(data_obj["test_dataloader"].dataset.indices) if data_obj.get("test_dataloader") else []

    # Sanity: chunk's vals dim must match expected V_paper
    _, _, first_vals, _, _ = ds.chunks[0]
    actual_v = int(first_vals.shape[-1])
    if actual_v != n_vars:
        raise ValueError(
            f"{dataset}: feature count mismatch (chunker={actual_v}, paper Table 1={n_vars})"
        )

    print(
        f"chunks: total={len(ds.chunks)}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    out_root = Path(args_cli.out_root) / regime_name(dataset)
    for split_name, idxs in splits.items():
        rows = []
        for i in idxs:
            chunk = ds.chunks[i]
            chunk_id = chunk[0]
            row = chunk_to_sparse_flat(chunk, n_vars=n_vars, item_id=chunk_id, history=history)
            rows.append(row)
        out_dir = out_root / split_name
        write_split(rows, out_dir)
        print(f"  wrote {split_name}: {len(rows)} rows -> {out_dir}")

    # norm_stats.json: values in Arrow are already z-score-normalized per feature
    # by the upstream chunker (parse_datasets.py:103-111). Set normalize_vals=False
    # so train_mv.py does NOT double-normalize. data_min/data_max are set to
    # placeholders that would be a no-op if accidentally applied.
    norm_stats = {
        "n_vars": n_vars,
        "history": history,
        "time_max": history + pred_window,
        "data_min": [0.0] * n_vars,
        "data_max": [1.0] * n_vars,
        "normalize_vals": False,
        "dataset": regime_name(dataset),
    }
    with open(out_root / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  wrote norm_stats.json -> {out_root / 'norm_stats.json'}")


def main() -> int:
    p = argparse.ArgumentParser(description="TIME-IMM -> sparse-flat Arrow adapter")
    p.add_argument("--dataset", choices=list(DATASETS.keys()) + ["ALL"], default="ALL")
    p.add_argument("--time_imm_root", default="_time_imm_repo/data",
                   help="Path to the Time-IMM repo's data/ directory")
    p.add_argument("--imm_tsf_root", default="_imm_tsf_repo",
                   help="Path to the IMM-TSF repo root (so we can import lib.parse_datasets)")
    p.add_argument("--out_root", default="time_imm_data",
                   help="Output root; per-dataset Arrow goes under {out_root}/imm_<name>/")
    args = p.parse_args()

    if args.dataset == "ALL":
        for dataset in DATASETS:
            run_dataset(args, dataset)
    else:
        run_dataset(args, args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
