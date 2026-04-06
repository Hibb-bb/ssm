"""
Convert model-agnostic .npz sinusoidal data to HuggingFace Arrow datasets
for MOIRAI training.

Reads from the raw .npz + meta.json produced by generate_sinusoidal_raw.py
and writes HuggingFace datasets with the fields MOIRAI expects:
  item_id, target, timestamp, past_feat_dynamic_real, n_obs_per_var, history

Usage:
    python convert_to_moirai.py \
        --input_root  /path/to/raw/sinusoidal_high_irreg \
        --output_root /path/to/moirai/sinusoidal_high_irreg

    # Or convert all levels at once:
    python convert_to_moirai.py \
        --input_root  /path/to/raw \
        --output_root /path/to/moirai \
        --all_levels
"""

import argparse
import json
from pathlib import Path

import datasets
import numpy as np
from datasets import Features, Sequence, Value


def npz_to_moirai_rows(npz_path, meta):
    """Convert a single .npz split to MOIRAI-format row dicts."""
    data = np.load(npz_path)
    timestamps = data["timestamps"]
    values_normed = data["values_normed"]
    n_samples, n_obs = timestamps.shape
    history = meta["history"]

    rows = []
    for i in range(n_samples):
        ts_i = timestamps[i]
        delta_t = np.zeros_like(ts_i)
        delta_t[1:] = np.diff(ts_i)

        rows.append({
            "item_id": f"sin_{i:04d}",
            "target": values_normed[i].tolist(),
            "timestamp": ts_i.tolist(),
            "past_feat_dynamic_real": delta_t.tolist(),
            "n_obs_per_var": [n_obs],
            "history": float(history),
        })
    return rows


def rows_to_hf(rows):
    features = Features({
        "item_id": Value("string"),
        "target": Sequence(Value("float32")),
        "timestamp": Sequence(Value("float32")),
        "past_feat_dynamic_real": Sequence(Value("float32")),
        "n_obs_per_var": Sequence(Value("int32")),
        "history": Value("float32"),
    })
    return datasets.Dataset.from_dict(
        {k: [r[k] for r in rows] for k in rows[0].keys()},
        features=features,
    )


def convert_one_level(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    with open(input_dir / "meta.json") as f:
        meta = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump({
            "data_min": [meta["data_min"]],
            "data_max": [meta["data_max"]],
            "time_max": meta["t_max"],
            "normalize_vals": True,
            "history": meta["history"],
            "n_vars": 1,
            "dataset": output_dir.name,
        }, f, indent=2)

    for split in ("train", "val", "test"):
        npz_path = input_dir / f"{split}.npz"
        if not npz_path.exists():
            print(f"  Skipping {split} (not found)")
            continue
        rows = npz_to_moirai_rows(npz_path, meta)
        hf_ds = rows_to_hf(rows)
        save_path = output_dir / split
        hf_ds.save_to_disk(str(save_path))
        print(f"  {split}: {len(hf_ds)} samples -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True,
                        help="Path to raw .npz data (single level dir, or parent if --all_levels)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path for MOIRAI HuggingFace output")
    parser.add_argument("--all_levels", action="store_true",
                        help="Convert all sinusoidal_* subdirectories")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if args.all_levels:
        for level_dir in sorted(input_root.glob("sinusoidal_*")):
            if not level_dir.is_dir():
                continue
            print(f"\nConverting {level_dir.name}...")
            convert_one_level(level_dir, output_root / level_dir.name)
    else:
        print(f"Converting {input_root.name}...")
        convert_one_level(input_root, output_root)

    print("\nDone!")


if __name__ == "__main__":
    main()
