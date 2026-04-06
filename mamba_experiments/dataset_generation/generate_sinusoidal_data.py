"""
Generate synthetic sinusoidal datasets with controlled temporal irregularity
for ablation study of RoPE + delta-t on MOIRAI.

Three irregularity levels (controlled by fraction of identical Δt):
  - high_irreg: 0% identical gaps (all random, like USHCN)
  - med_irreg:  30% identical gaps, 70% random
  - low_irreg:  80% identical gaps, 20% random

Signal: x(t) = A * sin(2π * f * t + φ) + ε
  - f ∈ [0.5, 3.0], A ∈ [0.5, 2.0], φ ∈ [0, 2π), ε ~ N(0, 0.05)
  - Univariate (n_vars=1), 80 obs per sample
  - Time window [0, 10], history=7.0

The same underlying signals (f, A, φ, noise) are used across all 3 irregularity
levels so the only difference is the temporal sampling pattern.

Usage:
    python generate_sinusoidal_data.py [--output_root PATH] [--n_train 1000]
"""

import argparse
import json
import os
from pathlib import Path

import datasets
import numpy as np
from datasets import Features, Sequence, Value

N_OBS = 120
T_MAX = 10.0
HISTORY = 7.0
NOISE_STD = 0.05

IRREGULARITY_LEVELS = {
    "high_irreg": 0.0,
    "med_irreg": 0.3,
    "low_irreg": 0.8,
    "regular": 1.0,
}


def generate_timestamps(n_obs, t_max, frac_regular, rng):
    """
    Generate n_obs sorted timestamps in [0, t_max] with a controlled
    fraction of identical time gaps.

    frac_regular: fraction of gaps (out of n_obs-1) that are set to a
                  fixed regular spacing. The rest are drawn uniformly.
    """
    n_gaps = n_obs - 1
    d_regular = t_max / n_gaps

    n_regular = int(round(frac_regular * n_gaps))
    n_random = n_gaps - n_regular

    gaps = np.empty(n_gaps)
    gaps[:n_regular] = d_regular

    gaps[n_regular:] = rng.uniform(d_regular * 0.1, d_regular * 3.0, size=n_random)

    rng.shuffle(gaps)

    # Rescale so gaps sum to t_max
    gaps = gaps * (t_max / gaps.sum())

    timestamps = np.concatenate([[0.0], np.cumsum(gaps)])
    return timestamps.astype(np.float32)


def generate_signal_params(n_samples, rng):
    """Generate random sinusoidal parameters for n_samples."""
    freqs = rng.uniform(0.5, 3.0, size=n_samples)
    amps = rng.uniform(0.5, 2.0, size=n_samples)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples)
    return freqs, amps, phases


def evaluate_signal(timestamps, freq, amp, phase, rng):
    """Evaluate x(t) = A * sin(2π * f * t + φ) + ε at given timestamps."""
    values = amp * np.sin(2 * np.pi * freq * timestamps + phase)
    noise = rng.normal(0, NOISE_STD, size=len(timestamps))
    return (values + noise).astype(np.float32)


def build_dataset(n_samples, freqs, amps, phases, frac_regular,
                  signal_rng, ts_rng, normalize=True, data_min=None, data_max=None):
    """
    Build a list of sample dicts for HuggingFace Dataset.

    signal_rng: RNG for noise (shared across irregularity levels for same noise)
    ts_rng: RNG for timestamp generation (different per irregularity level)
    """
    rows = []
    all_values = []

    for i in range(n_samples):
        timestamps = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)
        values = evaluate_signal(timestamps, freqs[i], amps[i], phases[i], signal_rng)
        all_values.append(values)

        delta_t = np.zeros_like(timestamps)
        delta_t[1:] = np.diff(timestamps)

        rows.append({
            "item_id": f"sin_{i:04d}",
            "target": values.tolist(),
            "timestamp": timestamps.tolist(),
            "past_feat_dynamic_real": delta_t.tolist(),
            "n_obs_per_var": [N_OBS],
            "history": HISTORY,
        })

    if normalize and data_min is not None:
        scale = data_max - data_min
        if scale < 1e-08:
            scale = 1.0
        for row, vals in zip(rows, all_values):
            normed = ((vals - data_min) / scale).tolist()
            row["target"] = normed

    return rows, np.array(all_values)


def rows_to_hf(rows):
    """Convert list of dicts to HuggingFace Dataset."""
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


def compute_irregularity_stats(rows):
    """Compute CV of Δt and mode fraction for a set of samples."""
    all_cv = []
    all_mode_frac = []
    for row in rows:
        dt = np.array(row["past_feat_dynamic_real"])[1:]
        if len(dt) < 2:
            continue
        mean_dt = np.mean(dt)
        std_dt = np.std(dt)
        cv = std_dt / mean_dt if mean_dt > 1e-10 else 0.0
        all_cv.append(cv)

        dt_rounded = np.round(dt, 4)
        unique, counts = np.unique(dt_rounded, return_counts=True)
        mode_frac = counts.max() / len(dt)
        all_mode_frac.append(mode_frac)

    return {
        "mean_cv": float(np.mean(all_cv)),
        "std_cv": float(np.std(all_cv)),
        "mean_mode_frac": float(np.mean(all_mode_frac)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--signal_seed", type=int, default=42,
                        help="Seed for signal params and noise (shared across irregularity levels)")
    args = parser.parse_args()

    if args.output_root is None:
        output_root = Path(__file__).resolve().parent.parent.parent / "tpatchgnn_data"
    else:
        output_root = Path(args.output_root)

    n_total = args.n_train + args.n_val + args.n_test

    # Generate shared signal parameters (same across all irregularity levels)
    param_rng = np.random.default_rng(args.signal_seed)
    freqs, amps, phases = generate_signal_params(n_total, param_rng)

    # Split indices
    train_idx = slice(0, args.n_train)
    val_idx = slice(args.n_train, args.n_train + args.n_val)
    test_idx = slice(args.n_train + args.n_val, n_total)

    all_stats = {}

    for level_name, frac_regular in IRREGULARITY_LEVELS.items():
        print(f"\n{'=' * 60}")
        print(f"Generating: sinusoidal_{level_name}")
        print(f"  frac_regular={frac_regular:.0%} of gaps identical")
        print(f"{'=' * 60}")

        ds_name = f"sinusoidal_{level_name}"
        out_dir = output_root / ds_name
        os.makedirs(out_dir, exist_ok=True)

        # Reset signal RNG so noise is identical across levels
        signal_rng = np.random.default_rng(args.signal_seed + 1000)

        ts_seed = {"high_irreg": 100, "med_irreg": 200, "low_irreg": 300, "regular": 400}[level_name]
        ts_rng = np.random.default_rng(ts_seed)

        all_rows, all_vals = build_dataset(
            n_total, freqs, amps, phases, frac_regular,
            signal_rng, ts_rng, normalize=False,
        )

        # Split into train / val / test
        train_rows = all_rows[train_idx]
        val_rows = all_rows[val_idx]
        test_rows = all_rows[test_idx]
        train_vals = all_vals[train_idx]
        val_vals = all_vals[val_idx]

        # Compute normalization stats from train + val only
        seen_vals = np.concatenate([train_vals.ravel(), val_vals.ravel()])
        data_min = float(seen_vals.min())
        data_max = float(seen_vals.max())
        scale = (data_max - data_min) if (data_max - data_min) > 1e-08 else 1.0

        # Normalize all splits with train+val stats
        for row_set in (train_rows, val_rows, test_rows):
            for row in row_set:
                vals = np.array(row["target"])
                row["target"] = ((vals - data_min) / scale).tolist()

        # Irregularity statistics
        stats = compute_irregularity_stats(list(train_rows))
        stats["frac_regular"] = frac_regular
        stats["n_train"] = len(train_rows)
        stats["n_val"] = len(val_rows)
        stats["n_test"] = len(test_rows)
        stats["n_obs"] = N_OBS
        stats["data_min"] = data_min
        stats["data_max"] = data_max
        all_stats[ds_name] = stats

        print(f"  CV of Δt: {stats['mean_cv']:.4f} ± {stats['std_cv']:.4f}")
        print(f"  Mode fraction: {stats['mean_mode_frac']:.4f}")

        # Save each split as HuggingFace dataset
        for split_name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            hf_ds = rows_to_hf(list(split_rows))
            save_path = out_dir / split_name
            hf_ds.save_to_disk(str(save_path))
            print(f"  Saved {split_name}: {len(hf_ds)} samples → {save_path}")

        norm_stats = {
            "data_min": [data_min],
            "data_max": [data_max],
            "time_max": T_MAX,
            "normalize_vals": True,
            "history": HISTORY,
            "n_vars": 1,
            "dataset": ds_name,
        }

        with open(out_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

    stats_path = output_root / "sinusoidal_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved irregularity stats → {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
