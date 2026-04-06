"""
Generate synthetic sinusoidal time series at four irregularity levels.

Produces raw .npz + meta.json files that can be converted to HuggingFace
Arrow datasets via convert_to_moirai.py.

Irregularity levels (controlled by --frac_regular):
  regular    (1.0) — all uniform timestamps
  low_irreg  (0.8) — 80% uniform, 20% jittered
  med_irreg  (0.3) — 30% uniform, 70% jittered
  high_irreg (0.0) — all jittered

Each sample is: value(t) = amplitude * sin(2π * freq * t + phase)
  freq  ~ Uniform(0.5, 5.0)
  amp   ~ Uniform(0.5, 2.0)
  phase ~ Uniform(0, 2π)

Usage:
    # Generate a single level:
    python generate_sinusoidal_raw.py \
        --output_dir /path/to/raw/sinusoidal_high_irreg \
        --frac_regular 0.0

    # Generate all four levels:
    python generate_sinusoidal_raw.py \
        --output_dir /path/to/raw \
        --all_levels
"""

import argparse
import json
import os

import numpy as np


def generate_timestamps(n_obs, t_max, regular, rng):
    """Generate sorted timestamps in [0, t_max]."""
    if regular:
        return np.linspace(0, t_max, n_obs)
    else:
        ts = rng.uniform(0, t_max, n_obs)
        ts.sort()
        ts[0] = 0.0
        ts[-1] = t_max
        return ts


def generate_sinusoidal_dataset(
    n_samples, n_obs, t_max, frac_regular, seed, rng=None,
):
    if rng is None:
        rng = np.random.default_rng(seed)

    n_regular = int(round(n_samples * frac_regular))

    timestamps = np.zeros((n_samples, n_obs), dtype=np.float32)
    values_raw = np.zeros((n_samples, n_obs), dtype=np.float32)

    for i in range(n_samples):
        is_regular = i < n_regular
        ts = generate_timestamps(n_obs, t_max, is_regular, rng)

        freq = rng.uniform(0.5, 5.0)
        amp = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)

        vals = amp * np.sin(2 * np.pi * freq * ts + phase)

        timestamps[i] = ts.astype(np.float32)
        values_raw[i] = vals.astype(np.float32)

    perm = rng.permutation(n_samples)
    timestamps = timestamps[perm]
    values_raw = values_raw[perm]

    return timestamps, values_raw


def generate_and_save(output_dir, frac_regular, n_obs, t_max, history,
                      n_train, n_val, n_test, seed):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = {"train": n_train, "val": n_val, "test": n_test}
    all_values = []

    split_data = {}
    for split_name, n in splits.items():
        ts, vals = generate_sinusoidal_dataset(
            n, n_obs, t_max, frac_regular, seed=None, rng=rng,
        )
        split_data[split_name] = (ts, vals)
        all_values.append(vals)

    all_vals_flat = np.concatenate([v.ravel() for v in all_values])
    data_min = float(all_vals_flat.min())
    data_max = float(all_vals_flat.max())
    val_range = data_max - data_min
    if val_range < 1e-8:
        val_range = 1.0

    for split_name, (ts, vals) in split_data.items():
        vals_normed = (vals - data_min) / val_range
        npz_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez(npz_path, timestamps=ts, values_normed=vals_normed)
        print(f"  {split_name}: {len(ts)} samples -> {npz_path}")

    meta = {
        "history": history,
        "t_max": t_max,
        "data_min": data_min,
        "data_max": data_max,
        "n_obs": n_obs,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "frac_regular": frac_regular,
        "seed": seed,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sinusoidal time series")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frac_regular", type=float, default=0.0,
                        help="Fraction of samples with uniform timestamps")
    parser.add_argument("--n_obs", type=int, default=160)
    parser.add_argument("--t_max", type=float, default=10.0)
    parser.add_argument("--history", type=float, default=7.0)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all_levels", action="store_true",
                        help="Generate all four irregularity levels")
    args = parser.parse_args()

    levels = {
        "sinusoidal_regular":    1.0,
        "sinusoidal_low_irreg":  0.8,
        "sinusoidal_med_irreg":  0.3,
        "sinusoidal_high_irreg": 0.0,
    }

    if args.all_levels:
        for name, frac in levels.items():
            print(f"\nGenerating {name} (frac_regular={frac})...")
            generate_and_save(
                os.path.join(args.output_dir, name), frac,
                args.n_obs, args.t_max, args.history,
                args.n_train, args.n_val, args.n_test, args.seed,
            )
    else:
        print(f"Generating (frac_regular={args.frac_regular})...")
        generate_and_save(
            args.output_dir, args.frac_regular,
            args.n_obs, args.t_max, args.history,
            args.n_train, args.n_val, args.n_test, args.seed,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
