"""
Phase 3: async-dense multivariate sinusoidal data generator.

Extends the Phase 2 generator (generate_multivariate_sinusodial_data.py) by
drawing INDEPENDENT timestamps per variate, instead of a single shared
timestamp grid. No long gaps: each variate still has N_OBS=120 observations.

The convex-mixture ground-truth relationship x3(t) = w*x1(t) + (1-w)*x2(t) is
preserved analytically — because we have closed-form sinusoid parameters for
variates 1 and 2, we re-evaluate them at variate 3's OWN timestamps and then
take the convex sum there. No interpolation needed.

Signals (async per-variate timestamps; sparse flat layout):
  - Variate 1 at ts1: x1(ts1) = A1 * sin(2π * f1 * ts1 + φ1) + ε1
  - Variate 2 at ts2: x2(ts2) = A2 * sin(2π * f2 * ts2 + φ2) + ε2
  - Variate 3 at ts3: x3(ts3) = w * [A1·sin(2π·f1·ts3 + φ1)]
                               + (1-w) * [A2·sin(2π·f2·ts3 + φ2)] + ε3

  Same f, A, φ, noise ranges as Phase 2. Same 4 irregularity regimes.

Note on v3 noise: Phase 2 inherits v3 noise from v1+v2 linear combination.
Phase 3 adds independent Gaussian noise to v3 (the "fresh" noise model).
Variance is NOISE_STD² for all three variates, not dependent on w.

Output tree (same schema as Phase 2, just different timestamps):
  ssm_dk/data_correct_async/
  ├── multisin_regular/
  ├── multisin_low_irreg/
  ├── multisin_med_irreg/
  ├── multisin_high_irreg/
  └── multisin_stats.json

Usage:
    python generate_multivariate_sinusodial_data_async.py \\
        [--output_root PATH] [--n_train 1000]
"""

import argparse
import json
import os
from pathlib import Path

import datasets
import numpy as np
from datasets import Features, Sequence, Value

N_OBS = 120
N_VARS = 3
T_MAX = 10.0
HISTORY = 8.0
NOISE_STD = 0.05
W_MIX_LO = 0.05
W_MIX_HI = 0.95

IRREGULARITY_LEVELS = {
    "high_irreg": 0.0,
    "med_irreg": 0.3,
    "low_irreg": 0.8,
    "regular": 1.0,
}


def generate_timestamps(n_obs, t_max, frac_regular, rng):
    n_gaps = n_obs - 1
    d_regular = t_max / n_gaps

    n_regular = int(round(frac_regular * n_gaps))
    n_random = n_gaps - n_regular

    gaps = np.empty(n_gaps)
    gaps[:n_regular] = d_regular
    gaps[n_regular:] = rng.uniform(d_regular * 0.1, d_regular * 3.0, size=n_random)
    rng.shuffle(gaps)
    gaps = gaps * (t_max / gaps.sum())

    timestamps = np.concatenate([[0.0], np.cumsum(gaps)])
    return timestamps.astype(np.float32)


def generate_signal_params(n_samples, rng):
    freqs = rng.uniform(0.5, 3.0, size=n_samples)
    amps = rng.uniform(0.5, 2.0, size=n_samples)
    phases = rng.uniform(0, 2 * np.pi, size=n_samples)
    return freqs, amps, phases


def generate_multivariate_signal_params(n_samples, rng):
    f1, a1, p1 = generate_signal_params(n_samples, rng)
    f2, a2, p2 = generate_signal_params(n_samples, rng)
    mix_w = rng.uniform(W_MIX_LO, W_MIX_HI, size=n_samples)
    return f1, a1, p1, f2, a2, p2, mix_w


def evaluate_sinusoid_clean(timestamps, freq, amp, phase):
    """Clean sinusoid evaluation (no noise). Used for v3's analytic mix."""
    return (amp * np.sin(2 * np.pi * freq * timestamps + phase)).astype(np.float32)


def evaluate_signal_noisy(timestamps, freq, amp, phase, rng):
    clean = evaluate_sinusoid_clean(timestamps, freq, amp, phase)
    noise = rng.normal(0, NOISE_STD, size=len(timestamps)).astype(np.float32)
    return clean + noise


def build_dataset(
    n_samples,
    freqs1,
    amps1,
    phases1,
    freqs2,
    amps2,
    phases2,
    mix_w,
    frac_regular,
    signal_rng,
    ts_rng,
    normalize=True,
    data_min=None,
    data_max=None,
):
    """Async-dense build. Each variate gets its own timestamp array."""
    rows = []
    all_values = []
    obs_per_var = [N_OBS] * N_VARS

    for i in range(n_samples):
        # Three independent timestamp arrays (sharing the same frac_regular
        # regime but drawn separately, so they generically do NOT collide).
        ts1 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)
        ts2 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)
        ts3 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)

        # Noisy observations of variate 1 at ts1, variate 2 at ts2.
        v1 = evaluate_signal_noisy(ts1, freqs1[i], amps1[i], phases1[i], signal_rng)
        v2 = evaluate_signal_noisy(ts2, freqs2[i], amps2[i], phases2[i], signal_rng)

        # Variate 3 is the analytic convex mix evaluated at ts3 (NOT derived
        # from v1, v2 -- they live on different time grids). Fresh noise.
        w = float(mix_w[i])
        x1_at_ts3 = evaluate_sinusoid_clean(ts3, freqs1[i], amps1[i], phases1[i])
        x2_at_ts3 = evaluate_sinusoid_clean(ts3, freqs2[i], amps2[i], phases2[i])
        v3_clean = w * x1_at_ts3 + (1.0 - w) * x2_at_ts3
        v3_noise = signal_rng.normal(0, NOISE_STD, size=N_OBS).astype(np.float32)
        v3 = (v3_clean + v3_noise).astype(np.float32)

        flat_vals = np.concatenate([v1, v2, v3])
        all_values.append(flat_vals)

        # Per-variate delta_t: first entry of each variate's block is 0.
        dt1 = np.zeros_like(ts1); dt1[1:] = np.diff(ts1)
        dt2 = np.zeros_like(ts2); dt2[1:] = np.diff(ts2)
        dt3 = np.zeros_like(ts3); dt3[1:] = np.diff(ts3)

        target = flat_vals.tolist()
        stamp = ts1.tolist() + ts2.tolist() + ts3.tolist()
        past = dt1.tolist() + dt2.tolist() + dt3.tolist()

        rows.append({
            "item_id": f"sin_{i:04d}",
            "target": target,
            "timestamp": stamp,
            "past_feat_dynamic_real": past,
            "n_obs_per_var": obs_per_var,
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
    """Per-variate CV of Δt averaged over first variate's block."""
    all_cv = []
    all_mode_frac = []
    for row in rows:
        dt = np.array(row["past_feat_dynamic_real"][:N_OBS])[1:]
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
    parser.add_argument("--signal_seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_root is None:
        output_root = Path(__file__).resolve().parent / "data_correct_async"
    else:
        output_root = Path(args.output_root)

    n_total = args.n_train + args.n_val + args.n_test

    # Shared signal-parameter RNG across regimes so the underlying signals
    # are identical across irregularity levels; only timestamps differ.
    param_rng = np.random.default_rng(args.signal_seed)
    f1, a1, p1, f2, a2, p2, mix_w = generate_multivariate_signal_params(n_total, param_rng)

    train_idx = slice(0, args.n_train)
    val_idx = slice(args.n_train, args.n_train + args.n_val)
    test_idx = slice(args.n_train + args.n_val, n_total)

    all_stats = {}

    for level_name, frac_regular in IRREGULARITY_LEVELS.items():
        print(f"\n{'=' * 60}")
        print(f"Generating (async): multisin_{level_name}")
        print(f"  frac_regular={frac_regular:.0%}, per-variate independent timestamps")
        print(f"{'=' * 60}")

        ds_name = f"multisin_{level_name}"
        out_dir = output_root / ds_name
        os.makedirs(out_dir, exist_ok=True)

        signal_rng = np.random.default_rng(args.signal_seed + 1000)
        ts_seed = {"high_irreg": 100, "med_irreg": 200, "low_irreg": 300, "regular": 400}[level_name]
        ts_rng = np.random.default_rng(ts_seed)

        all_rows, all_vals = build_dataset(
            n_total,
            f1, a1, p1, f2, a2, p2, mix_w,
            frac_regular,
            signal_rng,
            ts_rng,
            normalize=False,
        )

        train_rows = all_rows[train_idx]
        val_rows = all_rows[val_idx]
        test_rows = all_rows[test_idx]
        train_vals = all_vals[train_idx]
        val_vals = all_vals[val_idx]

        seen_vals = np.concatenate([train_vals.ravel(), val_vals.ravel()])
        data_min = float(seen_vals.min())
        data_max = float(seen_vals.max())
        scale = (data_max - data_min) if (data_max - data_min) > 1e-08 else 1.0

        for row_set in (train_rows, val_rows, test_rows):
            for row in row_set:
                vals = np.array(row["target"])
                row["target"] = ((vals - data_min) / scale).tolist()

        stats = compute_irregularity_stats(list(train_rows))
        stats["frac_regular"] = frac_regular
        stats["n_train"] = len(train_rows)
        stats["n_val"] = len(val_rows)
        stats["n_test"] = len(test_rows)
        stats["n_obs_per_var"] = N_OBS
        stats["n_vars"] = N_VARS
        stats["data_min"] = data_min
        stats["data_max"] = data_max
        all_stats[ds_name] = stats

        print(f"  CV of Δt (var 1): {stats['mean_cv']:.4f} ± {stats['std_cv']:.4f}")
        print(f"  Mode fraction (var 1): {stats['mean_mode_frac']:.4f}")

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
            "n_vars": N_VARS,
            "dataset": ds_name,
            "async_timestamps": True,
        }

        with open(out_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

    stats_path = output_root / "multisin_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved irregularity stats → {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
