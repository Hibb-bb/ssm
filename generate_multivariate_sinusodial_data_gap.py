"""
Phase 4: async-dense + long-gap multivariate sinusoidal data generator.

Extends the Phase 3 async generator (generate_multivariate_sinusodial_data_async.py)
by injecting long missing-observation gaps into a subset of variates per sample.

Pipeline:
  1. Generate per-variate async timestamps and analytic convex-mixture values
     exactly as in Phase 3 (preserves ground-truth x3 = w*x1 + (1-w)*x2 via
     re-evaluation of underlying sinusoids at each variate's own timestamps).
  2. Draw N_GAPS forbidden intervals per sample (default 1 gap per sample),
     with gap_start in [2.5, 6.0] and gap_len in [1.5, 3.0].
  3. Pick K variates per sample (default K=1) uniformly at random; drop their
     observations whose timestamps fall inside any forbidden interval.
  4. Emit sparse-flat HF Arrow records. n_obs_per_var is now VARIABLE per
     sample (gapped variates have fewer observations).
  5. Record gapped_variate_index: list[int] so downstream evaluation can
     slice gap-vs-non-gap metrics.

Gap semantics with HISTORY=8.0:
  * Gaps can span into the forecast region [8, 10] when gap_start > 5.0 AND
    gap_len > 3.0. This is intentional — the gap may remove observations
    AND/OR remove forecast targets. Non-gapped variates still have full
    forecast-region coverage at their own per-variate timestamps.

Output tree:
  ssm_dk/data_correct_gap/
  ├── multisin_regular/
  ├── multisin_low_irreg/
  ├── multisin_med_irreg/
  ├── multisin_high_irreg/
  └── multisin_stats.json

Usage:
    python generate_multivariate_sinusodial_data_gap.py \\
        [--output_root PATH] [--n_train 1000] [--gap_variates_per_sample 1]
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

# Gap constants (matching the deprecated long-gap generator).
GAP_START_RANGE = (2.5, 6.0)
GAP_LEN_RANGE = (1.5, 3.0)
N_GAPS = 1

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
    return (amp * np.sin(2 * np.pi * freq * timestamps + phase)).astype(np.float32)


def evaluate_signal_noisy(timestamps, freq, amp, phase, rng):
    clean = evaluate_sinusoid_clean(timestamps, freq, amp, phase)
    noise = rng.normal(0, NOISE_STD, size=len(timestamps)).astype(np.float32)
    return clean + noise


def draw_forbidden_intervals(n_gaps, gap_start_range, gap_len_range, t_max, rng):
    """Sample n_gaps non-overlapping forbidden intervals in [0, t_max]."""
    intervals = []
    max_retries = 50
    for _ in range(n_gaps):
        placed = False
        for _ in range(max_retries):
            g_start = rng.uniform(*gap_start_range)
            g_len = rng.uniform(*gap_len_range)
            g_end = min(g_start + g_len, t_max)
            if g_end <= g_start:
                continue
            if all((g_end <= s) or (g_start >= e) for s, e in intervals):
                intervals.append((float(g_start), float(g_end)))
                placed = True
                break
        if not placed:
            break
    intervals.sort()
    return intervals


def apply_forbidden_mask(ts, values, intervals):
    """Drop (timestamp, value) pairs where ts falls in any forbidden interval."""
    if not intervals:
        return ts, values
    keep = np.ones_like(ts, dtype=bool)
    for s, e in intervals:
        keep &= ~((ts >= s) & (ts < e))
    return ts[keep], values[keep]


def build_dataset(
    n_samples,
    freqs1, amps1, phases1,
    freqs2, amps2, phases2,
    mix_w,
    frac_regular,
    gap_variates_per_sample,
    n_gaps_per_sample,
    signal_rng,
    ts_rng,
    gap_rng,
    normalize=True,
    data_min=None,
    data_max=None,
    gap_variate_mode="fixed_v3",
):
    """Async + long-gap build."""
    rows = []
    all_values = []

    for i in range(n_samples):
        # Phase 3: three independent timestamp arrays.
        ts1 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)
        ts2 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)
        ts3 = generate_timestamps(N_OBS, T_MAX, frac_regular, ts_rng)

        # Phase 3: per-variate values with analytic convex mix for var 3.
        v1 = evaluate_signal_noisy(ts1, freqs1[i], amps1[i], phases1[i], signal_rng)
        v2 = evaluate_signal_noisy(ts2, freqs2[i], amps2[i], phases2[i], signal_rng)
        w = float(mix_w[i])
        x1_at_ts3 = evaluate_sinusoid_clean(ts3, freqs1[i], amps1[i], phases1[i])
        x2_at_ts3 = evaluate_sinusoid_clean(ts3, freqs2[i], amps2[i], phases2[i])
        v3_clean = w * x1_at_ts3 + (1.0 - w) * x2_at_ts3
        v3_noise = signal_rng.normal(0, NOISE_STD, size=N_OBS).astype(np.float32)
        v3 = (v3_clean + v3_noise).astype(np.float32)

        # Phase 4 (Option C): gap ALWAYS applied to v3 (variate index 2),
        # the convex-mixture-derived variate. This directly tests the paper's
        # cross-variate alignment claim: can the model use v1, v2 observations
        # to fill v3's gap, given that v3 = w*v1 + (1-w)*v2 analytically?
        intervals = draw_forbidden_intervals(
            n_gaps_per_sample, GAP_START_RANGE, GAP_LEN_RANGE, T_MAX, gap_rng
        )
        # gap_variate_mode controls which variate gets the gap:
        #   "fixed_v3"       : always gap v3 (Phase 4-1, forward-mixture test)
        #   "random_uniform" : uniformly pick variate ∈ {0,1,2} per sample
        #                      (Phase 4-2, includes inverse-mixture cases)
        if gap_variates_per_sample > 0 and intervals:
            if gap_variate_mode == "fixed_v3":
                gapped_vars = [2]
            elif gap_variate_mode == "random_uniform":
                gapped_vars = [int(gap_rng.integers(0, N_VARS))]
            else:
                raise ValueError(f"unknown gap_variate_mode: {gap_variate_mode}")
        else:
            gapped_vars = []

        ts_per_var = [ts1, ts2, ts3]
        val_per_var = [v1, v2, v3]
        for d in gapped_vars:
            ts_per_var[d], val_per_var[d] = apply_forbidden_mask(
                ts_per_var[d], val_per_var[d], intervals
            )

        obs_per_var = [int(len(t)) for t in ts_per_var]
        flat_vals = np.concatenate(val_per_var)
        all_values.append(flat_vals)

        # Per-variate delta_t. First entry of each variate's block is 0.
        dt_per_var = []
        for t in ts_per_var:
            dt = np.zeros_like(t)
            if len(t) > 1:
                dt[1:] = np.diff(t)
            dt_per_var.append(dt)

        target = flat_vals.tolist()
        stamp = np.concatenate(ts_per_var).tolist()
        past = np.concatenate(dt_per_var).tolist()

        # Flatten gap intervals into parallel float arrays (HF Arrow friendly)
        # so downstream eval can compute in-gap vs out-of-gap metrics on v3.
        gap_starts = [float(iv[0]) for iv in intervals]
        gap_ends = [float(iv[1]) for iv in intervals]

        rows.append({
            "item_id": f"sin_{i:04d}",
            "target": target,
            "timestamp": stamp,
            "past_feat_dynamic_real": past,
            "n_obs_per_var": obs_per_var,
            "history": HISTORY,
            "gapped_variate_index": gapped_vars,
            "gap_starts": gap_starts,
            "gap_ends": gap_ends,
            "_gap_intervals": [list(iv) for iv in intervals],  # kept for internal stats
        })

    if normalize and data_min is not None:
        scale = data_max - data_min
        if scale < 1e-08:
            scale = 1.0
        for row, vals in zip(rows, all_values):
            normed = ((vals - data_min) / scale).tolist()
            row["target"] = normed

    return rows, all_values


def rows_to_hf(rows):
    # gap_starts / gap_ends are parallel variable-length arrays encoding the
    # forbidden intervals applied to v3. Evaluation uses them to slice
    # in-gap vs out-of-gap metrics. `_gap_intervals` (nested list) is dropped
    # from the HF schema and only used for dataset-level stats.
    features = Features({
        "item_id": Value("string"),
        "target": Sequence(Value("float32")),
        "timestamp": Sequence(Value("float32")),
        "past_feat_dynamic_real": Sequence(Value("float32")),
        "n_obs_per_var": Sequence(Value("int32")),
        "history": Value("float32"),
        "gapped_variate_index": Sequence(Value("int32")),
        "gap_starts": Sequence(Value("float32")),
        "gap_ends": Sequence(Value("float32")),
    })
    public_rows = [
        {k: r[k] for k in features.keys()} for r in rows
    ]
    return datasets.Dataset.from_dict(
        {k: [r[k] for r in public_rows] for k in public_rows[0].keys()},
        features=features,
    )


def compute_gap_stats(rows):
    """Dataset-level gap statistics: mean obs per variate, gap length distribution."""
    n_vars = N_VARS
    obs_matrix = np.array([r["n_obs_per_var"] for r in rows])     # (n_rows, n_vars)
    gap_lens = []
    for r in rows:
        for s, e in r["_gap_intervals"]:
            gap_lens.append(e - s)
    gapped_counts = [len(r["gapped_variate_index"]) for r in rows]
    return {
        "n_samples": len(rows),
        "mean_obs_per_var": obs_matrix.mean(axis=0).tolist(),
        "min_obs_per_var": obs_matrix.min(axis=0).tolist(),
        "max_obs_per_var": obs_matrix.max(axis=0).tolist(),
        "mean_gap_len": float(np.mean(gap_lens)) if gap_lens else 0.0,
        "n_gaps_total": len(gap_lens),
        "mean_gapped_variates_per_sample": float(np.mean(gapped_counts)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--signal_seed", type=int, default=42)
    parser.add_argument("--gap_variates_per_sample", type=int, default=1,
                        help="K: number of variates per sample to apply the gap to (default 1)")
    parser.add_argument("--n_gaps_per_sample", type=int, default=N_GAPS)
    parser.add_argument("--gap_variate_mode", type=str, default="fixed_v3",
                        choices=["fixed_v3", "random_uniform"],
                        help="fixed_v3 (Phase 4-1): always gap v3 (forward-mixture test). "
                             "random_uniform (Phase 4-2): uniformly pick variate per sample "
                             "(stratified forward+inverse-mixture test).")
    args = parser.parse_args()

    if args.output_root is None:
        output_root = Path(__file__).resolve().parent / "data_correct_gap"
    else:
        output_root = Path(args.output_root)

    n_total = args.n_train + args.n_val + args.n_test

    param_rng = np.random.default_rng(args.signal_seed)
    f1, a1, p1, f2, a2, p2, mix_w = generate_multivariate_signal_params(n_total, param_rng)

    train_idx = slice(0, args.n_train)
    val_idx = slice(args.n_train, args.n_train + args.n_val)
    test_idx = slice(args.n_train + args.n_val, n_total)

    all_stats = {}

    for level_name, frac_regular in IRREGULARITY_LEVELS.items():
        print(f"\n{'=' * 60}")
        print(f"Generating (async+gap): multisin_{level_name}")
        print(f"  frac_regular={frac_regular:.0%}, "
              f"gap_variates_per_sample={args.gap_variates_per_sample}, "
              f"n_gaps={args.n_gaps_per_sample}")
        print(f"{'=' * 60}")

        ds_name = f"multisin_{level_name}"
        out_dir = output_root / ds_name
        os.makedirs(out_dir, exist_ok=True)

        signal_rng = np.random.default_rng(args.signal_seed + 1000)
        ts_seed = {"high_irreg": 100, "med_irreg": 200, "low_irreg": 300, "regular": 400}[level_name]
        ts_rng = np.random.default_rng(ts_seed)
        gap_rng = np.random.default_rng(args.signal_seed + 2000)

        all_rows, all_vals = build_dataset(
            n_total,
            f1, a1, p1, f2, a2, p2, mix_w,
            frac_regular,
            gap_variates_per_sample=args.gap_variates_per_sample,
            n_gaps_per_sample=args.n_gaps_per_sample,
            signal_rng=signal_rng,
            ts_rng=ts_rng,
            gap_rng=gap_rng,
            gap_variate_mode=args.gap_variate_mode,
            normalize=False,
        )

        train_rows = all_rows[train_idx]
        val_rows = all_rows[val_idx]
        test_rows = all_rows[test_idx]
        train_vals = all_vals[train_idx]
        val_vals = all_vals[val_idx]

        # Compute global min/max from train+val values (all concatenated).
        seen_vals = np.concatenate([np.concatenate(train_vals), np.concatenate(val_vals)])
        data_min = float(seen_vals.min())
        data_max = float(seen_vals.max())
        scale = (data_max - data_min) if (data_max - data_min) > 1e-08 else 1.0

        for row_set in (train_rows, val_rows, test_rows):
            for row in row_set:
                vals = np.array(row["target"])
                row["target"] = ((vals - data_min) / scale).tolist()

        stats = compute_gap_stats(list(train_rows))
        stats["frac_regular"] = frac_regular
        stats["n_train"] = len(train_rows)
        stats["n_val"] = len(val_rows)
        stats["n_test"] = len(test_rows)
        stats["gap_variates_per_sample"] = args.gap_variates_per_sample
        stats["n_gaps_per_sample"] = args.n_gaps_per_sample
        stats["data_min"] = data_min
        stats["data_max"] = data_max
        all_stats[ds_name] = stats

        print(f"  mean obs per var: {stats['mean_obs_per_var']}")
        print(f"  min obs per var:  {stats['min_obs_per_var']}")
        print(f"  mean gap length:  {stats['mean_gap_len']:.3f}")

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
            "long_gap_injection": True,
            "gap_start_range": list(GAP_START_RANGE),
            "gap_len_range": list(GAP_LEN_RANGE),
            "n_gaps_per_sample": args.n_gaps_per_sample,
            "gap_variates_per_sample": args.gap_variates_per_sample,
        }
        with open(out_dir / "norm_stats.json", "w") as f:
            json.dump(norm_stats, f, indent=2)

    stats_path = output_root / "multisin_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved gap stats → {stats_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
