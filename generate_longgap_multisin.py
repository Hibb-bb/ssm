"""
DEPRECATED as of v2 (Phase 2 rebuild, 2026-04-21).
This generator diverges from the reference convex-mix spec at
  ssm_model/ssm/mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py
Phase 2 uses the correct generator at
  ssm_dk/generate_multivariate_sinusodial_data.py (shared timestamps, convex mix, 4 irreg regimes).
This file is retained because Phase 4 (IMTS long-gap) will reuse its
`draw_forbidden_intervals` and `apply_forbidden_mask` helpers on top of the
correct async dense generator. Do NOT run this script for new experiments.

Generate synthetic multivariate sinusoidal datasets with:
  - asynchronous per-variate irregular timestamps
  - explicit long missing gaps (forbidden intervals)
  - two regimes: sparse_independent / sparse_dependent

Sparse multivariate flat layout (matches convert_tpatchgnn_data.py):
  - target: list of floats, all variates concatenated
  - timestamp: same length as target (per-variate timestamps concatenated)
  - past_feat_dynamic_real: per-variate delta_t (first entry per variate = 0)
  - n_obs_per_var: count of observations per variate
  - history: scalar (shared across samples)

Regimes
-------
sparse_independent:
    Each variate drawn independently from its own (f, A, phi, noise).
    No shared latent. Negative control.

sparse_dependent:
    One shared latent sinusoid s(t) = A0 * sin(2*pi*f0*t + phi0).
    Variate d observes: a_d * s(t - tau_d) + b_d * priv_d(t) + eps_d,
    where priv_d is a small private sinusoid. Positive control — other
    variates carry useful information across the long gap.

Long-gap mechanism
------------------
For each sample we draw n_gaps forbidden intervals [g_start, g_start + g_len].
For each variate we optionally drop all observations whose timestamps fall
inside any forbidden interval. Which variates receive the gap is controlled
by --gap_variates_per_sample K: exactly K variates (uniformly chosen without
replacement) receive the long gap, the remaining n_vars - K stay dense.
Set K = n_vars to blank every variate in the gap; set K = 1 to blank only
one variate (the "target" regime where the model must rely on the others).

Usage:
    python generate_longgap_multisin.py \
        --regime sparse_dependent \
        --n_train 1000 --n_val 200 --n_test 200 \
        --output_root ./data
"""

import argparse
import json
import os
from pathlib import Path

import datasets
import numpy as np
from datasets import Features, Sequence, Value


# --- defaults (match scales used in generate_multivariate_sinusodial_data.py) ---
T_MAX = 10.0
HISTORY = 7.0
N_VARS = 3
N_OBS_PER_VAR = 120
NOISE_STD = 0.05

# per-variate timestamp irregularity: fraction of gaps that are regular spacing
# (0.0 = all random, like high_irreg in the old file)
FRAC_REGULAR = 0.0

# independent regime params
FREQ_RANGE = (0.5, 3.0)
AMP_RANGE = (0.5, 2.0)
PHASE_RANGE = (0.0, 2.0 * np.pi)

# dependent regime params
SHARED_FREQ_RANGE = (0.5, 2.0)
SHARED_AMP_RANGE = (1.0, 2.0)
LAG_RANGE = (-0.5, 0.5)         # tau_d: per-variate time shift of shared source
MIX_SHARED_RANGE = (0.6, 0.9)   # a_d: weight on shared source
PRIV_AMP_RANGE = (0.1, 0.4)     # b_d: weight on private sinusoid
PRIV_FREQ_RANGE = (0.5, 4.0)

# long-gap controls
GAP_START_RANGE = (2.5, 6.0)    # where a gap can begin (in time units)
GAP_LEN_RANGE = (1.5, 3.0)      # how long a gap is
N_GAPS = 1                      # number of forbidden intervals per sample


def generate_timestamps(n_obs, t_max, frac_regular, rng):
    """Generate n_obs sorted timestamps in [0, t_max]. Lifted from the old
    generate_multivariate_sinusodial_data.py, kept byte-for-byte compatible."""
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


def draw_forbidden_intervals(n_gaps, gap_start_range, gap_len_range, t_max, rng):
    """Sample `n_gaps` non-overlapping forbidden intervals inside [0, t_max].

    Returns a list of (start, end) tuples sorted by start time.
    Non-overlap is enforced by rejection sampling with a small retry budget;
    if we cannot place all gaps we return whatever fit.
    """
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


def apply_forbidden_mask(ts, intervals):
    """Drop timestamps falling inside any forbidden interval."""
    if not intervals:
        return ts
    keep = np.ones_like(ts, dtype=bool)
    for s, e in intervals:
        keep &= ~((ts >= s) & (ts < e))
    return ts[keep]


# --- signal families -------------------------------------------------------

def sample_independent_params(n_samples, n_vars, rng):
    """Per-sample, per-variate (f, A, phi)."""
    shape = (n_samples, n_vars)
    return {
        "freq": rng.uniform(*FREQ_RANGE, size=shape),
        "amp": rng.uniform(*AMP_RANGE, size=shape),
        "phase": rng.uniform(*PHASE_RANGE, size=shape),
    }


def eval_independent(ts, sample_idx, var_idx, params, rng):
    f = params["freq"][sample_idx, var_idx]
    a = params["amp"][sample_idx, var_idx]
    p = params["phase"][sample_idx, var_idx]
    y = a * np.sin(2.0 * np.pi * f * ts + p)
    y = y + rng.normal(0.0, NOISE_STD, size=len(ts))
    return y.astype(np.float32)


def sample_dependent_params(n_samples, n_vars, rng):
    """Per-sample shared (f0, A0, phi0) + per-variate (tau, a_d, b_d, f_priv, p_priv)."""
    return {
        "shared_freq": rng.uniform(*SHARED_FREQ_RANGE, size=n_samples),
        "shared_amp": rng.uniform(*SHARED_AMP_RANGE, size=n_samples),
        "shared_phase": rng.uniform(*PHASE_RANGE, size=n_samples),
        "lag": rng.uniform(*LAG_RANGE, size=(n_samples, n_vars)),
        "mix_shared": rng.uniform(*MIX_SHARED_RANGE, size=(n_samples, n_vars)),
        "priv_amp": rng.uniform(*PRIV_AMP_RANGE, size=(n_samples, n_vars)),
        "priv_freq": rng.uniform(*PRIV_FREQ_RANGE, size=(n_samples, n_vars)),
        "priv_phase": rng.uniform(*PHASE_RANGE, size=(n_samples, n_vars)),
    }


def eval_dependent(ts, sample_idx, var_idx, params, rng):
    f0 = params["shared_freq"][sample_idx]
    a0 = params["shared_amp"][sample_idx]
    p0 = params["shared_phase"][sample_idx]
    tau = params["lag"][sample_idx, var_idx]
    a_d = params["mix_shared"][sample_idx, var_idx]
    b_d = params["priv_amp"][sample_idx, var_idx]
    f_priv = params["priv_freq"][sample_idx, var_idx]
    p_priv = params["priv_phase"][sample_idx, var_idx]

    shared = a0 * np.sin(2.0 * np.pi * f0 * (ts - tau) + p0)
    priv = np.sin(2.0 * np.pi * f_priv * ts + p_priv)
    y = a_d * shared + b_d * priv + rng.normal(0.0, NOISE_STD, size=len(ts))
    return y.astype(np.float32)


# --- sample assembly -------------------------------------------------------

def build_sample(
    sample_idx,
    n_vars,
    t_max,
    n_obs_per_var,
    frac_regular,
    gap_variates_per_sample,
    n_gaps,
    params,
    eval_fn,
    ts_rng,
    signal_rng,
    gap_rng,
    history,
):
    """Return a dict in sparse-flat format plus the raw per-variate arrays
    (so callers can compute dataset-wide min/max before normalization)."""
    intervals = draw_forbidden_intervals(
        n_gaps, GAP_START_RANGE, GAP_LEN_RANGE, t_max, gap_rng
    )

    # Pick exactly `gap_variates_per_sample` variates (no replacement) to blank.
    k = max(0, min(gap_variates_per_sample, n_vars))
    gapped_variates = gap_rng.choice(n_vars, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    gap_mask_per_var = np.zeros(n_vars, dtype=bool)
    gap_mask_per_var[gapped_variates] = True

    all_target = []
    all_timestamp = []
    all_delta_t = []
    obs_per_var = []
    raw_values = []  # flat concat of per-variate observed values (for min/max)

    for d in range(n_vars):
        ts_d = generate_timestamps(n_obs_per_var, t_max, frac_regular, ts_rng)
        if gap_mask_per_var[d] and intervals:
            ts_d = apply_forbidden_mask(ts_d, intervals)

        if len(ts_d) == 0:
            obs_per_var.append(0)
            continue

        vals_d = eval_fn(ts_d, sample_idx, d, params, signal_rng)

        delta_t = np.zeros_like(ts_d)
        if len(ts_d) > 1:
            delta_t[1:] = np.diff(ts_d)

        all_target.extend(vals_d.tolist())
        all_timestamp.extend(ts_d.tolist())
        all_delta_t.extend(delta_t.tolist())
        obs_per_var.append(int(len(ts_d)))
        raw_values.append(vals_d)

    row = {
        "item_id": f"sin_{sample_idx:05d}",
        "target": all_target,
        "timestamp": all_timestamp,
        "past_feat_dynamic_real": all_delta_t,
        "n_obs_per_var": obs_per_var,
        "history": history,
        "_gap_intervals": intervals,
        "_gap_variate_mask": gap_mask_per_var.tolist(),
    }
    flat_raw = np.concatenate(raw_values) if raw_values else np.array([], dtype=np.float32)
    return row, flat_raw


def rows_to_hf(rows):
    """Convert list of dicts to HF Dataset. Strips debug-only underscore keys."""
    keep_keys = (
        "item_id",
        "target",
        "timestamp",
        "past_feat_dynamic_real",
        "n_obs_per_var",
        "history",
    )
    features = Features({
        "item_id": Value("string"),
        "target": Sequence(Value("float32")),
        "timestamp": Sequence(Value("float32")),
        "past_feat_dynamic_real": Sequence(Value("float32")),
        "n_obs_per_var": Sequence(Value("int32")),
        "history": Value("float32"),
    })
    return datasets.Dataset.from_dict(
        {k: [r[k] for r in rows] for k in keep_keys},
        features=features,
    )


def summary_stats(rows, n_vars):
    """Quick sanity stats: avg obs per variate, gap coverage."""
    obs_matrix = np.array([r["n_obs_per_var"] for r in rows])  # (N, n_vars)
    gap_lens = []
    for r in rows:
        gap_lens.extend(e - s for s, e in r["_gap_intervals"])
    return {
        "n_samples": len(rows),
        "mean_obs_per_var": obs_matrix.mean(axis=0).tolist(),
        "min_obs_per_var": obs_matrix.min(axis=0).tolist(),
        "max_obs_per_var": obs_matrix.max(axis=0).tolist(),
        "mean_gap_len": float(np.mean(gap_lens)) if gap_lens else 0.0,
        "n_gaps_total": len(gap_lens),
    }


# --- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", required=True, choices=["sparse_independent", "sparse_dependent"])
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--out_subdir", type=str, default=None,
                        help="Override output sub-directory name (defaults to --regime). "
                             "Use to write no-gap variants under nogap_* without overwriting "
                             "the long-gap data at sparse_*.")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--n_vars", type=int, default=N_VARS)
    parser.add_argument("--n_obs_per_var", type=int, default=N_OBS_PER_VAR)
    parser.add_argument("--t_max", type=float, default=T_MAX)
    parser.add_argument("--history", type=float, default=HISTORY,
                        help="Forecast boundary stored in each sample's `history` field "
                             "(datamodule reads this to build pred_mask = timestamps >= history). "
                             "Does NOT affect signal generation, only metadata.")
    parser.add_argument("--frac_regular", type=float, default=FRAC_REGULAR,
                        help="Fraction of per-variate time gaps that are regular (0 = all random).")
    parser.add_argument("--gap_variates_per_sample", type=int, default=None,
                        help="Exactly K variates per sample receive the long gap "
                             "(0 = no gap, 1 = one variate, n_vars = all). "
                             "Default = n_vars.")
    parser.add_argument("--n_gaps", type=int, default=N_GAPS)
    parser.add_argument("--signal_seed", type=int, default=42)
    parser.add_argument("--ts_seed", type=int, default=101)
    parser.add_argument("--gap_seed", type=int, default=202)
    parser.add_argument("--noise_seed", type=int, default=303)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    if args.output_root is None:
        output_root = Path(__file__).resolve().parent / "data"
    else:
        output_root = Path(args.output_root)

    n_total = args.n_train + args.n_val + args.n_test

    # Separate RNGs so we can change one axis (e.g. gap placement) without
    # disturbing the others — mirrors the old generator's param/signal split.
    param_rng = np.random.default_rng(args.signal_seed)
    signal_rng = np.random.default_rng(args.noise_seed)
    ts_rng = np.random.default_rng(args.ts_seed)
    gap_rng = np.random.default_rng(args.gap_seed)

    if args.regime == "sparse_independent":
        params = sample_independent_params(n_total, args.n_vars, param_rng)
        eval_fn = eval_independent
    else:
        params = sample_dependent_params(n_total, args.n_vars, param_rng)
        eval_fn = eval_dependent

    gap_variates_per_sample = (
        args.gap_variates_per_sample
        if args.gap_variates_per_sample is not None
        else args.n_vars
    )

    all_rows = []
    all_raw = []
    for i in range(n_total):
        row, raw = build_sample(
            i,
            args.n_vars,
            args.t_max,
            args.n_obs_per_var,
            args.frac_regular,
            gap_variates_per_sample,
            args.n_gaps,
            params,
            eval_fn,
            ts_rng,
            signal_rng,
            gap_rng,
            args.history,
        )
        all_rows.append(row)
        all_raw.append(raw)

    # Split by absolute index so signal params line up with splits.
    train_idx = slice(0, args.n_train)
    val_idx = slice(args.n_train, args.n_train + args.n_val)
    test_idx = slice(args.n_train + args.n_val, n_total)

    train_rows = all_rows[train_idx]
    val_rows = all_rows[val_idx]
    test_rows = all_rows[test_idx]

    # Normalization uses only seen (train+val) values, like the old generator.
    if args.normalize:
        seen_vals = np.concatenate(all_raw[train_idx] + all_raw[val_idx]) \
            if any(x.size for x in all_raw[train_idx] + all_raw[val_idx]) else np.array([0.0, 1.0])
        data_min = float(seen_vals.min())
        data_max = float(seen_vals.max())
        scale = (data_max - data_min) if (data_max - data_min) > 1e-08 else 1.0
        for row_set in (train_rows, val_rows, test_rows):
            for row in row_set:
                vals = np.asarray(row["target"], dtype=np.float32)
                row["target"] = ((vals - data_min) / scale).astype(np.float32).tolist()
    else:
        data_min = None
        data_max = None

    ds_name = args.out_subdir if args.out_subdir is not None else args.regime
    out_dir = output_root / ds_name
    os.makedirs(out_dir, exist_ok=True)

    split_stats = {}
    for split_name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        hf_ds = rows_to_hf(list(split_rows))
        save_path = out_dir / split_name
        hf_ds.save_to_disk(str(save_path))
        split_stats[split_name] = summary_stats(list(split_rows), args.n_vars)
        print(f"  Saved {split_name}: {len(hf_ds)} samples -> {save_path}")

    norm_stats = {
        "data_min": [data_min] if data_min is not None else None,
        "data_max": [data_max] if data_max is not None else None,
        "time_max": args.t_max,
        "normalize_vals": bool(args.normalize),
        "history": args.history,
        "n_vars": args.n_vars,
        "dataset": ds_name,
        "regime": args.regime,
        "n_obs_per_var_target": args.n_obs_per_var,
        "frac_regular": args.frac_regular,
        "gap_start_range": list(GAP_START_RANGE),
        "gap_len_range": list(GAP_LEN_RANGE),
        "gap_variates_per_sample": gap_variates_per_sample,
        "n_gaps": args.n_gaps,
        "seeds": {
            "signal": args.signal_seed,
            "ts": args.ts_seed,
            "gap": args.gap_seed,
            "noise": args.noise_seed,
        },
        "split_stats": split_stats,
    }
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Wrote norm_stats.json -> {out_dir / 'norm_stats.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
