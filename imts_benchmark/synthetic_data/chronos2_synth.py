# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Synthetic time series generators inspired by the Chronos-2 pretraining
# recipe (Ansari et al., 2025, arXiv:2510.15821). Produces irregular
# univariate AND multivariate series, written in long format to a single
# Arrow IPC file.
#
# Base univariate generators:
#   - AR   : autoregressive processes parameterized via partial autocorrs
#            (guaranteed stable via the Levinson recursion).
#   - ETS  : additive innovations state-space exponential smoothing with
#            optional trend and seasonality.
#   - TSI  : trend + seasonality + irregularity composites, in the spirit
#            of Bahrpeyma et al. (2021).
#
# Multivariatizers (impose dependencies across variates):
#   - Cotemporaneous : random (sparse) linear mix with optional pointwise
#                      nonlinearities, same time step across variates.
#   - Sequential     : VAR-style recursion driven by the base series,
#                      with a spectral-radius-controlled transition matrix.
#
# Irregularity is applied per variate: each variate independently gets
# Gaussian timestamp jitter and Bernoulli point drops. This matches long
# format's natural handling of ragged multivariate data.

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed
from scipy.signal import lfilter
from tqdm.auto import tqdm


LENGTH = 1024
BASE_START = np.datetime64("2000-01-01 00:00", "s")
STEP_SECONDS = 3600  # nominal hourly spacing


# ---------------------------------------------------------------------------
# Univariate generators
# ---------------------------------------------------------------------------

def _pac_to_ar(pac: np.ndarray) -> np.ndarray:
    """Partial autocorrelations -> AR coefficients via Levinson-Durbin.

    Any pac vector in (-1, 1) maps to a stationary AR, so we can sample
    coefficients freely without stability checks.
    """
    p = len(pac)
    phi = np.zeros(p)
    for k in range(p):
        new = phi.copy()
        new[k] = pac[k]
        for j in range(k):
            new[j] = phi[j] - pac[k] * phi[k - 1 - j]
        phi = new
    return phi


def generate_ar(rng: np.random.Generator, length: int = LENGTH) -> np.ndarray:
    """Sample from an AR(p) process with p ~ U{1,...,5} and stable coeffs."""
    p = int(rng.integers(1, 6))
    pac = rng.uniform(-0.95, 0.95, size=p)
    phi = _pac_to_ar(pac)
    sigma = rng.uniform(0.3, 1.5)

    # y[t] = phi1 y[t-1] + ... + phip y[t-p] + eps[t]
    # lfilter form: b=[1], a=[1, -phi1, ..., -phip]
    burn = 100
    eps = rng.normal(0.0, sigma, size=length + burn)
    a = np.concatenate([[1.0], -phi])
    y = lfilter([1.0], a, eps)
    return y[burn:]


def generate_ets(rng: np.random.Generator, length: int = LENGTH) -> np.ndarray:
    """Additive innovations state-space ETS with optional trend + seasonality.

    State update (Hyndman & Athanasopoulos, ETS(A,A,A) form):
        y_t   = l_{t-1} + b_{t-1} + s_{t-m} + eps_t
        l_t   = l_{t-1} + b_{t-1} + alpha * eps_t
        b_t   = b_{t-1} + beta  * eps_t
        s_t   = s_{t-m} + gamma * eps_t
    """
    has_trend = rng.random() < 0.7
    has_season = rng.random() < 0.7
    m = int(rng.choice([4, 7, 12, 24, 52])) if has_season else 1

    alpha = rng.uniform(0.05, 0.6)
    beta = rng.uniform(0.01, 0.3) if has_trend else 0.0
    gamma = rng.uniform(0.01, 0.3) if has_season else 0.0
    sigma = rng.uniform(0.3, 1.5)

    l = rng.normal(0.0, 2.0)
    b = rng.normal(0.0, 0.1) if has_trend else 0.0
    s = rng.normal(0.0, 1.0, size=m) if has_season else np.zeros(m)
    if has_season:
        s -= s.mean()  # identifiability: seasonal pattern sums to 0

    y = np.empty(length)
    eps = rng.normal(0.0, sigma, size=length)
    for t in range(length):
        s_tm = s[t % m]
        y[t] = l + b + s_tm + eps[t]
        l, b = l + b + alpha * eps[t], b + beta * eps[t]
        s[t % m] = s_tm + gamma * eps[t]
    return y


def generate_tsi(rng: np.random.Generator, length: int = LENGTH) -> np.ndarray:
    """Trend + Seasonality + Irregularity composite (Bahrpeyma et al., 2021)."""
    t = np.arange(length, dtype=np.float64)
    tn = t / length  # normalized in [0, 1)

    # --- trend ---
    trend_type = rng.choice(["none", "linear", "quadratic", "exp", "piecewise"])
    if trend_type == "none":
        trend = np.zeros(length)
    elif trend_type == "linear":
        trend = rng.uniform(-3.0, 3.0) * tn
    elif trend_type == "quadratic":
        trend = rng.uniform(-3.0, 3.0) * tn ** 2
    elif trend_type == "exp":
        trend = np.sign(rng.normal()) * np.expm1(rng.uniform(0.5, 2.0) * tn)
    else:  # piecewise linear with random breakpoints
        n_breaks = int(rng.integers(1, 4))
        breaks = np.sort(rng.uniform(0.1, 0.9, size=n_breaks))
        slopes = rng.uniform(-3.0, 3.0, size=n_breaks + 1)
        segs = np.concatenate([[0.0], breaks, [1.0]])
        trend = np.zeros(length)
        level = 0.0
        for i, slope in enumerate(slopes):
            mask = (tn >= segs[i]) & (tn < segs[i + 1])
            trend[mask] = level + slope * (tn[mask] - segs[i])
            level += slope * (segs[i + 1] - segs[i])

    # --- seasonality: sum of sinusoids ---
    K = int(rng.integers(1, 4))
    season = np.zeros(length)
    for _ in range(K):
        period = float(rng.choice([7, 12, 24, 30, 52, 60, 96, 168, 365]))
        amp = rng.uniform(0.2, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        season += amp * np.sin(2.0 * np.pi * t / period + phase)

    # --- irregularity: AR(1) ---
    phi1 = rng.uniform(-0.8, 0.8)
    sigma = rng.uniform(0.2, 1.2)
    eps = rng.normal(0.0, sigma, size=length)
    irreg = lfilter([1.0], [1.0, -phi1], eps)

    return trend + season + irreg


UNIVARIATE_GENERATORS = {
    "ar": generate_ar,
    "ets": generate_ets,
    "tsi": generate_tsi,
}


# ---------------------------------------------------------------------------
# Multivariatizers
# ---------------------------------------------------------------------------

def cotemporaneous_mix(
    base: np.ndarray, rng: np.random.Generator, n_out: int
) -> np.ndarray:
    """Instantaneous mixing: Y_t = f(W @ X_t) + noise.

    Each output variate can get a different pointwise nonlinearity.
    A fraction of W entries are zeroed to produce sparse dependency graphs.
    """
    _, n_base = base.shape
    W = rng.normal(0.0, 1.0, size=(n_base, n_out))
    sparsity = rng.uniform(0.0, 0.5)
    if sparsity > 0:
        W[rng.random(W.shape) < sparsity] = 0.0

    Y = base @ W

    for j in range(n_out):
        nl = rng.choice(["id", "tanh", "sq", "sin"])
        if nl == "tanh":
            Y[:, j] = np.tanh(Y[:, j])
        elif nl == "sq":
            Y[:, j] = np.sign(Y[:, j]) * np.minimum(Y[:, j] ** 2 * 0.1, 1e4)
        elif nl == "sin":
            Y[:, j] = np.sin(Y[:, j])

    Y += rng.normal(0.0, 0.1, size=Y.shape)
    return Y


def sequential_mix(
    base: np.ndarray, rng: np.random.Generator, n_out: int
) -> np.ndarray:
    """VAR-style lagged coupling driven by the base series.

        Y_t = A @ Y_{t-1} + B @ X_t + noise

    A is rescaled so its spectral radius is in (0.3, 0.9) to keep the
    recursion stable but expressive. This naturally creates lead-lag
    and cointegration-like behaviour across variates.
    """
    L, n_base = base.shape
    A = rng.normal(0.0, 1.0, size=(n_out, n_out))
    eig_max = np.max(np.abs(np.linalg.eigvals(A))) + 1e-8
    A *= rng.uniform(0.3, 0.9) / eig_max

    B = rng.normal(0.0, 0.5, size=(n_out, n_base))
    sigma = rng.uniform(0.05, 0.3)

    Y = np.zeros((L, n_out))
    noise = rng.normal(0.0, sigma, size=(L, n_out))
    for t in range(1, L):
        Y[t] = A @ Y[t - 1] + B @ base[t] + noise[t]
    return Y


MULTIVARIATIZERS = {
    "cotemporaneous": cotemporaneous_mix,
    "sequential": sequential_mix,
}


# ---------------------------------------------------------------------------
# System-level generation + irregularity
# ---------------------------------------------------------------------------

def generate_system(
    system_id: int,
    length: int = LENGTH,
    max_variates: int = 5,
    generators: Sequence[str] = ("ar", "ets", "tsi"),
    multivariatizers: Sequence[str] = ("cotemporaneous", "sequential"),
    p_univariate: float = 0.3,
    jitter_std: float = 0.0,
    drop_prob: float = 0.0,
    seed: Optional[int] = None,
):
    """Generate one (possibly multivariate) synthetic system.

    Returns a list of per-variate dicts with numpy arrays, or None if the
    generator produced non-finite values (rare, but possible with
    composed nonlinearities).
    """
    rng = np.random.default_rng(seed)

    gen_name = str(rng.choice(list(generators)))
    gen_fn = UNIVARIATE_GENERATORS[gen_name]

    go_univariate = (rng.random() < p_univariate) or not multivariatizers
    if go_univariate:
        Y = gen_fn(rng, length)[:, None]
    else:
        n_base = int(rng.integers(1, max_variates + 1))
        n_out = int(rng.integers(2, max_variates + 1))
        base = np.column_stack([gen_fn(rng, length) for _ in range(n_base)])
        mv_name = str(rng.choice(list(multivariatizers)))
        Y = MULTIVARIATIZERS[mv_name](base, rng, n_out)

    if not np.all(np.isfinite(Y)):
        return None

    offsets_nominal = np.arange(length, dtype=np.int64) * STEP_SECONDS
    # Give each system a disjoint series_id range so ids stay globally unique
    series_id_base = system_id * (max_variates + 1)

    variates = []
    for j in range(Y.shape[1]):
        offsets = offsets_nominal.copy()
        vals = Y[:, j].copy()

        if jitter_std > 0:
            noise = rng.normal(0.0, jitter_std, size=length).astype(np.int64)
            offsets = offsets + noise
            order = np.argsort(offsets)
            offsets, vals = offsets[order], vals[order]

        if drop_prob > 0:
            keep = rng.random(size=len(vals)) > drop_prob
            offsets, vals = offsets[keep], vals[keep]

        if len(vals) == 0:
            continue

        timestamps = BASE_START + offsets.astype("timedelta64[s]")
        n = len(vals)
        variates.append({
            "series_id": np.full(n, series_id_base + j, dtype=np.int32),
            "group_id": np.full(n, system_id, dtype=np.int32),
            "timestamp": timestamps,
            "value": vals.astype(np.float32),
        })

    return variates or None


# ---------------------------------------------------------------------------
# Arrow IPC writing
# ---------------------------------------------------------------------------

SCHEMA = pa.schema([
    ("series_id", pa.int32()),
    ("group_id", pa.int32()),
    ("timestamp", pa.timestamp("s")),
    ("value", pa.float32()),
])


def systems_to_batch(systems) -> Optional[pa.RecordBatch]:
    """Flatten a list-of-systems into a single Arrow RecordBatch."""
    records = [v for sys in systems if sys for v in sys]
    if not records:
        return None
    return pa.RecordBatch.from_pydict(
        {
            "series_id": np.concatenate([r["series_id"] for r in records]),
            "group_id": np.concatenate([r["group_id"] for r in records]),
            "timestamp": np.concatenate([r["timestamp"] for r in records]),
            "value": np.concatenate([r["value"] for r in records]),
        },
        schema=SCHEMA,
    )


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate irregular multivariate synthetic time series in Arrow long format."
    )
    parser.add_argument("-N", "--num-systems", type=int, default=100_000)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument(
        "--max-variates", type=int, default=5,
        help="Max variates per multivariate system.",
    )
    parser.add_argument(
        "--generators", type=str, default="ar,ets,tsi",
        help="Comma-separated univariate generators to sample from.",
    )
    parser.add_argument(
        "--multivariatizers", type=str, default="cotemporaneous,sequential",
        help="Comma-separated multivariatizers. Pass '' for univariate-only.",
    )
    parser.add_argument(
        "--p-univariate", type=float, default=0.3,
        help="Probability a given system is emitted as univariate (single variate).",
    )
    parser.add_argument(
        "--jitter-std", type=float, default=0.0,
        help="Stddev (seconds) of Gaussian timestamp jitter, per point, per variate.",
    )
    parser.add_argument(
        "--drop-prob", type=float, default=0.0,
        help="Per-point drop probability in [0, 1), independent per variate.",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    gens = _parse_csv(args.generators)
    mvs = _parse_csv(args.multivariatizers)

    if not gens:
        parser.error("--generators must list at least one generator.")
    for g in gens:
        if g not in UNIVARIATE_GENERATORS:
            parser.error(
                f"unknown generator '{g}'. choose from {list(UNIVARIATE_GENERATORS)}."
            )
    for m in mvs:
        if m not in MULTIVARIATIZERS:
            parser.error(
                f"unknown multivariatizer '{m}'. choose from {list(MULTIVARIATIZERS)}."
            )
    if not 0.0 <= args.drop_prob < 1.0:
        parser.error("--drop-prob must be in [0, 1).")
    if args.jitter_std < 0.0:
        parser.error("--jitter-std must be >= 0.")
    if not 0.0 <= args.p_univariate <= 1.0:
        parser.error("--p-univariate must be in [0, 1].")

    path = (
        Path(args.output)
        if args.output
        else Path(__file__).parent / "chronos2-synth.arrow"
    )

    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        for start in tqdm(
            range(0, args.num_systems, args.batch_size), desc="batches"
        ):
            ids = range(start, min(start + args.batch_size, args.num_systems))
            systems = Parallel(n_jobs=args.n_jobs)(
                delayed(generate_system)(
                    system_id=i,
                    length=args.length,
                    max_variates=args.max_variates,
                    generators=gens,
                    multivariatizers=mvs,
                    p_univariate=args.p_univariate,
                    jitter_std=args.jitter_std,
                    drop_prob=args.drop_prob,
                    seed=i,
                )
                for i in ids
            )
            batch = systems_to_batch(systems)
            if batch is not None:
                writer.write_batch(batch)

    print(f"Wrote {path}")