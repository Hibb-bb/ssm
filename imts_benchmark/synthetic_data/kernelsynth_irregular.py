# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified version of kernel-synth.py that produces irregular time series
# in long format (series_id, timestamp, value) and writes them as an
# Arrow IPC file, streamed in batches.

import argparse
import functools
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm

LENGTH = 1024
BASE_START = np.datetime64("2000-01-01 00:00", "s")
STEP_SECONDS = 3600  # nominal hourly spacing between points

KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Y
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    WhiteKernel(noise_level=0.1),
    WhiteKernel(noise_level=1.0),
    ConstantKernel(),
]


def random_binary_map(a: Kernel, b: Kernel, rng: np.random.Generator):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return binary_maps[rng.integers(0, 2)](a, b)


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
):
    """
    Draw a sample from a GP prior using multivariate_normal with a fast
    factorization method (eigh/cholesky) instead of SVD.
    """
    if X.ndim == 1:
        X = X[:, None]
    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )
    return ts


def generate_time_series(
    series_id: int,
    max_kernels: int = 5,
    jitter_std: float = 0.0,
    drop_prob: float = 0.0,
    seed: Optional[int] = None,
):
    """
    Generate a single synthetic time series from KernelSynth, optionally
    jittering the timestamps and/or dropping points so the series becomes
    irregular.

    Parameters
    ----------
    series_id
        Integer id assigned to this series (used as the long-format key).
    max_kernels
        Max number of base kernels composed to build the GP kernel.
    jitter_std
        Standard deviation, in seconds, of Gaussian noise added to each
        nominal timestamp. 0 disables jitter.
    drop_prob
        Probability of independently dropping each point. 0 disables drops.
    seed
        Per-series seed. Using the series_id keeps generation reproducible
        under joblib parallelism.

    Returns
    -------
    dict with numpy arrays {"series_id", "timestamp", "value"}, or None if
    the GP sample failed (caller filters these out).
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0, 1, LENGTH)

    # Build a random composite kernel
    selected_kernels = rng.choice(
        KERNEL_BANK, size=rng.integers(1, max_kernels + 1), replace=True
    )
    kernel = functools.reduce(
        lambda a, b: random_binary_map(a, b, rng), selected_kernels
    )

    # Sample a series from the GP prior
    try:
        ts = sample_from_gp_prior_efficient(kernel=kernel, X=X, random_seed=seed)
    except np.linalg.LinAlgError as err:
        print(f"[series {series_id}] GP sample failed: {err}")
        return None

    # Nominal regular offsets, in seconds
    offsets = np.arange(LENGTH, dtype=np.int64) * STEP_SECONDS

    # Jitter timestamps
    if jitter_std > 0:
        noise = rng.normal(0.0, jitter_std, size=LENGTH).astype(np.int64)
        offsets = offsets + noise
        order = np.argsort(offsets)  # keep monotonic after jitter
        offsets = offsets[order]
        ts = ts[order]

    # Drop points
    if drop_prob > 0:
        keep = rng.random(size=len(ts)) > drop_prob
        offsets = offsets[keep]
        ts = ts[keep]

    if len(ts) == 0:
        return None

    timestamps = BASE_START + offsets.astype("timedelta64[s]")
    n = len(ts)
    return {
        "series_id": np.full(n, series_id, dtype=np.int32),
        "timestamp": timestamps,
        "value": ts.astype(np.float32),
    }


# Arrow IPC schema for the long-format output.
SCHEMA = pa.schema(
    [
        ("series_id", pa.int32()),
        ("timestamp", pa.timestamp("s")),
        ("value", pa.float32()),
    ]
)


def records_to_batch(records):
    """Concatenate per-series dicts into a single Arrow RecordBatch."""
    records = [r for r in records if r is not None]
    if not records:
        return None
    return pa.RecordBatch.from_pydict(
        {
            "series_id": np.concatenate([r["series_id"] for r in records]),
            "timestamp": np.concatenate([r["timestamp"] for r in records]),
            "value": np.concatenate([r["value"] for r in records]),
        },
        schema=SCHEMA,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=1000_000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    parser.add_argument(
        "--jitter-std",
        type=float,
        default=0.0,
        help="Stddev (seconds) of Gaussian noise added to each timestamp. "
        "Try ~600 for noticeable but sub-step jitter at hourly spacing.",
    )
    parser.add_argument(
        "--drop-prob",
        type=float,
        default=0.0,
        help="Per-point drop probability in [0, 1). E.g. 0.1 drops ~10%% of points.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of series generated per Arrow record batch.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="joblib n_jobs (-1 = all cores).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .arrow path. Defaults to kernelsynth-irregular.arrow next to this script.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.drop_prob < 1.0:
        parser.error("--drop-prob must be in [0, 1).")
    if args.jitter_std < 0.0:
        parser.error("--jitter-std must be >= 0.")

    path = Path(args.output) if args.output else (
        Path(__file__).parent / "kernelsynth-irregular.arrow"
    )

    # Stream record batches to disk so we never materialize the whole dataset.
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        for start in tqdm(
            range(0, args.num_series, args.batch_size),
            desc="batches",
        ):
            ids = range(start, min(start + args.batch_size, args.num_series))
            records = Parallel(n_jobs=args.n_jobs)(
                delayed(generate_time_series)(
                    series_id=i,
                    max_kernels=args.max_kernels,
                    jitter_std=args.jitter_std,
                    drop_prob=args.drop_prob,
                    seed=i,
                )
                for i in ids
            )
            batch = records_to_batch(records)
            if batch is not None:
                writer.write_batch(batch)

    print(f"Wrote {path}")