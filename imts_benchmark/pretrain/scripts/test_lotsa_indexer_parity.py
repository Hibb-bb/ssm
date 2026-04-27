"""Parity test: new ``LOTSAHFSource`` (Moirai-style indexer) vs the
old HF-default path on representative LOTSA datasets.

Verifies, per (dataset, idx) pair:
  1. shape equality
  2. dtype equality (must be float32 after to_2d)
  3. element-wise value equality (allclose with rtol=0, atol=0,
     equal_nan=True so structural NaN holes don't fail)
  4. yield-time speedup over the old path

This script must be re-run after any change to ``_HFArrowIndexer`` or
``LOTSAHFSource``.

Usage::

    python imts_benchmark/pretrain/scripts/test_lotsa_indexer_parity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from imts_benchmark.pretrain.sources import LOTSAHFSource


# Representative LOTSA datasets covering shape variations:
#   - 1 huge univariate record:   wind_power, solar_power
#   - many univariate records:    m4_yearly, covid_deaths
#   - many multivariate records:  residential_pv_power, PEMS04
#   - 1 medium univariate record: spain
TEST_CASES = [
    "wind_power",
    "solar_power",
    "residential_pv_power",
    "wind_farms_with_missing",
    "spain",
    "covid_deaths",
    "m4_yearly",
    "PEMS04",
]

LOTSA_ROOT = Path("/home/ubuntu/lotsa_data")
N_PER_DS = 5


def _slow_to_2d(target: Any) -> np.ndarray:
    """Mirror LOTSAHFSource._to_2d as it was before the indexer fix."""
    if isinstance(target, list):
        if target and isinstance(target[0], (list, np.ndarray)):
            return np.stack(
                [np.asarray(t, dtype=np.float32) for t in target]
            )
        return np.asarray(target, dtype=np.float32)[None, :]
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def main():
    from datasets import load_from_disk

    rng = np.random.default_rng(0)
    print(
        f"{'Dataset':<28s} {'records':>9s}  {'idx':>6s}  "
        f"{'shape':>16s}  {'dtype':>8s}  {'old ms':>8s}  "
        f"{'new ms':>8s}  status"
    )
    print("-" * 100)
    n_passed = 0
    n_total = 0
    for name in TEST_CASES:
        path = LOTSA_ROOT / name
        if not path.is_dir():
            print(f"{name:<28s}  (skip — directory not found)")
            continue

        # Separate handles: ``ds_old`` is *only* for the old path's
        # ``ds_old[idx]`` calls. ``LOTSAHFSource`` builds its own
        # internal handle and mutates its format — we don't share
        # that with ``ds_old``.
        ds_old = load_from_disk(str(path))
        L = len(ds_old)
        if L == 0:
            print(f"{name:<28s} {L:>9d}  (skip — empty)")
            continue

        src = LOTSAHFSource(path=str(path), name=f"lotsa:{name}", seed=0)

        budget = min(N_PER_DS, L)
        idxs = [int(i) for i in rng.integers(0, L, size=budget)]
        for idx in idxs:
            n_total += 1

            # Old slow path.
            t0 = time.perf_counter()
            ex_old = ds_old[idx]
            target_old = ex_old.get("target")
            arr_old = _slow_to_2d(target_old) if target_old is not None else None
            t1 = time.perf_counter()
            old_ms = 1000 * (t1 - t0)

            # New fast path (via indexer).
            t2 = time.perf_counter()
            ex_new = src._indexer.getitem(idx)
            target_new = ex_new.get("target")
            arr_new = (
                target_new if (target_new is not None and target_new.ndim == 2)
                else (target_new[None, :] if target_new is not None else None)
            )
            if arr_new is not None and arr_new.dtype != np.float32:
                arr_new = arr_new.astype(np.float32, copy=False)
            t3 = time.perf_counter()
            new_ms = 1000 * (t3 - t2)

            if arr_old is None or arr_new is None:
                status = (
                    "FAIL: target missing"
                    f" (old={arr_old is None}, new={arr_new is None})"
                )
            elif arr_old.shape != arr_new.shape:
                status = f"FAIL shape: old={arr_old.shape} new={arr_new.shape}"
            elif arr_old.dtype != arr_new.dtype:
                status = f"FAIL dtype: old={arr_old.dtype} new={arr_new.dtype}"
            elif not np.allclose(
                arr_old, arr_new, equal_nan=True, rtol=0.0, atol=0.0
            ):
                # Find a sample mismatch for debugging.
                diff = ~np.isclose(arr_old, arr_new, equal_nan=True, rtol=0, atol=0)
                n_bad = int(diff.sum())
                bad_idx = np.argwhere(diff)[0] if n_bad else None
                status = f"FAIL values: {n_bad} mismatches, first at {tuple(bad_idx)}"
            else:
                status = "OK"
                n_passed += 1

            print(
                f"{name:<28s} {L:>9d}  {idx:>6d}  "
                f"{str(arr_new.shape if arr_new is not None else '?'):>16s}  "
                f"{str(arr_new.dtype if arr_new is not None else '?'):>8s}  "
                f"{old_ms:>8.2f}  {new_ms:>8.2f}  {status}"
            )

    print("-" * 100)
    print(f"{n_passed}/{n_total} parity checks passed")
    return 0 if (n_total > 0 and n_passed == n_total) else 1


if __name__ == "__main__":
    sys.exit(main())
