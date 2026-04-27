"""A/B test: our LOTSAHFSource (slow) vs Moirai-style PyArrow-direct
zero-copy indexer (fast).

Result on 2026-04-26 (cold cache, sudo-dropped page cache before run):
    wind_power            1  2377.31    1.98  1202.8x   OK
    solar_power           1  2337.11    0.86  2726.4x   OK
    residential_pv_power  233 454.95    0.30  1510.4x   OK
    wind_farms_w_missing  337 140.86    0.35   403.8x   OK
    bull                  41   20.50    0.29    70.1x   OK
    spain                  1   20.23    0.62    32.6x   OK
    covid_deaths         266    0.13    0.22     0.6x   OK
    m4_yearly          22739    0.10    0.23     0.4x   OK
    OVERALL               28  279.19    0.37   753.0x

    Throughput (bs=32 nw=12 gpu=110ms):
      slow : 0.745 s/batch (14.8% GPU util)
      fast : 0.001 s/batch (100.0% GPU util)


Hypothesis (from §7.D.2): the bottleneck is HF Datasets' default
``self.dataset[i]`` path, which converts Arrow sequences through
Python lists (one full O(N) copy + Python overhead per record) before
we even get to crop a window.

Moirai's ``HuggingFaceDatasetIndexer._getitem_int`` bypasses this by
using ``datasets.formatting.query_table + pa.chunk.slice().flatten()
.to_numpy(False)``, which is effectively zero-copy on the Arrow
memmap.

This script samples N records from the heaviest LOTSA datasets
through both code paths and reports the speedup.

Usage::

    python imts_benchmark/pretrain/scripts/profile_lotsa_indexer_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Heaviest LOTSA datasets from §7.D.2 + a couple of medium-tier and a
# few cheap ones for the contrast.
TEST_SOURCES = [
    "wind_power",          # 2390 ms/sample (slow path)
    "solar_power",         # 2375 ms/sample (slow path)
    "residential_pv_power",  # 532 ms/sample
    "wind_farms_with_missing",  # 160 ms/sample
    "bull",                # 21 ms/sample
    "spain",               # 20 ms/sample
    "covid_deaths",        # 0.48 ms/sample
    "m4_yearly",           # 0.30 ms/sample
]

LOTSA_ROOT = Path("/home/ubuntu/lotsa_data")
N_SAMPLES_PER_SOURCE = 5  # small budget; the slow path is ~2 s each


def _slow_path_load(ds, idx: int) -> np.ndarray:
    """Mirror what LOTSAHFSource.__iter__ does today."""
    ex = ds[idx]
    target = ex.get("target")
    if target is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(target, list):
        if target and isinstance(target[0], (list, np.ndarray)):
            return np.stack([np.asarray(t, dtype=np.float32) for t in target])
        return np.asarray(target, dtype=np.float32)[None, :]
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _fast_path_load(fetch_fn, idx: int) -> np.ndarray:
    """Mirror what Moirai's HuggingFaceDatasetIndexer does."""
    out = fetch_fn(idx)
    target = out.get("target")
    if target is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _build_pa_indexer(ds):
    """Inline port of uni2ts/src/uni2ts/data/indexer/hf_dataset_indexer.py
    (Apache-2 by Salesforce, Inc.).  Bypasses HF Datasets' Python-list
    formatting for sequence columns by going directly through the
    PyArrow chunk slice → numpy view path.  Avoids one O(N) Python
    list copy per record, which on huge multivariate LOTSA records
    (wind_power, solar_power) was costing ~2 s per sample.

    Returns a callable ``(idx) -> dict[str, ndarray|scalar]``.
    """
    import pyarrow as pa
    from datasets.formatting import query_table
    from datasets.features import Sequence

    features = dict(ds.features)
    non_seq_cols = [n for n, f in features.items() if not isinstance(f, Sequence)]
    seq_cols = [n for n, f in features.items() if isinstance(f, Sequence)]
    ds.set_format("numpy", columns=non_seq_cols)

    def _pa_col_to_np(pa_table, col_name):
        pa_array = pa_table.column(col_name)
        feat = features[col_name]
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(feat.feature, Sequence):
                arrs = []
                for chunk in pa_array.chunks:
                    for i in range(len(chunk)):
                        flat_slice = chunk.slice(i, 1).flatten()
                        feat_len = feat.length if feat.length != -1 else len(flat_slice)
                        arrs.append(
                            flat_slice.flatten().to_numpy(False).reshape(feat_len, -1)
                        )
                return arrs
            else:
                return [
                    chunk.slice(i, 1).flatten().to_numpy(False)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                ]
        elif isinstance(pa_array, pa.ListArray):
            if isinstance(feat.feature, Sequence):
                flat_slice = pa_array.flatten()
                feat_len = feat.length if feat.length != -1 else len(flat_slice)
                return [flat_slice.flatten().to_numpy(False).reshape(feat_len, -1)]
            return [pa_array.flatten().to_numpy(False)]
        raise NotImplementedError(f"unsupported pa array type: {type(pa_array)}")

    def fetch(idx: int) -> dict:
        non_seqs = ds[idx]
        pa_subtable = query_table(ds.data, idx, indices=ds._indices)
        seqs = {col: _pa_col_to_np(pa_subtable, col)[0] for col in seq_cols}
        return non_seqs | seqs

    return fetch


def main():
    import json
    from datasets import load_from_disk

    rng = np.random.default_rng(0)

    print(
        f"{'Dataset':<28s} {'records':>10s}  "
        f"{'slow ms':>10s} {'fast ms':>10s} {'speedup':>10s}"
    )
    print("-" * 80)

    overall_slow_total = 0.0
    overall_fast_total = 0.0
    overall_n = 0

    per_source: dict[str, dict] = {}

    for ds_name in TEST_SOURCES:
        path = LOTSA_ROOT / ds_name
        if not path.is_dir():
            print(f"{ds_name:<28s}  (skip — directory not found)")
            continue

        # Use two separate dataset handles. The fast-path build calls
        # set_format("numpy", columns=non_seq_cols), which would silently
        # drop the ``target`` column from the slow path's ``ds[idx]``.
        ds_slow = load_from_disk(str(path))
        ds_fast = load_from_disk(str(path))
        L = len(ds_slow)
        if L == 0:
            print(f"{ds_name:<28s} {L:>10d}  (skip — empty)")
            continue

        fast_fetch = _build_pa_indexer(ds_fast)

        # Sample the same indices for both paths so we measure the same
        # records. Cap so really tiny datasets still get something.
        budget = min(N_SAMPLES_PER_SOURCE, L)
        idxs_slow = [int(i) for i in rng.integers(0, L, size=budget)]
        idxs_fast = [int(i) for i in rng.integers(0, L, size=budget)]

        # NO warm-up: we want to capture the cold-cache latency that
        # dominates real training, not the warm-cache best case.
        slow_times = []
        for i in idxs_slow:
            t0 = time.perf_counter()
            arr = _slow_path_load(ds_slow, i)
            _ = arr.shape
            slow_times.append(time.perf_counter() - t0)

        fast_times = []
        for i in idxs_fast:
            t0 = time.perf_counter()
            arr = _fast_path_load(fast_fetch, i)
            _ = arr.shape
            fast_times.append(time.perf_counter() - t0)

        slow_ms = 1000 * mean(slow_times)
        fast_ms = 1000 * mean(fast_times)
        speedup = slow_ms / fast_ms if fast_ms > 0 else float("nan")

        # Sanity-check shapes match between the two paths on idx=0.
        try:
            s = _slow_path_load(ds_slow, 0).shape
            f = _fast_path_load(fast_fetch, 0).shape
            shape_match = "OK" if s == f else f"MISMATCH(s={s},f={f})"
        except Exception as e:
            shape_match = f"err:{type(e).__name__}"

        print(
            f"{ds_name:<28s} {L:>10d}  "
            f"{slow_ms:>10.2f} {fast_ms:>10.2f} {speedup:>9.1f}x   {shape_match}"
        )

        per_source[ds_name] = {
            "records": L,
            "slow_ms_mean": slow_ms,
            "fast_ms_mean": fast_ms,
            "speedup": speedup,
            "shape_check": shape_match,
        }
        overall_slow_total += sum(slow_times)
        overall_fast_total += sum(fast_times)
        overall_n += len(slow_times)

    print("-" * 80)
    if overall_n:
        slow_overall_ms = 1000 * overall_slow_total / overall_n
        fast_overall_ms = 1000 * overall_fast_total / overall_n
        speedup = slow_overall_ms / fast_overall_ms
        print(
            f"{'OVERALL':<28s} {overall_n:>10d}  "
            f"{slow_overall_ms:>10.2f} {fast_overall_ms:>10.2f} "
            f"{speedup:>9.1f}x"
        )

        # Throughput projection.
        bs, nw, gpu_step_s = 32, 12, 0.110
        slow_batch_s = bs * slow_overall_ms / 1000 / max(1, nw)
        fast_batch_s = bs * fast_overall_ms / 1000 / max(1, nw)
        print()
        print(f"Throughput projection (bs={bs}, nw={nw}, gpu={gpu_step_s*1000:.0f}ms):")
        print(f"  slow path : {slow_batch_s:.3f} s/batch ({gpu_step_s/slow_batch_s*100:.1f}% GPU util)")
        print(f"  fast path : {fast_batch_s:.3f} s/batch ({min(100, gpu_step_s/fast_batch_s*100):.1f}% GPU util)")

        sidecar = {
            "n_samples_per_source": N_SAMPLES_PER_SOURCE,
            "overall": {
                "slow_ms_mean": slow_overall_ms,
                "fast_ms_mean": fast_overall_ms,
                "speedup": speedup,
            },
            "throughput_proj": {
                "bs": bs, "nw": nw, "gpu_step_s": gpu_step_s,
                "slow_batch_s": slow_batch_s,
                "fast_batch_s": fast_batch_s,
                "slow_gpu_util_pct": gpu_step_s / slow_batch_s * 100,
                "fast_gpu_util_pct": min(100, gpu_step_s / fast_batch_s * 100),
            },
            "per_source": per_source,
        }
        out_path = (
            _REPO_ROOT
            / "imts_benchmark/pretrain/runs/profile/lotsa_indexer_compare.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sidecar, indent=2))
        print(f"\nJSON sidecar : {out_path}")


if __name__ == "__main__":
    main()
