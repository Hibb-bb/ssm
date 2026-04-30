"""Uniform IterableSource interface for pretraining data sources.

All sources yield raw entries of shape::

    {
        "target":      np.ndarray [V, T]  (float32, V can be 1)
        "freq":        str (pandas freq string, e.g. "5T", "H", "D")
        "item_id":     str
        "source_name": str (e.g. "lotsa:PEMS04", "chronos2_synth",
                           "kernelsynth")
    }

Downstream, ``synthetic_data.degradation.LOTSAToIrregular`` consumes
these entries to produce the per-variate ragged irregular format that
the pretraining collate fn batches.

Three concrete sources:
  - ``LOTSAHFSource``       wraps any HuggingFace ``arrow`` dataset on
                            disk (one item == one (multivariate) series)
  - ``ChronosSynthSource``  wraps the long-format chronos2_synth Arrow
                            IPC file (one ``group_id`` == one
                            multivariate system)
  - ``KernelSynthSource``   wraps the long-format kernelsynth Arrow IPC
                            file (one ``series_id`` == one univariate
                            series)

Sources are designed to be used inside a multi-worker PyTorch DataLoader.
Each worker draws its own infinite stream, partitioned by
``(worker_id, num_workers)`` for chronos / kernel, and by per-worker
random sampling for LOTSA HF (which is loaded lazily and cheap to index).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa


# ---------------------------------------------------------------------------
# Zero-copy HuggingFace Arrow indexer
# ---------------------------------------------------------------------------
#
# Ported from Moirai / uni2ts (Apache-2, Salesforce, Inc.):
#   uni2ts/src/uni2ts/data/indexer/hf_dataset_indexer.py
#
# WHY: HuggingFace Datasets' default ``__getitem__`` formats Sequence
# columns by materializing a Python ``list[float]`` of the full record
# length first, which numpy then has to copy back into an ndarray.
# For LOTSA's heaviest records (e.g. ``wind_power`` is one univariate
# series of 7,397,147 floats) this is two ≈2.3 s passes per access,
# *independent of disk I/O*.  See README §7.D.2-7.D.4 for the diagnosis
# and the A/B benchmark (791× overall, up to 3091× on solar_power).
#
# This indexer goes through ``pa.ChunkedArray.chunk.slice(i, 1).flatten()
# .to_numpy(False)`` instead, which is a zero-copy numpy view on the
# underlying Arrow Float32 buffer.  No Python list is materialized.
# ---------------------------------------------------------------------------

class _HFArrowIndexer:
    """Zero-copy single-record fetch over a HuggingFace ``Dataset``.

    Constructor sets the dataset's __getitem__ format to ``numpy``
    restricted to *non-sequence* columns.  Sequence columns are read
    via the PyArrow direct path in :meth:`getitem`.

    The returned ``target`` is:
      - shape ``(T,)`` for ``Sequence(float)``                (univariate)
      - shape ``(V, T)`` for ``Sequence(Sequence(float))``    (multivariate)

    Side effect: mutates the underlying Dataset's format.  Anyone
    sharing the same Dataset object will then only see the non-sequence
    columns through ``ds[i]``.  All consumers in this codebase must go
    through the indexer's :meth:`getitem`.
    """

    def __init__(self, ds):
        from datasets.features import Sequence as HFSequence
        self.ds = ds
        self.features = dict(ds.features)
        self._is_seq: dict[str, bool] = {
            n: isinstance(f, HFSequence) for n, f in self.features.items()
        }
        self.non_seq_cols = [n for n, s in self._is_seq.items() if not s]
        self.seq_cols = [n for n, s in self._is_seq.items() if s]
        ds.set_format("numpy", columns=self.non_seq_cols)

    def __len__(self) -> int:
        return len(self.ds)

    def getitem(self, idx: int) -> dict[str, Any]:
        from datasets.formatting import query_table
        non_seqs = self.ds[idx]
        pa_subtable = query_table(self.ds.data, idx, indices=self.ds._indices)
        seqs = {col: self._pa_col_to_np(pa_subtable, col) for col in self.seq_cols}
        return {**non_seqs, **seqs}

    def _pa_col_to_np(self, pa_table: pa.Table, col: str) -> np.ndarray:
        from datasets.features import Sequence as HFSequence
        pa_array = pa_table.column(col)
        feat = self.features[col]
        is_nested = isinstance(feat.feature, HFSequence)

        if isinstance(pa_array, pa.ChunkedArray):
            for chunk in pa_array.chunks:
                if len(chunk) == 0:
                    continue
                flat = chunk.slice(0, 1).flatten()
                if is_nested:
                    v_len = feat.length if feat.length != -1 else len(flat)
                    return flat.flatten().to_numpy(False).reshape(v_len, -1)
                return flat.to_numpy(False)
            raise IndexError(f"empty chunked array for column {col!r}")

        if isinstance(pa_array, pa.ListArray):
            flat = pa_array.flatten()
            if is_nested:
                v_len = feat.length if feat.length != -1 else len(flat)
                return flat.flatten().to_numpy(False).reshape(v_len, -1)
            return flat.to_numpy(False)

        raise NotImplementedError(
            f"unsupported pa array type for column {col!r}: {type(pa_array).__name__}"
        )


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class IterableSource:
    """Abstract iterable source: produces raw entries indefinitely."""

    name: str

    def __iter__(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of distinct items this source can yield (per epoch)."""
        return 0

    def set_worker(self, worker_id: int, num_workers: int, seed: int) -> None:
        """Optional: set the per-worker shard / RNG."""
        pass


# ---------------------------------------------------------------------------
# LOTSA via HuggingFace ``datasets``
# ---------------------------------------------------------------------------

@dataclass
class LOTSAHFSource(IterableSource):
    """One LOTSA dataset on disk (HuggingFace arrow format).

    Each item has fields ``item_id, start, freq, target``. ``target``
    is either a 1D list (univariate) or a list-of-lists [V][T]
    (multivariate). We normalize to a ``[V, T]`` ``np.ndarray``.

    Reads go through :class:`_HFArrowIndexer` (Moirai's PyArrow-direct
    pattern, ported in-tree under Apache-2) so that large records load
    in O(ms) instead of O(s) — see README §7.D.

    We sample item indices uniformly across the dataset, infinitely.
    """

    path: str
    name: Optional[str] = None
    seed: int = 0

    # Internal state (set in __post_init__)
    _ds: Any = field(default=None, init=False, repr=False)
    _indexer: Any = field(default=None, init=False, repr=False)
    _len: int = field(default=0, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from datasets import load_from_disk  # lazy import
        self._ds = load_from_disk(self.path)
        self._len = len(self._ds)
        if self.name is None:
            self.name = f"lotsa:{Path(self.path).name}"
        self._rng = np.random.default_rng(self.seed)
        self._indexer = _HFArrowIndexer(self._ds)

    def __len__(self) -> int:
        return self._len

    def set_worker(self, worker_id: int, num_workers: int, seed: int) -> None:
        # Each worker draws its own subset of indices, with its own RNG.
        self._rng = np.random.default_rng(seed + worker_id * 9973)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._len == 0:
            return
        while True:
            i = int(self._rng.integers(0, self._len))
            try:
                ex = self._indexer.getitem(i)
            except Exception:
                continue
            target = ex.get("target")
            if target is None:
                continue
            # Indexer returns (T,) univariate or (V, T) multivariate.
            arr = target if target.ndim == 2 else target[None, :]
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            if arr.size == 0 or arr.shape[1] < 2:
                continue
            yield {
                "target": arr,
                "freq": str(ex.get("freq", "H")),
                "item_id": str(ex.get("item_id", i)),
                "source_name": self.name,
            }


# ---------------------------------------------------------------------------
# Moirai-style multi-sample stacking for univariate LOTSA datasets
# ---------------------------------------------------------------------------

# Moirai's curated MULTI_SAMPLE_DATASETS lists, copied verbatim from:
#   uni2ts/src/uni2ts/data/builder/lotsa_v1/{gluonts,proenfo,others,
#                                            buildings_bench,subseasonal,
#                                            lib_city}.py
#
# Each builder file has its own list; we union them.  Datasets in this
# union are passed through ``StackedLOTSASource`` instead of
# ``LOTSAHFSource``.  See README §7.N for the rationale and the
# diff to Moirai's exact behavior (we use OUR ``variate_count_dist``
# bucketing instead of beta_binomial(2, 5), since most of our IMM-TSF
# eval sets sit in V=4..11 — see Table 1 of the IMM-TSF paper).
MOIRAI_STACKING_DATASETS: tuple[str, ...] = (
    # gluonts.py
    "oikolab_weather",
    "kaggle_web_traffic_weekly",
    "extended_web_traffic_with_missing",
    "m5",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "traffic_hourly",
    "traffic_weekly",
    "rideshare_with_missing",
    "temperature_rain_with_missing",
    "car_parts_with_missing",
    "fred_md",
    "hospital",
    "covid_deaths",
    # proenfo.py
    "gfc12_load",
    "gfc17_load",
    "bull",
    "hog",
    # others.py
    "godaddy",
    "hierarchical_sales",
    "beijing_air_quality",
    # buildings_bench.py
    "bdg-2_panther",
    # lib_city.py and subseasonal.py: every dataset in the dataset_list
    # uses MultiSampleTimeSeriesDataset.  Most of those are already
    # multivariate-on-disk (LOOP_SEATTLE, PEMS_BAY, subseasonal etc.),
    # so stacking is a no-op for them — we keep them in the list anyway
    # for completeness and future-proofing.
    "BEIJING_SUBWAY_30MIN",
    "HZMETRO",
    "LOOP_SEATTLE",
    "LOS_LOOP",
    "M_DENSE",
    "PEMS03",
    "PEMS04",
    "PEMS07",
    "PEMS08",
    "PEMS_BAY",
    "Q-TRAFFIC",
    "SHMETRO",
    "SZ_TAXI",
    "subseasonal",
    "subseasonal_precip",
)


# Default per-bucket distribution; can be overridden per-source.
# Buckets: P[V=1], P[V=2..8], P[V=9..16], P[V=17..max_variates].
# This matches our ``stage_a_single_phase.yaml::variate_count_dist`` and
# is tilted toward V=9..16 because 5 of our 8 IMM-TSF eval datasets live
# there (StudentLife=9, RepoHealth=10, CESNET=10, ILINet=11,
# ClusterTrace=11; see IMM-TSF Table 1).
_DEFAULT_VARIATE_COUNT_DIST = (0.05, 0.35, 0.55, 0.05)


@dataclass
class StackedLOTSASource(LOTSAHFSource):
    """Univariate-on-disk LOTSA source that stacks K random series per item.

    Implements Moirai's ``MultiSampleTimeSeriesDataset`` recipe (Woo et
    al., 2024, §3.2: "constructing multivariate time series from
    sub-datasets with univariate time series, by randomly concatenating
    them"; ``uni2ts/src/uni2ts/data/dataset.py:MultiSampleTimeSeriesDataset``).

    Per ``__iter__`` step:
      1.  Peek at one random record.  If it's natively multivariate
          (``target.shape[0] > 1``), pass through unchanged via the
          parent's ``__iter__``.
      2.  Otherwise stack ``stack_K = min(max_variates, num_ts)`` random
          series into ``[stack_K, T]``, truncated to the shortest series
          length.  The downstream :class:`LOTSAToIrregular._sample_n_var`
          is then the single source of truth for the model-seen V
          distribution — see README §7.N "double-sampling" note.

    Why ``stack_K = max_variates`` and not a sample from
    ``variate_count_dist``?  ``LOTSAToIrregular`` already samples the
    model-facing V from ``variate_count_dist`` capped at ``total_var``.
    If we *also* sampled the stack size from the same distribution,
    the cap would compound: model-seen V ≈ ``min(K1_dist, K2_dist)``,
    which biases V low and underrepresents bucket 3 (V=9..16) — we
    measured 35% vs target 55% in the smoke test.  Stacking
    ``max_variates`` removes the upstream cap and lets the downstream
    transform produce the requested V exactly.

    Differences vs Moirai:
      * Moirai uses ``beta_binomial(a=2, b=5, n=128)`` for K once;
        the result enters ``ConcatDataset`` directly.  Our pipeline
        always reduces to the model's ``max_dim=20`` *after* stacking,
        so we use a fixed ``stack_K = max_variates`` (== 20 by default)
        instead.  The downstream transform's bucketed
        ``variate_count_dist`` is the equivalent of Moirai's
        ``beta_binomial`` re-sampling.
      * Moirai's recipe assumes "same start and end dates" across
        stacked series.  Our normalization is per-(b,v), so series
        with different absolute calendar starts still produce a
        coherent multivariate sample after we project to
        ``timestamps ∈ [0, 1]`` per-record in the collator.  Truncating
        to ``min(T_i)`` ensures all stacked series have the same length
        without padding.

    Output schema is identical to ``LOTSAHFSource``, so this is a
    drop-in replacement.
    """

    variate_count_dist: tuple = _DEFAULT_VARIATE_COUNT_DIST  # kept for API parity
    max_variates: int = 20
    # Internal: True/False/None (None until first peek, then memoized).
    _natively_multivariate: Any = field(default=None, init=False, repr=False)

    def _check_natively_multivariate(self) -> bool:
        """Inspect a few records to decide whether stacking applies.

        Cached: only run on first ``__iter__`` call.  Memoized as
        ``self._natively_multivariate``.
        """
        if self._natively_multivariate is not None:
            return self._natively_multivariate
        # Peek up to 5 records.  All-or-nothing for these LOTSA datasets,
        # but we check a few to be defensive against single-record
        # anomalies.
        n_to_check = min(5, self._len)
        is_mv = False
        for _ in range(n_to_check):
            i = int(self._rng.integers(0, self._len))
            try:
                ex = self._indexer.getitem(i)
            except Exception:
                continue
            target = ex.get("target")
            if target is None:
                continue
            if target.ndim == 2 and target.shape[0] > 1:
                is_mv = True
                break
        self._natively_multivariate = bool(is_mv)
        return is_mv

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._len == 0:
            return

        # Decide once per worker whether this source is natively MV.
        if self._check_natively_multivariate():
            # Pass through using the parent's iterator.
            yield from super().__iter__()
            return

        # Univariate-on-disk: stack max_variates random series per item.
        # The downstream transform's _sample_n_var picks the actual
        # model-facing V from variate_count_dist (no double-sampling).
        while True:
            n_avail = self._len
            K = max(1, min(self.max_variates, n_avail))
            indices = self._rng.choice(n_avail, size=K, replace=(K > n_avail))
            tracks: list[np.ndarray] = []
            freq_seen: Optional[str] = None
            ids_seen: list[str] = []
            for j in indices:
                try:
                    ex = self._indexer.getitem(int(j))
                except Exception:
                    continue
                t = ex.get("target")
                if t is None or t.size == 0:
                    continue
                arr = t.flatten() if t.ndim == 2 else t
                if arr.size < 2:
                    continue
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                tracks.append(arr)
                if freq_seen is None:
                    freq_seen = str(ex.get("freq", "H"))
                ids_seen.append(str(ex.get("item_id", int(j))))
            if not tracks:
                continue
            min_T = min(t.size for t in tracks)
            if min_T < 2:
                continue
            stacked = np.stack([t[:min_T] for t in tracks], axis=0)
            # Replace NaN with 0 (downstream LOTSAToIrregular also does
            # this; doing it here avoids a copy later).
            if not np.all(np.isfinite(stacked)):
                stacked = np.where(np.isfinite(stacked), stacked, 0.0).astype(
                    np.float32, copy=False
                )
            yield {
                "target": stacked,
                "freq": freq_seen or "H",
                "item_id": "stk_" + "_".join(ids_seen[:3])
                + (f"+{len(ids_seen) - 3}" if len(ids_seen) > 3 else ""),
                "source_name": self.name,
            }


# ---------------------------------------------------------------------------
# Synthetic Arrow IPC sources
# ---------------------------------------------------------------------------

def _read_arrow_table(path: str) -> pa.Table:
    """Read a streaming or file-format Arrow IPC file into one Table."""
    p = str(path)
    # Try file format first (chronos2_synth uses ipc.new_file)
    try:
        with pa.OSFile(p, "rb") as src, pa.ipc.open_file(src) as r:
            return r.read_all()
    except Exception:
        with pa.OSFile(p, "rb") as src, pa.ipc.open_stream(src) as r:
            return r.read_all()


def _group_indices_by(
    table: pa.Table, key_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_keys, sorted_indices_grouped_by_key).

    ``sorted_indices_grouped_by_key`` is a [N] int64 array of row indices
    such that rows for the same key are contiguous; ``unique_keys`` is a
    [G] array; the row-range for group ``g`` is in ``np.searchsorted``
    over the corresponding key column.
    """
    keys = table.column(key_col).to_numpy()
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    uniq, starts = np.unique(sorted_keys, return_index=True)
    return uniq, order, starts


@dataclass
class ChronosSynthSource(IterableSource):
    """Long-format Arrow file from ``chronos2_synth.py``.

    Each ``group_id`` represents one (possibly multivariate) system.
    Variate index within a system is the rank of ``series_id`` inside
    that group. Timestamps within a variate are already sorted, but
    across variates the spacing/length can differ (that's the irregular
    nature of the data).

    For pretraining we materialize the system as a dense [V, T] grid:
      - take the union of all timestamps in the group
      - place each variate's values at their respective timestamps
      - leave NaN where the variate has no observation (degradation
        will pick these up as missingness)

    This grid feeds the ``LOTSAToIrregular`` transform downstream, which
    will further degrade/sample from it.
    """

    path: str
    name: str = "chronos2_synth"
    freq: str = "H"
    seed: int = 0

    _table: pa.Table = field(default=None, init=False, repr=False)
    _group_keys: np.ndarray = field(default=None, init=False, repr=False)
    _row_order: np.ndarray = field(default=None, init=False, repr=False)
    _group_starts: np.ndarray = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._table = _read_arrow_table(self.path)
        keys = self._table.column("group_id").to_numpy()
        self._row_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self._row_order]
        self._group_keys, self._group_starts = np.unique(sorted_keys, return_index=True)
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return int(self._group_keys.size)

    def set_worker(self, worker_id: int, num_workers: int, seed: int) -> None:
        self._rng = np.random.default_rng(seed + worker_id * 1009)

    def _row_range(self, gi: int) -> tuple[int, int]:
        s = int(self._group_starts[gi])
        e = (
            int(self._group_starts[gi + 1])
            if gi + 1 < len(self._group_keys)
            else int(len(self._row_order))
        )
        return s, e

    def __iter__(self) -> Iterator[dict[str, Any]]:
        n_groups = len(self._group_keys)
        if n_groups == 0:
            return
        sid_col = self._table.column("series_id").to_numpy()
        ts_col = self._table.column("timestamp").to_numpy()
        val_col = self._table.column("value").to_numpy()
        while True:
            gi = int(self._rng.integers(0, n_groups))
            s, e = self._row_range(gi)
            rows = self._row_order[s:e]
            sids = sid_col[rows]
            tss = ts_col[rows]
            vals = val_col[rows]

            uniq_sids, sid_inv = np.unique(sids, return_inverse=True)
            V = uniq_sids.size

            # Union of all timestamps -> dense [V, T] grid
            uniq_ts, ts_inv = np.unique(tss, return_inverse=True)
            T = uniq_ts.size
            if T < 2:
                continue
            grid = np.full((V, T), np.nan, dtype=np.float32)
            grid[sid_inv, ts_inv] = vals.astype(np.float32, copy=False)

            yield {
                "target": grid,
                "freq": self.freq,
                "item_id": str(int(self._group_keys[gi])),
                "source_name": self.name,
            }


@dataclass
class KernelSynthSource(IterableSource):
    """Long-format Arrow file from ``kernelsynth_irregular.py``.

    Each ``series_id`` is one univariate series. We can either yield
    each series as a univariate sample (V=1), or group ``g`` series
    from the bag together to form a synthetic multivariate sample (V=g).
    The latter encourages the model to handle "unrelated" multivariate
    inputs, which is consistent with Chronos-2's KernelSynth-MV idea.
    """

    path: str
    name: str = "kernelsynth"
    freq: str = "H"
    seed: int = 0
    multivariate_max: int = 8
    multivariate_prob: float = 0.5

    _table: pa.Table = field(default=None, init=False, repr=False)
    _series_keys: np.ndarray = field(default=None, init=False, repr=False)
    _row_order: np.ndarray = field(default=None, init=False, repr=False)
    _series_starts: np.ndarray = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._table = _read_arrow_table(self.path)
        keys = self._table.column("series_id").to_numpy()
        self._row_order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self._row_order]
        self._series_keys, self._series_starts = np.unique(sorted_keys, return_index=True)
        self._rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return int(self._series_keys.size)

    def set_worker(self, worker_id: int, num_workers: int, seed: int) -> None:
        self._rng = np.random.default_rng(seed + worker_id * 7919)

    def _row_range(self, si: int) -> tuple[int, int]:
        s = int(self._series_starts[si])
        e = (
            int(self._series_starts[si + 1])
            if si + 1 < len(self._series_keys)
            else int(len(self._row_order))
        )
        return s, e

    def _draw_one_series(
        self, si: int, ts_col: np.ndarray, val_col: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        s, e = self._row_range(si)
        rows = self._row_order[s:e]
        return ts_col[rows], val_col[rows].astype(np.float32, copy=False)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        n_series = len(self._series_keys)
        if n_series == 0:
            return
        ts_col = self._table.column("timestamp").to_numpy()
        val_col = self._table.column("value").to_numpy()
        while True:
            if self._rng.random() < self.multivariate_prob and n_series > 1:
                V = int(self._rng.integers(2, self.multivariate_max + 1))
                V = min(V, n_series)
                sis = self._rng.choice(n_series, size=V, replace=False)
            else:
                sis = np.array([int(self._rng.integers(0, n_series))])
                V = 1

            per_var_ts: list[np.ndarray] = []
            per_var_vals: list[np.ndarray] = []
            for si in sis:
                ts, vs = self._draw_one_series(int(si), ts_col, val_col)
                if len(ts) < 2:
                    break
                per_var_ts.append(ts)
                per_var_vals.append(vs)
            else:
                # Union timestamps -> dense [V, T]
                all_ts = np.concatenate(per_var_ts)
                uniq_ts = np.unique(all_ts)
                T = uniq_ts.size
                if T < 2:
                    continue
                grid = np.full((V, T), np.nan, dtype=np.float32)
                for v, (ts, vs) in enumerate(zip(per_var_ts, per_var_vals)):
                    idx = np.searchsorted(uniq_ts, ts)
                    grid[v, idx] = vs

                yield {
                    "target": grid,
                    "freq": self.freq,
                    "item_id": "_".join(
                        str(int(self._series_keys[i])) for i in sis
                    ),
                    "source_name": self.name,
                }
                continue
            # if break in for loop -> retry
            continue


# ---------------------------------------------------------------------------
# Convenience: build a list of LOTSAHFSource from a directory listing
# ---------------------------------------------------------------------------

# Sentinel: pass this to ``discover_lotsa_sources(stacking_datasets=...)``
# to enable stacking on every dataset (auto-detect at runtime via
# ``StackedLOTSASource._check_natively_multivariate``).
STACK_ALL: object = object()


def discover_lotsa_sources(
    root: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    seed: int = 0,
    stacking_datasets: Optional[Any] = STACK_ALL,
    variate_count_dist: Optional[tuple] = None,
    max_variates: int = 20,
) -> list[LOTSAHFSource]:
    """Discover LOTSA datasets under ``root`` and wrap each in a source.

    Each subdirectory of ``root`` containing a ``dataset_info.json`` is
    treated as one HF dataset. ``include``/``exclude`` filter by name.

    Stacking policy (``stacking_datasets``):

    * :data:`STACK_ALL` (default): every dataset is wrapped in
      :class:`StackedLOTSASource`.  Stacking only fires for datasets
      that are univariate-on-disk; natively-multivariate datasets are
      passed through unchanged via the
      ``_check_natively_multivariate`` short-circuit.  This is more
      aggressive than Moirai (whose curated list misses many univariate
      LOTSA datasets like ``alibaba_cluster_trace_2018``,
      ``favorita_sales``, etc.).  It directly fixes the V=1 over-
      representation we observed in the smoke test (see README §7.N).
    * Sequence of names: only those datasets are wrapped; others use
      plain :class:`LOTSAHFSource`.  Pass
      :data:`MOIRAI_STACKING_DATASETS` to reproduce Moirai's exact
      behavior.
    * ``None`` or empty sequence: stacking disabled entirely.

    Args:
        root: LOTSA root directory.
        include / exclude: dataset name filters.
        seed: base seed for per-source RNGs.
        stacking_datasets: see policy above.
        variate_count_dist: P[V] bucketing for stacked sources.
            Defaults to the same distribution as ``task_sampler.py``.
        max_variates: hard cap on stacked V (matches ``collate.max_dim``).
    """
    if variate_count_dist is None:
        variate_count_dist = _DEFAULT_VARIATE_COUNT_DIST

    if stacking_datasets is STACK_ALL:
        stack_all = True
        stacking_set: set[str] = set()
    elif stacking_datasets is None or len(stacking_datasets) == 0:
        stack_all = False
        stacking_set = set()
    else:
        stack_all = False
        stacking_set = set(stacking_datasets)

    out: list[LOTSAHFSource] = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        if not os.path.exists(os.path.join(sub, "dataset_info.json")):
            continue
        if include is not None and entry not in include:
            continue
        if exclude is not None and entry in exclude:
            continue
        try:
            if stack_all or entry in stacking_set:
                out.append(
                    StackedLOTSASource(
                        path=sub,
                        name=f"lotsa:{entry}",
                        seed=seed,
                        variate_count_dist=tuple(variate_count_dist),
                        max_variates=max_variates,
                    )
                )
            else:
                out.append(LOTSAHFSource(path=sub, name=f"lotsa:{entry}", seed=seed))
        except Exception as e:
            print(f"[discover_lotsa_sources] skipping {entry}: {e}")
    return out
