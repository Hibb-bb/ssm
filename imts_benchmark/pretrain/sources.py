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

def discover_lotsa_sources(
    root: str,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> list[LOTSAHFSource]:
    """Discover LOTSA datasets under ``root`` and wrap each in a source.

    Each subdirectory of ``root`` containing a ``dataset_info.json`` is
    treated as one HF dataset. ``include``/``exclude`` filter by name.
    """
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
            out.append(LOTSAHFSource(path=sub, name=f"lotsa:{entry}", seed=seed))
        except Exception as e:
            print(f"[discover_lotsa_sources] skipping {entry}: {e}")
    return out
