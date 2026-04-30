"""Stage-weighted multi-source IterableDataset.

Two-level weighted sampling per yielded example:
  1. **Logical source** (e.g. ``chronos2_synth``, ``lotsa_degraded``)
     drawn by per-stage mix weights.
  2. **Physical source** within that logical source — for the LOTSA
     groups, this is sampled by the per-dataset ``weight_map``
     (Moirai-style dataset capping, reusing
     ``uni2ts/cli/conf/pretrain/data/lotsa_v1_weighted.yaml``).

The chosen physical source's iterator yields one raw entry, the
logical source's :class:`LOTSAToIrregular` transforms it, and the
result is yielded. Empty entries are filtered.

Designed to run with multiple PyTorch DataLoader workers — each worker
gets its own RNG and re-seeds every physical source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .sources import IterableSource
from ..synthetic_data.degradation import LOTSAToIrregular


@dataclass
class LogicalSource:
    """A named bundle of physical sources sharing a single transform.

    For LOTSA, ``physical_sources`` is the per-dataset list (e.g.
    one ``LOTSAHFSource`` per LOTSA arrow). For synthetic, it is
    typically a single source.

    ``physical_weights`` is normalized to a probability distribution
    over ``physical_sources``.
    """

    name: str
    physical_sources: list[IterableSource]
    physical_weights: np.ndarray
    transform: LOTSAToIrregular

    def __post_init__(self):
        w = np.asarray(self.physical_weights, dtype=np.float64)
        if w.size != len(self.physical_sources):
            raise ValueError(
                f"[{self.name}] weights size {w.size} != #sources "
                f"{len(self.physical_sources)}"
            )
        s = w.sum()
        if s <= 0:
            raise ValueError(f"[{self.name}] all weights zero")
        self.physical_weights = w / s

    def num_sources(self) -> int:
        return len(self.physical_sources)


def make_logical_source(
    name: str,
    physical_sources: list[IterableSource],
    transform: LOTSAToIrregular,
    weight_map: Optional[dict[str, float]] = None,
) -> LogicalSource:
    """Build a LogicalSource, looking up per-source weights from
    ``weight_map`` (keyed by ``IterableSource.name`` after the
    ``"lotsa:"`` prefix is stripped). Sources not in the map fall back
    to weight 1.0.

    **Moirai weighting (Woo et al., 2024, §3.2)**

    Moirai's published ``lotsa_v1_weighted.yaml`` lists per-series
    multipliers, NOT per-dataset probabilities.  Their effective dataset
    sampling probability is

        p(D_k) ∝ num_ts_k × yaml_weight_k

    because their ``ConcatDataset(TimeSeriesDataset(__len__ =
    num_ts × dataset_weight))`` samples uniformly over indices.  The
    per-series multiplier is precomputed so that ``num_ts × yaml_weight
    = ω_k`` where ``ω_k = min(num_observations_k / total_obs, ε)`` for
    ``ε = 0.001`` — that is, the cap is *baked into the YAML*.

    Treating ``yaml_weight`` directly as the sampling probability (our
    pre-2026-04-29 behavior) is wrong.  It gave ``solar_power`` (1
    series, weight 33,835) and ``wind_power`` (same) a combined ~90 %
    of LOTSA mass, vs Moirai's intended ~2.2 % combined.

    Fix: weight by ``len(source) × yaml_weight`` so the resulting
    sampling probability equals Moirai's ``ω_k_capped``.  See README
    §7.M / §7.N.
    """
    if not physical_sources:
        raise ValueError(f"[{name}] no physical sources")
    if weight_map is None:
        # No weight map -> equal mass per dataset (sub-dataset uniform).
        # We do NOT multiply by num_ts here because the user explicitly
        # opted out of any prior over datasets.
        w = np.ones(len(physical_sources), dtype=np.float64)
    else:
        # Moirai-faithful weighting: yaml_weight is a per-series
        # multiplier; multiply by num_ts to get the dataset's
        # contribution to the (capped) sampling probability.
        w = np.array(
            [
                _lookup_weight(s.name, weight_map) * max(1, len(s))
                for s in physical_sources
            ],
            dtype=np.float64,
        )
        if (w == 0).all():
            w = np.ones_like(w)
    return LogicalSource(
        name=name,
        physical_sources=physical_sources,
        physical_weights=w,
        transform=transform,
    )


def _lookup_weight(source_name: str, weight_map: dict[str, float]) -> float:
    if source_name in weight_map:
        return float(weight_map[source_name])
    short = source_name.split(":", 1)[-1]
    if short in weight_map:
        return float(weight_map[short])
    return 1.0


class MixedTorchDataset(IterableDataset):
    """Stage-weighted mix of LogicalSources, transform-applied.

    Yields the per-variate ragged dict produced by
    ``LOTSAToIrregular``. Filters out empty entries (from degenerate
    series or transforms that produced no observations).

    Multi-worker-safe: each worker sets a unique seed on every
    physical source.
    """

    def __init__(
        self,
        logical_sources: list[LogicalSource],
        logical_weights: dict[str, float],
        seed: int = 0,
        max_retries_per_step: int = 8,
    ):
        super().__init__()
        if not logical_sources:
            raise ValueError("logical_sources must not be empty")
        self.logical_sources = logical_sources
        names = [ls.name for ls in logical_sources]
        for n in logical_weights:
            if n not in names:
                raise ValueError(
                    f"weight given for unknown logical source '{n}'. "
                    f"known={names}"
                )
        w = np.array(
            [float(logical_weights.get(n, 0.0)) for n in names],
            dtype=np.float64,
        )
        s = w.sum()
        if s <= 0:
            raise ValueError(f"logical_weights sum to {s}; need positive")
        self.logical_weights = w / s
        self.seed = int(seed)
        self.max_retries_per_step = int(max_retries_per_step)

    def _setup_worker(self) -> tuple[np.random.Generator, list[Iterator]]:
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers
        worker_seed = self.seed + worker_id * 31337

        # seed each physical source so worker iterators don't lock-step
        for ls in self.logical_sources:
            for k, src in enumerate(ls.physical_sources):
                src.set_worker(worker_id, num_workers, worker_seed + k * 5701)

        # build one infinite iterator per (logical, physical) source
        iterators: list[list[Iterator]] = []
        for ls in self.logical_sources:
            iterators.append([iter(src) for src in ls.physical_sources])

        rng = np.random.default_rng(worker_seed)
        return rng, iterators

    def __iter__(self):
        rng, iterators = self._setup_worker()
        n_logical = len(self.logical_sources)
        while True:
            li = int(rng.choice(n_logical, p=self.logical_weights))
            ls = self.logical_sources[li]
            pi = int(rng.choice(ls.num_sources(), p=ls.physical_weights))
            it = iterators[li][pi]
            for _ in range(self.max_retries_per_step):
                try:
                    raw = next(it)
                except StopIteration:
                    iterators[li][pi] = iter(ls.physical_sources[pi])
                    continue
                try:
                    out = ls.transform(raw)
                except Exception:
                    continue
                if out.get("_empty"):
                    continue
                if out["n_obs_per_var"].sum() < 4:
                    continue
                out["logical_source"] = ls.name
                yield out
                break


# ---------------------------------------------------------------------------
# Helpers for loading uni2ts's weight map yaml
# ---------------------------------------------------------------------------

def load_lotsa_weight_map(path: str) -> dict[str, float]:
    """Flatten the uni2ts weighted yaml into a single
    ``{dataset_name: weight}`` dict. Other yaml structure is ignored.

    The yaml has shape ``[{ _target_, datasets, weight_map: {...} }, ...]``;
    we union all the inner ``weight_map``s.
    """
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    out: dict[str, float] = {}
    items = cfg.get("_args_", []) if isinstance(cfg, dict) else cfg
    for entry in items:
        wm = entry.get("weight_map", {}) if isinstance(entry, dict) else {}
        for k, v in wm.items():
            out[k] = float(v)
    return out
