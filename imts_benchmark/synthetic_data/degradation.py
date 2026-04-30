"""Bridge transform: convert (regular or already-irregular) multivariate
time series into the irregular format consumed by the pretraining
pipeline.

Used uniformly across:
  - regular LOTSA series (apply degradation augmentations to make them
    look irregular),
  - synthetic series produced by ``chronos2_synth.py`` and
    ``kernelsynth_irregular.py`` (already irregular by construction;
    augmentations add extra missingness / asynchrony).

Output format (one entry per call) — consumed by
``imts_benchmark.pretrain.collate``:

    {
        "values_per_var":     list[np.ndarray (float32)] of len V,
        "timestamps_per_var": list[np.ndarray (float32)] of len V,
                                  normalized to [0, 1] over the sample
                                  window
        "deltat_per_var":     list[np.ndarray (float32)] of len V,
                                  also normalized to [0, 1]; first entry
                                  per variate is 0
        "n_obs_per_var":      np.ndarray (int32) of shape (V,),
        "target_variate_mask": np.ndarray (bool) of shape (V,),
                                  True = variate is a forecast target
        "history":            float in [0, 1] (normalized),
        "time_scale":         float, original window duration in hours,
                                  preserved so the model can condition
                                  on absolute scale,
        "freq":               str (pandas freq string),
        "source_tag":         str (e.g. "lotsa:PEMS04",
                                  "chronos2_synth", "kernelsynth"),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ._base import Transformation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_param(val: Union[float, tuple, list]) -> float:
    """Sample a scalar from a value or (min, max) range."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(np.random.uniform(val[0], val[1]))
    return float(val)


def _freq_to_hours(freq: str) -> float:
    """Convert a pandas freq string to hours.

    Falls back to 1.0h on unknown frequencies so we never crash on
    obscure LOTSA freq strings; pretraining can still sample windows,
    they just inherit a generic time scale.
    """
    if freq is None:
        return 1.0
    _COMPAT = {"H": "h", "T": "min", "S": "s"}
    f = _COMPAT.get(freq, freq)
    try:
        return pd.Timedelta(1, unit=f).total_seconds() / 3600.0
    except Exception:
        try:
            return pd.Timedelta(pd.tseries.frequencies.to_offset(f)).total_seconds() / 3600.0
        except Exception:
            return 1.0


# ---------------------------------------------------------------------------
# Augmentations: each takes (n_var, time_len, timestamps_norm, target)
# and returns a [n_var, time_len] bool drop-mask (True = drop).
# ---------------------------------------------------------------------------

@dataclass
class RandomDrops:
    """Randomly drop a fraction of observations.

    sync_variates=True: same indices dropped across all variates
                        (sync irregular).
    sync_variates=False: each variate gets its own dropped indices
                        (async irregular).
    """

    drop_fraction: Union[float, tuple, list] = (0.1, 0.5)
    sync_variates: bool = True

    def __call__(self, n_var, time_len, timestamps_norm, target):
        frac = _sample_param(self.drop_fraction)
        n_drop = min(int(round(frac * time_len)), max(time_len - 1, 0))
        mask = np.zeros((n_var, time_len), dtype=bool)
        if n_drop <= 0 or time_len == 0:
            return mask
        if self.sync_variates:
            idx = np.random.choice(time_len, size=n_drop, replace=False)
            mask[:, idx] = True
        else:
            for v in range(n_var):
                idx = np.random.choice(time_len, size=n_drop, replace=False)
                mask[v, idx] = True
        return mask


@dataclass
class RegularGaps:
    """Periodically drop windows of observations.

    Frequency-invariant: parameters are *fractions of the window*, not
    absolute hours. ``period_fraction`` controls the spacing of the
    dropped intervals; ``gap_duration_fraction`` controls each interval's
    width. Both are in [0, 1] of the segment length.

    sync_variates=True: drops the same intervals across all variates.
    sync_variates=False: each interval picks one variate to drop.
    """

    gap_duration_fraction: Union[float, tuple, list] = (0.05, 0.20)
    period_fraction: Union[float, tuple, list] = (0.20, 0.60)
    sync_variates: bool = True

    def __call__(self, n_var, time_len, timestamps_norm, target):
        if time_len <= 1:
            return np.zeros((n_var, time_len), dtype=bool)

        gap = _sample_param(self.gap_duration_fraction)
        period = _sample_param(self.period_fraction)
        if period <= gap:
            period = min(1.0, gap * 2.0)

        offset = float(np.random.uniform(0.0, period))
        mask = np.zeros((n_var, time_len), dtype=bool)
        t = timestamps_norm  # normalized to [0, 1]

        k = 0
        while True:
            g_start = offset + k * period
            g_end = g_start + gap
            if g_start >= 1.0:
                break
            in_gap = (t >= g_start) & (t < g_end)
            if self.sync_variates:
                mask[:] |= in_gap[None, :]
            else:
                v = int(np.random.randint(0, n_var))
                mask[v] |= in_gap
            k += 1

        return mask


@dataclass
class ThresholdCutoff:
    """Drop observations above/below a per-variate quantile threshold.

    Caps total drop fraction per variate at ``max_drop_fraction`` so a
    rare extreme variate doesn't lose its entire history.
    """

    quantile: Union[float, tuple, list] = (0.85, 0.95)
    mode: str = "upper"
    max_drop_fraction: float = 0.4
    sync_variates: bool = True

    def __call__(self, n_var, time_len, timestamps_norm, target):
        q = _sample_param(self.quantile)
        mask = np.zeros((n_var, time_len), dtype=bool)
        if time_len == 0:
            return mask

        if self.sync_variates:
            cutoff = float(np.nanquantile(target.ravel(), q))
            mask[:] = (target > cutoff) if self.mode == "upper" else (target < cutoff)
        else:
            for v in range(n_var):
                cutoff = float(np.nanquantile(target[v], q))
                mask[v] = (target[v] > cutoff) if self.mode == "upper" else (target[v] < cutoff)

        for v in range(n_var):
            drop_frac = mask[v].sum() / max(time_len, 1)
            if drop_frac > self.max_drop_fraction:
                drop_idx = np.where(mask[v])[0]
                n_allowed = int(self.max_drop_fraction * time_len)
                keep = np.random.choice(
                    drop_idx, size=max(0, len(drop_idx) - n_allowed), replace=False
                )
                mask[v, keep] = False
        return mask


@dataclass
class StartDelay:
    """Per-variate prefix drop.

    For each variate independently, drop the first
    ``u ~ Uniform(*max_prefix_fraction)`` of the segment. Models the
    common real-world case where one sensor starts logging much later
    than another.

    Restricted to context region (timestamps_norm < history) by the
    caller; here we just produce the mask over the full segment.
    """

    max_prefix_fraction: Union[float, tuple, list] = (0.0, 0.5)
    apply_prob: float = 0.6

    def __call__(self, n_var, time_len, timestamps_norm, target):
        mask = np.zeros((n_var, time_len), dtype=bool)
        if time_len <= 1:
            return mask
        for v in range(n_var):
            if np.random.random() > self.apply_prob:
                continue
            u = _sample_param(self.max_prefix_fraction)
            cut_idx = int(round(u * time_len))
            if cut_idx > 0:
                mask[v, :cut_idx] = True
        return mask


@dataclass
class TimestampJitter:
    """Per-variate Gaussian jitter on retained timestamps.

    Output: a *post-processor* that perturbs timestamps after drop
    masks have been applied; expressed as a return value rather than a
    mask, since it modifies timestamps_per_var in place.

    jitter_fraction: stddev as a fraction of the per-variate median
        spacing; sample uniformly in this range. After perturbation,
        timestamps within each variate are re-sorted.
    """

    jitter_fraction: Union[float, tuple, list] = (0.1, 0.3)
    apply_prob: float = 0.5

    def __call__(self, ts_per_var: list[np.ndarray], vals_per_var: list[np.ndarray]):
        if np.random.random() > self.apply_prob:
            return ts_per_var, vals_per_var
        out_ts = []
        out_vals = []
        scale = _sample_param(self.jitter_fraction)
        for ts, vals in zip(ts_per_var, vals_per_var):
            n = len(ts)
            if n < 2:
                out_ts.append(ts)
                out_vals.append(vals)
                continue
            base_dt = float(np.median(np.diff(ts)))
            sigma = scale * base_dt
            noise = np.random.normal(0.0, sigma, size=n).astype(ts.dtype)
            jittered = np.clip(ts + noise, 0.0, 1.0 - 1e-6).astype(ts.dtype)
            order = np.argsort(jittered)
            out_ts.append(jittered[order])
            out_vals.append(vals[order])
        return out_ts, out_vals


@dataclass
class ResampleNoise:
    """Multiplicative Gaussian noise on values."""

    uncertainty: Union[float, tuple, list] = 0.05

    def __call__(self, target):
        frac = _sample_param(self.uncertainty)
        per_point_std = np.clip(frac * np.abs(target), a_min=1e-8, a_max=None)
        return target + np.random.normal(0, 1, size=target.shape) * per_point_std


# ---------------------------------------------------------------------------
# Regime presets — irregularity dialed by sampling across regimes.
# Each entry returns a list of (augmentation, kwargs) to compose for
# context-only degradation. ``noise`` is a separate per-sample noise op.
# ---------------------------------------------------------------------------

REGIMES = {
    "regular": {
        "augmentations": [],   # no drops
        "noise": ResampleNoise(uncertainty=(0.0, 0.02)),
        "jitter": None,
    },
    "sync": {
        "augmentations": [
            ("random_drops_sync", RandomDrops(drop_fraction=(0.05, 0.20), sync_variates=True)),
        ],
        "noise": ResampleNoise(uncertainty=(0.0, 0.05)),
        "jitter": None,
    },
    "mixed": {
        "augmentations": [
            ("random_drops_sync",  RandomDrops(drop_fraction=(0.05, 0.15), sync_variates=True)),
            ("random_drops_async", RandomDrops(drop_fraction=(0.10, 0.30), sync_variates=False)),
            ("regular_gaps_async", RegularGaps(
                gap_duration_fraction=(0.05, 0.20),
                period_fraction=(0.20, 0.60),
                sync_variates=False,
            )),
            # ThresholdCutoff used rarely; informative-missingness signal.
            ("threshold",          ThresholdCutoff(quantile=(0.85, 0.95), mode="upper")),
        ],
        # In mixed, we apply 1-2 augmentations sampled from this list
        # with ThresholdCutoff probability capped at ~10%.
        "noise": ResampleNoise(uncertainty=(0.0, 0.05)),
        "jitter": TimestampJitter(jitter_fraction=(0.05, 0.15), apply_prob=0.3),
    },
    "async": {
        # NOTE: ``StartDelay`` was removed on 2026-04-28 after the
        # diagnostics in README §Q4 / §start_delay_removal.  StartDelay
        # could drop up to 50% of the per-variate prefix on top of
        # RandomDrops + RegularGaps, leaving as few as 3 context obs
        # for "good" samples (= just barely passing the old
        # min_ctx_obs_per_target=3 floor).  Removing it makes async still
        # ~50% per-variate worst case (RandomDrops 0.20-0.50 +
        # RegularGaps ~50%), but no longer catastrophic.
        "augmentations": [
            ("random_drops_async", RandomDrops(drop_fraction=(0.20, 0.50), sync_variates=False)),
            ("regular_gaps_async", RegularGaps(
                gap_duration_fraction=(0.05, 0.30),
                period_fraction=(0.20, 0.60),
                sync_variates=False,
            )),
        ],
        "noise": ResampleNoise(uncertainty=(0.0, 0.05)),
        "jitter": TimestampJitter(jitter_fraction=(0.10, 0.30), apply_prob=0.5),
    },
}


def _apply_regime_mask(
    regime_name: str,
    n_var: int,
    time_len: int,
    timestamps_norm: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compose a context drop-mask from a regime's augmentation list.

    Sampling rule:
      - regular: empty mask.
      - sync:    apply the single augmentation in the list.
      - mixed:   apply 1 sync drop + 1 async/gap aug; threshold ~10%.
      - async:   apply ALL listed augmentations.
    """
    cfg = REGIMES[regime_name]
    augs = cfg["augmentations"]
    mask = np.zeros((n_var, time_len), dtype=bool)
    if not augs or time_len == 0 or n_var == 0:
        return mask

    if regime_name == "regular":
        return mask

    if regime_name == "sync":
        _, aug = augs[0]
        return mask | aug(n_var, time_len, timestamps_norm, target)

    if regime_name == "mixed":
        # one sync drop + one async/gap aug + 10% chance of threshold
        sync_aug = next(a for n, a in augs if n == "random_drops_sync")
        mask |= sync_aug(n_var, time_len, timestamps_norm, target)
        async_choice = augs[np.random.randint(1, 3)][1]  # random_drops_async or regular_gaps_async
        mask |= async_choice(n_var, time_len, timestamps_norm, target)
        if np.random.random() < 0.10:
            thresh = next(a for n, a in augs if n == "threshold")
            mask |= thresh(n_var, time_len, timestamps_norm, target)
        return mask

    if regime_name == "async":
        for _, aug in augs:
            mask |= aug(n_var, time_len, timestamps_norm, target)
        return mask

    raise ValueError(f"unknown regime '{regime_name}'")


# ---------------------------------------------------------------------------
# Main bridge transform
# ---------------------------------------------------------------------------

@dataclass
class LOTSAToIrregular(Transformation):
    """Convert a (multi-variate) regular series into the irregular
    pretraining format.

    Behaviour:
      1. Optionally subsample variates to ``max_variates`` (will be
         further capped by the per-sample task sampler).
      2. Crop a window of length ``seg_len ~ U(min_length, max_length)``.
      3. Build normalized timestamps ``t in [0, 1]`` and per-variate
         delta-t (also in [0, 1]).
      4. Apply value noise globally.
      5. Apply per-regime context-region degradation (drops). If
         ``clean_target_mode=True``, drops are restricted to the context
         region (t < history); horizon observations are kept clean.
      6. Apply timestamp jitter (per regime).
      7. Enforce per-variate observation minima separately for context
         and horizon (different rules for target vs auxiliary variates).
      8. Sample ``target_variate_mask`` from one of three task types:
         univariate, all-target, partial-target.

    Sampling parameters are typically *overridden* per-call by the task
    sampler in ``imts_benchmark.pretrain.task_sampler`` — this dataclass
    sets sensible defaults that work standalone too.
    """

    min_length: int = 64
    max_length: int = 512
    max_variates: int = 20
    context_fraction: tuple = (0.5, 0.95)
    tail_prediction_prob: float = 0.05
    tail_prediction_range: tuple = (0.92, 0.98)

    # observation minima
    min_ctx_obs_per_target: int = 3
    min_pred_obs_per_target: int = 1
    min_ctx_obs_per_aux: int = 1

    # task type distribution (univariate / all-target / partial-target)
    task_type_probs: tuple = (0.20, 0.50, 0.30)

    # variate count distribution: (P[V=1], P[V in 2..8], P[V in 9..16], P[V in 17..max_variates])
    variate_count_dist: tuple = (0.10, 0.60, 0.25, 0.05)

    # irregularity regime distribution (regular / sync / mixed / async)
    regime_dist: tuple = (0.15, 0.15, 0.45, 0.25)

    # whether to keep the clean future as supervision (recommended for
    # degraded LOTSA; harmless for synthetic).
    clean_target_mode: bool = True

    # Whether timestamp jitter is allowed for this source. We default to
    # True because synthetic sources benefit from it; set False on real
    # data (LOTSA) to avoid distorting accurately-recorded observation
    # times. See pretrain/README.md §3.3 for the discussion.
    apply_jitter: bool = True

    source_tag_prefix: str = "lotsa"

    def __post_init__(self):
        if abs(sum(self.task_type_probs) - 1.0) > 1e-3:
            raise ValueError("task_type_probs must sum to 1")
        if abs(sum(self.variate_count_dist) - 1.0) > 1e-3:
            raise ValueError("variate_count_dist must sum to 1")
        if abs(sum(self.regime_dist) - 1.0) > 1e-3:
            raise ValueError("regime_dist must sum to 1")

    # ---- helpers ----------------------------------------------------

    def _sample_n_var(self, total_var: int) -> int:
        """Stratified discrete distribution over V; capped at total_var."""
        cap = min(total_var, self.max_variates)
        bucket = int(np.random.choice(4, p=np.asarray(self.variate_count_dist)))
        if bucket == 0:
            n = 1
        elif bucket == 1:
            lo, hi = 2, min(8, cap)
            n = int(np.random.randint(lo, hi + 1)) if hi >= lo else lo
        elif bucket == 2:
            lo, hi = 9, min(16, cap)
            n = int(np.random.randint(lo, hi + 1)) if hi >= lo else lo
        else:
            lo, hi = 17, cap
            n = int(np.random.randint(lo, hi + 1)) if hi >= lo else lo
        return max(1, min(n, cap))

    def _sample_history_norm(self) -> float:
        """Sample history fraction in [0, 1]."""
        if np.random.random() < self.tail_prediction_prob:
            return _sample_param(self.tail_prediction_range)
        return _sample_param(self.context_fraction)

    def _sample_target_mask(self, n_var: int) -> np.ndarray:
        """Choose which variates are forecast targets (rest are aux)."""
        if n_var == 1:
            return np.array([True], dtype=bool)
        task_type = int(np.random.choice(3, p=np.asarray(self.task_type_probs)))
        if task_type == 0:  # univariate target
            mask = np.zeros(n_var, dtype=bool)
            mask[np.random.randint(n_var)] = True
            return mask
        if task_type == 1:  # all-target
            return np.ones(n_var, dtype=bool)
        # partial-target: between 1 and n_var-1 inclusive
        n_targets = int(np.random.randint(1, n_var))
        idx = np.random.choice(n_var, size=n_targets, replace=False)
        mask = np.zeros(n_var, dtype=bool)
        mask[idx] = True
        return mask

    # ---- core call --------------------------------------------------

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target_in = data_entry["target"]
        freq = data_entry.get("freq", "H")
        item_id = data_entry.get("item_id", "")

        # Stack to [V, T] regardless of input shape (univariate or list).
        if isinstance(target_in, list):
            target_2d = np.stack([np.asarray(t, dtype=np.float32) for t in target_in])
        else:
            target_2d = np.asarray(target_in, dtype=np.float32)
            if target_2d.ndim == 1:
                target_2d = target_2d[None, :]

        total_var, total_len = target_2d.shape
        n_var = self._sample_n_var(total_var)
        if n_var < total_var:
            chosen = np.random.choice(total_var, size=n_var, replace=False)
            target_2d = target_2d[chosen]
        else:
            n_var = total_var

        # Crop temporal window
        seg_max = min(self.max_length, total_len)
        seg_min = min(self.min_length, seg_max)
        if seg_max < 2:
            # Degenerate series: nothing to do, return empty entry that
            # the caller can filter.
            return self._empty(freq, item_id, source_tag=f"{self.source_tag_prefix}:empty")
        seg_len = int(np.random.randint(seg_min, seg_max + 1))
        start_idx = int(np.random.randint(0, max(total_len - seg_len + 1, 1)))
        target_crop = target_2d[:, start_idx : start_idx + seg_len].astype(np.float32, copy=True)

        # Replace NaNs with 0 in inputs (LOTSA has missing values for some
        # series); we still let the mask flow.
        nan_mask = ~np.isfinite(target_crop)
        if nan_mask.any():
            target_crop = np.where(nan_mask, 0.0, target_crop)

        # Normalized timestamps
        freq_hours = _freq_to_hours(freq)
        ts_norm = np.linspace(0.0, 1.0, seg_len, dtype=np.float32, endpoint=False)
        time_scale = float(seg_len * freq_hours)  # window duration in hours

        # Sample task setup
        regime = ["regular", "sync", "mixed", "async"][
            int(np.random.choice(4, p=np.asarray(self.regime_dist)))
        ]
        history_norm = self._sample_history_norm()
        target_variate_mask = self._sample_target_mask(n_var)

        # 1) Value noise (always)
        noise_op = REGIMES[regime]["noise"]
        if noise_op is not None:
            target_crop = noise_op(target_crop)

        # 2) Drop mask, restricted to context region if clean_target_mode
        drop_mask = _apply_regime_mask(
            regime, n_var, seg_len, ts_norm, target_crop
        )
        if self.clean_target_mode:
            ctx_region = ts_norm < history_norm
            drop_mask = drop_mask & ctx_region[None, :]

        # 3) Inherit existing missingness (NaNs in source) into the drop
        #    mask so we never emit fake observations for missing data.
        drop_mask |= nan_mask

        # 4) Enforce per-variate minima before splitting / outputting.
        #    Only "augmentation drops" are restorable — i.e. positions
        #    that were NOT NaN at the source. Restoring a NaN position
        #    would emit a fake zero observation as supervision.
        ctx_idx_set = np.where(ts_norm < history_norm)[0]
        pred_idx_set = np.where(ts_norm >= history_norm)[0]
        restorable = drop_mask & ~nan_mask  # [V, L]
        for v in range(n_var):
            is_target = bool(target_variate_mask[v])
            min_ctx = self.min_ctx_obs_per_target if is_target else self.min_ctx_obs_per_aux
            min_pred = self.min_pred_obs_per_target if is_target else 0

            ctx_kept = (~drop_mask[v, ctx_idx_set]).sum()
            if ctx_kept < min_ctx and len(ctx_idx_set) > 0:
                drops_in_ctx = np.where(restorable[v, ctx_idx_set])[0]
                if len(drops_in_ctx) > 0:
                    n_restore = min(min_ctx - int(ctx_kept), len(drops_in_ctx))
                    restore = np.random.choice(drops_in_ctx, size=n_restore, replace=False)
                    drop_mask[v, ctx_idx_set[restore]] = False

            if is_target and min_pred > 0 and len(pred_idx_set) > 0:
                pred_kept = (~drop_mask[v, pred_idx_set]).sum()
                if pred_kept < min_pred:
                    drops_in_pred = np.where(restorable[v, pred_idx_set])[0]
                    if len(drops_in_pred) > 0:
                        n_restore = min(min_pred - int(pred_kept), len(drops_in_pred))
                        restore = np.random.choice(drops_in_pred, size=n_restore, replace=False)
                        drop_mask[v, pred_idx_set[restore]] = False

        # 5) Split into per-variate ragged lists
        values_per_var: list[np.ndarray] = []
        timestamps_per_var: list[np.ndarray] = []
        n_obs_per_var = np.zeros(n_var, dtype=np.int32)
        for v in range(n_var):
            keep = ~drop_mask[v]
            vals = target_crop[v, keep].astype(np.float32, copy=True)
            ts = ts_norm[keep].astype(np.float32, copy=True)
            values_per_var.append(vals)
            timestamps_per_var.append(ts)
            n_obs_per_var[v] = len(vals)

        # 6) Timestamp jitter (post-drop, so we don't disturb context
        #    masking for the regime). Disabled per-source via
        #    ``apply_jitter=False`` for LOTSA (real-data observation
        #    times are not ambiguous downstream).
        jitter_op = REGIMES[regime].get("jitter") if self.apply_jitter else None
        if jitter_op is not None:
            timestamps_per_var, values_per_var = jitter_op(
                timestamps_per_var, values_per_var
            )

        # 7) Per-variate delta-t (also normalized)
        deltat_per_var: list[np.ndarray] = []
        for ts in timestamps_per_var:
            dt = np.zeros(len(ts), dtype=np.float32)
            if len(ts) > 1:
                dt[1:] = np.diff(ts)
            deltat_per_var.append(dt)

        return {
            "values_per_var": values_per_var,
            "timestamps_per_var": timestamps_per_var,
            "deltat_per_var": deltat_per_var,
            "n_obs_per_var": n_obs_per_var,
            "target_variate_mask": target_variate_mask,
            "history": float(history_norm),
            "time_scale": float(time_scale),
            "freq": str(freq),
            "source_tag": f"{self.source_tag_prefix}:{item_id}" if item_id else self.source_tag_prefix,
            "regime": regime,
        }

    @staticmethod
    def _empty(freq, item_id, source_tag) -> dict[str, Any]:
        """Empty entry; collate filters these out."""
        return {
            "values_per_var": [np.zeros(0, dtype=np.float32)],
            "timestamps_per_var": [np.zeros(0, dtype=np.float32)],
            "deltat_per_var": [np.zeros(0, dtype=np.float32)],
            "n_obs_per_var": np.zeros(1, dtype=np.int32),
            "target_variate_mask": np.array([True], dtype=bool),
            "history": 0.5,
            "time_scale": 1.0,
            "freq": str(freq),
            "source_tag": source_tag,
            "regime": "regular",
            "_empty": True,
        }
