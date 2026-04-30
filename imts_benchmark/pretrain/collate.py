"""Collate function for the multivariate Mamba pretraining pipeline.

Inputs are per-variate ragged dicts produced by
``LOTSAToIrregular``::

    {
        "values_per_var":     list[np.ndarray] of len V_active,
        "timestamps_per_var": list[np.ndarray] of len V_active (in [0, 1]),
        "deltat_per_var":     list[np.ndarray] of len V_active (in [0, 1]),
        "n_obs_per_var":      np.ndarray (int32) of shape (V_active,),
        "target_variate_mask": np.ndarray (bool)  of shape (V_active,),
        "history":            float in [0, 1],
        "time_scale":         float (window duration in source units),
        "freq":               str,
        "source_tag":         str,
        ...
    }

Outputs the dense batch the
``MultivariateMambaForecaster.forward`` method consumes, plus a
``valid_variate_mask`` (and ``target_variate_mask``) so the refactored
model can ignore padded variates::

    {
        "values":              [B, V_pad, L_max]   float32
        "timestamps":          [B, V_pad, L_max]   float32 in [0, 1]
        "deltat":              [B, V_pad, L_max]   float32 in [0, 1]
        "valid_mask":          [B, V_pad, L_max]   bool
        "pred_mask":           [B, V_pad, L_max]   bool
        "valid_variate_mask":  [B, V_pad]          bool
        "target_variate_mask": [B, V_pad]          bool
        "history":             [B]                 float32
        "time_scale":          [B]                 float32
        "n_obs_per_var":       [B, V_pad]          int32
        "var_slot":            [B, V_pad]          int32
                                  (which physical variate from the
                                   sample lives in this slot, -1 if pad)
        "item_id":             list[str]
        "source_name":         list[str]
        "regime":              list[str]
    }

Two non-trivial steps:
  1. **Variate slot randomization**: each sample's V_active active
     variates are placed at random slots inside V_pad; padded slots get
     ``valid_variate_mask = False``. This decouples the model's variate
     embeddings from "the first variate is always 0", which Moirai
     showed is necessary for a pretraining FM.
  2. **Per-sample value standardization**: per (b, v), values are
     standardized using mean/std of *context-only* observations
     (``timestamps < history``). Standardization with at least 3
     context observations; otherwise zero-mean / unit-std fallback.
     This is essential when mixing datasets at vastly different scales
     (LOTSA wind_power vs. cmip6 vs. synthetic).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch


def _stack_pad(
    arrays: list[np.ndarray], pad_to: int, dtype=np.float32, fill: float = 0.0
) -> np.ndarray:
    """Right-pad each 1D array to ``pad_to`` length, stack to [n, pad_to]."""
    out = np.full((len(arrays), pad_to), fill, dtype=dtype)
    for i, a in enumerate(arrays):
        n = len(a)
        if n > 0:
            out[i, :n] = a.astype(dtype, copy=False)
    return out


def make_collate(
    max_dim: int = 20,
    randomize_slots: bool = True,
    standardize_per_var: bool = True,
    min_std: float = 1e-3,
    use_asinh: bool = True,
):
    """Return a collate fn closure for use in a PyTorch DataLoader."""

    def collate(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
        samples = list(samples)
        B = len(samples)
        if B == 0:
            raise ValueError("empty batch")

        # Decide L_max across the batch (max per-variate observations)
        L_max = 1
        for s in samples:
            for n in s["n_obs_per_var"]:
                if int(n) > L_max:
                    L_max = int(n)

        V_pad = int(max_dim)

        values = np.zeros((B, V_pad, L_max), dtype=np.float32)
        timestamps = np.zeros((B, V_pad, L_max), dtype=np.float32)
        deltat = np.zeros((B, V_pad, L_max), dtype=np.float32)
        valid_mask = np.zeros((B, V_pad, L_max), dtype=bool)
        valid_var = np.zeros((B, V_pad), dtype=bool)
        target_var = np.zeros((B, V_pad), dtype=bool)
        n_obs = np.zeros((B, V_pad), dtype=np.int32)
        var_slot = np.full((B, V_pad), -1, dtype=np.int32)
        history = np.zeros((B,), dtype=np.float32)
        time_scale = np.zeros((B,), dtype=np.float32)
        item_id: list[str] = []
        source_name: list[str] = []
        regime: list[str] = []

        for b, s in enumerate(samples):
            V_act = int(min(len(s["n_obs_per_var"]), V_pad))
            if V_act <= 0:
                continue

            if randomize_slots and V_pad > V_act:
                slots = np.random.choice(V_pad, size=V_act, replace=False)
            else:
                slots = np.arange(V_act, dtype=np.int64)

            vals_list = s["values_per_var"][:V_act]
            ts_list = s["timestamps_per_var"][:V_act]
            dt_list = s["deltat_per_var"][:V_act]
            tgt_mask = np.asarray(s["target_variate_mask"][:V_act], dtype=bool)
            n_obs_v = np.asarray(s["n_obs_per_var"][:V_act], dtype=np.int32)

            # Per-variate stack to [V_act, L_max]
            vals_padded = _stack_pad(vals_list, L_max, dtype=np.float32)
            ts_padded = _stack_pad(ts_list, L_max, dtype=np.float32)
            dt_padded = _stack_pad(dt_list, L_max, dtype=np.float32)
            valid_padded = np.zeros((V_act, L_max), dtype=bool)
            for v in range(V_act):
                valid_padded[v, : n_obs_v[v]] = True

            # Per-(b, v) standardization, context-only, then optional
            # asinh squashing (Chronos-2 §3.1 "robust scaling").
            #
            # NOTE 1: stats are computed in float64 to avoid
            # ``RuntimeWarning: overflow encountered in square`` when raw
            # LOTSA values are large enough that ``(x - mu) ** 2`` saturates
            # the float32 range during ``.std()``.  See README §7.D.6.
            #
            # NOTE 2: when ``use_asinh=True`` (default), we apply
            #     z_tilde = arcsinh((x - mu) / max(std, 1e-6))
            # so that near-constant context (USHCN-style precipitation)
            # which would z-score future values into the thousands gets
            # tamed to ~9 (asinh(5000) ≈ 9.21).  This eliminates the
            # train-loss explosions logged in README §Q4 and is the same
            # robust-scaling step Chronos-2 uses.  The ``min_std`` floor
            # drops to 1e-6 since asinh handles whatever blowup remains.
            # When ``use_asinh=False``, fall back to plain z-scoring with
            # the old ``min_std=1e-3`` floor.
            #
            # 2026-04-29: bumped min_std back from 1e-6 to 1e-3 (F2 fix,
            # README §7.O).  The asinh trick handles loss-side stability
            # but the pre-asinh z-score still produced |a|=20 outliers
            # for heavy-tailed lotsa series with σ < 1e-3 (1 spike per
            # 300 obs at original-units value ~3e8 stds away from the
            # bulk mean).  Bumping the floor caps these at |a|≈9 without
            # losing any data.  No measurable effect on training loss
            # but reduces wasted gradient capacity on outlier targets.
            if standardize_per_var:
                hist_b = float(s["history"])
                std_floor = min_std if use_asinh else 1e-3
                for v in range(V_act):
                    ctx = valid_padded[v] & (ts_padded[v] < hist_b)
                    if ctx.sum() >= 3:
                        x64 = vals_padded[v, ctx].astype(np.float64, copy=False)
                        mu = float(x64.mean())
                        std = float(x64.std())
                    elif valid_padded[v].sum() >= 1:
                        x64 = vals_padded[v, valid_padded[v]].astype(np.float64, copy=False)
                        mu = float(x64.mean())
                        std = float(x64.std())
                    else:
                        mu, std = 0.0, 1.0
                    std = max(std, std_floor)
                    z = (vals_padded[v] - mu) / std
                    if use_asinh:
                        z = np.arcsinh(z).astype(np.float32)
                    vals_padded[v] = z.astype(np.float32)
                    vals_padded[v] = np.where(valid_padded[v], vals_padded[v], 0.0)

            # Scatter active variates into their (possibly randomized) slots.
            for v in range(V_act):
                k = int(slots[v])
                values[b, k] = vals_padded[v]
                timestamps[b, k] = ts_padded[v]
                deltat[b, k] = dt_padded[v]
                valid_mask[b, k] = valid_padded[v]
                target_var[b, k] = bool(tgt_mask[v])
                n_obs[b, k] = int(n_obs_v[v])
                var_slot[b, k] = v
            valid_var[b, slots] = True

            history[b] = float(s["history"])
            time_scale[b] = float(s.get("time_scale", 1.0))
            item_id.append(str(s.get("source_tag", b)))
            source_name.append(str(s.get("logical_source", s.get("source_tag", ""))))
            regime.append(str(s.get("regime", "regular")))

        # pred_mask: predict where valid AND in horizon AND variate is target
        ts_t = torch.from_numpy(timestamps)
        valid_t = torch.from_numpy(valid_mask)
        history_t = torch.from_numpy(history)
        target_var_t = torch.from_numpy(target_var)
        in_horizon = ts_t >= history_t.view(B, 1, 1)
        pred_mask = valid_t & in_horizon & target_var_t.view(B, V_pad, 1)

        return {
            "values":              torch.from_numpy(values),
            "timestamps":          ts_t,
            "deltat":              torch.from_numpy(deltat),
            "valid_mask":          valid_t,
            "pred_mask":           pred_mask,
            "valid_variate_mask":  torch.from_numpy(valid_var),
            "target_variate_mask": target_var_t,
            "n_obs_per_var":       torch.from_numpy(n_obs),
            "var_slot":            torch.from_numpy(var_slot),
            "history":             history_t,
            "time_scale":          torch.from_numpy(time_scale),
            "item_id":             item_id,
            "source_name":         source_name,
            "regime":              regime,
        }

    return collate
