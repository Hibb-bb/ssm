"""Shared multivariate Lightning DataModule for the sparse-flat HF format
produced by `ssm_dk/generate_longgap_multisin.py`.

Sparse-flat schema (one row per sample):
    target:                 list[float], all variates concatenated
    timestamp:              list[float], per-variate ts concatenated
    past_feat_dynamic_real: list[float], per-variate delta_t, first entry per
                            variate = 0 (so it does NOT leak across variate
                            boundaries; critical for the Mamba delta channel)
    n_obs_per_var:          list[int], length V (=3 here)
    history:                scalar (7.0)

Two batch formats are supported, selected via `format=`:

    "per_variate"  — Mamba MV consumes this. Returns per-variate padded tensors
                     of shape [B, V, L_max] plus length masks.

    "flat_tokens"  — RoMAE consumes this. Returns one token per observation
                     with features [value, timestamp, variate_id] plus a
                     padding mask.

Prediction mask is computed as `timestamps >= history` on the un-padded tokens
(identical in both formats). item_id is preserved for result-matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


BatchFormat = Literal["per_variate", "flat_tokens"]


def _split_per_variate(flat: np.ndarray, n_obs_per_var: np.ndarray) -> list[np.ndarray]:
    """Split a flat array into per-variate slices using cumulative counts."""
    split_points = np.cumsum(n_obs_per_var)[:-1]
    return np.split(flat, split_points)


def _pad_per_variate(
    per_var: list[np.ndarray],
    max_len: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of per-variate arrays to [V, max_len]. Returns (data, mask)."""
    V = len(per_var)
    out = np.zeros((V, max_len), dtype=dtype)
    mask = np.zeros((V, max_len), dtype=np.bool_)
    for d, arr in enumerate(per_var):
        n = len(arr)
        if n > 0:
            out[d, :n] = arr
            mask[d, :n] = True
    return out, mask


class MultivariateSparseFlat(Dataset):
    def __init__(self, hf_dataset: datasets.Dataset, format: BatchFormat = "per_variate"):
        self.data = hf_dataset
        self.format = format

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        target = np.asarray(row["target"], dtype=np.float32)
        timestamp = np.asarray(row["timestamp"], dtype=np.float32)
        deltat = np.asarray(row["past_feat_dynamic_real"], dtype=np.float32)
        n_obs = np.asarray(row["n_obs_per_var"], dtype=np.int64)
        history = float(row["history"])

        vals_pv = _split_per_variate(target, n_obs)
        ts_pv = _split_per_variate(timestamp, n_obs)
        dt_pv = _split_per_variate(deltat, n_obs)

        # Phase 4 only: forward gap_starts/gap_ends + gapped_variate_index so
        # test-step can compute in-gap vs out-of-gap metrics for v3. Phase 2/3
        # data has no such fields -- return empty lists for compatibility.
        gap_starts = [float(x) for x in row["gap_starts"]] if "gap_starts" in row else []
        gap_ends = [float(x) for x in row["gap_ends"]] if "gap_ends" in row else []
        gapped_variate_index = (
            [int(x) for x in row["gapped_variate_index"]]
            if "gapped_variate_index" in row else []
        )

        return dict(
            item_id=row["item_id"],
            values_per_var=vals_pv,
            timestamps_per_var=ts_pv,
            deltat_per_var=dt_pv,
            n_obs_per_var=n_obs,
            history=history,
            gap_starts=gap_starts,
            gap_ends=gap_ends,
            gapped_variate_index=gapped_variate_index,
        )


def collate_per_variate(items: list[dict]) -> dict:
    """Pad each variate to the batch's max per-variate length. Output shapes:
        values       [B, V, L_max]
        timestamps   [B, V, L_max]
        deltat       [B, V, L_max]
        valid_mask   [B, V, L_max]  bool
        pred_mask    [B, V, L_max]  bool  (valid AND timestamp >= history)
        n_obs        [B, V]
        history      [B]
    """
    B = len(items)
    V = len(items[0]["values_per_var"])
    L_max = max(
        len(items[i]["values_per_var"][d]) for i in range(B) for d in range(V)
    )
    L_max = max(L_max, 1)

    values = np.zeros((B, V, L_max), dtype=np.float32)
    timestamps = np.zeros((B, V, L_max), dtype=np.float32)
    deltat = np.zeros((B, V, L_max), dtype=np.float32)
    valid = np.zeros((B, V, L_max), dtype=np.bool_)

    n_obs_out = np.zeros((B, V), dtype=np.int64)
    history_out = np.zeros((B,), dtype=np.float32)
    item_ids = []

    for i, it in enumerate(items):
        item_ids.append(it["item_id"])
        history_out[i] = it["history"]
        n_obs_out[i] = it["n_obs_per_var"]
        for d in range(V):
            n = len(it["values_per_var"][d])
            if n == 0:
                continue
            values[i, d, :n] = it["values_per_var"][d]
            timestamps[i, d, :n] = it["timestamps_per_var"][d]
            deltat[i, d, :n] = it["deltat_per_var"][d]
            valid[i, d, :n] = True

    pred_mask = valid & (timestamps >= history_out[:, None, None])
    # Forward gap fields as Python lists (per-sample variable-length). Only
    # populated for Phase 4 data; empty for Phase 2/3.
    gap_starts = [it.get("gap_starts", []) for it in items]
    gap_ends = [it.get("gap_ends", []) for it in items]
    gapped_variate_index = [it.get("gapped_variate_index", []) for it in items]
    return dict(
        item_id=item_ids,
        values=torch.from_numpy(values),
        timestamps=torch.from_numpy(timestamps),
        deltat=torch.from_numpy(deltat),
        valid_mask=torch.from_numpy(valid),
        pred_mask=torch.from_numpy(pred_mask),
        n_obs_per_var=torch.from_numpy(n_obs_out),
        history=torch.from_numpy(history_out),
        gap_starts=gap_starts,
        gap_ends=gap_ends,
        gapped_variate_index=gapped_variate_index,
    )


def collate_flat_tokens(items: list[dict]) -> dict:
    """Flatten all observations of a sample into a single token sequence.
    Each token: (value, timestamp, variate_id).

    Length padding: all samples padded to N_max = max(real_count) + max(pred_count).
    This guarantees every sample has enough padded slots to carry `extra`
    forecast-target markers in pred_mask so that every sample ends up with the
    SAME number of True positions in pred_mask -- a requirement of RoMAE's
    MAE-style forward (`x[mask].reshape(b, -1, F)`).

    Outputs:
        values           [B, N_max]
        timestamps       [B, N_max]
        variate_id       [B, N_max]  long
        pad_mask         [B, N_max]  bool  (True  = real token, OUR convention)
        pred_mask        [B, N_max]  bool  (True  = forecast target; includes
                                             `extra` dummies in the padding
                                             region so the count is uniform)
        pred_real_count  [B]         int   # of *real* forecast targets per sample
        history          [B]
    """
    B = len(items)
    V = len(items[0]["values_per_var"])

    real_counts = np.array(
        [sum(len(it["values_per_var"][d]) for d in range(V)) for it in items],
        dtype=np.int64,
    )
    # per-sample count of real forecast targets (ts >= history)
    pred_real = np.zeros(B, dtype=np.int64)
    for i, it in enumerate(items):
        hist = it["history"]
        cnt = 0
        for d in range(V):
            ts = it["timestamps_per_var"][d]
            cnt += int(np.sum(ts >= hist))
        pred_real[i] = cnt

    real_max = int(real_counts.max(initial=0))
    pred_max = int(pred_real.max(initial=0))
    N_max = max(real_max + pred_max, 1)

    values = np.zeros((B, N_max), dtype=np.float32)
    timestamps = np.zeros((B, N_max), dtype=np.float32)
    variate_id = np.zeros((B, N_max), dtype=np.int64)
    pad_mask = np.zeros((B, N_max), dtype=np.bool_)
    pred_mask = np.zeros((B, N_max), dtype=np.bool_)
    history_out = np.zeros((B,), dtype=np.float32)
    item_ids = []

    for i, it in enumerate(items):
        item_ids.append(it["item_id"])
        history_out[i] = it["history"]
        hist = it["history"]
        cursor = 0
        for d in range(V):
            vals = it["values_per_var"][d]
            ts = it["timestamps_per_var"][d]
            n = len(vals)
            if n == 0:
                continue
            values[i, cursor : cursor + n] = vals
            timestamps[i, cursor : cursor + n] = ts
            variate_id[i, cursor : cursor + n] = d
            pad_mask[i, cursor : cursor + n] = True
            pred_mask[i, cursor : cursor + n] = ts >= hist
            cursor += n

        # Pad pred_mask with `extra` dummies in the padding region so every
        # sample has exactly pred_max True positions. The dummies live on
        # padded slots (pad_mask=False), which RoMAE's loss zeroes out.
        extra = pred_max - int(pred_real[i])
        if extra > 0:
            pred_mask[i, cursor : cursor + extra] = True

    # Forward gap fields (Phase 4 only; empty lists for Phase 2/3).
    gap_starts = [it.get("gap_starts", []) for it in items]
    gap_ends = [it.get("gap_ends", []) for it in items]
    gapped_variate_index = [it.get("gapped_variate_index", []) for it in items]
    return dict(
        item_id=item_ids,
        values=torch.from_numpy(values),
        timestamps=torch.from_numpy(timestamps),
        variate_id=torch.from_numpy(variate_id),
        pad_mask=torch.from_numpy(pad_mask),
        pred_mask=torch.from_numpy(pred_mask),
        pred_real_count=torch.from_numpy(pred_real),
        history=torch.from_numpy(history_out),
        gap_starts=gap_starts,
        gap_ends=gap_ends,
        gapped_variate_index=gapped_variate_index,
    )


class MultivariateSinusoidalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        regime: str,
        format: BatchFormat = "per_variate",
        train_batch_size: int = 128,
        val_batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.regime = regime
        self.format = format
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        base = self.data_root / self.regime
        self.train_ds = MultivariateSparseFlat(
            datasets.load_from_disk(str(base / "train")), format=self.format
        )
        self.val_ds = MultivariateSparseFlat(
            datasets.load_from_disk(str(base / "val")), format=self.format
        )
        self.test_ds = MultivariateSparseFlat(
            datasets.load_from_disk(str(base / "test")), format=self.format
        )

    def _collate(self):
        if self.format == "per_variate":
            return collate_per_variate
        return collate_flat_tokens

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate(),
        )
