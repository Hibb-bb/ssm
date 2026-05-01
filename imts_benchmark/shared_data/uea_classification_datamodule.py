"""UEA classification DataModule with Kidger 30% sync drop.

Loads UEA .ts files, applies the Kidger et al. (2020) NCDE protocol of
synchronously dropping 30% of timesteps per sample, and emits a per-variate
batch dict identical in shape to the forecasting datamodule (plus a `label`
field, minus pred_mask and gap fields).

Expected on-disk layout:
    <data_root>/<DatasetName>/<DatasetName>_TRAIN.ts
    <data_root>/<DatasetName>/<DatasetName>_TEST.ts

Five datasets used in this benchmark (matching RoMAE Table 4):
    BasicMotions, CharacterTrajectories, Epilepsy, Heartbeat, LSST.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Minimal .ts parser. Supports:
#   - equalLength true/false
#   - missing values represented by '?' (replaced with NaN, then mean-filled
#     per channel before drop)
#   - multivariate (channels separated by ':', values within a channel by ',')
#   - classLabel true with named labels at end of each data line
# This covers BM, CT, EP, HB, LSST.
# ============================================================


def _parse_ts_file(path: Path) -> tuple[list[list[np.ndarray]], list[str], dict]:
    """Parse a .ts file. Returns (X, y, header) where:
        X[i] is a list of length V; X[i][v] is a 1-D float numpy array (length L_iv).
        y[i] is the string class label.
        header is the parsed @-prefixed metadata.
    """
    header: dict = {}
    samples_x: list[list[np.ndarray]] = []
    samples_y: list[str] = []
    in_data = False

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                in_data = True
                continue
            if not in_data and line.startswith("@"):
                # Header line, e.g. "@dimensions 6"
                parts = line.split(None, 1)
                key = parts[0][1:].lower()
                val = parts[1].strip() if len(parts) > 1 else ""
                header[key] = val
                continue
            if not in_data:
                continue

            # Data line: V channels split by ":", optional final ":label".
            # Values within a channel are comma-separated; "?" means missing.
            tokens = line.split(":")
            # If classLabel is true, last token is the label.
            has_label = header.get("classlabel", "false").lower().startswith("true")
            if has_label:
                y = tokens[-1].strip()
                channels = tokens[:-1]
            else:
                y = ""
                channels = tokens

            channel_arrs: list[np.ndarray] = []
            for ch in channels:
                ch = ch.strip()
                if not ch:
                    channel_arrs.append(np.zeros(0, dtype=np.float32))
                    continue
                vals = []
                for v in ch.split(","):
                    v = v.strip()
                    if v == "?" or v == "":
                        vals.append(np.nan)
                    else:
                        vals.append(float(v))
                channel_arrs.append(np.asarray(vals, dtype=np.float32))
            samples_x.append(channel_arrs)
            samples_y.append(y)
    return samples_x, samples_y, header


def _equalize_length(
    samples: list[list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack a list of [V] -> [L_v] samples into a [N, V, L_max] tensor with
    a [N, V, L_max] valid mask. Pads with zeros on the right.

    Note for UEA: most datasets have equal length within a sample (all V have
    same L); CharacterTrajectories can vary across samples.
    """
    N = len(samples)
    V = len(samples[0])
    L_max = 0
    for s in samples:
        for ch in s:
            if len(ch) > L_max:
                L_max = len(ch)
    X = np.zeros((N, V, L_max), dtype=np.float32)
    valid = np.zeros((N, V, L_max), dtype=np.bool_)
    for i, s in enumerate(samples):
        for v, ch in enumerate(s):
            n = len(ch)
            if n > 0:
                X[i, v, :n] = ch
                valid[i, v, :n] = True
    return X, valid


def _impute_nans(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace NaN values in valid positions with per-channel mean of the
    sample's valid non-NaN values (fallback to 0)."""
    X = X.copy()
    nan_mask = ~np.isfinite(X) & valid
    if not nan_mask.any():
        return X
    N, V, L = X.shape
    for i in range(N):
        for v in range(V):
            m = valid[i, v] & np.isfinite(X[i, v])
            if m.any():
                fill = float(X[i, v, m].mean())
            else:
                fill = 0.0
            bad = nan_mask[i, v]
            if bad.any():
                X[i, v, bad] = fill
    return X


def _normalize_zscore(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-(sample, channel) zscore over valid positions. Stable for tiny
    constant channels (returns zeros). Mamba-MV doesn't have built-in input
    norm, so we standardize externally to match the baselines' convention."""
    X = X.copy()
    N, V, L = X.shape
    for i in range(N):
        for v in range(V):
            m = valid[i, v]
            if not m.any():
                continue
            vals = X[i, v, m]
            mu = float(vals.mean())
            sd = float(vals.std())
            if sd < 1e-8:
                X[i, v, m] = 0.0
            else:
                X[i, v, m] = (vals - mu) / sd
    return X


# ============================================================
# Kidger 30% sync drop.
# ============================================================


def _drop_seed(dataset_name: str, split: str, sample_idx: int, base_seed: int) -> int:
    """Deterministic per-sample seed: stable across reruns and machines."""
    h = hashlib.md5(
        f"{dataset_name}|{split}|{sample_idx}|{base_seed}".encode()
    ).hexdigest()
    return int(h[:8], 16)


def _kidger_drop_mask(L: int, sample_seed: int, drop_rate: float) -> np.ndarray:
    """Generate a sync drop mask of length L. True = keep, False = drop.
    Uses np.random.default_rng for deterministic per-sample reproducibility.
    """
    rng = np.random.default_rng(sample_seed)
    n_drop = int(round(L * drop_rate))
    n_drop = min(max(n_drop, 0), L - 1)  # always leave at least one obs
    drop_idx = rng.choice(L, size=n_drop, replace=False)
    keep = np.ones(L, dtype=np.bool_)
    keep[drop_idx] = False
    return keep


# ============================================================
# Dataset.
# ============================================================


class UEAClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        split: str,                   # "train" | "test"
        drop_rate: float = 0.3,
        drop_seed: int = 0,
        normalize: bool = True,
        select_indices: Optional[list[int]] = None,
        label_to_idx: Optional[dict[str, int]] = None,
        t_max: float = 1.0,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.t_max = t_max

        path = Path(data_root) / dataset_name / f"{dataset_name}_{split.upper()}.ts"
        if not path.exists():
            raise FileNotFoundError(
                f"UEA file not found: {path}. "
                f"Place the {dataset_name}_{split.upper()}.ts file at that path. "
                f"UEA datasets can be downloaded from "
                f"https://www.timeseriesclassification.com/."
            )
        samples_x, samples_y, header = _parse_ts_file(path)
        self.header = header

        # Stack to [N, V, L_max]; impute NaNs with per-channel mean.
        X, valid = _equalize_length(samples_x)
        X = _impute_nans(X, valid)
        if normalize:
            X = _normalize_zscore(X, valid)

        # Build label map if not provided (fixed by sorted unique training labels).
        if label_to_idx is None:
            label_to_idx = {lab: i for i, lab in enumerate(sorted(set(samples_y)))}
        y_idx = np.asarray([label_to_idx[s] for s in samples_y], dtype=np.int64)

        if select_indices is not None:
            X = X[select_indices]
            valid = valid[select_indices]
            y_idx = y_idx[select_indices]

        self.X = X                                # [N, V, L]
        self.valid = valid                        # [N, V, L]
        self.y = y_idx                            # [N]
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        self.drop_rate = drop_rate
        self.drop_seed = drop_seed

    @property
    def n_classes(self) -> int:
        return len(self.label_to_idx)

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> dict:
        x_full = self.X[idx]              # [V, L]
        v_full = self.valid[idx]          # [V, L]
        V, L = x_full.shape

        # Kidger sync drop: per-sample mask shared across variates.
        sample_seed = _drop_seed(self.dataset_name, self.split, int(idx), self.drop_seed)
        keep_mask = _kidger_drop_mask(L, sample_seed, self.drop_rate)  # [L]

        # Original timestamps are 0..L-1 rescaled to [0, t_max].
        ts_full = np.linspace(0.0, self.t_max, L, dtype=np.float32)

        # Apply drop: keep timestamps where keep_mask=True AND original valid.
        # Build per-variate kept arrays (variable length per variate).
        values_per_var: list[np.ndarray] = []
        timestamps_per_var: list[np.ndarray] = []
        deltat_per_var: list[np.ndarray] = []
        n_obs_per_var = np.zeros(V, dtype=np.int64)

        for v in range(V):
            keep_v = keep_mask & v_full[v]
            ts_v = ts_full[keep_v]
            vals_v = x_full[v, keep_v]
            if len(ts_v) == 0:
                # Shouldn't happen with Kidger 30%, but degrade gracefully:
                # one zero observation at t=0.
                ts_v = np.asarray([0.0], dtype=np.float32)
                vals_v = np.asarray([0.0], dtype=np.float32)
            # Per-variate delta-t. First entry = 0 so delta does not leak
            # across variate boundaries (matches forecasting convention).
            dt_v = np.zeros_like(ts_v)
            if len(ts_v) > 1:
                dt_v[1:] = np.diff(ts_v)
            values_per_var.append(vals_v.astype(np.float32))
            timestamps_per_var.append(ts_v.astype(np.float32))
            deltat_per_var.append(dt_v.astype(np.float32))
            n_obs_per_var[v] = len(ts_v)

        return dict(
            item_id=int(idx),
            label=int(self.y[idx]),
            values_per_var=values_per_var,
            timestamps_per_var=timestamps_per_var,
            deltat_per_var=deltat_per_var,
            n_obs_per_var=n_obs_per_var,
            history=float(self.t_max),
        )


# ============================================================
# Collate (mirrors collate_per_variate; adds label, drops pred_mask/gaps).
# ============================================================


def collate_uea_per_variate(items: list[dict]) -> dict:
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
    labels = np.zeros((B,), dtype=np.int64)
    item_ids = []

    for i, it in enumerate(items):
        item_ids.append(it["item_id"])
        history_out[i] = it["history"]
        labels[i] = it["label"]
        n_obs_out[i] = it["n_obs_per_var"]
        for d in range(V):
            n = len(it["values_per_var"][d])
            if n == 0:
                continue
            values[i, d, :n] = it["values_per_var"][d]
            timestamps[i, d, :n] = it["timestamps_per_var"][d]
            deltat[i, d, :n] = it["deltat_per_var"][d]
            valid[i, d, :n] = True

    return dict(
        item_id=item_ids,
        values=torch.from_numpy(values),
        timestamps=torch.from_numpy(timestamps),
        deltat=torch.from_numpy(deltat),
        valid_mask=torch.from_numpy(valid),
        n_obs_per_var=torch.from_numpy(n_obs_out),
        history=torch.from_numpy(history_out),
        label=torch.from_numpy(labels),
    )


# ============================================================
# DataModule.
# ============================================================


class UEAClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for UEA classification.

    Train/val: 80/20 stratified split from UEA train. Test: UEA test split.
    Drop is applied at __getitem__ time so different epochs see the *same*
    drop pattern (deterministic per (sample, drop_seed)), matching Kidger's
    fixed-protocol convention.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        batch_size: int = 16,
        num_workers: int = 2,
        val_frac: float = 0.2,
        split_seed: int = 42,
        drop_rate: float = 0.3,
        drop_seed: int = 0,
        normalize: bool = True,
        t_max: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        # Build the full train and test datasets.
        full_train = UEAClassificationDataset(
            data_root=self.hparams.data_root,
            dataset_name=self.hparams.dataset_name,
            split="train",
            drop_rate=self.hparams.drop_rate,
            drop_seed=self.hparams.drop_seed,
            normalize=self.hparams.normalize,
            t_max=self.hparams.t_max,
        )
        # Stratified 80/20 split on the train indices.
        rng = np.random.default_rng(self.hparams.split_seed)
        y = full_train.y
        train_idx_all = np.arange(len(full_train))
        val_indices: list[int] = []
        for c in np.unique(y):
            cls_idx = train_idx_all[y == c]
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * self.hparams.val_frac)))
            val_indices.extend(cls_idx[:n_val].tolist())
        val_set = set(val_indices)
        train_indices = [i for i in train_idx_all if i not in val_set]

        # Use the same label_to_idx mapping for all three splits.
        label_to_idx = full_train.label_to_idx

        self.train_ds = UEAClassificationDataset(
            data_root=self.hparams.data_root,
            dataset_name=self.hparams.dataset_name,
            split="train",
            drop_rate=self.hparams.drop_rate,
            drop_seed=self.hparams.drop_seed,
            normalize=self.hparams.normalize,
            select_indices=train_indices,
            label_to_idx=label_to_idx,
            t_max=self.hparams.t_max,
        )
        self.val_ds = UEAClassificationDataset(
            data_root=self.hparams.data_root,
            dataset_name=self.hparams.dataset_name,
            split="train",       # val comes from train file
            drop_rate=self.hparams.drop_rate,
            drop_seed=self.hparams.drop_seed,
            normalize=self.hparams.normalize,
            select_indices=val_indices,
            label_to_idx=label_to_idx,
            t_max=self.hparams.t_max,
        )
        self.test_ds = UEAClassificationDataset(
            data_root=self.hparams.data_root,
            dataset_name=self.hparams.dataset_name,
            split="test",
            drop_rate=self.hparams.drop_rate,
            drop_seed=self.hparams.drop_seed,
            normalize=self.hparams.normalize,
            label_to_idx=label_to_idx,
            t_max=self.hparams.t_max,
        )

        # Compute class weights from training subset (inverse frequency,
        # normalized to sum to n_classes).
        n_classes = len(label_to_idx)
        counts = np.zeros(n_classes, dtype=np.float64)
        for i in train_indices:
            counts[full_train.y[i]] += 1
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        weights = inv * (n_classes / inv.sum())
        self.class_weights = torch.from_numpy(weights.astype(np.float32))

        self.n_classes = n_classes
        self.n_vars = full_train.n_vars
        self.label_to_idx = label_to_idx

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_uea_per_variate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_uea_per_variate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate_uea_per_variate,
            persistent_workers=self.hparams.num_workers > 0,
        )
