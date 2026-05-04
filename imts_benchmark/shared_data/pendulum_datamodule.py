"""Pendulum image-regression Lightning DataModule.

Sparse-flat HF Arrow schema, one row per sample (V=1, single image stream):

    item_id:                  string         e.g. "pendulum_00042"
    images:                   list[float]    flattened, length T * 24 * 24
    timestamp:                list[float]    length T, irregular times in [0, 100]
    past_feat_dynamic_real:   list[float]    length T, per-step delta_t,
                                             first entry = 0 (no leakage)
    target:                   list[float]    length T * 2, [sin0,cos0,sin1,cos1,...]
    n_obs:                    int            T (variable per sample)

Output batch (collate_pendulum):
    images       [B, V=1, T_max, 576]   float32   flattened images, padded
    timestamps   [B, V=1, T_max]        float32
    deltat       [B, V=1, T_max]        float32
    valid_mask   [B, V=1, T_max]        bool      True = real observation
    pred_mask    [B, V=1, T_max]        bool      same as valid_mask
                                                  (no history split)
    target       [B, V=1, T_max, 2]     float32   (sin, cos)
    history      [B]                    float32   stub (unused; carried for
                                                  trainer-shape parity)
    n_obs        [B]                    int64     real T per sample
    item_id      list[str]              length B

The image side is emitted *flattened* (576-dim per timestep) so the
patch-embed Linear(576, d_hidden) in PendulumMambaForecaster can apply
without an extra reshape. To recover the spatial layout, .reshape(..., 24, 24).
"""

from __future__ import annotations

from pathlib import Path

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_FLAT_DIM = 24 * 24  # 576
TARGET_DIM = 2            # (sin, cos)


class PendulumSparseFlat(Dataset):
    def __init__(self, hf_dataset: datasets.Dataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        n_obs = int(row["n_obs"])

        images = np.asarray(row["images"], dtype=np.float32).reshape(
            n_obs, IMAGE_FLAT_DIM
        )                                                                # [T, 576]
        timestamp = np.asarray(row["timestamp"], dtype=np.float32)       # [T]
        deltat = np.asarray(row["past_feat_dynamic_real"], dtype=np.float32)  # [T]
        target = np.asarray(row["target"], dtype=np.float32).reshape(
            n_obs, TARGET_DIM
        )                                                                # [T, 2]

        return dict(
            item_id=row["item_id"],
            images=images,
            timestamps=timestamp,
            deltat=deltat,
            target=target,
            n_obs=n_obs,
        )


def collate_pendulum(items: list[dict]) -> dict:
    B = len(items)
    T_max = max(it["n_obs"] for it in items)
    T_max = max(T_max, 1)
    V = 1

    images = np.zeros((B, V, T_max, IMAGE_FLAT_DIM), dtype=np.float32)
    timestamps = np.zeros((B, V, T_max), dtype=np.float32)
    deltat = np.zeros((B, V, T_max), dtype=np.float32)
    target = np.zeros((B, V, T_max, TARGET_DIM), dtype=np.float32)
    valid = np.zeros((B, V, T_max), dtype=np.bool_)
    n_obs_out = np.zeros((B,), dtype=np.int64)
    history_out = np.zeros((B,), dtype=np.float32)
    item_ids = []

    for i, it in enumerate(items):
        T = it["n_obs"]
        item_ids.append(it["item_id"])
        n_obs_out[i] = T
        if T == 0:
            continue
        images[i, 0, :T] = it["images"]
        timestamps[i, 0, :T] = it["timestamps"]
        deltat[i, 0, :T] = it["deltat"]
        target[i, 0, :T] = it["target"]
        valid[i, 0, :T] = True

    pred_mask = valid  # no history split for pendulum regression

    return dict(
        item_id=item_ids,
        images=torch.from_numpy(images),
        timestamps=torch.from_numpy(timestamps),
        deltat=torch.from_numpy(deltat),
        valid_mask=torch.from_numpy(valid),
        pred_mask=torch.from_numpy(pred_mask),
        target=torch.from_numpy(target),
        history=torch.from_numpy(history_out),
        n_obs=torch.from_numpy(n_obs_out),
    )


class PendulumDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        regime: str = "pendulum",
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.regime = regime
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        base = self.data_root / self.regime
        self.train_ds = PendulumSparseFlat(
            datasets.load_from_disk(str(base / "train"))
        )
        self.val_ds = PendulumSparseFlat(
            datasets.load_from_disk(str(base / "val"))
        )
        self.test_ds = PendulumSparseFlat(
            datasets.load_from_disk(str(base / "test"))
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_pendulum,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_pendulum,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_pendulum,
        )
