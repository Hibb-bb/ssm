"""
Lightning DataModule for sinusoidal datasets stored as HuggingFace Arrow.
Returns dict batches with: values, timestamps, delta_t.
"""

from pathlib import Path

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class SinusoidalHFDataset(Dataset):
    """Wraps a HuggingFace Arrow dataset for Mamba forecasting."""

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        values = np.array(row["target"], dtype=np.float32)
        timestamps = np.array(row["timestamp"], dtype=np.float32)
        delta_t = np.zeros_like(timestamps)
        delta_t[1:] = np.diff(timestamps)
        return dict(
            values=torch.from_numpy(values),
            timestamps=torch.from_numpy(timestamps),
            delta_t=torch.from_numpy(delta_t),
        )


class SinusoidalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        irregularity: str = "high_irreg",
        train_batch_size: int = 128,
        val_batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.irregularity = irregularity
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        base = self.data_root / f"sinusoidal_{self.irregularity}"
        self.train_ds = SinusoidalHFDataset(
            datasets.load_from_disk(str(base / "train"))
        )
        self.val_ds = SinusoidalHFDataset(
            datasets.load_from_disk(str(base / "val"))
        )
        self.test_ds = SinusoidalHFDataset(
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
