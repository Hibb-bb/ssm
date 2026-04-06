"""
PyTorch Dataset that loads model-agnostic sinusoidal .npz files
and presents them in the format expected by NCDSSM.

Each sample returns:
  past_target:   (n_ctx, 1)     values in context window [0, history)
  past_times:    (n_ctx,)       timestamps in context window
  past_mask:     (n_ctx, 1)     all 1.0 (observations are fully present)
  future_target: (n_pred, 1)    values in prediction window [history, T_MAX]
  future_times:  (n_pred,)      timestamps in prediction window
  future_mask:   (n_pred, 1)    all 1.0

Since each sample has different timestamps, use `collate_fn` which
unions timestamps across the batch (climate-dataset style from NCDSSM).

Usage:
    from sinusoidal_dataset_ncdssm import SinusoidalNCDSSMDataset

    ds = SinusoidalNCDSSMDataset("/path/to/sinusoidal_high_irreg/train.npz")
    loader = DataLoader(ds, batch_size=32, collate_fn=ds.collate_fn)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SinusoidalNCDSSMDataset(Dataset):
    """Loads raw .npz and serves data in NCDSSM-compatible dict format."""

    def __init__(self, npz_path, meta_path=None, normalize=True):
        """
        Args:
            npz_path:  Path to train.npz / val.npz / test.npz
            meta_path: Path to meta.json (defaults to sibling of npz_path)
            normalize: Whether to use pre-normalized values
        """
        npz_path = Path(npz_path)
        data = np.load(npz_path)

        self.timestamps = data['timestamps']
        if normalize:
            self.values = data['values_normed']
        else:
            self.values = data['values']
        self.freqs = data['freqs']
        self.amps = data['amps']
        self.phases = data['phases']

        if meta_path is None:
            meta_path = npz_path.parent / 'meta.json'
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.history = self.meta['history']

    def __len__(self):
        return self.timestamps.shape[0]

    def __getitem__(self, idx):
        ts = self.timestamps[idx]
        vals = self.values[idx]

        ctx_mask = ts < self.history
        pred_mask = ts >= self.history

        past_times = ts[ctx_mask]
        past_values = vals[ctx_mask][:, None]
        past_obs_mask = np.ones_like(past_values)

        future_times = ts[pred_mask]
        future_values = vals[pred_mask][:, None]
        future_obs_mask = np.ones_like(future_values)

        return dict(
            past_target=torch.as_tensor(past_values, dtype=torch.float32),
            past_times=torch.as_tensor(past_times, dtype=torch.float32),
            past_mask=torch.as_tensor(past_obs_mask, dtype=torch.float32),
            future_target=torch.as_tensor(future_values, dtype=torch.float32),
            future_times=torch.as_tensor(future_times, dtype=torch.float32),
            future_mask=torch.as_tensor(future_obs_mask, dtype=torch.float32),
        )

    @staticmethod
    def _listofdict2dictoflist(list_of_dicts):
        keys = list_of_dicts[0].keys()
        return {k: [d[k] for d in list_of_dicts] for k in keys}

    def collate_fn(self, list_of_samples):
        """
        Climate-dataset-style collation: union all timestamps across the
        batch into a shared time grid, then scatter each sample's values
        and masks to the correct positions.
        """
        d = self._listofdict2dictoflist(list_of_samples)
        batch_size = len(list_of_samples)
        target_dim = d['past_target'][0].shape[-1]

        comb_past_times, past_inv = torch.unique(
            torch.cat(d['past_times']), sorted=True, return_inverse=True
        )

        comb_past_target = torch.zeros(batch_size, comb_past_times.shape[0], target_dim)
        comb_past_mask = torch.zeros_like(comb_past_target)
        offset = 0
        for i, (tgt, mask, time) in enumerate(
            zip(d['past_target'], d['past_mask'], d['past_times'])
        ):
            indices = past_inv[offset:offset + time.shape[0]]
            offset += time.shape[0]
            comb_past_target[i, indices] = tgt
            comb_past_mask[i, indices] = mask

        comb_future_target = None
        comb_future_times = None
        comb_future_mask = None
        if d['future_target'][0] is not None:
            comb_future_times, future_inv = torch.unique(
                torch.cat(d['future_times']), sorted=True, return_inverse=True
            )
            comb_future_target = torch.zeros(
                batch_size, comb_future_times.shape[0], target_dim)
            comb_future_mask = torch.zeros_like(comb_future_target)
            offset = 0
            for i, (tgt, mask, time) in enumerate(
                zip(d['future_target'], d['future_mask'], d['future_times'])
            ):
                indices = future_inv[offset:offset + time.shape[0]]
                offset += time.shape[0]
                comb_future_target[i, indices] = tgt
                comb_future_mask[i, indices] = mask

        return dict(
            past_target=comb_past_target,
            past_times=comb_past_times,
            past_mask=comb_past_mask,
            future_target=comb_future_target,
            future_times=comb_future_times,
            future_mask=comb_future_mask,
        )
