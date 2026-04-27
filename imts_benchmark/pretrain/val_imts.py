"""Downstream IMTS validation loop for the pretraining pipeline.

Wraps ``tpatchgnn_data/{regime}/{val,test}`` HF datasets (the same
sparse-flat schema used by ``MultivariateSinusoidalDataModule``) so we
can validate the pretraining model against real IMTS forecasting MSE
*during* pretraining.

We deliberately keep the validation pipeline separate from
``pretrain/collate.py`` because:

  - val data is *not* degraded (no ``LOTSAToIrregular``);
  - val MSE/MAE is reported in the dataset's **original units**
    (denormalized after the model's forward), so numbers are directly
    comparable to the existing IMTS benchmark numbers from
    ``imts_benchmark/mamba_mv/train_mv.py``;
  - val data carries an extra ``dataset_name`` tag so the model can
    log ``val/mse_<ds>`` / ``val/mae_<ds>`` per dataset.

See ``pretrain/README.md`` §3.4 for the discussion that led to this
design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _split_per_variate(flat: np.ndarray, n_obs_per_var: np.ndarray) -> list[np.ndarray]:
    split_points = np.cumsum(n_obs_per_var)[:-1]
    return [a.copy() for a in np.split(flat, split_points)]


@dataclass
class IMTSValSample:
    """One IMTS validation sample.

    All arrays are float32; timestamps and history are normalized to
    [0, 1] so the model (trained with ``t_max=1.0``) consumes them
    directly.  ``value_mu`` / ``value_std`` are computed from
    context-only observations and used at val time to denormalize the
    model's predictions back to original units before MSE/MAE.
    """

    item_id: str
    n_obs_per_var: np.ndarray            # (V_real,)
    target_variate_mask: np.ndarray      # (V_real,) all True for IMTS
    values_per_var: list[np.ndarray]     # normalized (z-scored) values
    values_orig_per_var: list[np.ndarray]  # original-units values
    timestamps_per_var: list[np.ndarray]
    deltat_per_var: list[np.ndarray]
    value_mu_per_var: np.ndarray         # (V_real,)
    value_std_per_var: np.ndarray        # (V_real,)
    history: float                        # in [0, 1]
    time_scale: float                     # original time_max (e.g. 4000h)
    freq: str
    source_tag: str
    regime: str
    dataset_name: str


class IMTSValDataset(Dataset):
    """Wrap ``tpatchgnn_data/{regime}/{split}`` as a val Dataset.

    Per-sample, the dataset:
      1. splits the flat sparse-flat row into per-variate ragged arrays;
      2. normalizes timestamps and history by ``time_max`` (read from
         ``norm_stats.json``);
      3. computes context-only mean/std per variate (matching the
         pretraining collate); stores both standardized and original
         values so val_step can denormalize predictions.

    No padding here — ``imts_val_collate`` does the padding to V_pad.

    Parameters
    ----------
    data_root : str
        Path to ``tpatchgnn_data`` root.
    regime : str
        Subdirectory name, e.g. ``"activity"`` or ``"ushcn"``.
    split : str
        ``"val"`` or ``"test"``.
    subset_size : int | None
        If set, deterministically subsample the split.  Used to keep
        live val cheap during pretraining (default 1024).
    seed : int
        Subsampling seed.
    min_std : float
        Floor for std to avoid division by 0 on near-constant variates.
    """

    def __init__(
        self,
        data_root: str,
        regime: str,
        split: str = "val",
        subset_size: int | None = 1024,
        seed: int = 12345,
        min_std: float = 1e-3,
        norm_mode: str | None = None,
        dataset_name: str | None = None,
    ):
        super().__init__()
        self.regime = regime
        self.split = split
        self.min_std = float(min_std)

        ds_dir = Path(data_root) / regime / split
        meta_path = Path(data_root) / regime / "norm_stats.json"
        if not ds_dir.exists():
            raise FileNotFoundError(f"missing IMTS val dataset at {ds_dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing norm_stats.json at {meta_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.time_max = float(self.meta["time_max"])
        self.history_default = float(self.meta.get("history", self.time_max / 2.0))
        self.n_vars_real = int(self.meta["n_vars"])

        # Normalization mode:
        #   - "context_only": compute per-(b,v) mean/std from context
        #     observations (default for tpatchgnn_data activity/ushcn,
        #     matches the pretraining collate behavior).
        #   - "precomputed_per_record": use ``value_mu_per_var`` /
        #     ``value_std_per_var`` columns stored in the parquet row
        #     (produced by IMM-TSF converter, mirrors paper z-score).
        # If not given, fall back to the meta's ``norm_mode`` field, or
        # "context_only" if absent.
        self.norm_mode = (
            norm_mode
            if norm_mode is not None
            else str(self.meta.get("norm_mode", "context_only"))
        )
        if self.norm_mode not in ("context_only", "precomputed_per_record"):
            raise ValueError(
                f"unknown norm_mode {self.norm_mode!r}; expected "
                f"'context_only' or 'precomputed_per_record'"
            )

        # Optional override for the dataset_name attached to each sample
        # (so the model can log e.g. ``val/mse_EPA-Air`` instead of
        # ``val/mse_<regime>``).  Defaults to ``regime``.
        self.dataset_name = dataset_name if dataset_name is not None else regime

        self.data = datasets.load_from_disk(str(ds_dir))
        N = len(self.data)
        if subset_size is not None and subset_size < N:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(N, size=subset_size, replace=False))
        else:
            self.indices = np.arange(N, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> IMTSValSample:
        row = self.data[int(self.indices[i])]
        target = np.asarray(row["target"], dtype=np.float32)
        timestamp = np.asarray(row["timestamp"], dtype=np.float32)
        deltat = np.asarray(row["past_feat_dynamic_real"], dtype=np.float32)
        n_obs = np.asarray(row["n_obs_per_var"], dtype=np.int32)
        history = float(row.get("history", self.history_default))

        vals_pv = _split_per_variate(target, n_obs)
        ts_pv = _split_per_variate(timestamp, n_obs)
        dt_pv = _split_per_variate(deltat, n_obs)

        # Normalize timestamps to [0, 1] to match pretraining time_max=1.0.
        scale = self.time_max if self.time_max > 0 else 1.0
        ts_pv = [t / scale for t in ts_pv]
        dt_pv = [d / scale for d in dt_pv]
        history_norm = history / scale

        V = len(vals_pv)
        mu = np.zeros(V, dtype=np.float32)
        std = np.ones(V, dtype=np.float32)
        vals_z: list[np.ndarray] = []

        if self.norm_mode == "precomputed_per_record":
            # IMM-TSF mirror: use the per-record global (mu, std) stored
            # at convert time.  This matches IMM-TSF/lib/parse_datasets.py
            # lines 103-111 (per-record per-feature z-score over the full
            # record, computed before chunking), so MSE/MAE on the test
            # split is byte-comparable to the IMM-TSF paper Table.
            mu_row = np.asarray(row.get("value_mu_per_var", []), dtype=np.float32)
            sd_row = np.asarray(row.get("value_std_per_var", []), dtype=np.float32)
            if mu_row.size != V or sd_row.size != V:
                raise RuntimeError(
                    f"{self.regime}: expected V={V} pre-computed mu/std, "
                    f"got mu={mu_row.size}, std={sd_row.size}.  Did the "
                    f"converter run with --norm_mode=precomputed_per_record?"
                )
            for v in range(V):
                m = float(mu_row[v])
                s = max(float(sd_row[v]), self.min_std)
                mu[v] = m
                std[v] = s
                if vals_pv[v].size > 0:
                    vals_z.append(((vals_pv[v] - m) / s).astype(np.float32))
                else:
                    vals_z.append(vals_pv[v].copy())
        else:
            # Context-only per-variate standardization, mirroring
            # pretrain.collate behavior so the model sees inputs in the
            # same space it trained on.  Default for tpatchgnn_data
            # (activity / ushcn).
            for v in range(V):
                vals = vals_pv[v]
                ts_n = ts_pv[v]
                n = len(vals)
                if n == 0:
                    vals_z.append(vals.copy())
                    continue
                ctx_idx = ts_n < history_norm
                # Promote to float64 so squaring the deviations cannot
                # overflow float32 -- mirror collate.py.  See README §7.D.6.
                if ctx_idx.sum() >= 3:
                    x64 = vals[ctx_idx].astype(np.float64, copy=False)
                    m = float(x64.mean())
                    s = float(x64.std())
                elif n >= 1:
                    x64 = vals.astype(np.float64, copy=False)
                    m = float(x64.mean())
                    s = float(x64.std())
                else:
                    m, s = 0.0, 1.0
                s = max(s, self.min_std)
                mu[v] = m
                std[v] = s
                vals_z.append(((vals - m) / s).astype(np.float32))

        return IMTSValSample(
            item_id=str(row.get("item_id", i)),
            n_obs_per_var=n_obs,
            target_variate_mask=np.ones(V, dtype=bool),
            values_per_var=vals_z,
            values_orig_per_var=[v.copy() for v in vals_pv],
            timestamps_per_var=ts_pv,
            deltat_per_var=dt_pv,
            value_mu_per_var=mu,
            value_std_per_var=std,
            history=float(history_norm),
            time_scale=float(self.time_max),
            freq=str(self.meta.get("freq", "")),
            source_tag=f"imts:{self.regime}",
            regime=f"imts_{self.regime}",
            dataset_name=self.dataset_name,
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _stack_pad(arrays: list[np.ndarray], pad_to: int, dtype=np.float32, fill: float = 0.0) -> np.ndarray:
    out = np.full((len(arrays), pad_to), fill, dtype=dtype)
    for i, a in enumerate(arrays):
        n = len(a)
        if n > 0:
            out[i, :n] = a.astype(dtype, copy=False)
    return out


def make_imts_val_collate(max_dim: int = 20) -> Callable[[Iterable[IMTSValSample]], dict[str, Any]]:
    """Collate IMTSValSamples to the same dense batch shape as
    ``pretrain.collate``, plus extras the val_step needs to denormalize:

      - ``values_orig``  [B, V_pad, L_max]
      - ``value_mu``     [B, V_pad]
      - ``value_std``    [B, V_pad]
      - ``dataset_name`` list[str]  (one entry per sample)
    """

    def collate(samples: Iterable[IMTSValSample]) -> dict[str, Any]:
        samples = list(samples)
        B = len(samples)
        if B == 0:
            raise ValueError("empty batch")

        # Variate slot here is FIXED (not randomized): variates 0..V_real-1
        # always live in slots 0..V_real-1, padded slots after. This is the
        # canonical layout the IMTS benchmark uses, so val numbers are
        # directly comparable. (The pretraining collate randomizes slots
        # to teach the model to be slot-invariant.)
        V_pad = int(max_dim)

        L_max = 1
        for s in samples:
            for n in s.n_obs_per_var:
                if int(n) > L_max:
                    L_max = int(n)

        values = np.zeros((B, V_pad, L_max), dtype=np.float32)
        values_orig = np.zeros((B, V_pad, L_max), dtype=np.float32)
        timestamps = np.zeros((B, V_pad, L_max), dtype=np.float32)
        deltat = np.zeros((B, V_pad, L_max), dtype=np.float32)
        valid_mask = np.zeros((B, V_pad, L_max), dtype=bool)
        valid_var = np.zeros((B, V_pad), dtype=bool)
        target_var = np.zeros((B, V_pad), dtype=bool)
        n_obs = np.zeros((B, V_pad), dtype=np.int32)
        value_mu = np.zeros((B, V_pad), dtype=np.float32)
        value_std = np.ones((B, V_pad), dtype=np.float32)
        history = np.zeros((B,), dtype=np.float32)
        time_scale = np.zeros((B,), dtype=np.float32)
        item_id: list[str] = []
        dataset_name: list[str] = []
        regime_list: list[str] = []
        source_name: list[str] = []

        for b, s in enumerate(samples):
            V_act = int(min(len(s.n_obs_per_var), V_pad))
            for v in range(V_act):
                n = int(s.n_obs_per_var[v])
                if n > 0:
                    values[b, v, :n] = s.values_per_var[v]
                    values_orig[b, v, :n] = s.values_orig_per_var[v]
                    timestamps[b, v, :n] = s.timestamps_per_var[v]
                    deltat[b, v, :n] = s.deltat_per_var[v]
                    valid_mask[b, v, :n] = True
                target_var[b, v] = bool(s.target_variate_mask[v])
                n_obs[b, v] = n
                value_mu[b, v] = float(s.value_mu_per_var[v])
                value_std[b, v] = float(s.value_std_per_var[v])
            valid_var[b, :V_act] = True
            history[b] = float(s.history)
            time_scale[b] = float(s.time_scale)
            item_id.append(str(s.item_id))
            dataset_name.append(str(s.dataset_name))
            regime_list.append(str(s.regime))
            source_name.append(str(s.source_tag))

        ts_t = torch.from_numpy(timestamps)
        valid_t = torch.from_numpy(valid_mask)
        history_t = torch.from_numpy(history)
        target_var_t = torch.from_numpy(target_var)
        in_horizon = ts_t >= history_t.view(B, 1, 1)
        pred_mask = valid_t & in_horizon & target_var_t.view(B, V_pad, 1)

        return {
            "values":              torch.from_numpy(values),
            "values_orig":         torch.from_numpy(values_orig),
            "value_mu":            torch.from_numpy(value_mu),
            "value_std":           torch.from_numpy(value_std),
            "timestamps":          ts_t,
            "deltat":              torch.from_numpy(deltat),
            "valid_mask":          valid_t,
            "pred_mask":           pred_mask,
            "valid_variate_mask":  torch.from_numpy(valid_var),
            "target_variate_mask": target_var_t,
            "n_obs_per_var":       torch.from_numpy(n_obs),
            "history":             history_t,
            "time_scale":          torch.from_numpy(time_scale),
            "item_id":             item_id,
            "source_name":         source_name,
            "regime":              regime_list,
            "dataset_name":        dataset_name,
        }

    return collate


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_imts_val_loaders(
    data_root: str,
    datasets_list: list[str],
    split: str = "val",
    subset_size: int | None = 1024,
    batch_size: int = 64,
    num_workers: int = 2,
    max_dim: int = 20,
    seed: int = 12345,
    norm_mode: str | None = None,
    name_prefix: str = "",
) -> list[DataLoader]:
    """Build one DataLoader per regime in ``datasets_list``.

    Parameters
    ----------
    data_root, datasets_list, split, subset_size, batch_size, num_workers,
    max_dim, seed
        Standard val plumbing.
    norm_mode : str | None
        Override normalization mode for *all* datasets in this batch.
        ``None`` lets each dataset's ``norm_stats.json`` decide.  Used by
        the IMM-TSF builder which always wants ``precomputed_per_record``.
    name_prefix : str
        Prefix for the logged metric name, e.g. ``"imm_tsf:"`` so we get
        ``val/mse_imm_tsf:EPA-Air`` rather than colliding with the
        existing ``val/mse_<regime>`` from tpatchgnn_data.

    Pretraining script registers all returned loaders as a multi-loader
    val set; Lightning iterates them in order each ``val_check_interval``.
    """
    out: list[DataLoader] = []
    for regime in datasets_list:
        ds = IMTSValDataset(
            data_root=data_root,
            regime=regime,
            split=split,
            subset_size=subset_size,
            seed=seed,
            norm_mode=norm_mode,
            dataset_name=f"{name_prefix}{regime}" if name_prefix else regime,
        )
        out.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
                collate_fn=make_imts_val_collate(max_dim=max_dim),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
        )
    return out
