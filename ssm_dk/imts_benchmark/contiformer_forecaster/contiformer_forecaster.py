"""ContiFormer adapted for irregular-multivariate forecasting.

Adapter strategy
----------------
The vendored `ContiFormer` (Chen et al. 2023, NeurIPS) is an encoder-only
transformer whose attention is parametrized by a Neural ODE (`ODELinear`)
defined on continuous time. Forward signature: `model(x, t, mask)`.

For our forecasting task we frame it as MAE-style interpolation:
  * Build per-sample sorted union of (timestamp, variate) tuples.
  * Encoder input x: at each union token, value goes in column var_id of a
    [2V] vector (other columns 0); mask channel is one-hot on var_id.
    Forecast positions are HIDDEN: value=0, mask=0.
  * Encoder output: [B, T_max, d_model].
  * Linear head: d_model -> V (predicts all variates per query time).
  * Loss: MSE on (forecast_time, forecast_variate) cells only.

Sized to ~7.65M params via d_model=320, d_inner=1024, n_layers=6, n_head=4,
d_k=80 (within +/-10% of the 7.8M shared budget).

Memory caveat
-------------
ContiFormer's attention projection allocates an [B, L, L, hidden, n_head, d_k]
tensor before attention, which is O(L^2) memory. For L ~ 360 (3 variates *
~120 obs/var) at B=128 this can reach 10s of GB. If we OOM on H100, the
fallback is to (a) drop to B=64, (b) subsample the union token set per sample.
Both are documented escape hatches; the default attempts B=128 with full L.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

_THIS_DIR = Path(__file__).resolve().parent
_UPSTREAM = _THIS_DIR / "_upstream"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from physiopro.network.contiformer import ContiFormer, AttrDict


class ContiFormerForecaster(pl.LightningModule):
    def __init__(
        self,
        n_vars: int = 3,
        d_model: int = 320,
        d_inner: int = 1024,
        n_layers: int = 6,
        n_head: int = 4,
        d_k: int = 80,
        dropout: float = 0.1,
        atol_ode: float = 1e-1,
        rtol_ode: float = 1e-1,
        method_ode: str = "rk4",
        actfn_ode: str = "tanh",
        layer_type_ode: str = "concat",
        interpolate_ode: str = "linear",
        approximate_method: str = "bilinear",
        nlinspace: int = 1,
        zero_init_ode: bool = True,
        linear_type_ode: str = "before",
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
        history: float = 8.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_vars = n_vars
        self.history = history

        # Input dim = 2V (value channels + mask channels, one-hot on variate)
        self.encoder = ContiFormer(
            input_size=2 * n_vars,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_k,
            dropout=dropout,
            actfn_ode=actfn_ode,
            layer_type_ode=layer_type_ode,
            zero_init_ode=zero_init_ode,
            atol_ode=atol_ode,
            rtol_ode=rtol_ode,
            method_ode=method_ode,
            linear_type_ode=linear_type_ode,
            regularize=False,
            approximate_method=approximate_method,
            nlinspace=nlinspace,
            interpolate_ode=interpolate_ode,
            itol_ode=1e-2,
            add_pe=False,
            normalize_before=False,
        )
        self.head = nn.Linear(d_model, n_vars)

        self._test_outputs = []

    # --------------- input transformation ---------------

    def _build_inputs(self, batch):
        """Convert per_variate batch -> ContiFormer inputs (mirror of mTAN
        adapter's _build_inputs).

        Returns:
            x_enc          [B, T_max, 2V]  encoder input (one-hot variate
                            channels for value + mask; zero at forecast tokens)
            time_steps     [B, T_max]      sorted union of observation times
            pred_mask_full [B, T_max, V]   True iff (t, var) is forecast target
            true_full      [B, T_max, V]   ground-truth at forecast cells
            valid_token    [B, T_max]      True for real (non-padding) tokens
        """
        values = batch["values"]
        timestamps = batch["timestamps"]
        valid = batch["valid_mask"]
        pred = batch["pred_mask"]

        B, V, L_max = values.shape
        device = values.device

        ts_flat = timestamps.reshape(B, -1)
        vl_flat = values.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)
        pred_flat = pred.reshape(B, -1)
        var_grid = torch.arange(V, device=device).unsqueeze(1).expand(V, L_max).reshape(-1)
        var_flat = var_grid.unsqueeze(0).expand(B, -1)

        ts_for_sort = torch.where(
            valid_flat,
            ts_flat,
            torch.full_like(ts_flat, float("inf")),
        )
        order = torch.argsort(ts_for_sort, dim=1)
        ts_sorted = torch.gather(ts_flat, 1, order)
        vl_sorted = torch.gather(vl_flat, 1, order)
        var_sorted = torch.gather(var_flat, 1, order)
        valid_sorted = torch.gather(valid_flat, 1, order)
        pred_sorted = torch.gather(pred_flat, 1, order)

        n_valid_per_sample = valid_sorted.sum(dim=1)
        T_max = int(n_valid_per_sample.max().item()) if B > 0 else 0
        T_max = max(T_max, 1)

        ts_sorted = ts_sorted[:, :T_max]
        vl_sorted = vl_sorted[:, :T_max]
        var_sorted = var_sorted[:, :T_max]
        valid_sorted = valid_sorted[:, :T_max]
        pred_sorted = pred_sorted[:, :T_max]

        x_val = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        x_msk = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        true_full = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        pred_mask_full = torch.zeros(B, T_max, V, device=device, dtype=torch.bool)

        idx0 = torch.arange(B, device=device).unsqueeze(1).expand(B, T_max)
        idx1 = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
        is_pred_real = pred_sorted & valid_sorted
        is_hist_real = valid_sorted & (~pred_sorted)

        flat_b = idx0[is_hist_real]
        flat_t = idx1[is_hist_real]
        flat_v = var_sorted[is_hist_real]
        x_val[flat_b, flat_t, flat_v] = vl_sorted[is_hist_real]
        x_msk[flat_b, flat_t, flat_v] = 1.0

        flat_b = idx0[is_pred_real]
        flat_t = idx1[is_pred_real]
        flat_v = var_sorted[is_pred_real]
        true_full[flat_b, flat_t, flat_v] = vl_sorted[is_pred_real]
        pred_mask_full[flat_b, flat_t, flat_v] = True

        x_enc = torch.cat([x_val, x_msk], dim=-1)
        time_steps = torch.where(
            valid_sorted, ts_sorted, torch.zeros_like(ts_sorted)
        )

        return x_enc, time_steps, pred_mask_full, true_full, valid_sorted

    # --------------- forward ---------------

    def forward(self, batch):
        x_enc, time_steps, pred_mask_full, true_full, valid_token = self._build_inputs(batch)
        # ContiFormer's mask convention: True = "should be masked out" (per
        # the masked_fill in ScaledDotProductAttention). For self-attention we
        # broadcast a [B, T_max, T_max] bool mask, blocking attention to
        # padding tokens.
        invalid = ~valid_token  # [B, T_max], True at padding
        attn_mask = invalid.unsqueeze(1).expand(-1, valid_token.shape[1], -1)
        # `enc_output, _last = encoder(x, t, mask)`
        enc_out, _ = self.encoder(x_enc, time_steps, attn_mask)
        pred = self.head(enc_out)  # [B, T_max, V]
        return pred, time_steps, pred_mask_full, true_full

    def _compute_loss(self, batch, prefix):
        pred, _, pred_mask_full, true_full = self.forward(batch)
        diff_sq = (pred - true_full) ** 2
        loss = (diff_sq * pred_mask_full.float()).sum() / pred_mask_full.sum().clamp(min=1)
        self.log(f"{prefix}/mse", loss, prog_bar=True, batch_size=batch["values"].shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        pred, _, pred_mask_full, true_full = self.forward(batch)
        item_ids = batch["item_id"]
        B = pred.shape[0]
        pred_np = pred.detach().cpu().numpy()
        true_np = true_full.detach().cpu().numpy()
        mask_np = pred_mask_full.detach().cpu().numpy()

        for i in range(B):
            mask_i = mask_np[i]
            if not mask_i.any():
                continue
            y_true = true_np[i][mask_i]
            y_pred = pred_np[i][mask_i]
            var_grid = np.broadcast_to(
                np.arange(self.n_vars), mask_i.shape
            )[mask_i]

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            pearson = (
                float(sp_stats.pearsonr(y_true, y_pred).statistic)
                if len(y_true) > 2 else 0.0
            )

            per_var = {}
            for d in range(self.n_vars):
                m_d = var_grid == d
                if int(m_d.sum()) == 0:
                    continue
                y_t = y_true[m_d]; y_p = y_pred[m_d]
                per_var[f"mse_v{d}"] = float(np.mean((y_t - y_p) ** 2))
                per_var[f"n_pred_v{d}"] = int(m_d.sum())

            out = dict(item_id=item_ids[i], mse=mse, mae=mae, r2=r2, pearson=pearson)
            out.update(per_var)
            self._test_outputs.append(out)

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        keys = ("mse", "mae", "r2", "pearson")
        metrics = {k: float(np.mean([o[k] for o in self._test_outputs])) for k in keys}
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=True)
        self._test_agg = metrics

    def configure_optimizers(self):
        no_decay = set()
        for name, _ in self.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.add(name)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in self.named_parameters() if n not in no_decay],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if n in no_decay],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.hparams.lr,
        )

        def lr_lambda(step):
            if step < self.hparams.num_warmup_steps:
                return step / max(1, self.hparams.num_warmup_steps)
            progress = (step - self.hparams.num_warmup_steps) / max(
                1, self.hparams.num_training_steps - self.hparams.num_warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
