"""mTAN adapted for irregular-multivariate forecasting.

Adapter strategy
----------------
mTAN's `enc_mtan_rnn` + `dec_mtan_rnn` (Shukla & Marlin 2021, ICLR) expect:
  - x : [B, T, 2V] -- value channels concat with mask channels
        (`x[:, :, :V]` = values; `x[:, :, V:]` = observation mask, 1 if observed).
  - time_steps : [B, T] -- shared time axis across variates within a sample.

Our `per_variate` batch has *async* per-variate timestamps. To feed mTAN we
build a per-sample sorted union of all observed timestamps; at each union
time only ONE variate is observed (the one that produced this token), so the
value goes in column `var_id` and mask is one-hot on `var_id`.

Forecasting framing
-------------------
We use the standard "interpolation as masked-target prediction": forecast
positions are HIDDEN from the encoder (value=0, mask=0 in encoder input)
and then reconstructed by the decoder. The decoder is queried at the same
union time axis and outputs a value for ALL V variates per query time; the
loss is MSE on the (forecast_time, forecast_variate) cells only.

We use the deterministic mean of `qz0` (no IWAE / KL); this turns mTAN into
a deterministic encoder->latent->decoder regressor.

Device caveat
-------------
The vendored mTAN encoder/decoder hardcode `self.device` at __init__ for
internal `.to(device)` calls. We monkey-patch `.device` on each forward so
Lightning's device placement remains authoritative.
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
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _upstream.models import enc_mtan_rnn, dec_mtan_rnn


class MTANForecaster(pl.LightningModule):
    def __init__(
        self,
        n_vars: int = 3,
        rec_hidden: int = 576,
        gen_hidden: int = 576,
        latent_dim: int = 64,
        embed_time: int = 128,
        num_ref_points: int = 64,
        num_heads: int = 1,
        learn_emb: bool = True,
        t_max: float = 10.0,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
        history: float = 8.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_vars = n_vars
        self.latent_dim = latent_dim
        self.history = history

        # Reference points span the full data window so the model can attend
        # over both history and forecast regions.
        ref_pts = torch.linspace(0.0, t_max, num_ref_points)

        # device='cpu' here is just stored on the module; we monkey-patch it
        # in forward() to match the actual Lightning device.
        self.rec = enc_mtan_rnn(
            input_dim=n_vars,
            query=ref_pts,
            latent_dim=latent_dim,
            nhidden=rec_hidden,
            embed_time=embed_time,
            num_heads=num_heads,
            learn_emb=learn_emb,
            device="cpu",
        )
        self.dec = dec_mtan_rnn(
            input_dim=n_vars,
            query=ref_pts,
            latent_dim=latent_dim,
            nhidden=gen_hidden,
            embed_time=embed_time,
            num_heads=num_heads,
            learn_emb=learn_emb,
            device="cpu",
        )

        self._test_outputs = []

    # --------------- input transformation ---------------

    def _build_inputs(self, batch):
        """per_variate batch -> mTAN inputs.

        Returns:
            x_enc          [B, T_max, 2V]  encoder input (value + mask one-hot
                            on the variate axis; zero at forecast positions)
            time_steps     [B, T_max]      union of all observed timestamps,
                            sorted, padded with zeros
            pred_mask_full [B, T_max, V]   True iff (t, var) is a forecast target
            true_full      [B, T_max, V]   ground-truth values at forecast
                            positions, zero elsewhere
            tok_var        [B, T_max]      which variate produced each token
                            (long; 0 for padding -- only used when valid_token)
            valid_token    [B, T_max]      True for real (non-padding) tokens
        """
        values = batch["values"]      # [B, V, L_max]
        timestamps = batch["timestamps"]  # [B, V, L_max]
        valid = batch["valid_mask"]   # [B, V, L_max]
        pred = batch["pred_mask"]     # [B, V, L_max]

        B, V, L_max = values.shape
        device = values.device

        # Per-sample, gather (t, val, var, is_pred) tuples across variates,
        # sort by t, then pad to T_max. Pure tensor ops, no Python loop over
        # individual observations.

        # Flatten per sample: shape [B, V * L_max]
        ts_flat = timestamps.reshape(B, -1)
        vl_flat = values.reshape(B, -1)
        valid_flat = valid.reshape(B, -1)
        pred_flat = pred.reshape(B, -1)
        # Variate id per slot: [V, L_max] then broadcast to [B, V*L_max]
        var_grid = torch.arange(V, device=device).unsqueeze(1).expand(V, L_max).reshape(-1)
        var_flat = var_grid.unsqueeze(0).expand(B, -1)

        # For sorting: send invalid slots to +inf so they cluster at the tail
        ts_for_sort = torch.where(
            valid_flat,
            ts_flat,
            torch.full_like(ts_flat, float("inf")),
        )
        order = torch.argsort(ts_for_sort, dim=1)  # [B, V*L_max]

        ts_sorted = torch.gather(ts_flat, 1, order)
        vl_sorted = torch.gather(vl_flat, 1, order)
        var_sorted = torch.gather(var_flat, 1, order)
        valid_sorted = torch.gather(valid_flat, 1, order)
        pred_sorted = torch.gather(pred_flat, 1, order)

        # Trim to T_max = max valid count across the batch
        n_valid_per_sample = valid_sorted.sum(dim=1)  # [B]
        T_max = int(n_valid_per_sample.max().item()) if B > 0 else 0
        T_max = max(T_max, 1)

        ts_sorted = ts_sorted[:, :T_max]
        vl_sorted = vl_sorted[:, :T_max]
        var_sorted = var_sorted[:, :T_max]
        valid_sorted = valid_sorted[:, :T_max]
        pred_sorted = pred_sorted[:, :T_max]

        # Build [B, T_max, V] one-hot scatter for value+mask channels.
        x_val = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        x_msk = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        true_full = torch.zeros(B, T_max, V, device=device, dtype=values.dtype)
        pred_mask_full = torch.zeros(B, T_max, V, device=device, dtype=torch.bool)

        # Scatter values into per-variate columns. Padding slots have
        # var_sorted = (whatever, but valid_sorted=False) so we mask afterwards.
        idx0 = torch.arange(B, device=device).unsqueeze(1).expand(B, T_max)
        idx1 = torch.arange(T_max, device=device).unsqueeze(0).expand(B, T_max)
        # Forecast positions: ground truth goes to true_full, mask cleared.
        is_real = valid_sorted
        is_pred_real = pred_sorted & valid_sorted
        is_hist_real = is_real & (~pred_sorted)

        # History: x_val[i, t, var] = value, x_msk[i, t, var] = 1
        # Pred: true_full[i, t, var] = value, pred_mask_full[i, t, var] = True
        # Use index_put_ via scatter:
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

        x_enc = torch.cat([x_val, x_msk], dim=-1)  # [B, T_max, 2V]
        time_steps = torch.where(
            valid_sorted, ts_sorted, torch.zeros_like(ts_sorted)
        )

        return x_enc, time_steps, pred_mask_full, true_full, valid_sorted

    # --------------- forward ---------------

    def _patch_device(self):
        cur = next(self.parameters()).device
        self.rec.device = cur
        self.dec.device = cur

    def forward(self, batch):
        self._patch_device()
        x_enc, time_steps, pred_mask_full, true_full, valid_token = self._build_inputs(batch)

        z_out = self.rec(x_enc, time_steps)         # [B, num_ref, 2*latent]
        z_mean = z_out[:, :, :self.latent_dim]      # deterministic
        pred = self.dec(z_mean, time_steps)         # [B, T_max, V]
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
            mask_i = mask_np[i]                  # [T_max, V]
            if not mask_i.any():
                continue
            y_true = true_np[i][mask_i]
            y_pred = pred_np[i][mask_i]
            # Re-derive the variate id of each forecast position so per-variate
            # MSE is meaningful.
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
