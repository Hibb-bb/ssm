"""RoMAE adapted for irregular-multivariate forecasting.

We reuse `RoMAEForPreTraining` unmodified. The adaptation is entirely in
how we build its inputs:

  * Each observation (value at some timestamp from some variate) is ONE token.
  * tubelet_size=(1,1,1), n_channels=1  ->  `patchify` is a no-op.
  * positions = [timestamp, variate_id] as float coords (both RoPEND dims).
  * `mask = pred_mask` (True = forecast target; encoder hides them, decoder reconstructs them).
  * `pad_mask` convention in the RoMAE code: True == PADDING (verified via
    `utils.py:209  per_sample_n = (~pad_mask).sum(dim=1)` and the loss-zeroing
    at `model.py:399`). Our shared datamodule emits pad_mask with True==real,
    so we flip it here.

Loss: the built-in `MSELoss(reduction='none')` already runs only on masked
tokens and zeroes padding -- this is exactly our forecasting loss (MSE on
`timestamp >= history AND real-token`).

For evaluation we rebuild per-sample y_true/y_pred from the logits by noting
that `logits[b, i]` corresponds to the i-th True position in `mask[b]`.
"""

from __future__ import annotations

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

from romae.model import (
    EncoderConfig,
    RoMAEForPreTraining,
    RoMAEForPreTrainingConfig,
)


class RoMAEForecaster(pl.LightningModule):
    def __init__(
        self,
        enc_d_model: int = 288,
        enc_nhead: int = 6,
        enc_depth: int = 7,
        dec_d_model: int = 180,
        dec_nhead: int = 3,
        dec_depth: int = 2,
        max_len: int = 1500,
        p_rope_val: float = 0.75,
        n_vars: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
        history: float = 7.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.history = history
        self.n_vars = n_vars

        cfg = RoMAEForPreTrainingConfig(
            encoder_config=EncoderConfig(
                d_model=enc_d_model, nhead=enc_nhead, depth=enc_depth
            ),
            decoder_config=EncoderConfig(
                d_model=dec_d_model, nhead=dec_nhead, depth=dec_depth
            ),
            use_cls=True,
            pos_encoding="ropend",
            max_len=max_len,
            tubelet_size=(1, 1, 1),
            n_channels=1,
            n_pos_dims=2,
            p_rope_val=p_rope_val,
            normalize_targets=False,
        )
        self.model = RoMAEForPreTraining(cfg)
        self.model.set_loss_fn(nn.MSELoss(reduction="none"))

        self._test_outputs = []

    # -------------------------- input builder --------------------------

    def _build_inputs(self, batch):
        """Turn a flat-tokens batch into (values, positions, mask, pad_mask_romae).

        Inputs:
          batch["values"]:     [B, N]
          batch["timestamps"]: [B, N]
          batch["variate_id"]: [B, N]  long
          batch["pad_mask"]:   [B, N]  bool  (True = real token, OUR convention)
          batch["pred_mask"]:  [B, N]  bool
        Outputs:
          values_5d:    [B, N, 1, 1, 1]
          positions:    [B, 2, N]      float (timestamp, variate_id_as_float)
          mask:         [B, N]         True = forecast target
          pad_mask_romae: [B, N]       True = padding  (RoMAE convention)
        """
        values = batch["values"]
        timestamps = batch["timestamps"]
        variate_id = batch["variate_id"]
        pred_mask = batch["pred_mask"]
        pad_mask_ours = batch["pad_mask"]

        B, N = values.shape

        # Zero out values in the forecast region so nothing leaks through
        # the input projection or the RoPE positions (cf. paper adaptation).
        masked_values = values.clone()
        masked_values[pred_mask] = 0.0

        values_5d = masked_values.view(B, N, 1, 1, 1)
        positions = torch.stack(
            [timestamps, variate_id.to(timestamps.dtype)], dim=1
        )  # [B, 2, N]

        mask = pred_mask                      # True = forecast target
        pad_mask_romae = ~pad_mask_ours       # True = padding
        return values_5d, positions, mask, pad_mask_romae

    # -------------------------- forward --------------------------

    def forward(self, batch):
        values_5d, positions, mask, pad_mask_romae = self._build_inputs(batch)
        logits, loss = self.model(
            values=values_5d,
            mask=mask,
            positions=positions,
            pad_mask=pad_mask_romae,
        )
        return logits, loss

    # -------------------------- lightning hooks --------------------------

    def _compute_loss(self, batch, prefix):
        _, loss = self.forward(batch)
        # RoMAE may return `None` if no positions are masked. Replace with 0.
        if loss is None:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.log(f"{prefix}/mse", loss, prog_bar=True, batch_size=batch["values"].shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    # Rebuild per-sample y_true / y_pred for the eval metrics.
    def test_step(self, batch, batch_idx):
        logits, _ = self.forward(batch)
        # logits: [B, pred_max, 1]  where pred_max = batch-wide constant.
        values = batch["values"]
        pred_mask = batch["pred_mask"]
        pred_real_count = batch["pred_real_count"]
        pad_mask = batch["pad_mask"]
        variate_id = batch["variate_id"]
        item_ids = batch["item_id"]

        B = values.shape[0]
        # Real forecast targets are the FIRST `pred_real_count[i]` True entries
        # of pred_mask[i] (dummies live in the trailing padding region). Since
        # RoMAE's `x[mask]` iterates in positional order, the first
        # `pred_real_count[i]` rows of logits[i] correspond to real targets.
        for i in range(B):
            n_true = int(pred_real_count[i].item())
            if n_true == 0:
                continue
            real_mask = pred_mask[i] & pad_mask[i]
            y_true = values[i][real_mask].detach().cpu().numpy()
            variates = variate_id[i][real_mask].detach().cpu().numpy()
            assert y_true.shape[0] == n_true, (y_true.shape, n_true)
            y_pred = logits[i, :n_true, 0].detach().cpu().float().numpy()

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            pearson = (
                float(sp_stats.pearsonr(y_true, y_pred).statistic)
                if len(y_true) > 2
                else 0.0
            )

            per_var = {}
            for d in range(self.n_vars):
                m_d = variates == d
                if int(m_d.sum()) == 0:
                    continue
                y_t = y_true[m_d]
                y_p = y_pred[m_d]
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
