"""
PyTorch Lightning module for Mamba-based time series forecasting.

Three variants controlled by `dt_mode`:
  "learned"  — vanilla Mamba, delta fully learned (Variant 1)
  "replace"  — delta = true time gap (Option A / Variant 2)
  "additive" — delta = softplus(learned + true dt) (Option C / Variant 3)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy import stats as sp_stats

from .mamba_block import MambaBlock, MambaIrregularBlock


class MambaForecaster(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 384,
        n_layer: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "learned",
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
        history: float = 7.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)

        if dt_mode == "learned":
            block_cls = MambaBlock
            block_kwargs = dict(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            block_cls = MambaIrregularBlock
            block_kwargs = dict(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                dt_mode=dt_mode,
            )

        self.layers = nn.ModuleList(
            [block_cls(**block_kwargs) for _ in range(n_layer)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.history = history
        self._test_outputs = []

    def forward(self, values, delta_t=None):
        """
        Args:
            values:  (B, L)  observation values
            delta_t: (B, L)  time gaps (needed for replace/additive modes)
        Returns:
            preds:   (B, L)  predicted values at each position
        """
        x = self.input_proj(values.unsqueeze(-1))
        for layer in self.layers:
            x = layer(x, delta_t=delta_t)
        x = self.final_norm(x)
        preds = self.output_proj(x).squeeze(-1)
        return preds

    def _compute_loss(self, batch, prefix):
        values = batch["values"]
        timestamps = batch["timestamps"]
        delta_t = batch["delta_t"]

        preds = self.forward(values, delta_t=delta_t)

        pred_mask = timestamps >= self.history
        if pred_mask.sum() == 0:
            return torch.tensor(0.0, device=values.device, requires_grad=True)

        loss = F.mse_loss(preds[pred_mask], values[pred_mask])

        self.log(
            f"{prefix}/mse",
            loss,
            prog_bar=True,
            batch_size=values.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        values = batch["values"]
        timestamps = batch["timestamps"]
        delta_t = batch["delta_t"]

        preds = self.forward(values, delta_t=delta_t)

        pred_mask = timestamps >= self.history
        B = values.shape[0]
        for i in range(B):
            m = pred_mask[i]
            if m.sum() == 0:
                continue
            y_true = values[i][m].detach().cpu().numpy()
            y_pred = preds[i][m].detach().cpu().numpy()

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1 - ss_res / (ss_tot + 1e-10))

            if len(y_true) > 2:
                pearson = float(sp_stats.pearsonr(y_true, y_pred).statistic)
            else:
                pearson = 0.0

            self._test_outputs.append(dict(mse=mse, mae=mae, r2=r2, pearson=pearson))

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        metrics = {
            k: np.mean([o[k] for o in self._test_outputs])
            for k in ("mse", "mae", "r2", "pearson")
        }
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=True)
        self._test_outputs.clear()
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
