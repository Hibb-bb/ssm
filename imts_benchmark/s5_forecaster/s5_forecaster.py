"""
S5 forecaster (per-variate SSM stack) using Kwaijtaal's `s5-pytorch` port of
Smith, Warrington, Linderman — "Simplified State Space Layers for Sequence
Modeling" (ICLR 2023, arXiv:2208.04933).

The `s5-pytorch` package exposes a raw `S5` module whose `forward(signal,
step_scale)` accepts per-step Δt in continuous time. We use the raw `S5` (not
`S5Block`, which doesn't thread step_scale through its forward) and build our
own Pre-LN + FFN block around it — standard transformer-style block layout
but with the SSM replacing attention.

Forecasting framing
-------------------
Same MAE-style forecast-positions-hidden-from-encoder pattern as the other
baselines:
  * Mask values where timestamps >= history (set to 0 before SSM stack).
  * Each variate is processed independently through the stacked S5 blocks
    (per-variate SSM, batched as [B*V, L, D]).
  * Per-time-step linear head projects d_model -> 1 value.
  * MSE loss restricted to pred_mask (= timestamps >= history AND valid).
  * Test metrics use global target-weighted aggregation (SSE / count).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import scipy.stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

# s5-pytorch upstream.
from s5 import S5


class S5TemporalBlock(nn.Module):
    """Pre-LN block: S5(Δt) + residual, then FFN + residual.

    Mirrors the transformer-style block layout but with SSM in place of
    self-attention. step_scale is threaded in from the per-step Δt tensor.
    """

    def __init__(self, d_model: int, state_dim: int, ff_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.s5 = S5(width=d_model, state_width=state_dim)
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = int(ff_mult * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, step_scale: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)   step_scale: (B, L)
        x = x + self.s5(self.ln1(x), step_scale=step_scale)
        x = x + self.ffn(self.ln2(x))
        return x


class S5Forecaster(pl.LightningModule):
    """Multivariate irregular-TS forecaster using per-variate S5 stacks."""

    def __init__(
        self,
        n_vars: int = 3,
        # --- paper-native config (Smith et al. ICLR 2023, S5 pendulum run_train.py) ---
        # Upstream defaults: d_model=128, ssm_size_base=256, n_layers=6,
        # ssm_lr_base=1e-3, weight_decay=0.05 (with SSM params excluded).
        # Our previous 384/96 config (chosen to match a 7.8M param budget)
        # with uniform weight-decay on SSM spectral params (Lambda, log_step)
        # caused 0/5 seeds to escape mean-prediction; reverting to native.
        d_model: int = 128,
        state_dim: int = 256,
        n_layers: int = 6,
        ff_mult: float = 4.0,
        time_emb_dim: int = 64,
        dropout: float = 0.0,
        t_max: float = 10.0,
        history: float = 8.0,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_vars = n_vars
        self.history = history

        # Time embedding: project scalar timestamp to time_emb_dim via sin-cos.
        # Learnable frequency is a simple and standard choice; we just use a
        # linear layer (value embedding is simpler; time info flows via the
        # s5 step_scale too, so duplicate encoding is tolerable).
        self.time_emb_dim = time_emb_dim
        self.time_emb = nn.Linear(1, time_emb_dim)

        self.input_proj = nn.Linear(1 + time_emb_dim, d_model)
        self.blocks = nn.ModuleList(
            [S5TemporalBlock(d_model, state_dim, ff_mult=ff_mult, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        self._test_outputs: list[dict] = []

    # -------------------------- masking --------------------------

    def _mask_prediction_inputs(self, values, timestamps, valid_mask, history):
        """Zero values where timestamps >= per-sample history. Matches the
        datamodule's pred_mask semantics (via batch["history"]), so training
        and evaluation masks agree even if CLI defaults drift."""
        pred_region = timestamps >= history[:, None, None]
        masked = values.clone()
        masked[pred_region] = 0.0
        return masked, pred_region & valid_mask

    # -------------------------- forward --------------------------

    def forward(self, batch):
        values = batch["values"]           # [B, V, L]
        timestamps = batch["timestamps"]
        deltat = batch["deltat"]
        valid_mask = batch["valid_mask"]
        history = batch["history"]         # [B]

        masked_values, _ = self._mask_prediction_inputs(values, timestamps, valid_mask, history)

        B, V, L = values.shape
        # Per-(variate, time) embedding: [value, time_embed] -> d_model.
        val_feat = masked_values.unsqueeze(-1)                        # [B, V, L, 1]
        t_feat = self.time_emb(timestamps.unsqueeze(-1))              # [B, V, L, E]
        x = torch.cat([val_feat, t_feat], dim=-1)                     # [B, V, L, 1+E]
        x = self.input_proj(x)                                        # [B, V, L, D]

        # Batched per-variate SSM: flatten (B, V) into batch dim.
        x = x.reshape(B * V, L, -1)                                   # [B*V, L, D]
        step = deltat.reshape(B * V, L)                               # [B*V, L]

        for block in self.blocks:
            x = block(x, step_scale=step)

        x = self.norm(x)                                              # [B*V, L, D]
        preds = self.head(x).squeeze(-1).reshape(B, V, L)             # [B, V, L]
        return preds

    # -------------------------- loss --------------------------

    def _compute_loss(self, batch, prefix):
        preds = self.forward(batch)
        values = batch["values"]
        pred_mask = batch["pred_mask"]
        if pred_mask.sum() == 0:
            return torch.tensor(0.0, device=values.device, requires_grad=True)
        loss = F.mse_loss(preds[pred_mask], values[pred_mask])
        self.log(f"{prefix}/mse", loss, prog_bar=True, batch_size=values.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        values = batch["values"]
        pred_mask = batch["pred_mask"]
        n_obs_per_var = batch["n_obs_per_var"]
        timestamps = batch["timestamps"]
        preds = self.forward(batch)

        B, V, L = values.shape
        item_ids = batch["item_id"]
        # Phase 4 gap-region info (empty lists for Phase 2/3 data).
        gap_starts_list = batch.get("gap_starts", [[]] * B)
        gap_ends_list = batch.get("gap_ends", [[]] * B)
        gapped_vars_list = batch.get("gapped_variate_index", [[]] * B)
        for i in range(B):
            m_all = pred_mask[i]
            if m_all.sum() == 0:
                continue
            y_true = values[i][m_all].detach().cpu().numpy()
            y_pred = preds[i][m_all].detach().cpu().numpy()

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            abs_err_sum = float(np.sum(np.abs(y_true - y_pred)))
            count = int(len(y_true))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            pearson = (
                float(sp_stats.pearsonr(y_true, y_pred).statistic)
                if len(y_true) > 2 else 0.0
            )

            per_var_mse = {}
            obs_counts = n_obs_per_var[i].tolist()
            for d in range(V):
                m_d = pred_mask[i, d]
                if m_d.sum() == 0:
                    continue
                y_t = values[i, d][m_d].detach().cpu().numpy()
                y_p = preds[i, d][m_d].detach().cpu().numpy()
                per_var_mse[f"mse_v{d}"] = float(np.mean((y_t - y_p) ** 2))
                per_var_mse[f"mae_v{d}"] = float(np.mean(np.abs(y_t - y_p)))
                per_var_mse[f"ss_res_v{d}"] = float(np.sum((y_t - y_p) ** 2))
                per_var_mse[f"abs_err_sum_v{d}"] = float(np.sum(np.abs(y_t - y_p)))
                per_var_mse[f"count_v{d}"] = int(len(y_t))
                per_var_mse[f"n_obs_v{d}"] = int(obs_counts[d])

            # Phase 4 gap-region metrics for the gapped variate d ∈ {0,1,2}.
            # Phase 4-1 (fixed_v3): always d=2. Phase 4-2 (random_uniform):
            # d varies per sample per gapped_variate_index. Empty for Phase
            # 2/3 data. Fields are keyed by the actual gapped variate index
            # so the aggregator can stratify (ss_res_v{d}_{in_gap|out_gap}).
            if (i < len(gap_starts_list) and gap_starts_list[i]
                    and i < len(gapped_vars_list) and gapped_vars_list[i]):
                g_s = float(gap_starts_list[i][0])
                g_e = float(gap_ends_list[i][0])
                d = int(gapped_vars_list[i][0])
                ts_vd = timestamps[i, d]
                mask_vd = pred_mask[i, d]
                in_interval = (ts_vd >= g_s) & (ts_vd < g_e)
                in_gap = mask_vd & in_interval
                out_gap = mask_vd & ~in_interval
                for label, m in (("in_gap", in_gap), ("out_gap", out_gap)):
                    if m.sum() == 0:
                        continue
                    y_t = values[i, d][m].detach().cpu().numpy()
                    y_p = preds[i, d][m].detach().cpu().numpy()
                    per_var_mse[f"ss_res_v{d}_{label}"] = float(np.sum((y_t - y_p) ** 2))
                    per_var_mse[f"abs_err_sum_v{d}_{label}"] = float(np.sum(np.abs(y_t - y_p)))
                    per_var_mse[f"count_v{d}_{label}"] = int(len(y_t))
                per_var_mse["gapped_variate"] = d

            out = dict(
                item_id=item_ids[i], mse=mse, mae=mae, r2=r2, pearson=pearson,
                ss_res=ss_res, abs_err_sum=abs_err_sum, count=count,
            )
            out.update(per_var_mse)
            self._test_outputs.append(out)

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        # Target-weighted MSE/MAE matches training-loss convention; R^2 and
        # Pearson stay sample-averaged (per-sample correlation scores).
        total_ss_res = sum(o["ss_res"] for o in self._test_outputs)
        total_abs_err = sum(o["abs_err_sum"] for o in self._test_outputs)
        total_count = max(sum(o["count"] for o in self._test_outputs), 1)
        metrics = {
            "mse": total_ss_res / total_count,
            "mae": total_abs_err / total_count,
            "r2": float(np.mean([o["r2"] for o in self._test_outputs])),
            "pearson": float(np.mean([o["pearson"] for o in self._test_outputs])),
        }
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=True)
        self._test_agg = metrics

    # -------------------------- optim --------------------------

    def configure_optimizers(self):
        # S5 paper (Smith et al. 2023) excludes SSM-structured params from
        # weight decay: Lambda (diagonal HiPPO eigenvalues), log_step
        # (per-state discretization timestep), D (skip connection), and
        # B/C init factors (BH/BP/CH/CP). These carry the structured bias;
        # decaying them toward zero collapses the SSM's frequency response.
        ssm_keywords = ("Lambda", "log_step", ".D", "BH", "BP", "CH", "CP")
        no_decay = set()
        for name, _ in self.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.add(name)
            elif any(k in name for k in ssm_keywords):
                no_decay.add(name)
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.named_parameters() if n not in no_decay],
                 "weight_decay": self.hparams.weight_decay},
                {"params": [p for n, p in self.named_parameters() if n in no_decay],
                 "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )

        num_warmup = self.hparams.num_warmup_steps
        num_total = self.hparams.num_training_steps

        def lr_lambda(step):
            if step < num_warmup:
                return float(step) / float(max(1, num_warmup))
            progress = float(step - num_warmup) / float(max(1, num_total - num_warmup))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
