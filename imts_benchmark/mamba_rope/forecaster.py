"""Hybrid: per-variate Mamba SSM (front) → axial-RoPE attention (back).

Design:
  Stage 1: Per-variate Mamba SSM (channel-independent, dt_mode='learned')
           Reuses imts_benchmark.mamba_mv.irregular_ssm.PerVariateIrregularSSM.
           Produces [B, V, L_max, D] hidden states.
  Stage 2: Flatten to [B, V*L_max, D] tokens with axial position (t, variate_id).
           Apply L axial-RoPE transformer layers (romae library's Encoder + NDPRope).
  Stage 3: Linear readout per token. Loss = MSE on pred_mask positions.

Predictions: SSM input is zero-masked at pred positions (same as Mamba-MV); the
encoder then contextualizes with cross-variate info via axial RoPE attention.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

from romae.model import Encoder, EncoderConfig
from romae.positional_embeddings import NDPRope

from imts_benchmark.mamba_mv.irregular_ssm import PerVariateIrregularSSM
from imts_benchmark.shared_config.global_metrics import (
    aggregate_global_metrics,
    per_sample_variable_sums,
)


class MambaRoPEForecaster(pl.LightningModule):
    """Per-variate Mamba SSM + axial-RoPE flat attention.

    Args:
        d_model: hidden dim, shared between SSM and attention encoder.
        n_perv_layer: number of Mamba blocks in per-variate SSM stack.
        n_attn_layer: number of axial-RoPE transformer layers.
        nhead: attention heads.
        d_state, d_conv, expand: Mamba SSM hyperparameters.
        dt_mode: 'learned' (default), 'replace', or 'concat'. Default 'learned'
            because attention's RoPE handles temporal positioning; SSM stays
            time-agnostic.
        n_vars: number of variates (V).
        max_len_per_var: pad length per variate (L_max).
        history: forecast cutoff time (anything >= history is a target).
        lr, weight_decay, num_warmup_steps, num_training_steps: optimizer.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_perv_layer: int = 3,
        n_attn_layer: int = 4,
        nhead: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "learned",
        p_rope: float = 0.75,
        n_vars: int = 5,
        max_len_per_var: int = 256,
        history: float = 24.0,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.history = history
        self.n_vars = n_vars
        self.d_model = d_model

        # Stage 1: per-variate Mamba SSM
        self.perv_ssm = PerVariateIrregularSSM(
            d_model=d_model,
            n_layer=n_perv_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_mode=dt_mode,
        )

        # Stage 2: axial-RoPE transformer encoder (romae's Encoder + NDPRope)
        enc_cfg = EncoderConfig(
            d_model=d_model,
            nhead=nhead,
            depth=n_attn_layer,
        )
        self.encoder = Encoder(enc_cfg)

        # Axial RoPE on (timestamp, variate_id) — n_dims=2
        head_dim = d_model // nhead
        self.pos_emb = NDPRope(
            head_dim=head_dim,
            base=10000,
            p=p_rope,
            n_dims=2,
        )

        # Stage 3: scalar readout per token
        self.readout = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

        self._test_outputs = []

    # -------------------- helpers --------------------

    def _mask_prediction_inputs(self, values, timestamps, valid_mask, history):
        """Zero out values at pred positions so the SSM doesn't see future info.
        Mirrors mamba_mv.MultivariateMambaForecaster._mask_prediction_inputs."""
        is_future = (timestamps >= history) & valid_mask
        masked_values = values.clone()
        masked_values[is_future] = 0.0
        return masked_values, is_future

    # -------------------- forward --------------------

    def forward(self, batch):
        values = batch["values"]              # [B, V, L]
        timestamps = batch["timestamps"]      # [B, V, L]
        deltat = batch["deltat"]              # [B, V, L]
        valid_mask = batch["valid_mask"]      # [B, V, L]
        history = float(self.history)

        B, V, L = values.shape
        D = self.d_model

        # Stage 1: per-variate SSM (with prediction inputs masked to zero)
        masked_values, _ = self._mask_prediction_inputs(
            values, timestamps, valid_mask, history
        )
        h_pv = self.perv_ssm(masked_values, deltat, valid_mask)   # [B, V, L, D]

        # Stage 2: flatten to flat tokens
        h_flat = h_pv.reshape(B, V * L, D)                        # [B, N, D]

        # Build axial positions: (timestamp, variate_id) for each token
        ts_flat = timestamps.reshape(B, V * L)                    # [B, N]
        var_id = torch.arange(V, device=values.device, dtype=ts_flat.dtype)
        var_id = var_id.view(1, V, 1).expand(B, V, L).reshape(B, V * L)
        positions = torch.stack([ts_flat, var_id], dim=1)         # [B, 2, N]

        # Padding-aware attention mask: True = ignore (False = real token)
        valid_flat = valid_mask.reshape(B, V * L)                 # [B, N]
        # romae Attention's `mask` arg is added to attn_logits; use SDPA-style
        # additive mask: 0 for real positions, -inf for padding.
        # Build [B, 1, 1, N] so it broadcasts over (heads, query_seq).
        attn_mask = torch.zeros(B, 1, 1, V * L, device=values.device, dtype=h_flat.dtype)
        attn_mask = attn_mask.masked_fill(~valid_flat[:, None, None, :], float("-inf"))

        # Stage 3: encoder. NDPRope caches sin/cos from the first call and never
        # invalidates — must reset per batch since positions vary.
        self.pos_emb.reset_cache()
        h_enc = self.encoder(h_flat, positions, self.pos_emb, attn_mask=attn_mask)
        h_enc = self.norm(h_enc)

        # Stage 4: scalar readout
        preds_flat = self.readout(h_enc).squeeze(-1)              # [B, N]
        preds = preds_flat.reshape(B, V, L)
        return preds

    # -------------------- lightning hooks --------------------

    def _compute_loss(self, batch, prefix):
        values = batch["values"]
        pred_mask = batch["pred_mask"]
        preds = self.forward(batch)

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
        pred_mask = batch["pred_mask"]
        timestamps = batch["timestamps"]
        item_ids = batch["item_id"]
        preds = self.forward(batch)

        B, V, L = values.shape
        # Phase-4 gap info (empty if not provided)
        gap_starts_list = batch.get("gap_starts", [[]] * B)
        gap_ends_list = batch.get("gap_ends", [[]] * B)
        gapped_vars_list = batch.get("gapped_variate_index", [[]] * B)

        for i in range(B):
            sample_pred_mask = pred_mask[i]
            if sample_pred_mask.sum() == 0:
                continue

            y_true = values[i][sample_pred_mask].detach().cpu().numpy()
            y_pred = preds[i][sample_pred_mask].detach().cpu().float().numpy()
            ts = timestamps[i].detach().cpu().numpy()                 # [V, L]

            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            abs_err_sum = float(np.sum(np.abs(y_true - y_pred)))
            count = int(len(y_true))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1 - ss_res / (ss_tot + 1e-10))
            pearson = (
                float(sp_stats.pearsonr(y_true, y_pred).statistic)
                if len(y_true) > 2
                else 0.0
            )

            per_var = {}
            for d in range(self.n_vars):
                vd_mask = sample_pred_mask[d]
                if int(vd_mask.sum()) == 0:
                    continue
                y_t = values[i, d][vd_mask].detach().cpu().numpy()
                y_p = preds[i, d][vd_mask].detach().cpu().float().numpy()
                per_var[f"mse_v{d}"] = float(np.mean((y_t - y_p) ** 2))
                per_var[f"mae_v{d}"] = float(np.mean(np.abs(y_t - y_p)))
                per_var[f"n_pred_v{d}"] = int(vd_mask.sum())
                for k, v in per_sample_variable_sums(y_t, y_p).items():
                    per_var[f"{k}_v{d}"] = v

            # Phase-4 gap-region metrics (if provided)
            if (i < len(gap_starts_list) and gap_starts_list[i]
                    and i < len(gapped_vars_list) and gapped_vars_list[i]):
                g_s = float(gap_starts_list[i][0])
                g_e = float(gap_ends_list[i][0])
                d = int(gapped_vars_list[i][0])
                vd_mask = sample_pred_mask[d]
                if vd_mask.sum() > 0:
                    ts_vd = ts[d][vd_mask.cpu().numpy()]
                    y_t_vd = values[i, d][vd_mask].detach().cpu().numpy()
                    y_p_vd = preds[i, d][vd_mask].detach().cpu().float().numpy()
                    in_interval = (ts_vd >= g_s) & (ts_vd < g_e)
                    for label, region_mask in (("in_gap", in_interval), ("out_gap", ~in_interval)):
                        if region_mask.sum() == 0:
                            continue
                        y_t = y_t_vd[region_mask]
                        y_p = y_p_vd[region_mask]
                        per_var[f"ss_res_v{d}_{label}"] = float(np.sum((y_t - y_p) ** 2))
                        per_var[f"abs_err_sum_v{d}_{label}"] = float(np.sum(np.abs(y_t - y_p)))
                        per_var[f"count_v{d}_{label}"] = int(len(y_t))
                per_var["gapped_variate"] = d

            out = dict(
                item_id=item_ids[i], mse=mse, mae=mae, r2=r2, pearson=pearson,
                ss_res=ss_res, abs_err_sum=abs_err_sum, count=count,
            )
            out.update(per_var)
            self._test_outputs.append(out)

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return
        n_vars = int(self.n_vars)
        metrics = aggregate_global_metrics(self._test_outputs, n_vars)
        for k in ("mse", "mae", "mse_tpg", "mae_tpg", "r2", "pearson"):
            self.log(f"test/{k}", metrics[k], prog_bar=True)
        self._test_agg = {k: metrics[k] for k in
                          ("mse", "mae", "mse_tpg", "mae_tpg", "r2", "pearson")}
        self._test_per_var = {k: metrics[k] for k in
                              ("r2_per_var", "pearson_per_var",
                               "mse_per_var", "mae_per_var", "n_per_var")}

    def configure_optimizers(self):
        no_decay = set()
        for name, _ in self.named_parameters():
            if "bias" in name or "norm" in name:
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

        def lr_lambda(step):
            if step < self.hparams.num_warmup_steps:
                return step / max(1, self.hparams.num_warmup_steps)
            progress = (step - self.hparams.num_warmup_steps) / max(
                1, self.hparams.num_training_steps - self.hparams.num_warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
