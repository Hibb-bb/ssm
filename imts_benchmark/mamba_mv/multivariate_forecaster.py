"""Multivariate Mamba forecaster (PyTorch Lightning).

Pipeline:
    values_per_var, dt_per_var, valid_mask
            |
            v  (per-variate irregular SSM)
    h_per_var : [B, V, L, D_h]
            |
            v  (shared-grid alignment with learnable exp-decay)
    H0 : [B, K, V, D_model], avail, rho
            |
            v  L times: VariableAxisAttention then TemporalMambaOnGrid
    H^(L) : [B, K, V, D_model]
    (``MultivariateMambaSandwichForecaster`` adds extra grid-only Mamba tail.)
            |
            v  (query readout: per-(variate,time) prediction via learnable decay + per-variate head)
    preds : [B, V, L]

Loss is MSE on positions where pred_mask == True
(pred_mask = valid_mask AND timestamps >= history).

Optimizer / scheduler copied verbatim from the existing univariate
`MambaForecaster` for fair comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

from .irregular_ssm import PerVariateIrregularSSM
from .query_readout import QueryReadout
from .shared_grid import SharedGridAligner
from .temporal_mamba import TemporalMambaOnGrid
from .variable_axis_attention import VariableAxisAttention


class MultivariateMambaForecaster(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 256,
        n_vars: int = 3,
        n_perv_layer: int = 3,
        n_fusion_blocks: int = 3,
        n_heads_varattn: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "replace",
        grid_K: int = 128,
        t_max: float = 10.0,
        n_freq: int = 8,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        num_warmup_steps: int = 100,
        num_training_steps: int = 1600,
        history: float = 7.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.history = history

        self.perv_ssm = PerVariateIrregularSSM(
            d_model=d_hidden,
            n_layer=n_perv_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_mode=dt_mode,
        )
        self.grid = SharedGridAligner(
            d_hidden=d_hidden,
            d_model=d_model,
            K=grid_K,
            t_max=t_max,
            n_vars=n_vars,
            n_freq=n_freq,
        )
        self.fusion_attn = nn.ModuleList([
            VariableAxisAttention(
                d_model=d_model,
                n_vars=n_vars,
                n_heads=n_heads_varattn,
                n_freq_rho=n_freq,
            )
            for _ in range(n_fusion_blocks)
        ])
        self.fusion_mamba = nn.ModuleList([
            TemporalMambaOnGrid(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_fusion_blocks)
        ])
        self.readout = QueryReadout(
            d_model=d_model,
            n_vars=n_vars,
            t_max=t_max,
            K=grid_K,
            n_freq=n_freq,
        )

        self._test_outputs = []

    # -------------------------- forward --------------------------

    def _mask_prediction_inputs(self, values, timestamps, valid_mask, history):
        """Zero-out values in the prediction region.

        Use the per-sample history carried by the dataset so masking stays
        aligned with the datamodule's pred_mask even if CLI defaults drift.
        """
        pred_region = timestamps >= history[:, None, None]
        masked = values.clone()
        masked[pred_region] = 0.0
        return masked, pred_region & valid_mask

    def forward(self, batch):
        values = batch["values"]           # [B, V, L]
        timestamps = batch["timestamps"]   # [B, V, L]
        deltat = batch["deltat"]           # [B, V, L]
        valid_mask = batch["valid_mask"]   # [B, V, L]
        history = batch["history"]         # [B]

        masked_values, _ = self._mask_prediction_inputs(
            values, timestamps, valid_mask, history
        )

        h_pv = self.perv_ssm(masked_values, deltat, valid_mask)       # [B,V,L,D_h]
        grid_out = self.grid(h_pv, timestamps, valid_mask)
        H = grid_out["h0"]                                            # [B,K,V,D]
        avail = grid_out["avail"]
        rho = grid_out["rho"]

        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho)
            H = mamba(H)

        B, V, L = values.shape
        query_times = timestamps.reshape(B, V * L)                    # [B, VL]
        query_variate = (
            torch.arange(V, device=values.device)
            .view(1, V, 1)
            .expand(B, V, L)
            .reshape(B, V * L)
        )
        query_valid = valid_mask.reshape(B, V * L)

        preds_flat = self.readout(H, query_times, query_variate, query_valid)
        preds = preds_flat.view(B, V, L)
        return preds

    # -------------------------- lightning hooks --------------------------

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
                if len(y_true) > 2
                else 0.0
            )

            # Per-variate metrics with target-weighted-ready fields
            # (ss_res_v, abs_err_sum_v, count_v). Aggregator reads these
            # to compute global target-weighted per-variate MSE/MAE across
            # all test samples. Useful for the Mamba-vs-baselines cross-
            # variate analysis — particularly v3, which is the convex-mix
            # variate, and particularly Phase 4 where v3 has a gap.
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
            # Phase 4-1 (fixed_v3): d always = 2. Phase 4-2 (random_uniform):
            # d varies per sample per gapped_variate_index. For Phase 2/3 the
            # gap_starts/gap_ends lists are empty and we skip. Keys are indexed
            # by the actual gapped variate so the aggregator can stratify.
            if (i < len(gap_starts_list) and gap_starts_list[i]
                    and i < len(gapped_vars_list) and gapped_vars_list[i]):
                g_s = float(gap_starts_list[i][0])
                g_e = float(gap_ends_list[i][0])
                d = int(gapped_vars_list[i][0])
                ts_vd = timestamps[i, d]                          # [L]
                mask_vd = pred_mask[i, d]                         # [L]
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
        # Global target-weighted aggregation for MSE and MAE so test reporting
        # uses the same SSE/N denominator as the training loss (F.mse_loss on
        # pred_mask positions). R^2 and Pearson stay sample-averaged (they are
        # per-sample correlation-style scores).
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

    def configure_optimizers(self):
        no_decay = set()
        for name, _ in self.named_parameters():
            if "bias" in name or "norm" in name or "_no_weight_decay" in name:
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


class MultivariateMambaSandwichForecaster(MultivariateMambaForecaster):
    """Sandwich stack: irregular per-variate SSM → grid → fusion → tail grid Mamba.

    Default layout (all overridable via the same kwargs as
    ``MultivariateMambaForecaster``):

    - ``n_perv_layer=2``: two irregular (per-variate) Mamba layers before alignment.
    - ``n_fusion_blocks=2``: two cross-variate ``VariableAxisAttention`` +
      ``TemporalMambaOnGrid`` pairs on the shared grid.
    - ``n_tail_grid_mamba=2``: two extra ``TemporalMambaOnGrid`` blocks on the
      grid only (no attention between them), after the fusion pairs.

    Lightning hooks, loss, and readout match the parent class; only ``__init__``
    and ``forward`` differ.
    """

    def __init__(
        self,
        n_tail_grid_mamba: int = 2,
        n_perv_layer: int = 2,
        n_fusion_blocks: int = 2,
        **kwargs,
    ):
        super().__init__(
            n_perv_layer=n_perv_layer,
            n_fusion_blocks=n_fusion_blocks,
            **kwargs,
        )
        self.tail_grid_mamba = nn.ModuleList([
            TemporalMambaOnGrid(
                d_model=self.hparams["d_model"],
                d_state=self.hparams["d_state"],
                d_conv=self.hparams["d_conv"],
                expand=self.hparams["expand"],
            )
            for _ in range(n_tail_grid_mamba)
        ])

    def forward(self, batch):
        values = batch["values"]
        timestamps = batch["timestamps"]
        deltat = batch["deltat"]
        valid_mask = batch["valid_mask"]
        history = batch["history"]

        masked_values, _ = self._mask_prediction_inputs(
            values, timestamps, valid_mask, history
        )

        h_pv = self.perv_ssm(masked_values, deltat, valid_mask)
        grid_out = self.grid(h_pv, timestamps, valid_mask)
        H = grid_out["h0"]
        avail = grid_out["avail"]
        rho = grid_out["rho"]

        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho)
            H = mamba(H)
        for m in self.tail_grid_mamba:
            H = m(H)

        B, V, L = values.shape
        query_times = timestamps.reshape(B, V * L)
        query_variate = (
            torch.arange(V, device=values.device)
            .view(1, V, 1)
            .expand(B, V, L)
            .reshape(B, V * L)
        )
        query_valid = valid_mask.reshape(B, V * L)

        preds_flat = self.readout(H, query_times, query_variate, query_valid)
        preds = preds_flat.view(B, V, L)
        return preds
