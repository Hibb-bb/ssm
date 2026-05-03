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
        # ---- Pretraining / FM extensions ----
        # If ``max_dim`` is set, it overrides ``n_vars`` for the size of
        # the variate-embedding tables. The pretraining datamodule pads
        # batches to this dimension and randomizes slot assignment so
        # the model treats variate IDs as anonymous slot IDs.
        max_dim: "int | None" = None,
        # Stable point loss (Time-MoE §3.2.2). Default stays "mse" for
        # IMTS-benchmark backward compatibility; pretraining passes
        # "huber" via the train script.
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        lr_schedule: str = "cosine",
    ):
        super().__init__()
        # Resolve effective variate-table size before save_hyperparameters
        if max_dim is not None and int(max_dim) > 0:
            n_vars = int(max_dim)
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

    def _build_var_id(self, B: int, V: int, *, device) -> torch.Tensor:
        """Per-batch-element variate IDs for any-variate attention bias.

        Shape ``[B, V]``.  Each batch element is one window with V *distinct*
        variates (we never pack multiple series into one batch element), so
        slot indices ``arange(V)`` are unique within an element.  Under this
        choice the binary same/different bias reduces to a learned self-vs-other
        bias on the V x V attention matrix at each grid step — a simple but
        useful inductive prior, equivalent to a learned diagonal vs off-diagonal
        offset per head.  See Moirai-1 §3.2 for the general protocol; in their
        case multiple packed series motivate per-series random IDs, which does
        not apply here.
        """
        return (
            torch.arange(V, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, V)
        )

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

        B, V, L = values.shape
        var_id = self._build_var_id(B, V, device=values.device)        # [B, V]

        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho, var_id=var_id)
            H = mamba(H)

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

        if self.hparams.loss_type == "huber":
            loss = F.smooth_l1_loss(
                preds[pred_mask],
                values[pred_mask],
                beta=self.hparams.huber_delta,
            )
            log_key = f"{prefix}/huber"
        else:
            loss = F.mse_loss(preds[pred_mask], values[pred_mask])
            log_key = f"{prefix}/mse"
        self.log(
            log_key,
            loss,
            prog_bar=True,
            batch_size=values.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        is_imts_val = (
            "values_orig" in batch
            and "value_mu" in batch
            and "value_std" in batch
        )
        if is_imts_val:
            # Downstream IMTS val: only log the *denormalized* MSE/MAE
            # in the dataset's original units. The default normalized
            # huber would be misleading here because per-(b,v) std can
            # collapse to ``min_std`` for near-constant contexts (e.g.
            # USHCN precipitation), inflating z-scored future values.
            self._log_imts_val(batch)
            return None
        # Pretraining-style synthetic val: keep existing behavior so
        # the IMTS benchmark code paths and our pretrain val logs are
        # both consistent with what was here before.
        return self._compute_loss(batch, "val")

    def _log_imts_val(self, batch) -> None:
        """Compute val metrics for IMTS downstream eval.

        The downstream val loader produces a batch where ``values`` has
        already been per-(b,v) standardized:

        - For ``tpatchgnn_data`` (activity, ushcn) the standardization is
          *context-only*, computed inside ``IMTSValDataset.__getitem__``.
        - For Time-IMM / IMM-TSF the standardization is *per-record
          global*, precomputed by the converter to mirror the IMM-TSF
          paper's normalization (``parse_datasets.py:103-111``).

        We log **two** metric variants so each is comparable to the
        right reference numbers:

        - ``val/mse_<ds>``    : MSE in **original units** (preds and
          truth both denormalized with the carried ``value_mu`` /
          ``value_std``).  Matches what the existing IMTS benchmark
          (``train_mv.py``) reports for activity/ushcn.
        - ``val/mse_z_<ds>``  : MSE in **z-scored space** (no
          denormalization).  Matches what IMM-TSF's
          ``lib/evaluation.py::compute_error`` reports for "MSE" /
          "MAE" -- those work on the model's output in z-scored space
          directly, see lines 27-30.  Required for paper-Table parity
          on the eight Time-IMM datasets.

        ``val/mae_*`` mirrors the same split.
        """
        # The model is trained on asinh-z inputs (see collate.py +
        # val_imts.py), so its outputs live in asinh-z space.  We invert
        # with ``sinh`` to recover raw z, then denormalize for original
        # units.  The (mu, std) carried in batch are the *raw-z* mu/std,
        # i.e. (orig - mu) / std = raw_z.
        #
        # IMM-TSF paper-comparable metric requires raw z-MSE, NOT
        # asinh-z-MSE, so we always compare in raw z and orig units.
        # See README §3.5 + §Q4 for derivation.
        #
        # Bounded inversion: sinh grows exponentially.  Across the eight
        # IMM-TSF val sets the true |asinh-z| target tops out at 4.05
        # (FNSPID) and is below 3.5 for everything else.  An untrained or
        # off-distribution model can occasionally emit asinh-z=20-30 for
        # a single timestep on a single sample, which sinh() inflates to
        # 1e8-1e13 and dominates the average MSE (we observed inf at
        # step 44K of single_phase, see README §7.K).  Clamping
        # preds_asinh to ±10 keeps the worst sample-level z-error at
        # sinh(10)≈1.1e4 — bad numerically but no longer "infinity",
        # so EarlyStopping/ckpt-selection still works.  This does NOT
        # change training behavior, only the eval metric robustness.
        with torch.no_grad():
            preds_asinh = self.forward(batch)        # [B, V, L]  asinh-z space
            preds_asinh = preds_asinh.clamp(-10.0, 10.0)
        preds_z = torch.sinh(preds_asinh)            # raw z
        mu = batch["value_mu"].to(preds_z.dtype)     # [B, V]
        std = batch["value_std"].to(preds_z.dtype)   # [B, V]
        truth_orig = batch["values_orig"].to(preds_z.dtype)
        # Recompute raw-z truth from orig + (mu, std) — more numerically
        # stable than sinh(batch["values"]) which round-trips through asinh.
        truth_z = (truth_orig - mu.unsqueeze(-1)) / std.unsqueeze(-1).clamp_min(1e-12)
        preds_orig = preds_z * std.unsqueeze(-1) + mu.unsqueeze(-1)
        pm = batch["pred_mask"]
        if pm.sum() == 0:
            return

        diff_z = (preds_z - truth_z)[pm]
        diff_orig = (preds_orig - truth_orig)[pm]
        mse_z = (diff_z.float() ** 2).mean()
        mae_z = diff_z.float().abs().mean()
        mse = (diff_orig.float() ** 2).mean()
        mae = diff_orig.float().abs().mean()

        # All samples in a batch come from one dataset (val loaders are
        # per-dataset). Take the first tag.
        ds_names = batch.get("dataset_name", [])
        ds = ds_names[0] if ds_names else "imts"
        bs = preds_orig.shape[0]
        # Original-units metrics: prog_bar on so we can eyeball model
        # health during training; not paper-comparable for IMM-TSF.
        self.log(f"val/mse_{ds}", mse, batch_size=bs, prog_bar=True, add_dataloader_idx=False)
        self.log(f"val/mae_{ds}", mae, batch_size=bs, prog_bar=False, add_dataloader_idx=False)
        # z-scored metrics: only for IMM-TSF (where it's paper-comparable
        # via parse_datasets.py:103-111 + evaluation.py:27-30).
        # Activity/USHCN have known fixed scales and the published baselines
        # report original-unit MSE, so per-sample z is uninformative there
        # (and historically inflates to 1e10 because per-(b,v) std collapses
        # for near-constant series — see README §Q4).
        if ds.startswith("imm_"):
            self.log(f"val/mse_z_{ds}", mse_z, batch_size=bs, prog_bar=False, add_dataloader_idx=False)
            self.log(f"val/mae_z_{ds}", mae_z, batch_size=bs, prog_bar=False, add_dataloader_idx=False)

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
                # Streaming sums for the global per-variable R² / Pearson
                # aggregator in on_test_epoch_end (see shared_config.global_metrics).
                # Cheap (O(L) per sample, per variable) and JSON-serialisable.
                per_var_mse[f"sum_y_v{d}"]   = float(y_t.sum())
                per_var_mse[f"sum_y2_v{d}"]  = float((y_t * y_t).sum())
                per_var_mse[f"sum_yh_v{d}"]  = float(y_p.sum())
                per_var_mse[f"sum_yh2_v{d}"] = float((y_p * y_p).sum())
                per_var_mse[f"sum_yyh_v{d}"] = float((y_t * y_p).sum())

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
        # MSE/MAE: global target-weighted (SSE/N over all observed targets)
        # so the reported number matches F.mse_loss's denominator.
        # R^2 / Pearson: per-variable global, then averaged across variables.
        # See shared_config.global_metrics for the rationale (the previous
        # per-sample averaging blew up on USHCN where many samples have
        # ss_tot ≈ 0 because count_per_sample is 1–5).
        from imts_benchmark.shared_config.global_metrics import (
            aggregate_global_metrics,
        )
        agg = aggregate_global_metrics(
            self._test_outputs, n_vars=int(self.hparams.n_vars)
        )
        # Keep the legacy per-sample averages too for diagnostics; they are
        # NOT logged as the canonical test/r2 / test/pearson any more.
        legacy_r2_per_sample = float(np.mean([o["r2"] for o in self._test_outputs]))
        legacy_pearson_per_sample = float(
            np.nanmean([o["pearson"] for o in self._test_outputs])
        )
        metrics = {
            "mse": agg["mse"],
            "mae": agg["mae"],
            "mse_tpg": agg["mse_tpg"],
            "mae_tpg": agg["mae_tpg"],
            "r2": agg["r2"],
            "pearson": agg["pearson"],
            "r2_per_sample_legacy": legacy_r2_per_sample,
            "pearson_per_sample_legacy": legacy_pearson_per_sample,
        }
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=k in {"mse", "mae", "r2", "pearson"})
        # Stash extras (per-variable breakdowns, pooled counts) for downstream
        # tooling. Lists are kept off the live logger because W&B doesn't
        # natively log Python lists; train_mv writes them to test_metrics.csv.
        self._test_agg = metrics
        self._test_agg_extra = {
            "r2_per_var":      agg["r2_per_var"],
            "pearson_per_var": agg["pearson_per_var"],
            "n_per_var":       agg["n_per_var"],
        }

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

        # ``lr_schedule`` controls the post-warmup behavior:
        #   "cosine"    : decay 1.0 -> 0.0 over (num_training_steps - num_warmup_steps).
        #                 Default; matches the original imts_benchmark behavior.
        #   "constant"  : stay at peak LR for the entire post-warmup span.
        #   "multistep" : DeepSeek-LLM (Bi et al., 2024, §3.3) recipe.  Stay at
        #                 peak LR until 80% of training, drop to 31.6% (= √0.1)
        #                 of peak for next 10%, then drop to 10% of peak for the
        #                 last 10%.  Designed for continuation-training: the
        #                 first 80% is constant LR, so extending a run by
        #                 raising num_training_steps lets the constant phase
        #                 grow without replaying earlier steps.
        schedule_name = getattr(self.hparams, "lr_schedule", "cosine")

        def lr_lambda(step):
            warmup = self.hparams.num_warmup_steps
            total = self.hparams.num_training_steps
            if step < warmup:
                return step / max(1, warmup)
            if schedule_name == "constant":
                return 1.0
            if schedule_name == "multistep":
                # DeepSeek-LLM 80/90 schedule
                if step < 0.8 * total:
                    return 1.0
                elif step < 0.9 * total:
                    return 0.316  # √0.1
                else:
                    return 0.10
            # cosine (default)
            progress = (step - warmup) / max(1, total - warmup)
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

        B, V, L = values.shape
        var_id = self._build_var_id(B, V, device=values.device)        # [B, V]

        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho, var_id=var_id)
            H = mamba(H)
        for m in self.tail_grid_mamba:
            H = m(H)

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
