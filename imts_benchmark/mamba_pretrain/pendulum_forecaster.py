"""Pendulum image-regression variant of mamba_pretrain.

Per-timestep regression on irregularly-sampled 24x24 pendulum images
(Becker et al. 2019; Schirmer et al. 2022; Smith et al. 2023; Zivanovic
et al. 2025). Predicts (sin theta_t, cos theta_t) at every observed
timestep -- NOT future-forecasting.

Pipeline (V=1 throughout, single image stream):

    images [B, T, 24, 24]
            |
            v  Linear(576 -> d_hidden) image patch-embed (RoMAE-style)
    emb    [B, T, d_hidden]
            |
            v  reshape to [B, V=1, T, d_hidden] (skip scalar input_proj)
            v  per-variate irregular SSM stack (n_perv_layer)
    h_pv   [B, V=1, T, d_hidden]
            |
            v  shared-grid alignment with learnable exp-decay
    H0     [B, K, V=1, d_model], avail, rho
            |
            v  L x (VariableAxisAttention + TemporalMambaOnGrid)
    H^(L)  [B, K, V=1, d_model]
            |
            v  query readout at the original (irregular) timestamps,
            v  2-output head Linear(d_model + 2*n_freq, 2)
    preds  [B, V=1, T, 2]

Loss: per-timestep MSE on (sin, cos) over all observed timesteps.
No `history` masking -- every observed image is a target.

Reuses MambaBlock / MambaIrregularBlock / SharedGridAligner /
VariableAxisAttention / TemporalMambaOnGrid from mamba_pretrain so
the per-variate SSM and fusion stack are byte-identical to what
PhysioNet trains. Only the input front-end and the query head differ.

Optional Gaussian-likelihood head (head_type='gaussian') matches
S5/CRU/RKN's protocol for an apples-to-apples ablation row.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import MambaBlock, MambaIrregularBlock
from .shared_grid import SharedGridAligner, sinusoidal_encode
from .temporal_mamba import TemporalMambaOnGrid
from .variable_axis_attention import VariableAxisAttention


IMAGE_FLAT_DIM = 24 * 24  # 576
IMAGE_SIDE = 24


class _SchirmerCNNEncoder(nn.Module):
    """Schirmer 2022 CNN encoder, used verbatim by S5 / CRU / RKN / RKN-Δt /
    GRU / GRU-Δt / ODE-RNN / GRU-ODE-Bayes / Latent ODE / mTAND / f-CRU on
    the Pendulum dataset.

    Spec from S5 paper App. G.3.8 (verbatim):
        convolution, ReLU, max pool, convolution, ReLU, max pool,
        dense, ReLU, dense

      Conv2d(1 → 12, kernel=5, padding=(2, 2))     ReLU   MaxPool(2, s=2)
      Conv2d(12 → 12, kernel=3, stride=2, padding=(1, 1))  ReLU  MaxPool(2, s=2)
      Flatten → Linear(108 → 30) ReLU → Linear(30 → 30)

    Output H = 30 features (S5/Schirmer convention). To use with our v1
    body width (d_hidden = 64), an `adapter = Linear(30 → d_hidden)` is
    appended. When d_hidden == 30 the adapter is Identity (no extra params).

    Param count: ~5.8K encoder + ~2K adapter (at d=64).
    """

    def __init__(self, d_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # After conv1 (24x24) + pool (12x12), conv2 stride2+pad1 (6x6), pool (3x3).
        self.fc1 = nn.Linear(12 * 3 * 3, 30)
        self.fc2 = nn.Linear(30, 30)
        self.adapter = nn.Linear(30, d_out) if d_out != 30 else nn.Identity()

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: [B, V, T, 576]   (matches the existing image_embed contract)
        # We reshape into [B*V*T, 1, 24, 24] for Conv2d, then back to
        # [B, V, T, d_out] so the rest of the forward path is unchanged.
        B, V, T, _ = x_flat.shape
        x = x_flat.reshape(B * V * T, 1, IMAGE_SIDE, IMAGE_SIDE)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)                                   # [B*V*T, 108]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                                    # [B*V*T, 30]
        x = self.adapter(x)                                # [B*V*T, d_out]
        return x.reshape(B, V, T, -1)


class _QueryReadout2D(nn.Module):
    """Query-time readout producing 2 outputs per (variate, query_time).

    Mirrors mamba_pretrain.query_readout.QueryReadout exactly except the
    final head is Linear(d_model + 2*n_freq, 2) instead of (..., 1).
    Kept local so QueryReadout (used by PhysioNet) is not modified.
    """

    def __init__(self, d_model: int, t_max: float, K: int, n_freq: int = 8,
                 out_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.K = K
        self.t_max = t_max
        self.n_freq = n_freq
        self.out_dim = out_dim
        grid = torch.linspace(0.0, t_max, K)
        self.register_buffer("grid", grid, persistent=False)
        self.gamma_raw = nn.Parameter(torch.full((d_model,), -3.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d_model + 2 * n_freq, out_dim)

    def forward(
        self,
        H: torch.Tensor,               # [B, K, V, D]
        query_times: torch.Tensor,     # [B, Q] float
        query_variate: torch.Tensor,   # [B, Q] long
        query_valid: torch.Tensor,     # [B, Q] bool
    ) -> torch.Tensor:
        B, K, V, D = H.shape
        Q = query_times.shape[1]
        device = H.device
        grid = self.grid.to(device)

        dt = grid.view(1, 1, K) - query_times.view(B, Q, 1)
        leq = dt <= 0
        kappa = leq.sum(dim=-1).clamp(min=1) - 1
        omega = query_times - grid[kappa]

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, Q)
        h_at = H[batch_idx, kappa, query_variate]                       # [B, Q, D]

        gamma = F.softplus(self.gamma_raw).view(1, 1, D)
        decay = torch.exp(-gamma * omega.unsqueeze(-1))
        xi = h_at * decay

        omega_feat = sinusoidal_encode(omega, n_freq=self.n_freq)
        feat = torch.cat([xi, omega_feat], dim=-1)
        feat = self.dropout(feat)                                       # regularization at head input
        out = self.head(feat)                                           # [B, Q, out_dim]

        out = torch.where(query_valid.unsqueeze(-1), out, torch.zeros_like(out))
        return out


def _cosine_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PendulumMambaForecaster(pl.LightningModule):
    """Pendulum-regression Lightning module.

    Reuses mamba_pretrain's per-variate Mamba stack (MambaIrregularBlock /
    MambaBlock), grid aligner, and fusion stack. Differs from
    MultivariateMambaForecaster only in:
      - input front-end: Linear(576, d_hidden) image patch-embed
      - V=1 hard-coded
      - 2-output query head (Linear or Gaussian-NLL)
      - loss: per-timestep MSE on (sin, cos), no history mask
    """

    def __init__(
        self,
        # ---- image / task ----
        image_dim: int = IMAGE_FLAT_DIM,
        out_dim: int = 2,                 # (sin, cos)
        head_type: str = "gaussian",      # "gaussian" (SSM family) or "linear" (RoMAE-style ablation)
        # ---- body (matches mamba_pretrain SMALL on PhysioNet by default) ----
        d_model: int = 64,
        d_hidden: int = 64,
        n_perv_layer: int = 3,
        n_fusion_blocks: int = 3,
        n_heads_varattn: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "replace",
        grid_K: int = 256,
        t_max: float = 100.0,             # CRU/S5 use T=100 for pendulum
        n_freq: int = 8,
        # ---- optim ----
        # Defaults aligned with S5 (Table 11) / CRU (Schirmer 2022) — same SSM
        # family our architecture lives in. RoMAE used wd=0.01, epochs=50; we
        # don't match those because RoMAE is a Transformer (different family).
        lr: float = 5e-3,
        weight_decay: float = 0.0,        # S5/CRU pendulum default; v2 sweep
        warmup_pct: float = 0.10,         # 10% of total steps; computed at runtime
        warmup_min_steps: int = 50,
        lr_schedule: str = "cosine",
        # ---- regularization (added in v2 to address v1 overfitting) ----
        dropout: float = 0.0,             # applied after image_embed and at head input
        # ---- front-end (added in v3 to address v1 noise-rejection failure) ----
        # "linear" = single Linear(576, d_hidden), matches RoMAE's patch-embed
        #            (default; v1 behavior, byte-identical when unset).
        # "schirmer_cnn" = Schirmer 2022 CNN encoder used by S5/CRU/RKN/etc.
        #                  See _SchirmerCNNEncoder docstring + S5 App. G.3.8.
        front_end: str = "linear",
    ):
        super().__init__()
        self.save_hyperparameters()

        n_vars = 1                        # hard-coded for pendulum

        # Image front-end. v1 default = single Linear (RoMAE-style); v3 = Schirmer CNN.
        if front_end == "linear":
            # Bias=True matches Conv2d(1, d_hidden, kernel=24, stride=24) defaults.
            self.image_embed = nn.Linear(image_dim, d_hidden, bias=True)
        elif front_end == "schirmer_cnn":
            self.image_embed = _SchirmerCNNEncoder(d_out=d_hidden)
        else:
            raise ValueError(f"Unknown front_end: {front_end!r}")
        self.image_embed_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Per-variate Mamba stack -- built directly from MambaBlock /
        # MambaIrregularBlock so we can feed pre-embedded inputs without
        # touching PerVariateIrregularSSM (used by PhysioNet path).
        if dt_mode == "learned":
            block_cls = MambaBlock
            block_kw = dict(d_model=d_hidden, d_state=d_state,
                            d_conv=d_conv, expand=expand)
        else:
            block_cls = MambaIrregularBlock
            block_kw = dict(d_model=d_hidden, d_state=d_state,
                            d_conv=d_conv, expand=expand, dt_mode=dt_mode)
        self.perv_layers = nn.ModuleList(
            [block_cls(**block_kw) for _ in range(n_perv_layer)]
        )
        self.perv_final_norm = nn.LayerNorm(d_hidden)

        # Grid aligner + fusion stack (reused as-is from mamba_pretrain).
        self.grid = SharedGridAligner(
            d_hidden=d_hidden, d_model=d_model, K=grid_K,
            t_max=t_max, n_vars=n_vars, n_freq=n_freq,
        )
        self.fusion_attn = nn.ModuleList([
            VariableAxisAttention(
                d_model=d_model, n_vars=n_vars,
                n_heads=n_heads_varattn, n_freq_rho=n_freq,
            )
            for _ in range(n_fusion_blocks)
        ])
        self.fusion_mamba = nn.ModuleList([
            TemporalMambaOnGrid(
                d_model=d_model, d_state=d_state,
                d_conv=d_conv, expand=expand,
            )
            for _ in range(n_fusion_blocks)
        ])

        # Output head. Default = gaussian (matches S5/CRU/RKN — every
        # probabilistic-SSM baseline beats every deterministic-head baseline
        # on Pendulum, see S5 paper Table 9). Linear is the ablation that
        # isolates whether any of our gain is head-induced.
        # Gaussian: separate mu and log-var heads, both 2-dim; loss is NLL.
        # Linear: predict (sin, cos) directly; loss is MSE (matches RoMAE).
        if head_type == "linear":
            self.head_mu = _QueryReadout2D(
                d_model=d_model, t_max=t_max, K=grid_K,
                n_freq=n_freq, out_dim=out_dim, dropout=dropout,
            )
            self.head_logvar = None
        elif head_type == "gaussian":
            self.head_mu = _QueryReadout2D(
                d_model=d_model, t_max=t_max, K=grid_K,
                n_freq=n_freq, out_dim=out_dim, dropout=dropout,
            )
            self.head_logvar = _QueryReadout2D(
                d_model=d_model, t_max=t_max, K=grid_K,
                n_freq=n_freq, out_dim=out_dim, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type!r}")

        self._test_outputs: list[dict] = []

    # -------------------------- forward --------------------------

    def _build_var_id(self, B: int, V: int, *, device) -> torch.Tensor:
        return (
            torch.arange(V, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, V)
        )

    def forward(self, batch) -> dict:
        # Datamodule emits images at [B, V=1, T_max, image_dim] (flattened).
        images = batch["images"]          # [B, V, T, image_dim] float
        timestamps = batch["timestamps"]  # [B, V, T]
        deltat = batch["deltat"]          # [B, V, T]
        valid_mask = batch["valid_mask"]  # [B, V, T] bool

        B, V, T, _ = images.shape
        assert V == 1, f"PendulumMambaForecaster expects V=1, got V={V}"

        # 1. Patch-embed each image. Zero out padding so it doesn't
        #    contribute to the embed (matches PerVariateIrregularSSM's
        #    'x = x * m' on scalar inputs).
        m = valid_mask.to(images.dtype).unsqueeze(-1)                   # [B, V, T, 1]
        emb = self.image_embed(images * m)                              # [B, V, T, D_h]
        emb = self.image_embed_drop(emb)                                # regularization

        # 2. Per-variate Mamba: collapse (B, V) for the block, run, restore.
        h = emb.reshape(B * V, T, self.hparams.d_hidden)
        dt_flat = deltat.reshape(B * V, T)
        for layer in self.perv_layers:
            if self.hparams.dt_mode == "learned":
                h = layer(h)
            else:
                h = layer(h, delta_t=dt_flat)
        h = self.perv_final_norm(h)
        h_pv = h.reshape(B, V, T, self.hparams.d_hidden)

        # 3. Grid + fusion (V=1 makes VariableAxisAttention degenerate, but
        #    TemporalMambaOnGrid still does useful work over the K grid).
        grid_out = self.grid(h_pv, timestamps, valid_mask)
        H = grid_out["h0"]                                              # [B, K, V, D]
        avail = grid_out["avail"]
        rho = grid_out["rho"]

        var_id = self._build_var_id(B, V, device=images.device)
        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho, var_id=var_id)
            H = mamba(H)

        # 4. Per-token query readout to original (irregular) timestamps.
        query_times = timestamps.reshape(B, V * T)
        query_variate = (
            torch.arange(V, device=images.device).view(1, V, 1).expand(B, V, T)
            .reshape(B, V * T)
        )
        query_valid = valid_mask.reshape(B, V * T)

        mu = self.head_mu(H, query_times, query_variate, query_valid)   # [B, Q, 2]
        mu = mu.view(B, V, T, self.hparams.out_dim)

        out = {"mu": mu}
        if self.head_logvar is not None:
            logvar = self.head_logvar(H, query_times, query_variate, query_valid)
            out["logvar"] = logvar.view(B, V, T, self.hparams.out_dim)
        return out

    # -------------------------- loss ----------------------------

    def _compute_loss(self, batch, prefix: str):
        target = batch["target"]                # [B, V=1, T, 2]
        valid_mask = batch["valid_mask"]        # [B, V=1, T]
        out = self.forward(batch)
        mu = out["mu"]                          # [B, V=1, T, 2]

        m = valid_mask.unsqueeze(-1)            # [B, V=1, T, 1]
        if m.sum() == 0:
            return torch.tensor(0.0, device=mu.device, requires_grad=True)

        if self.head_logvar is None:
            # Deterministic: MSE on (sin, cos) at observed timesteps.
            diff = (mu - target) * m
            loss = (diff ** 2).sum() / (2 * m.sum())  # /2 because mean over both dims
            self.log(f"{prefix}/mse", loss, prog_bar=True, batch_size=mu.shape[0])
        else:
            # Gaussian NLL: 0.5 * (logvar + (target - mu)^2 / exp(logvar))
            # Using elu+1 on raw logvar for positivity, matching S5 App. G.3.8.
            logvar_raw = out["logvar"]
            var = F.elu(logvar_raw) + 1.0 + 1e-6                         # > 0
            sq = (mu - target) ** 2
            nll_per_elem = 0.5 * (torch.log(var) + sq / var)
            nll = (nll_per_elem * m).sum() / (2 * m.sum())
            # Also log MSE on mu for selection / reporting parity.
            with torch.no_grad():
                mse_mu = ((mu - target) ** 2 * m).sum() / (2 * m.sum())
            self.log(f"{prefix}/nll", nll, prog_bar=True, batch_size=mu.shape[0])
            self.log(f"{prefix}/mse", mse_mu, prog_bar=False, batch_size=mu.shape[0])
            loss = nll

        return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        return self._compute_loss(batch, "val")

    # -------------------------- test ----------------------------

    def test_step(self, batch, batch_idx):
        target = batch["target"]                # [B, V=1, T, 2]
        valid_mask = batch["valid_mask"]
        out = self.forward(batch)
        mu = out["mu"]                          # [B, V=1, T, 2]
        item_ids = batch["item_id"]

        B = mu.shape[0]
        for i in range(B):
            m = valid_mask[i, 0]                # [T]
            if m.sum() == 0:
                continue
            y_true = target[i, 0][m].detach().cpu().numpy()    # [t_obs, 2]
            y_pred = mu[i, 0][m].detach().cpu().numpy()        # [t_obs, 2]

            sq = (y_true - y_pred) ** 2
            mse = float(sq.mean())                              # over (t, dim)
            mae = float(np.abs(y_true - y_pred).mean())
            ss_res_sin = float(sq[:, 0].sum())
            ss_res_cos = float(sq[:, 1].sum())
            count = int(y_true.shape[0])

            self._test_outputs.append({
                "item_id": item_ids[i],
                "mse": mse,
                "mae": mae,
                "mse_sin": float(sq[:, 0].mean()),
                "mse_cos": float(sq[:, 1].mean()),
                "ss_res_sin": ss_res_sin,
                "ss_res_cos": ss_res_cos,
                "count": count,
            })

    def on_test_epoch_end(self):
        if not self._test_outputs:
            return

        rows = self._test_outputs
        out_dir = Path(self.trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-sample dump (matches PhysioNet trainer convention).
        per_sample_path = out_dir / "per_sample.jsonl"
        with open(per_sample_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        # Aggregate metrics. Two reportings:
        # - sample-averaged MSE (mean over per-sample MSEs)
        # - target-weighted MSE (sum-of-squared-residuals / total-count)
        n_samples = len(rows)
        sample_avg_mse = float(np.mean([r["mse"] for r in rows]))
        sample_avg_mae = float(np.mean([r["mae"] for r in rows]))
        total_count = sum(r["count"] for r in rows)
        ss_res_total = sum(r["ss_res_sin"] + r["ss_res_cos"] for r in rows)
        # /2 for averaging over the 2 output dims to match the train/val MSE.
        target_weighted_mse = ss_res_total / (2 * total_count) if total_count else 0.0

        summary = {
            "n_samples": n_samples,
            "n_total_obs": total_count,
            "test_mse_sample_avg": sample_avg_mse,
            "test_mae_sample_avg": sample_avg_mae,
            "test_mse_target_weighted": float(target_weighted_mse),
        }
        # Headline number is the sample-averaged MSE (matches the paper
        # convention: mean MSE per sample, then mean across samples).
        self.log("test/mse", summary["test_mse_sample_avg"])
        self.log("test/mae", summary["test_mae_sample_avg"])
        self.log("test/mse_target_weighted", summary["test_mse_target_weighted"])

        # CSV for the aggregator (single-row, matches PhysioNet
        # test_metrics.csv convention so the eval scripts can glob it).
        csv_path = out_dir / "test_metrics.csv"
        with open(csv_path, "w") as f:
            f.write("model,head_type,dt_mode,seed,n_samples,n_total_obs,"
                    "test_mse,test_mae,test_mse_target_weighted\n")
            f.write(
                f"mamba_pretrain_pendulum,{self.hparams.head_type},"
                f"{self.hparams.dt_mode},{getattr(self, '_seed', '')},"
                f"{n_samples},{total_count},"
                f"{sample_avg_mse:.8f},{sample_avg_mae:.8f},"
                f"{target_weighted_mse:.8f}\n"
            )
        print(f"[pendulum] wrote {csv_path}")
        print(f"[pendulum] test/mse (sample-avg) = {sample_avg_mse:.6f}  "
              f"(={sample_avg_mse * 1e3:.3f} x10^-3)")
        self._test_outputs = []

    # -------------------------- optim ----------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )
        if self.hparams.lr_schedule == "constant":
            return opt
        if self.hparams.lr_schedule == "cosine":
            # Compute warmup + total steps at runtime so the schedule scales
            # correctly across all (B, max_epochs) cells. Lightning gives us
            # the true number of optimizer steps the run will execute,
            # accounting for max_epochs * steps_per_epoch * accumulation.
            total_steps = int(self.trainer.estimated_stepping_batches)
            warmup_steps = max(
                self.hparams.warmup_min_steps,
                int(self.hparams.warmup_pct * total_steps),
            )
            sched = _cosine_with_warmup(
                opt,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        raise ValueError(f"Unknown lr_schedule: {self.hparams.lr_schedule!r}")
