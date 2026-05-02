"""Multivariate Mamba classifier — pretrain-architecture variant.

Wires the pretrain encoder (slot-anonymous variates with Moirai-style binary
attention bias, no per-variate ID embedding in shared-grid, shared-head
forecaster's pieces — but here we attach the same hierarchical attention pool
classification head as the supervised mamba_mv classifier so the head-to-head
isolates the encoder change as the only architectural axis).

Differences vs imts_benchmark.mamba_mv.multivariate_classifier:
- Imports the encoder pieces from `.irregular_ssm`, `.shared_grid`,
  `.temporal_mamba`, `.variable_axis_attention` — all under mamba_pretrain.
- The pretrain `VariableAxisAttention.forward` REQUIRES a `var_id: [B, V]`
  argument (Moirai any-variate binary bias). The supervised version does not.
  We build `var_id = arange(V)` per batch element via `_build_var_id`,
  matching the convention the pretrain forecaster uses.
- Otherwise identical: same loss, same optimizer/schedule, same val/test
  metrics (val/macro_f1 + collapse-detector inputs), same `_test_agg` API
  consumed by `train_cls.py`.

Design contract:
    forward(batch) -> dict(z [B, D], logits [B, C])
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classification_head import ClassificationHead
from .irregular_ssm import PerVariateIrregularSSM
from .shared_grid import SharedGridAligner
from .temporal_mamba import TemporalMambaOnGrid
from .variable_axis_attention import VariableAxisAttention


class MultivariateMambaClassifier(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 256,
        n_vars: int = 3,
        n_classes: int = 2,
        n_perv_layer: int = 3,
        n_fusion_blocks: int = 3,
        n_heads_varattn: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "replace",
        grid_K: int = 128,
        t_max: float = 1.0,
        n_freq: int = 8,
        head_use_avail_mask: bool = True,
        head_dropout: float = 0.0,
        # Optimizer / schedule.
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        num_warmup_steps: int = 200,
        num_training_steps: int = 4000,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

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
        self.head = ClassificationHead(
            d_model=d_model,
            n_classes=n_classes,
            use_avail_mask=head_use_avail_mask,
            head_dropout=head_dropout,
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(n_classes, dtype=torch.float32),
            )

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self.class_weights = weights.to(dtype=torch.float32, device=self.device)

    @staticmethod
    def _build_var_id(B: int, V: int, *, device) -> torch.Tensor:
        """Per-batch-element variate IDs for any-variate attention bias.

        Shape ``[B, V]``. Each batch element is one window with V *distinct*
        variates (we never pack multiple series into one batch element), so
        slot indices ``arange(V)`` are unique within an element. Same convention
        used by the pretrain forecaster.
        """
        return (
            torch.arange(V, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, V)
        )

    # -------------------------- forward --------------------------

    def forward(self, batch) -> dict:
        values = batch["values"]            # [B, V, L]
        timestamps = batch["timestamps"]    # [B, V, L]
        deltat = batch["deltat"]            # [B, V, L]
        valid_mask = batch["valid_mask"]    # [B, V, L]

        h_pv = self.perv_ssm(values, deltat, valid_mask)
        grid_out = self.grid(h_pv, timestamps, valid_mask)
        H = grid_out["h0"]                                         # [B,K,V,D]
        avail = grid_out["avail"]                                  # [B,K,V] bool
        rho = grid_out["rho"]

        B, V = values.shape[0], values.shape[1]
        var_id = self._build_var_id(B, V, device=values.device)    # [B, V]

        for attn, mamba in zip(self.fusion_attn, self.fusion_mamba):
            H = attn(H, avail, rho, var_id=var_id)
            H = mamba(H)

        z, logits = self.head(H, avail)
        return dict(z=z, logits=logits)

    # ---------------------- loss / metrics ----------------------

    def _shared_step(self, batch, prefix: str) -> torch.Tensor:
        labels = batch["label"].long()
        out = self.forward(batch)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        bs = labels.shape[0]
        self.log(f"{prefix}/loss", loss, prog_bar=(prefix != "train"), batch_size=bs)
        self.log(f"{prefix}/acc", acc, prog_bar=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        labels = batch["label"].long()
        out = self.forward(batch)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        bs = labels.shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self.log("val/acc", acc, prog_bar=True, batch_size=bs)
        if not hasattr(self, "_val_outputs"):
            self._val_outputs = []
        self._val_outputs.append({
            "labels": labels.detach().cpu(),
            "preds": preds.detach().cpu(),
        })
        return loss

    def on_validation_epoch_end(self):
        if not getattr(self, "_val_outputs", None):
            return
        labels = torch.cat([d["labels"] for d in self._val_outputs])
        preds = torch.cat([d["preds"] for d in self._val_outputs])
        n_classes = int(self.hparams.n_classes)
        f1s = []
        for c in range(n_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            denom = (2 * tp + fp + fn)
            f1s.append(2 * tp / denom if denom > 0 else 0.0)
        macro_f1 = sum(f1s) / max(n_classes, 1)
        self.log("val/macro_f1", macro_f1, prog_bar=True)
        self._val_outputs = []

    def test_step(self, batch, batch_idx):
        labels = batch["label"].long()
        out = self.forward(batch)
        logits = out["logits"]
        loss = F.cross_entropy(
            logits, labels, weight=self.class_weights,
            label_smoothing=self.hparams.label_smoothing,
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        bs = labels.shape[0]
        self.log("test/loss", loss, batch_size=bs)
        self.log("test/acc", acc, prog_bar=True, batch_size=bs)
        if not hasattr(self, "_test_outputs"):
            self._test_outputs = []
        self._test_outputs.append({
            "labels": labels.detach().cpu(),
            "preds": preds.detach().cpu(),
            "logits": logits.detach().cpu(),
        })
        return loss

    def on_test_epoch_end(self):
        if not getattr(self, "_test_outputs", None):
            return
        labels = torch.cat([d["labels"] for d in self._test_outputs])
        preds = torch.cat([d["preds"] for d in self._test_outputs])
        n_classes = int(self.hparams.n_classes)
        f1s = []
        for c in range(n_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            denom = (2 * tp + fp + fn)
            f1s.append(2 * tp / denom if denom > 0 else 0.0)
        macro_f1 = sum(f1s) / max(n_classes, 1)
        acc = (preds == labels).float().mean().item()
        self.log("test/acc_final", acc)
        self.log("test/macro_f1", macro_f1)
        self._test_agg = {"acc": acc, "macro_f1": macro_f1}

    # -------------------------- optimizer --------------------------

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
            betas=(0.9, 0.95),
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
