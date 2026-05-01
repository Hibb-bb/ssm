"""Classification head for Mamba-MV.

Hierarchical attention pooling over H^(L) [B, K, V, D]:
    1. Per-variate temporal attention pool: single learnable query q_t,
       softmax over K with availability mask zeroing null-slot bins.
    2. Variate-axis attention pool: single learnable query q_v, softmax over V.
    3. LayerNorm + Linear -> logits [B, C].

Returns (z [B, D], logits [B, C]). The pooled vector z is exposed so a frozen
encoder + linear probe track can reuse the same forward pass without rebuilding
the head. No dropout in the head; weight decay (set in the optimizer) is the
sole regularization, matching the dropout-free head convention of S5
(Smith et al. 2023) and RoMAE (Zivanovic et al. 2025).

Score function is q^T h / sqrt(d) (Bahdanau-style single-query attention pool),
following the Hierarchical Attention Network pattern of Yang et al. 2016 (NAACL)
applied at two levels (time-within-channel, then channels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_classes: int,
        use_avail_mask: bool = True,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_avail_mask = use_avail_mask
        self._scale = math.sqrt(d_model)

        # Learnable query for temporal attention pool. Shared across variates:
        # one query vector for "what makes a timestep informative?". Initialized
        # to small magnitude so initial attention is approximately uniform.
        self.q_t = nn.Parameter(torch.randn(d_model) / self._scale)

        # LayerNorm on per-variate pooled vector before stage 2.
        self.ln_v = nn.LayerNorm(d_model)

        # Learnable query for variate-axis attention pool.
        self.q_v = nn.Parameter(torch.randn(d_model) / self._scale)

        # Final classifier: LN -> (optional Dropout) -> Linear.
        # Dropout sits between LN_out and the linear classifier, mirroring
        # RoMAE Table 12 where dropout=0.2 is enabled on EP and LSST only.
        self.ln_out = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        H: torch.Tensor,         # [B, K, V, D]
        avail: torch.Tensor,     # [B, K, V] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Step 1: per-variate temporal attention pool ----
        # Score(k, v) = q_t . H[k, v] / sqrt(d). Shared q_t across variates.
        scores_t = torch.einsum("d,bkvd->bkv", self.q_t, H) / self._scale

        if self.use_avail_mask:
            # Mask null-slot bins out of the temporal softmax. If a (b, v) has
            # no available bins (degenerate), fall back to uniform over K to
            # avoid NaN.
            has_any = avail.any(dim=1, keepdim=True)              # [B, 1, V]
            scores_masked = scores_t.masked_fill(~avail, float("-inf"))
            scores_t_eff = torch.where(
                has_any.expand_as(scores_t), scores_masked, torch.zeros_like(scores_t)
            )
        else:
            scores_t_eff = scores_t

        alpha_t = F.softmax(scores_t_eff, dim=1)                  # [B, K, V]
        H_v = torch.einsum("bkv,bkvd->bvd", alpha_t, H)           # [B, V, D]

        # ---- Step 2: variate-axis attention pool ----
        H_v_norm = self.ln_v(H_v)
        scores_v = torch.einsum("d,bvd->bv", self.q_v, H_v_norm) / self._scale
        alpha_v = F.softmax(scores_v, dim=1)                      # [B, V]
        z = torch.einsum("bv,bvd->bd", alpha_v, H_v_norm)         # [B, D]

        # ---- Step 3: classify ----
        z_norm = self.ln_out(z)
        z_drop = self.head_dropout(z_norm)
        logits = self.classifier(z_drop)                          # [B, C]
        return z, logits
