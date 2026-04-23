"""Temporal Mamba along the shared grid (uniform spacing delta_k).

We reuse the MambaBlock with dt_mode='learned' since the grid is uniform
(and the SSM already handles learnable time-scale). Per-variate stacks
share no weights; we apply the *same* block across all variates by
reshaping (B, K, V, D) -> (B*V, K, D).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_block import MambaBlock


class TemporalMambaOnGrid(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.block = MambaBlock(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: [B, K, V, D]  ->  [B, K, V, D]"""
        B, K, V, D = H.shape
        # Per-variate temporal model: reshape to (B*V, K, D).
        x = H.permute(0, 2, 1, 3).contiguous().view(B * V, K, D)
        x = self.block(x)
        return x.view(B, V, K, D).permute(0, 2, 1, 3).contiguous()
