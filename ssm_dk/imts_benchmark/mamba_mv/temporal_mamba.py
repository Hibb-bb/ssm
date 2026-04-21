"""Temporal Mamba along the shared grid (uniform spacing delta_k).

We reuse the MambaBlock with dt_mode='learned' since the grid is uniform
(and the SSM already handles learnable time-scale). Per-variate stacks
share no weights; we apply the *same* block across all variates by
reshaping (B, K, V, D) -> (B*V, K, D).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_MAMBA_EXP = (
    Path(__file__).resolve().parents[3]
    / "ssm_model" / "ssm" / "mamba_experiments"
)
if str(_MAMBA_EXP) not in sys.path:
    sys.path.insert(0, str(_MAMBA_EXP))

from forecaster.mamba_block import MambaBlock  # noqa: E402


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
