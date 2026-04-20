"""Per-variate irregular-step state space encoder.

Wraps a stack of `MambaIrregularBlock`s from the existing ssm_model package.
Each variate is processed independently with its true time gaps feeding the
SSM discretization step, matching the paper's
    bar_A = exp(Delta A), bar_B = (Delta A)^-1 (exp(Delta A) - I) Delta B

formulation (the mamba_ssm CUDA kernel handles the discretisation; we only
supply Delta = true_dt when dt_mode != 'learned').
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import the blocks from the project's canonical location.
# Path: .../uni2ts_hongyu/ssm_model/ssm/mamba_experiments/forecaster/mamba_block.py
_MAMBA_EXP = (
    Path(__file__).resolve().parents[3]
    / "ssm_model" / "ssm" / "mamba_experiments"
)
if str(_MAMBA_EXP) not in sys.path:
    sys.path.insert(0, str(_MAMBA_EXP))

from forecaster.mamba_block import MambaBlock, MambaIrregularBlock  # noqa: E402


class PerVariateIrregularSSM(nn.Module):
    """Encode each variate independently with an irregular-step Mamba stack.

    Input:
        values_per_var:  [B, V, L_max]    padded observation values
        deltat_per_var:  [B, V, L_max]    true per-variate time gaps
        valid_mask:      [B, V, L_max]    bool (True = real token)
    Output:
        h_per_var:       [B, V, L_max, D] per-variate hidden states
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_mode: str = "replace",
    ):
        super().__init__()
        self.d_model = d_model
        self.dt_mode = dt_mode
        self.input_proj = nn.Linear(1, d_model)

        if dt_mode == "learned":
            block_cls = MambaBlock
            kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            block_cls = MambaIrregularBlock
            kw = dict(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                dt_mode=dt_mode,
            )
        self.layers = nn.ModuleList([block_cls(**kw) for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        values_per_var: torch.Tensor,
        deltat_per_var: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, V, L = values_per_var.shape
        x = values_per_var.reshape(B * V, L)
        dt = deltat_per_var.reshape(B * V, L)
        m = valid_mask.reshape(B * V, L).to(x.dtype)

        # Zero out padding values so they don't contribute to the input_proj.
        x = x * m
        h = self.input_proj(x.unsqueeze(-1))

        for layer in self.layers:
            if self.dt_mode == "learned":
                h = layer(h)
            else:
                h = layer(h, delta_t=dt)

        h = self.final_norm(h)
        h = h.reshape(B, V, L, self.d_model)
        return h
