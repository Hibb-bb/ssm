"""Align per-variate hidden states onto a shared uniform grid.

For each (b, d, k) slot:
  pi_d(k)         = max{i : t_i^{(d)} <= s_k}    (latest obs index in variate d before s_k)
  avail_{b,d,k}   = 1[pi_d(k) exists]
  rho_{b,d,k}     = s_k - t_{pi_d(k)}            (staleness) if avail else s_k - t_0
  z_{b,d,k}       = h_{b,d,pi} * exp(-gamma_d * rho)  if avail else h_null_d
  e_{b,k,d}       = [z, variate_embed, avail_mask, phi(rho)]
  H^(0)_{b,k,d}   = W_e e_{b,k,d}

We use a learnable per-variate, per-channel decay rate (gamma_d) initialised
small so freshly observed slots are barely attenuated. This is a pragmatic
approximation of `bar_A_{rho} h` (which would require rerunning the SSM).

Output tensor: H0 of shape [B, K, V, D].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_encode(x: torch.Tensor, n_freq: int = 8, max_period: float = 100.0) -> torch.Tensor:
    """Classic sinusoidal positional encoding applied to a scalar feature.
    Input  x:   [...]
    Output:     [..., 2*n_freq]
    """
    device = x.device
    freqs = torch.exp(
        torch.linspace(0.0, math.log(max_period), n_freq, device=device)
    )
    ang = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class SharedGridAligner(nn.Module):
    """Align per-variate states onto a shared uniform grid of size K.

    Args:
        d_hidden: dim of the per-variate hidden states coming in
        d_model : dim of the aligned representation H^(0)
        K       : grid size
        t_max   : time horizon (grid = linspace(0, t_max, K))
        n_vars  : V
        n_freq  : # frequencies for staleness sinusoidal encoding
    """

    def __init__(
        self,
        d_hidden: int,
        d_model: int,
        K: int,
        t_max: float,
        n_vars: int,
        n_freq: int = 8,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.K = K
        self.t_max = t_max
        self.n_vars = n_vars
        self.n_freq = n_freq

        grid = torch.linspace(0.0, t_max, K)
        self.register_buffer("grid", grid, persistent=False)

        self.null_state = nn.Parameter(torch.zeros(n_vars, d_hidden))
        nn.init.normal_(self.null_state, std=0.02)

        # per-variate, per-channel decay rate gamma_d in [0, +) via softplus
        self.gamma_raw = nn.Parameter(torch.full((n_vars, d_hidden), -3.0))

        proj_in = d_hidden + 1 + 2 * n_freq
        self.proj = nn.Linear(proj_in, d_model)

    def forward(
        self,
        h_per_var: torch.Tensor,        # [B, V, L, D_h]
        timestamps_per_var: torch.Tensor,  # [B, V, L]
        valid_mask: torch.Tensor,       # [B, V, L]
    ) -> dict:
        B, V, L, D_h = h_per_var.shape
        K = self.K
        device = h_per_var.device

        grid = self.grid.to(device).view(1, 1, K, 1).expand(B, V, K, 1).squeeze(-1)  # [B,V,K]

        # For each (b, d, k) find pi = max{i : t_i <= s_k and valid}.
        # Replace timestamps of invalid tokens with -inf so they never win.
        ts = torch.where(
            valid_mask, timestamps_per_var, torch.full_like(timestamps_per_var, -float("inf"))
        )  # [B, V, L]

        # Count how many valid tokens satisfy t_i <= s_k.  This equals pi+1 when
        # at least one valid token is <= s_k.
        leq = ts.unsqueeze(-1) <= grid.unsqueeze(-2)  # [B, V, L, K] (True iff valid and t_i<=s_k)
        # Guard: we require valid_mask as well (True + -inf<=anything still True for -inf); fix:
        leq = leq & valid_mask.unsqueeze(-1)
        count_leq = leq.sum(dim=2)  # [B, V, K]
        avail = count_leq > 0      # [B, V, K]
        pi = (count_leq - 1).clamp(min=0)  # [B, V, K]

        # Gather the latent and timestamp at index pi along the L axis.
        pi_idx = pi.unsqueeze(-1).expand(B, V, K, D_h)
        h_pi = torch.gather(h_per_var, dim=2, index=pi_idx)  # [B, V, K, D_h]
        ts_pi = torch.gather(timestamps_per_var, dim=2, index=pi)  # [B, V, K]

        # Staleness rho = s_k - t_pi  (when available); else s_k - t_0 (=s_k).
        rho_avail = grid - ts_pi
        rho = torch.where(avail, rho_avail, grid)  # >=0

        # Decay h_pi by exp(-gamma_d * rho)
        gamma = F.softplus(self.gamma_raw).view(1, V, 1, D_h)  # [1,V,1,D_h]
        decay = torch.exp(-gamma * rho.unsqueeze(-1))           # [B,V,K,D_h]
        z = h_pi * decay

        # Replace with null state where unavailable.
        null = self.null_state.view(1, V, 1, D_h).expand(B, V, K, D_h)
        z = torch.where(avail.unsqueeze(-1), z, null)

        # Availability scalar + staleness sinusoid (no variate-ID embedding).
        avail_feat = avail.to(z.dtype).unsqueeze(-1)                # [B,V,K,1]
        rho_feat = sinusoidal_encode(rho, n_freq=self.n_freq)       # [B,V,K,2*n_freq]

        e = torch.cat([z, avail_feat, rho_feat], dim=-1)            # [B,V,K,*]
        h0 = self.proj(e)                                           # [B,V,K,D]

        # Reshape to [B, K, V, D] for downstream variable-axis attention.
        h0 = h0.permute(0, 2, 1, 3).contiguous()
        avail_kv = avail.permute(0, 2, 1).contiguous()              # [B,K,V]
        rho_kv = rho.permute(0, 2, 1).contiguous()                  # [B,K,V]

        return dict(h0=h0, avail=avail_kv, rho=rho_kv)
