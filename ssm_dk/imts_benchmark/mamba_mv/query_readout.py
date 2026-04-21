"""Query-time readout: predict each (variate, query_time) pair.

For a query q_j = (n_j, a_j):
    kappa(j) = max{k : s_k <= a_j}
    omega_j  = a_j - s_{kappa(j)}
    xi_j     = H^{(L)}_{kappa(j), n_j}  decayed by exp(-gamma_n * omega_j)
    x_hat_j  = g_{n_j}( [xi_j, phi(omega_j)] )

Here g_{n_j} is a *per-variate* linear head, which lets each variate carry its
own scale/offset for the decoded value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_grid import sinusoidal_encode


class QueryReadout(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vars: int,
        t_max: float,
        K: int,
        n_freq: int = 8,
        hidden: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.K = K
        self.t_max = t_max
        self.n_freq = n_freq
        grid = torch.linspace(0.0, t_max, K)
        self.register_buffer("grid", grid, persistent=False)

        self.gamma_raw = nn.Parameter(torch.full((n_vars, d_model), -3.0))
        # Per-variate readout heads: [V, (d_model + 2*n_freq) -> 1]
        self.head_weight = nn.Parameter(
            torch.empty(n_vars, d_model + 2 * n_freq, 1)
        )
        self.head_bias = nn.Parameter(torch.zeros(n_vars, 1))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(
        self,
        H: torch.Tensor,               # [B, K, V, D]
        query_times: torch.Tensor,     # [B, Q]  float, query time for each point
        query_variate: torch.Tensor,   # [B, Q]  long,  variate index for each point
        query_valid: torch.Tensor,     # [B, Q]  bool
    ) -> torch.Tensor:
        B, K, V, D = H.shape
        Q = query_times.shape[1]
        device = H.device
        grid = self.grid.to(device)

        # kappa(j) = # of grid slots with s_k <= a_j  - 1, clamped into [0, K-1].
        dt = grid.view(1, 1, K) - query_times.view(B, Q, 1)       # [B, Q, K]
        leq = dt <= 0
        kappa = leq.sum(dim=-1).clamp(min=1) - 1                   # [B, Q] in [0, K-1]
        omega = query_times - grid[kappa]                          # [B, Q], >=0

        # Gather H[b, kappa, v, :].
        # H: [B, K, V, D] -> pick (B, Q, D) per (kappa, variate).
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, Q)
        h_at = H[batch_idx, kappa, query_variate]                  # [B, Q, D]

        gamma = F.softplus(self.gamma_raw)[query_variate]          # [B, Q, D]
        decay = torch.exp(-gamma * omega.unsqueeze(-1))            # [B, Q, D]
        xi = h_at * decay

        omega_feat = sinusoidal_encode(omega, n_freq=self.n_freq)  # [B, Q, 2*n_freq]
        feat = torch.cat([xi, omega_feat], dim=-1)                 # [B, Q, D+2*n_freq]

        W = self.head_weight[query_variate]                        # [B, Q, D+2f, 1]
        b = self.head_bias[query_variate].squeeze(-1)              # [B, Q]
        pred = torch.einsum("bqd,bqdo->bqo", feat, W).squeeze(-1) + b

        pred = torch.where(query_valid, pred, torch.zeros_like(pred))
        return pred  # [B, Q]
