"""Variable-axis (cross-variate) attention on the shared grid.

At each grid time s_k, variates attend to each other. Availability mask
biases keys/values so unavailable variates do not provide evidence.
Staleness + variate-id + availability are re-injected as additive
embeddings before the QKV projection (matches the paper's `tilde H`).

Block: pre-norm MHA + residual + LayerNorm.

``VariableAxisMLP`` is a DeepSets-style alternative (shared φ, masked mean
pool, ρ, per-slot ψ); permutation-equivariant over the variate axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_grid import sinusoidal_encode


class VariableAxisAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_vars: int,
        n_heads: int = 4,
        n_freq_rho: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.norm = nn.LayerNorm(d_model)
        self.variate_embed = nn.Embedding(n_vars, d_model)
        self.avail_embed = nn.Linear(1, d_model)
        self.rho_embed = nn.Linear(2 * n_freq_rho, d_model)
        self.n_freq_rho = n_freq_rho

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,          # [B, K, V, D]
        avail: torch.Tensor,      # [B, K, V]  bool
        rho: torch.Tensor,        # [B, K, V]  float
    ) -> torch.Tensor:
        B, K, V, D = H.shape
        device = H.device

        x = self.norm(H)
        var_ids = torch.arange(V, device=device).view(1, 1, V).expand(B, K, V)
        v_emb = self.variate_embed(var_ids)                     # [B,K,V,D]
        a_emb = self.avail_embed(avail.to(x.dtype).unsqueeze(-1))
        r_emb = self.rho_embed(sinusoidal_encode(rho, n_freq=self.n_freq_rho))
        x_tilde = x + v_emb + a_emb + r_emb                     # additive injections

        # Reshape to (B*K) sequences of V tokens each for per-slot attention.
        x_tilde = x_tilde.view(B * K, V, D)
        q = self.q_proj(x_tilde).view(B * K, V, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = self.k_proj(x_tilde).view(B * K, V, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = self.v_proj(x_tilde).view(B * K, V, self.n_heads, D // self.n_heads).transpose(1, 2)

        # Unavailable variates cannot provide evidence as keys/values.
        # We apply an additive bias on attention scores; queries from
        # unavailable slots still see some content and are later ignored by
        # downstream modules via `avail`.
        key_mask = avail.view(B * K, 1, 1, V)  # True = allowed
        scale = (D // self.n_heads) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale              # [B*K, H, V, V]
        scores = scores.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        y = torch.matmul(attn, v)                                          # [B*K, H, V, d_head]

        y = y.transpose(1, 2).contiguous().view(B * K, V, D)
        y = self.o_proj(y).view(B, K, V, D)

        return self.out_norm(H + y)



class VariableAxisMLP(nn.Module):
    """Cross-variate mixing with DeepSets (φ → masked mean → ρ → ψ).

    Same interface as ``VariableAxisAttention`` (``[B, K, V, D]`` in/out).
    No variate-index embedding: only per-slot ``avail`` and ``rho`` (plus
    normalized ``H``) feed shared φ; pooling is a masked mean over ``V``,
    so the block is permutation-equivariant over variates.

    ``n_heads`` is accepted for API parity with ``VariableAxisAttention`` but
    is not used here.
    """

    def __init__(
        self,
        d_model: int,
        n_vars: int,
        n_heads: int = 4,
        n_freq_rho: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars
        self.n_freq_rho = n_freq_rho
        d_h = 2 * d_model

        self.norm = nn.LayerNorm(d_model)
        self.avail_embed = nn.Linear(1, d_model)
        self.rho_embed = nn.Linear(2 * n_freq_rho, d_model)

        self.phi = nn.Sequential(
            nn.Linear(d_model, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_model),
        )
        self.rho_net = nn.Sequential(
            nn.Linear(d_model, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_model),
        )
        self.psi = nn.Sequential(
            nn.Linear(2 * d_model, d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,          # [B, K, V, D]
        avail: torch.Tensor,      # [B, K, V]  bool
        rho: torch.Tensor,        # [B, K, V]  float
    ) -> torch.Tensor:
        B, K, V, D = H.shape

        x = self.norm(H)
        a_emb = self.avail_embed(avail.to(dtype=x.dtype).unsqueeze(-1))
        r_emb = self.rho_embed(sinusoidal_encode(rho, n_freq=self.n_freq_rho))
        x_tilde = x + a_emb + r_emb

        x_flat = x_tilde.reshape(B * K, V, D)
        phi_out = self.phi(x_flat)

        m = avail.reshape(B * K, V, 1).to(dtype=phi_out.dtype)
        num = (phi_out * m).sum(dim=1)
        den = m.sum(dim=1).clamp(min=1e-6)
        pooled = num / den

        rho_vec = self.rho_net(pooled)
        rho_broadcast = rho_vec.unsqueeze(1).expand(-1, V, -1)
        psi_in = torch.cat([phi_out, rho_broadcast], dim=-1)
        y_flat = self.psi(psi_in)
        y = y_flat.view(B, K, V, D)

        return self.out_norm(H + y)
