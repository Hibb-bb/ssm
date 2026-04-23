"""Variable-axis (cross-variate) attention on the shared grid.

At each grid time s_k, variates attend to each other. Availability mask
biases keys/values so unavailable variates do not provide evidence.
Staleness + variate-id + availability are re-injected as additive
embeddings before the QKV projection (matches the paper's `tilde H`).

Block: pre-norm MHA + residual + LayerNorm.
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
