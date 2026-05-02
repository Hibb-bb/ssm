"""Variable-axis (cross-variate) attention on the shared grid.

At each grid time s_k, variates attend to each other. Availability mask
biases keys/values so unavailable variates do not provide evidence.
Staleness + availability are re-injected as additive embeddings before
the QKV projection (matches the paper's `tilde H`).

Variate-identity is *not* injected as a per-slot learned embedding —
that approach binds meaning to slot indices, but our pretraining
collator randomizes slot assignments per batch, making per-slot
embeddings nothing but noise.  Instead, we expose a Moirai-style
optional ``var_id`` argument and add a learned per-head bias keyed only
on whether two tokens share the same variate ID (``_BinaryVariateAttentionBias``).
This is permutation-equivariant over variates and generalizes to any
variate count.  Reference: Woo et al., 2024, §3.2 ("Any-Variate
Attention"), arXiv:2402.02592.

Block: pre-norm MHA + residual + LayerNorm.

``VariableAxisMLP`` is a DeepSets-style alternative (shared φ, masked mean
pool, ρ, per-slot ψ); permutation-equivariant over the variate axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shared_grid import sinusoidal_encode


class _BinaryVariateAttentionBias(nn.Module):
    """Learned per-head bias for (same-variate) vs (different-variate) token pairs.

    This avoids a per-variate embedding table and can generalize to unseen
    variate IDs, as it depends only on equality of IDs.
    """

    def __init__(self, *, n_heads: int):
        super().__init__()
        # weight[0, h] = bias for different-variate pairs
        # weight[1, h] = bias for same-variate pairs
        self.weight = nn.Embedding(num_embeddings=2, embedding_dim=n_heads)

    def forward(self, *, query_var_id: torch.Tensor, kv_var_id: torch.Tensor) -> torch.Tensor:
        """Return bias shaped [N, H, Q, K] for attention scores.

        Args:
            query_var_id: [N, Q] integer IDs
            kv_var_id:    [N, K] integer IDs
        """
        # [N, Q, K] bool
        same = query_var_id[:, :, None].eq(kv_var_id[:, None, :])
        w = self.weight.weight  # [2, H]
        # [N, Q, K, H] -> [N, H, Q, K]
        bias = torch.where(same[..., None], w[1], w[0]).permute(0, 3, 1, 2).contiguous()
        return bias


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
        self.avail_embed = nn.Linear(1, d_model)
        self.rho_embed = nn.Linear(2 * n_freq_rho, d_model)
        self.n_freq_rho = n_freq_rho

        # Standard multi-head attention (matches Moirai-1's any-variate
        # attention).  We deliberately do NOT use MQA here: the variable
        # axis only has V <= max_dim tokens per grid step, so attention
        # cost is already negligible (V^2 = 400 ops at most), and the KV
        # cache motivation for MQA does not apply.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.var_bias = _BinaryVariateAttentionBias(n_heads=n_heads)

        self.dropout = dropout
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        H: torch.Tensor,          # [B, K, V, D]
        avail: torch.Tensor,      # [B, K, V]  bool
        rho: torch.Tensor,        # [B, K, V]  float
        var_id: torch.Tensor | None = None,  # [B, K, V] or [B, V] or None
    ) -> torch.Tensor:
        B, K, V, D = H.shape
        device = H.device

        x = self.norm(H)
        a_emb = self.avail_embed(avail.to(x.dtype).unsqueeze(-1))
        r_emb = self.rho_embed(sinusoidal_encode(rho, n_freq=self.n_freq_rho))
        x_tilde = x + a_emb + r_emb  # slot-local injections (zero-shot friendly)

        # Reshape to (B*K) sequences of V tokens each for per-slot attention.
        x_tilde = x_tilde.view(B * K, V, D)
        d_head = D // self.n_heads

        q = self.q_proj(x_tilde).view(B * K, V, self.n_heads, d_head).transpose(1, 2)
        k = self.k_proj(x_tilde).view(B * K, V, self.n_heads, d_head).transpose(1, 2)
        v = self.v_proj(x_tilde).view(B * K, V, self.n_heads, d_head).transpose(1, 2)

        # Optional variate-ID bias. If var_id is not provided, we do not inject
        # any non-permutation-equivariant signal — i.e. attention becomes fully
        # symmetric across variates.  Pass per-batch random IDs in training and
        # slot indices at inference for the bias to learn "same vs different
        # variate" structure (Moirai any-variate attention protocol).
        bias = None
        if var_id is not None:
            if var_id.ndim == 2:  # [B, V] -> [B, K, V]
                var_id = var_id[:, None, :].expand(B, K, V)
            var_id_bk = var_id.reshape(B * K, V).to(device=device, dtype=torch.long)
            bias = self.var_bias(query_var_id=var_id_bk, kv_var_id=var_id_bk)

        # Unavailable variates cannot provide evidence as keys/values.
        # We apply an additive bias on attention scores; queries from
        # unavailable slots still see some content and are later ignored by
        # downstream modules via `avail`.
        key_mask = avail.view(B * K, 1, 1, V)  # True = allowed
        scale = d_head ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*K, H, V, V]
        if bias is not None:
            scores = scores + bias
        scores = scores.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        y = torch.matmul(attn, v)  # [B*K, H, V, d_head]

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
