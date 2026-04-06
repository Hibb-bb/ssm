"""
Mamba-1 variants for irregular Time Series Classification.

Controlled study of temporal update mechanisms:
  - LTI:            Delta, B, C all input-independent (equivalent to linear SSM / S4-like)
  - Selective (TV): Delta, B, C all input-dependent (standard Mamba-1)
  - GapDelta:       Delta derived from inter-observation time gaps
  - GapSelective:   Delta from both input + time gaps
  - GapFeature:     Time gaps concatenated as input feature (control baseline)

Pure PyTorch implementation (no mamba_ssm CUDA kernels required).
Designed for short astronomical light curves (seq_len <= 200).
"""

import math
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MambaTSCConfig:
    # Input
    input_dim: int = 1              # 1 for univariate magnitude
    seq_len: int = 200              # max sequence length
    num_classes: int = 7            # ZTF 7-class

    # Mamba block
    d_model: int = 64               # model dimension
    d_state: int = 16               # SSM state dimension N
    d_conv: int = 4                 # local conv width
    expand: int = 2                 # expansion factor for inner dim
    n_layers: int = 4               # number of Mamba blocks
    dt_rank: str = "auto"           # rank of dt projection ("auto" = d_model // 16)
    dt_min: float = 0.001
    dt_max: float = 0.1

    # Delta mode: controls how dt (Delta) is parameterized
    # - "selective":      dt from input x (standard Mamba-1)
    # - "lti":            dt is a fixed learned parameter
    # - "gap":            dt from inter-observation time gaps only
    # - "gap_selective":  dt from input x + time gaps (additive)
    delta_mode: Literal["selective", "lti", "gap", "gap_selective"] = "selective"

    # B, C selectivity: controls whether B and C are input-dependent
    # - "selective": B, C projected from input (standard Mamba-1)
    # - "lti":       B, C are fixed learned parameters
    bc_mode: Literal["selective", "lti"] = "selective"

    # Gap-as-feature: concatenate time gaps to input (control baseline for H2)
    gap_as_feature: bool = False    # if True, input_dim effectively +1

    # Classification head
    pool_mode: str = "mean"         # "mean" over valid tokens, or "last"
    classifier_hidden: int = 128
    dropout: float = 0.1

    def resolve_dt_rank(self) -> int:
        if self.dt_rank == "auto":
            return max(math.ceil(self.d_model / 16), 1)
        return int(self.dt_rank)


# ---------------------------------------------------------------------------
# Pure PyTorch Selective Scan
# ---------------------------------------------------------------------------

def selective_scan_ref(x, dt, A, B_mat, C_mat, D_skip, z=None, mask=None):
    """
    Pure PyTorch selective scan (sequential, no CUDA kernel).

    Args:
        x:      (B, d, L)  input after conv
        dt:     (B, d, L)  delta (already softplus-ed)
        A:      (d, N)     state matrix (negative real)
        B_mat:  (B, N, L)  input matrix
        C_mat:  (B, N, L)  output matrix
        D_skip: (d,)       skip connection
        z:      (B, d, L)  gate (optional, for Mamba gating)
        mask:   (B, L)     1 for valid, 0 for padding

    Returns:
        y:      (B, d, L)
    """
    bsz, d, L = x.shape
    N = A.shape[1]

    # Discretize: dA = exp(A * dt), dB = dt * B  (ZOH approximation)
    dA = torch.exp(torch.einsum("bdl,dn->bdln", dt, A))
    dB = torch.einsum("bdl,bnl->bdln", dt, B_mat)

    # Sequential scan
    h = torch.zeros(bsz, d, N, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        if mask is not None:
            m = mask[:, t].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            h = m * (dA[:, :, t, :] * h + dB[:, :, t, :] * x[:, :, t].unsqueeze(2)) + (1 - m) * h
        else:
            h = dA[:, :, t, :] * h + dB[:, :, t, :] * x[:, :, t].unsqueeze(2)
        y_t = torch.einsum("bdn,bn->bd", h, C_mat[:, :, t])
        ys.append(y_t)

    y = torch.stack(ys, dim=2)  # (B, d, L)

    # Skip connection
    y = y + x * D_skip.unsqueeze(0).unsqueeze(2)

    # Gating
    if z is not None:
        y = y * F.silu(z)

    return y


# ---------------------------------------------------------------------------
# Mamba Block with configurable Delta
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Single Mamba-1 block with configurable Delta/B/C modes.
    """
    def __init__(self, config: MambaTSCConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.d_inner = config.d_model * config.expand
        self.dt_rank = config.resolve_dt_rank()
        self.delta_mode = config.delta_mode
        self.bc_mode = config.bc_mode

        # Input projection: d_model -> 2 * d_inner (x and z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Local convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )

        # --- Delta (dt) parameterization ---
        if self.delta_mode in ("selective", "gap_selective"):
            # Project from input x to dt_rank, then to d_inner
            self.x_to_dt = nn.Linear(self.d_inner, self.dt_rank, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        if self.delta_mode == "lti":
            # Learned fixed dt parameter (no input dependence)
            self.dt_param = nn.Parameter(torch.empty(self.d_inner))
            self._init_dt_param()

        if self.delta_mode in ("gap", "gap_selective"):
            # Project time gap (scalar per timestep) to d_inner
            self.gap_to_dt = nn.Linear(1, self.d_inner, bias=True)

        if self.delta_mode == "selective":
            self._init_dt_proj()

        if self.delta_mode == "gap_selective":
            self._init_dt_proj()

        # --- B, C parameterization ---
        if self.bc_mode == "selective":
            self.x_to_bc = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        else:  # lti
            self.B_param = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)
            self.C_param = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)

        # State matrix A (S4D real initialization)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Layer norm (pre-norm)
        self.norm = nn.LayerNorm(self.d_model)

    def _init_dt_proj(self):
        """Initialize dt projection bias so softplus(bias) ~ Uniform(dt_min, dt_max)."""
        cfg = self.config
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def _init_dt_param(self):
        """Initialize fixed dt parameter for LTI mode."""
        cfg = self.config
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_param.copy_(inv_dt)

    def _compute_dt(self, x, time_gaps=None):
        """
        Compute dt based on delta_mode.

        Args:
            x: (B, D, L) post-conv activated features
            time_gaps: (B, L) inter-observation time gaps (in days)

        Returns:
            dt: (B, D, L) discretization step sizes (after softplus)
        """
        B, D, L = x.shape

        if self.delta_mode == "selective":
            # Standard Mamba-1: dt from input
            x_dt = self.x_to_dt(rearrange(x, "b d l -> (b l) d"))
            dt = self.dt_proj(x_dt)  # (BL, d_inner)
            dt = rearrange(dt, "(b l) d -> b d l", b=B, l=L)
            dt = F.softplus(dt)

        elif self.delta_mode == "lti":
            # Fixed dt, broadcast across batch and time
            dt = F.softplus(self.dt_param)  # (D,)
            dt = dt.unsqueeze(0).unsqueeze(2).expand(B, -1, L)

        elif self.delta_mode == "gap":
            # dt purely from time gaps
            assert time_gaps is not None, "gap mode requires time_gaps input"
            gaps = time_gaps.unsqueeze(2)  # (B, L, 1)
            dt = self.gap_to_dt(gaps)  # (B, L, D)
            dt = rearrange(dt, "b l d -> b d l")
            dt = F.softplus(dt)

        elif self.delta_mode == "gap_selective":
            # dt from both input and time gaps (additive in pre-softplus space)
            assert time_gaps is not None, "gap_selective mode requires time_gaps input"
            # Input contribution
            x_dt = self.x_to_dt(rearrange(x, "b d l -> (b l) d"))
            dt_from_x = self.dt_proj(x_dt)  # (BL, d_inner)
            dt_from_x = rearrange(dt_from_x, "(b l) d -> b d l", b=B, l=L)
            # Gap contribution
            gaps = time_gaps.unsqueeze(2)  # (B, L, 1)
            dt_from_gap = self.gap_to_dt(gaps)  # (B, L, D)
            dt_from_gap = rearrange(dt_from_gap, "b l d -> b d l")
            # Combine in pre-softplus space
            dt = F.softplus(dt_from_x + dt_from_gap)

        else:
            raise ValueError(f"Unknown delta_mode: {self.delta_mode}")

        return dt

    def _compute_bc(self, x):
        """
        Compute B and C matrices.

        Args:
            x: (B, D, L) post-conv activated features

        Returns:
            B: (B, N, L)
            C: (B, N, L)
        """
        B_batch, D, L = x.shape

        if self.bc_mode == "selective":
            x_bc = self.x_to_bc(rearrange(x, "b d l -> (b l) d"))  # (BL, 2N)
            bc = rearrange(x_bc, "(b l) d -> b d l", b=B_batch, l=L)
            B_mat, C_mat = bc.split(self.d_state, dim=1)  # each (B, N, L)
        else:
            # LTI: fixed B, C. Average across d_inner to get (N,), then broadcast.
            # Actually, B and C in Mamba are per-batch per-time, shape (B, N, L).
            # For LTI, we use a single learned (N,) vector broadcast over B and L.
            B_mat = self.B_param.mean(dim=0)  # (N,)
            B_mat = B_mat.unsqueeze(0).unsqueeze(2).expand(B_batch, -1, L)
            C_mat = self.C_param.mean(dim=0)  # (N,)
            C_mat = C_mat.unsqueeze(0).unsqueeze(2).expand(B_batch, -1, L)

        return B_mat, C_mat

    def forward(self, hidden_states, time_gaps=None, mask=None):
        """
        Args:
            hidden_states: (B, L, D)
            time_gaps:     (B, L) inter-observation time gaps
            mask:          (B, L) 1 for valid, 0 for padding

        Returns:
            output: (B, L, D)
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        B_batch, L, D = hidden_states.shape

        # Project to 2 * d_inner
        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  # each (B, d_inner, L)

        # Local convolution
        x = self.conv1d(x)[..., :L]
        x = F.silu(x)

        # Compute dt, B, C
        dt = self._compute_dt(x, time_gaps)
        A = -torch.exp(self.A_log.float())
        B_mat, C_mat = self._compute_bc(x)

        # Selective scan
        y = selective_scan_ref(x, dt, A, B_mat, C_mat, self.D, z=z, mask=mask)

        # Output projection
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out + residual


# ---------------------------------------------------------------------------
# Full TSC Model
# ---------------------------------------------------------------------------

class MambaTSC(nn.Module):
    """
    Mamba-1 based Time Series Classification model.

    Input:  raw magnitudes (B, L) + time_gaps (B, L) + mask (B, L)
    Output: class logits (B, num_classes)
    """
    def __init__(self, config: MambaTSCConfig):
        super().__init__()
        self.config = config

        # Determine actual input dimension
        actual_input_dim = config.input_dim
        if config.gap_as_feature:
            actual_input_dim += 1  # concatenate time gap

        # Input projection
        self.input_proj = nn.Linear(actual_input_dim, config.d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.num_classes),
        )

    def forward(self, magnitudes, time_gaps=None, mask=None):
        """
        Args:
            magnitudes: (B, L) raw magnitude values
            time_gaps:  (B, L) inter-observation time gaps (days).
                        First timestep gap is 0.
            mask:       (B, L) 1 for valid observations, 0 for padding

        Returns:
            logits: (B, num_classes)
        """
        B, L = magnitudes.shape

        # Build input features
        x = magnitudes.unsqueeze(-1)  # (B, L, 1)
        if self.config.gap_as_feature:
            assert time_gaps is not None, "gap_as_feature=True requires time_gaps"
            x = torch.cat([x, time_gaps.unsqueeze(-1)], dim=-1)  # (B, L, 2)

        # Project to model dimension
        h = self.input_proj(x)  # (B, L, d_model)

        # Pass through Mamba blocks
        for layer in self.layers:
            h = layer(h, time_gaps=time_gaps, mask=mask)

        h = self.final_norm(h)

        # Pool over valid tokens
        if mask is not None:
            if self.config.pool_mode == "mean":
                mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
                h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            elif self.config.pool_mode == "last":
                # Get index of last valid token per sample
                lengths = mask.sum(dim=1).long()  # (B,)
                h = h[torch.arange(B, device=h.device), lengths - 1]
        else:
            if self.config.pool_mode == "mean":
                h = h.mean(dim=1)
            else:
                h = h[:, -1, :]

        # Classify
        logits = self.classifier(h)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Dual-Band Model (per-band encoder → concat embeddings → classifier)
# ---------------------------------------------------------------------------

class MambaTSCDual(nn.Module):
    """
    Two independent Mamba encoders (one per band), pooled embeddings
    concatenated, then classified. Matches StarEmbed concat convention.

    Input:  mags_g (B, L), mags_r (B, L), gaps_g, gaps_r, mask_g, mask_r
    Output: logits (B, num_classes)
    """
    def __init__(self, config: MambaTSCConfig):
        super().__init__()
        self.config = config

        # Determine actual input dimension
        actual_input_dim = config.input_dim
        if config.gap_as_feature:
            actual_input_dim += 1

        # Shared architecture, independent weights per band
        self.input_proj_g = nn.Linear(actual_input_dim, config.d_model)
        self.input_proj_r = nn.Linear(actual_input_dim, config.d_model)

        self.layers_g = nn.ModuleList([MambaBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.layers_r = nn.ModuleList([MambaBlock(config, layer_idx=i) for i in range(config.n_layers)])

        self.norm_g = nn.LayerNorm(config.d_model)
        self.norm_r = nn.LayerNorm(config.d_model)

        # Classifier takes concatenated embeddings (2 * d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model * 2, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, config.num_classes),
        )

    def _encode_band(self, mags, gaps, mask, input_proj, layers, norm):
        B, L = mags.shape
        x = mags.unsqueeze(-1)
        if self.config.gap_as_feature:
            x = torch.cat([x, gaps.unsqueeze(-1)], dim=-1)
        h = input_proj(x)
        for layer in layers:
            h = layer(h, time_gaps=gaps, mask=mask)
        h = norm(h)
        # Mean pool over valid tokens
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            h = (h * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return h  # (B, d_model)

    def forward(self, mags_g, mags_r, gaps_g, gaps_r, mask_g, mask_r):
        emb_g = self._encode_band(mags_g, gaps_g, mask_g,
                                   self.input_proj_g, self.layers_g, self.norm_g)
        emb_r = self._encode_band(mags_r, gaps_r, mask_r,
                                   self.input_proj_r, self.layers_r, self.norm_r)
        combined = torch.cat([emb_g, emb_r], dim=-1)  # (B, 2*d_model)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Convenience constructors for each experimental variant
# ---------------------------------------------------------------------------

def build_mamba_lti(num_classes=7, **kwargs) -> MambaTSC:
    """Mamba with all LTI parameters (S4-like baseline)."""
    config = MambaTSCConfig(
        delta_mode="lti", bc_mode="lti",
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


def build_mamba_selective(num_classes=7, **kwargs) -> MambaTSC:
    """Standard Mamba-1 (fully selective Delta/B/C)."""
    config = MambaTSCConfig(
        delta_mode="selective", bc_mode="selective",
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


def build_mamba_gap_delta(num_classes=7, **kwargs) -> MambaTSC:
    """Mamba with Delta derived from inter-observation time gaps."""
    config = MambaTSCConfig(
        delta_mode="gap", bc_mode="selective",
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


def build_mamba_gap_selective(num_classes=7, **kwargs) -> MambaTSC:
    """Mamba with Delta from both input + time gaps."""
    config = MambaTSCConfig(
        delta_mode="gap_selective", bc_mode="selective",
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


def build_mamba_gap_feature(num_classes=7, **kwargs) -> MambaTSC:
    """Mamba with time gaps as input feature (control for H2)."""
    config = MambaTSCConfig(
        delta_mode="selective", bc_mode="selective",
        gap_as_feature=True,
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


def build_mamba_both(num_classes=7, **kwargs) -> MambaTSC:
    """Mamba with gap-as-feature AND gap-informed Delta."""
    config = MambaTSCConfig(
        delta_mode="gap_selective", bc_mode="selective",
        gap_as_feature=True,
        num_classes=num_classes, **kwargs,
    )
    return MambaTSC(config)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    B, L = 4, 200
    mags = torch.randn(B, L)
    gaps = torch.abs(torch.randn(B, L)) * 5  # simulated gaps in days
    gaps[:, 0] = 0
    mask = torch.ones(B, L)
    mask[:, -20:] = 0  # last 20 are padding

    variants = {
        "LTI":           build_mamba_lti,
        "Selective":     build_mamba_selective,
        "GapDelta":      build_mamba_gap_delta,
        "GapSelective":  build_mamba_gap_selective,
        "GapFeature":    build_mamba_gap_feature,
        "Both":          build_mamba_both,
    }

    for name, builder in variants.items():
        model = builder(num_classes=7, d_model=64, n_layers=2)
        n_params = sum(p.numel() for p in model.parameters())
        logits = model(mags, time_gaps=gaps, mask=mask)
        print(f"{name:15s} | params: {n_params:>7,} | output: {logits.shape}")
