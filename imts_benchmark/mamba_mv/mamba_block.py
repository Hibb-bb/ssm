"""
Mamba block with CUDA-accelerated selective scan (falls back to pure PyTorch).

Provides:
  MambaBlock         - standard Mamba (fully-learned delta)
  MambaIrregularBlock - accepts external delta_t for irregular time series
                        mode="replace": use true dt as delta (Option A)
                        mode="additive": delta = softplus(learned + true dt) (Option C)
                        mode="concat": project [dt_raw || delta_t] -> learned delta
                                      (no additive merge; true gap is an input feature)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_CUDA_SCAN = True
except ImportError:
    HAS_CUDA_SCAN = False


def _selective_scan_pytorch(u, delta, A, B, C, D=None, z=None):
    """Pure PyTorch selective scan (sequential for-loop fallback)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    B = B.float()
    C = C.float()

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    L = u.shape[2]

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    C_t = C.transpose(1, 2)

    x = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
    y = torch.empty(batch, dim, L, device=u.device, dtype=torch.float32)
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y[:, :, i] = (x * C_t[:, i].unsqueeze(1)).sum(-1)

    out = y if D is None else y + u * D.unsqueeze(-1)
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


def selective_scan(u, delta, A, B, C, D=None, z=None):
    """Dispatch to CUDA kernel if available, otherwise pure PyTorch."""
    if HAS_CUDA_SCAN and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D=D, z=z, delta_softplus=False)
    return _selective_scan_pytorch(u, delta, A, B, C, D=D, z=z)


class MambaBlock(nn.Module):
    """Standard Mamba block with fully-learned delta."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.norm = nn.LayerNorm(self.d_model)

    def _compute_delta(self, x, seqlen):
        """Standard: project from input, add bias, softplus."""
        x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, 'd (b l) -> b d l', l=seqlen)
        dt = F.softplus(dt + self.dt_proj.bias[:, None])
        B = rearrange(B, '(b l) n -> b n l', l=seqlen).contiguous()
        C = rearrange(C, '(b l) n -> b n l', l=seqlen).contiguous()
        return dt, B, C

    def forward(self, hidden_states, delta_t=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
            'd (b l) -> b d l',
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias, 'd -> d 1')

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        A = -torch.exp(self.A_log.float())

        dt, B, C = self._compute_delta(x, seqlen)

        # Mamba's CUDA kernel requires delta.dtype == u.dtype. Under
        # bf16-mixed autocast the dt_proj.bias addition silently promotes
        # ``dt`` back to fp32 even though x is bf16, which the kernel
        # rejects with "Expected delta.scalar_type() == input_type".
        if dt.dtype != x.dtype:
            dt = dt.to(x.dtype)

        y = selective_scan(x, dt, A, B, C, self.D.float(), z=z)
        y = rearrange(y, 'b d l -> b l d')
        out = self.out_proj(y)
        return out + residual


class MambaIrregularBlock(MambaBlock):
    """
    Mamba block that injects true time gaps into the SSM discretization.

    mode="replace":  delta = broadcast(delta_t_real) to all d_inner channels
    mode="additive": delta = softplus(learned_dt + broadcast(delta_t_real))
    mode="concat":   concat per-step delta_t to dt_raw; dt_proj is (dt_rank+1) -> d_inner;
                     delta = softplus(projection + bias) (learned path depends on true gap)
    """

    def __init__(self, *args, dt_mode: str = 'replace', **kwargs):
        super().__init__(*args, **kwargs)
        assert dt_mode in ('replace', 'additive', 'concat')
        self.dt_mode = dt_mode

        if dt_mode == 'concat':
            old = self.dt_proj
            self.dt_proj = nn.Linear(self.dt_rank + 1, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj.weight[:, : self.dt_rank].copy_(old.weight)
                self.dt_proj.bias.copy_(old.bias)
            nn.init.zeros_(self.dt_proj.weight[:, self.dt_rank :])
            if hasattr(old.bias, '_no_reinit'):
                self.dt_proj.bias._no_reinit = True

    def forward(self, hidden_states, delta_t=None):
        assert delta_t is not None, 'MambaIrregularBlock requires delta_t'

        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
            'd (b l) -> b d l',
            l=seqlen,
        )

        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias, 'd -> d 1')

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        A = -torch.exp(self.A_log.float())

        x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))
        _, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        B = rearrange(B, '(b l) n -> b n l', l=seqlen).contiguous()
        C = rearrange(C, '(b l) n -> b n l', l=seqlen).contiguous()

        dt_real = repeat(delta_t, 'b l -> b d l', d=self.d_inner).float()
        dt_real = dt_real.clamp(min=1e-6)

        if self.dt_mode == 'replace':
            dt = dt_real
        elif self.dt_mode == 'additive':
            dt_raw = x_dbl[:, : self.dt_rank]
            dt_learned = self.dt_proj.weight @ dt_raw.t()
            dt_learned = rearrange(dt_learned, 'd (b l) -> b d l', l=seqlen)
            dt_learned = dt_learned + self.dt_proj.bias[:, None]
            dt = F.softplus(dt_learned + dt_real)
        elif self.dt_mode == 'concat':
            dt_raw = x_dbl[:, : self.dt_rank]
            dt_feat = rearrange(delta_t.float().clamp(min=1e-6), 'b l -> (b l) 1')
            dt_feat = dt_feat.to(dtype=dt_raw.dtype)
            dt_in = torch.cat([dt_raw, dt_feat], dim=-1)
            dt_learned = self.dt_proj.weight @ dt_in.t()
            dt_learned = rearrange(dt_learned, 'd (b l) -> b d l', l=seqlen)
            dt_learned = dt_learned + self.dt_proj.bias[:, None]
            dt = F.softplus(dt_learned)
        else:
            raise RuntimeError(f'unexpected dt_mode={self.dt_mode!r}')

        # Mamba's CUDA kernel requires delta.dtype == u.dtype (and z.dtype).
        # Under bf16-mixed autocast, x is bf16 while delta_t arrives as fp32,
        # so the `.float()` casts above leave dt in fp32 and the kernel
        # rejects it. Cast back to x's dtype here.
        if dt.dtype != x.dtype:
            dt = dt.to(x.dtype)

        y = selective_scan(x, dt, A, B, C, self.D.float(), z=z)
        y = rearrange(y, 'b d l -> b l d')
        out = self.out_proj(y)
        return out + residual
