# Mamba — Linear-Time Sequence Modeling with Selective State Spaces
Gu & Dao. 2023. arXiv:2312.00752.

## Status: **PENDING user attachment of the PDF.**

The user said they will attach the Mamba SSM paper in the next turn. Until then, this stub contains only what we can rely on from common knowledge + the `mamba-ssm` codebase we already import.

Fill in the formal section references AFTER the PDF is attached, for citations in:
- [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md) §4 (Mechanism 3 — `replace` vs `learned` reversal)
- [HPO_PLAN.md](../docs/HPO_PLAN.md) Stage 5 (`multiplicative` dt_mode semantics)
- [AUDIT_C_gamma_rho.md](../audit/AUDIT_C_gamma_rho.md) (references to d_state capacity comparison with S5)

## What we already understand from the code and established knowledge

### Selective scan — Δ is a selection gate, not a discretization step
In `MambaBlock.forward` at [mamba_block.py:140-144](../../../../ssm_model/ssm/mamba_experiments/forecaster/mamba_block.py):
```
dt_raw = x_dbl[:, :self.dt_rank]
dt = self.dt_proj.weight @ dt_raw.t()
dt = F.softplus(dt + self.dt_proj.bias[:, None])
```

Δ is data-dependent and softplus-bounded. It controls how much of each input token gets "mixed in" via the discretization `Ā = exp(Δ·A)`, `B̄ ≈ Δ·B` (for small Δ). A small Δ means "carry previous state, ignore this input"; a large Δ means "reset toward this input".

This is fundamentally different from S5's Δ, which is a ZOH discretization step in continuous time. S5's Δ and Mamba's Δ have the same mathematical role in `exp(A·Δ)` but completely different semantic meaning.

### `dt_min`, `dt_max` init range
`MambaBlock.__init__` at line 110-113 initializes `dt` from a log-uniform over `[dt_min=0.001, dt_max=0.1]`, then inverse-softplusses to set `dt_proj.bias`. This bounds the effective Δ range for a trained model to roughly `[0.001, 0.1]`.

**Implication for `replace` mode on Phase 4**: raw physical `delta_t` in Phase 4 can reach ~3.0 (gap length). That's 30× outside the trained-regime upper bound, which is why `learned` > `replace` on gapped data (per Part 1 §4 of [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md)).

### `expand=2`
Default throughout the codebase. The Mamba paper (per common-knowledge summary) fixes this architectural constant in every experiment.

### `d_state=16`
Much smaller than S5's 256. The Mamba paper uses 16 throughout for language modeling — but that is known to be tuned for language, not for continuous-time regression where larger state helps.

## Sections to fill in once PDF is available

| Stub heading | Expected section from paper | Needed in |
|---|---|---|
| Selection mechanism formal definition | §3.1 or §3.2 | INTERPRETATION §4 |
| `expand=2` rationale | §3.4 | HPO_PLAN non-axes |
| `dropout=0` convention | §E.2.1 | HPO_PLAN non-axes |
| `d_state` ablation results | §6 (language-only) or appendix | AUDIT_C implications |
| Selective vs non-selective SSM comparison | §3.1 or §4 | HPO_PLAN Stage 4 |
| Note on Δ as selection not discretization | §3.2 | INTERPRETATION §4 |

## Action items after the PDF arrives
1. Replace "common-knowledge summary" with direct section quotes in this stub.
2. Update [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md) §4.1 with formal Mamba-Δ semantics quotation.
3. Confirm `dt_mode=multiplicative` (HPO Stage 5) matches the paper's stated design intent for Δ.
