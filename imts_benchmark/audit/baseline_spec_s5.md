# S5 Baseline Spec — `s5-pytorch` Port

**Date**: 2026-04-22 (initial) · 2026-04-23 (post Phase-2 revision) · 2026-04-25 (Phase-5 real-data revision)
**Status**: Paper-native config (d_model=128, state_dim=256, 6 layers, ~1.4M params) used in Phases 2/3/4-1/4-2 and Phase-5 real-data runs.
**Verdict**: **GREEN** — kernel is unmistakably S5 (not S4). Two stylistic deviations from the paper (LN vs BatchNorm, GELU FFN vs `half_glu1`) are documented and minor. The non-trivial design choice — *per-variate channel-independence* — is a forced consequence of IMTS data shape and is explicitly justified below.

---

## 1. Config currently in training (paper-native)

| Knob | Value | Source |
|---|---|---|
| `d_model` | 128 | Smith et al. 2023 pendulum config |
| `state_dim` | 256 | Smith et al. 2023 pendulum config |
| `n_layers` | 6 | Smith et al. 2023 |
| LR | 1e-3 | paper-native |
| Weight decay | 0.05 | paper-native |
| Physical batch | 32 (accumulate 4× → effective 128) | H100-80GB memory at B·V=384 scan |
| WD routing | `Lambda` / `log_step` / `D` / `B` / `C` **excluded** from decay | paper-native; critical |
| Patience (early stop) | 10 (Phase-5 p10 runs) / 50 (Phase 2-4 baseline runs) | matches T-PatchGNN protocol for Phase 5 |
| **Total params** | **1,395,969 (~1.40M)** | counted via `tools/count_params` |

The earlier *forced-7.8M* config (`d_model=384, state_dim=96`, lr=5e-4, wd=0.01 applied to all params) collapsed 3–5/5 seeds in Phase 2 and was abandoned. Paper-native config + SSM-param WD exclusion + patience=50 made S5 a competitive baseline on Phase 2 irregular regimes.

---

## 2. Provenance — what this wrapper inherits, what it changes

### 2.1 Source

- **Paper**: Smith, Warrington, Linderman. *Simplified State Space Layers for Sequence Modeling*. ICLR 2023. arXiv:2208.04933.
- **Reference code**: https://github.com/lindermanlab/S5 (JAX/Flax). The pendulum task in `_upstream/S5_pendulum/` is the only continuous-time / irregular-Δt benchmark in the original repo and is what we mirror.
- **Our PyTorch dependency**: `s5-pytorch==0.2.1` (Kwaijtaal port), installed in `pythonenvs/mamba`.
- **API we use**: `from s5 import S5; m = S5(width=128, state_width=256); y = m(signal, step_scale=delta_t)`.

### 2.2 Why we do NOT use `S5Block`

The port ships an `S5Block` class (with GLU FFN + LayerNorm + residual) but its `forward` does **not** thread `step_scale`. Since per-step Δt is the entire reason we use S5 for IMTS, we use the raw `S5` module and build our own block:

```python
class S5TemporalBlock(nn.Module):              # s5_forecaster.py:41
    # Pre-LN → S5(step_scale=Δt) + residual → Pre-LN → GELU FFN(×4) + residual
    def forward(self, x, step_scale):
        x = x + self.s5(self.ln1(x), step_scale=step_scale)
        x = x + self.ffn(self.ln2(x))
        return x
```

### 2.3 Per-(variate, time) feature pipeline

Each observation contributes one token. For variate v at time t with value x:
1. Mask future values where `timestamp ≥ history` (set to 0).
2. Token feature = concat([masked_value, time_emb(t)]) → linear projection → d_model=128.
3. Flatten `(B, V, L)` → `(B·V, L, D)` and pass through 6 stacked `S5TemporalBlock`s.
   **Weights are shared across variates** (one set of parameters, applied independently to each variate's stream).
4. Per-step linear head: d_model=128 → 1.
5. MSE loss over `pred_mask` (timestamps ≥ history & valid).

---

## 3. Is this S5 or S4? **It is S5.**

A common confusion: per-variate stacking *looks* like S4-style channel-independent SISO. The kernel is what defines the model class, not the stacking pattern. Verified in `s5/s5_model.py`:

| Aspect | S4 (Gu et al. 2022) | S5 (Smith et al. 2023) | **Our wrapper** |
|---|---|---|---|
| State matrix | DPLR (Λ − PQ*) | **Diagonal Λ ∈ ℂᴾ** | **Diagonal Λ** ([s5_model.py:166](../../../../pythonenvs/mamba/lib/python3.12/site-packages/s5/s5_model.py)) |
| Recurrence solver | Cauchy kernel / FFT | **Parallel associative scan** | **Parallel scan** ([s5_model.py:44](../../../../pythonenvs/mamba/lib/python3.12/site-packages/s5/s5_model.py)) |
| Per-step Δt | single learnable scalar `step` | **`step_scale[B, T] · exp(log_step)`** ([s5_model.py:263](../../../../pythonenvs/mamba/lib/python3.12/site-packages/s5/s5_model.py)) | **identical to S5** |
| Initialization | HiPPO-LegS (DPLR) | HiPPO-N (diagonalize HiPPO eigenvalues) | **HiPPO-N** (`make_DPLR_HiPPO`, keep diagonal) |
| Native multivariate use | SISO + dense channel mixer | MIMO at `d_model` level | MIMO **at d_model level**, applied per-variate |

Our SSM operator is the S5 SSM operator. It happens to be wrapped in a *channel-independent* (per-variate) loop, but that is a wrapping choice, not a kernel choice. PatchTST uses channel-independence with Transformers; nobody calls it "non-Transformer."

---

## 4. Deviations from the S5 paper — three things, one is forced

### 4.1 Stylistic (minor)

| Aspect | S5 paper (pendulum) | Our wrapper | Impact |
|---|---|---|---|
| FFN activation | `half_glu1` (gated) | plain GELU FFN(×4) | minor; both are common Pre-LN block choices |
| Norm | BatchNorm | LayerNorm | minor; LN is the more common default in 2026 codebases |

### 4.2 Multivariate handling — *per-variate channel-independence* (forced)

The pendulum task in the S5 paper is a **single-stream regression** problem, so the paper does not prescribe a multivariate recipe. For multivariate IMTS, we have to choose:

- **Option A (channel-independent, ours)**: Each variate's stream is processed independently by a parameter-shared S5 stack. No cross-variate fusion. Forces the model to use *only intra-variate temporal structure*. This is identical to PatchTST's design.
- **Option B (true MIMO across variates)**: Stack all variates' values at each timestep into a `V`-dim vector and let S5 mix them via its B/C matrices. Requires variates to share the same observation timestamps — i.e., a pre-aligned regular grid. **Not usable on raw IMTS**, where each variate has its own irregular timestamps. Would require a pre-alignment step (which is exactly what Mamba-MV does via its shared grid + VarAttention).
- **Option C (hybrid)**: Channel-independent S5 + a separate cross-variate fusion layer (e.g., attention). Defensible but expands the baseline beyond "pure S5."

**We chose Option A** to keep the S5 baseline an unmodified per-variate SSM. This is the honest characterization: our S5 baseline tests the per-variate SSM hypothesis, and Mamba-MV's contribution lies in the shared-grid fusion that S5 does not have.

The cost is real: S5 cannot fill a gap in one variate using information from the other variates. Phase 4-2 demonstrated this (random-variate gap, `RESULTS_phase4_2.md`), and Phase-5 Activity (V=12, where cross-variate accelerometer correlations matter) shows S5 underperforming RoMAE / Mamba-MV by ~6× on test MSE — the same channel-independence cost.

---

## 5. Per-dimension audit verdicts

| Dimension | Status | Notes |
|---|---|---|
| Task adaptation (forecasting) | GREEN | Per-variate SSM + forecast-position-masked MSE. Standard pattern. |
| Forward signature | GREEN | S5 API verified; shapes match. |
| Per-step Δt handling | GREEN | `step_scale=delta_t` tensor flows through; matches paper's continuous-time formulation. |
| HiPPO-N initialization | GREEN | `make_DPLR_HiPPO` confirmed in port source. |
| Parallel scan correctness | GREEN | Port v0.2.1 includes the 2026-04-26 "bad state carrying" fix. Phase-2 training loss curves are smooth and decreasing. |
| Kernel = S5 (not S4) | GREEN | Diagonal Λ, parallel scan, per-step Δt — all S5 markers. |
| Multivariate handling | YELLOW (by design) | Per-variate channel-independent. Justified above as the honest S5 baseline; the cost shows up exactly as expected on cross-variate-coupled data. |
| Param count | GREEN | 1,395,969 (paper-native config). |

---

## 6. Reference block (for the paper)

> "We benchmark against S5 (Smith et al., ICLR 2023) using the community PyTorch port `s5-pytorch` v0.2.1. Our wrapper uses the raw `S5` module (not the port's pre-built `S5Block`, which does not thread per-step Δt through its forward) and builds a Pre-LN + GELU FFN block around it. We use the paper-native pendulum config (`d_model=128, state_dim=256, n_layers=6`, AdamW with weight decay 0.05 excluding the SSM-structured parameters Λ, log_step, B, C, D), totaling 1.40M parameters. Each variate is processed independently by the same parameter-shared S5 stack — i.e., channel-independent SISO at the variate level (analogous to PatchTST's design), MIMO at the `d_model` level. This is a forced choice for irregular multivariate data, where variates do not share observation timestamps and true MIMO across variates would require pre-alignment. Forecast positions are masked before the SSM stack (values zeroed where timestamp ≥ history) and MSE loss is computed only over those positions. The S5 kernel itself — diagonal HiPPO-N initialization, parallel associative scan, per-step `Δt` via `step_scale` — is exactly as described in Smith et al. The two stylistic departures from the paper are LayerNorm in place of BatchNorm and a plain GELU FFN in place of `half_glu1`; both are minor."

---

## 7. Honest limitations to disclose in the paper

1. **No public LRA reproduction of the port** vs. the JAX reference. We treat this as a qualitative S5 baseline; numbers may differ from a JAX-native re-implementation by hyperparameter-tuning amount.
2. **Channel-independence is an architectural ceiling on coupled data** (Activity V=12, USHCN V=5). When the gold-standard answer requires cross-variate information, S5 cannot recover it. This is a feature of the comparison (it isolates Mamba-MV's cross-variate fusion contribution), not a bug to be fixed.
