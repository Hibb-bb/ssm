# ContiFormer Baseline Spec — Upstream-Contract Audit

**Date**: 2026-04-21
**Verdict**: **YELLOW** — structurally correct for forecasting, but three execution risks worth tracking during smoke runs.

## Upstream summary

- **Paper**: ContiFormer (Chen et al., NeurIPS 2023) — continuous-time transformer via ODE-attention for irregular time series classification.
- **Source**: Microsoft `physiopro` repo, vendored at `contiformer_forecaster/_upstream/physiopro/`.
- **Main class**: `ContiFormer` at `contiformer.py:246–335` (encoder-only).
- **Forward signature**: `forward(x, t, mask)`
  - `x`: `[B, T, input_size]` — pre-embedded features
  - `t`: `[B, T]` — raw timestamps
  - `mask`: `[B, T, T]` bool — True = "block this attention position" (padding)
- **Output**: `(enc_output [B, T, d_model], last_hidden [B, d_model])`
- **Attention mechanism**: Queries are linear projections (`InterpLinear`). Keys/values are ODE-integrated from each observation time to each query time (`ODELinear`) via `torchdiffeq.odeint_adjoint`.
- **Time encoding**: Sinusoidal positional encoding (transformer-style absolute, not relative) added per layer.
- **Original loss**: Cross-entropy for UEA classification benchmarks (encoder-only; loss defined by downstream task).

## Our wrapper's adaptation for forecasting

- **File**: `contiformer_forecaster/contiformer_forecaster.py`
- **Input construction** (`_build_inputs`, lines 112–188):
  1. Flatten `(B, V, L_max)` per-variate batch to `(B, V*L_max)`
  2. Sort by timestamp (invalid tokens pushed to `inf`)
  3. Pad to `T_max = max(n_valid per sample)`
  4. Scatter values into `[B, T_max, V]`, masks into `[B, T_max, V]`
  5. Concatenate → `[B, T_max, 2V]`
  6. Zero forecast positions in **both** value and mask channels (MAE-style hide)
- **Attention mask**: `[B, T_max, T_max]` with True at padding positions, matching upstream convention.
- **Head**: linear `d_model → V` applied at each time step; predictions at forecast positions used for loss.
- **Loss**: `MSE * pred_mask_full` summed and normalized by `pred_mask_full.sum()` — MSE restricted to forecast positions.

## Diff table

| Upstream arg | Wrapper value | Notes |
|---|---|---|
| `x` | `[B, T_max, 2V]` | Pre-embedded by our _build_inputs (value + mask concat per variate) |
| `t` | `[B, T_max]` sorted-union | Raw timestamps, unnormalized (0–10); ODE solver remaps internally to [-1, 1] |
| `mask` | `[B, T_max, T_max]`, True at pad | Correct upstream semantic |
| — | Head (linear d_model → V) | Added by wrapper; upstream is encoder-only |
| — | MSE on pred_mask | Added by wrapper (upstream had no loss defined) |

## Per-dimension verdicts

| Dimension | Status | Notes |
|---|---|---|
| Task adaptation (classification→forecasting) | YELLOW | Valid extension; encoder is task-agnostic, but untested against paper's published results |
| Forward signature | GREEN | Shapes and semantics correctly aligned |
| Mask semantics (True = block) | GREEN | Matches upstream's `attn.masked_fill(mask, -1e9)` |
| Time scale (unnormalized) | GREEN | ODE solver internally remaps to [-1,1]; absolute-time sinusoidal PE handles any scale |
| ODE tolerances | **RED** | `atol = rtol = 1e-1` — ~10–1000× looser than typical DiffEq practice; not justified in code or paper |
| Memory | YELLOW | O(L²) × hidden × n_head × d_k ≈ 15 GB for B=128, L=360; documented escape hatches but no auto-trigger |
| Hyperparameters | YELLOW | 320/6/4 plausible but not cross-checked against any published config |
| Loss | GREEN | MSE on forecast positions is appropriate for regression |

## Hyperparameters

Our defaults (`d_model=320, n_layers=6, n_head=4, d_k=80, d_inner=1024, dropout=0.1, method=rk4, atol=rtol=1e-1`). Constraint `d_model = n_head × d_k` holds (320 = 4×80). **Param count**: 7,652,867 (−1.89% of 7.8M target).

## Three concrete risks to track during smoke runs

### 1. ODE tolerances too loose (RED)

`atol = rtol = 1e-1` means the ODE solver may prematurely converge and emit inaccurate key/value projections. Standard DiffEq practice is 1e-3 to 1e-6.

**Mitigation options**:
- Tighten to 1e-4 for smoke runs; if wall-clock rises sharply but metrics unchanged, keep 1e-1; otherwise adopt tighter.
- Alternative: instrument `torchdiffeq` to log step-size statistics; compare on smoke vs. full run.

### 2. Memory O(L²) risk (YELLOW)

For batch=128, L≈360 (3 variates × ~120 obs/var), attention projection allocates `[B, L, L, hidden, n_head, d_k]`:
`128 × 360 × 360 × 1 × 4 × 80 ≈ 660 MB per head → ~15 GB total before activations/gradients`.

**Mitigation**: if OOM on smoke, drop `train_batch_size` to 64. Document as a per-model override (breaks fairness: note this as caveat in RESULTS).

### 3. Hyperparameter choices unvalidated against paper (YELLOW)

No paper-reported config in the vendored repo. Our `d_model=320` etc. are our own choice, tuned only to hit the ~7.8M param budget.

**Mitigation**: acceptable for this work (user's priority is fair intra-paper comparison, not reproducing ContiFormer's originally-published numbers). Document in RESULTS as a design caveat.

## Bottom-line

ContiFormer is being used correctly for forecasting in broad outline (masking, time encoding, loss are sound). Launch smoke tests with current hyperparameters **but instrument**:
1. ODE solver step-size / convergence warnings
2. Peak GPU memory per batch

If smoke runs 2× slower than expected OR show anomalous test MSE, first suspect: loose ODE tolerances. Run a single-seed ablation with `atol=rtol=1e-4` before launching the 20-run array.
