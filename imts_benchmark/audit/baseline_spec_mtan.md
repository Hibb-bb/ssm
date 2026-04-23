# mTAN Baseline Spec — Upstream-Contract Audit

**Date**: 2026-04-21 (initial) · 2026-04-22 (status revision)
**Verdict**: **DROPPED** — wrapper is still implemented correctly against the upstream contract (the original GREEN audit stands on code-contract grounds), but every training config we tried in Phase 2 collapsed to constant-output predictions. Removed from Phase 3 / 4 / 4-1 / 4-2 runs.

## Why dropped (2026-04-22 decision)

Five configurations tried on `multisin_med_irreg`, all 4–5/5 seeds collapsed (test MSE ≈ `Var(y)`):

1. Forced 7.8M (`latent_dim=128, rec_hidden=384`) at `lr=5e-4, wd=0.01`, patience=20
2. Paper-native (`latent_dim=40, rec_hidden=64, gen_hidden=50`) at `lr=5e-4, wd=0.01`, patience=50
3. Paper-native at `lr=1e-4, wd=0.01`, patience=50
4. Paper-native with timestamp normalization `t → t / t_max ∈ [0, 1]`, patience=50
5. Paper-native with `sin(Linear(1, 127)(t))` time embedding removed (hot-fix), patience=50

**Suspected root cause**: mTAN's `sin(Linear(1, 127)(t))` reference-attention time encoding is scale-sensitive, and the architecture is designed for the *imputation* objective (MSE on randomly missing interior points with a VAE prior), not strict forecasting. The deterministic-forecasting use-case fights the VAE decoder's expectation of latent stochasticity at generation time. Even with time normalized to [0, 1] and the VAE KL suppressed, the model did not learn.

**Status of the wrapper code.** Preserved under `mtan_forecaster/` for reference and potential future revival. The `baseline_spec` below documents the original contract audit (still correct); it is retained so anyone reviving mTAN has the wrapper design notes.

---

## Upstream summary

- **Paper**: Shukla & Marlin, "Multi-Time Attention Networks for Irregularly Sampled Time Series", ICLR 2021.
- **Source**: https://github.com/reml-lab/mTAN, vendored at `mtan_forecaster/_upstream/models.py`
- **Original tasks**: (1) **Interpolation** on MIMIC-III / PhysioNet (MSE on unobserved positions); (2) **Classification** on PhysioNet (cross-entropy on sepsis/mortality labels).
- **Architecture**:
  - `enc_mtan_rnn` (lines 71–122) — reference-point attention → bidirectional GRU over reference latents → linear projection to `2·latent_dim` (mean + log-variance, VAE-style).
  - `dec_mtan_rnn` (lines 125–176) — latent at reference points → attention query at decode times → GRU → linear projection to `input_dim`.
  - **Reference points**: fixed learned query positions spanning `[0, t_max]` (default 64 points).

## Our wrapper's adaptation

- **File**: `mtan_forecaster/mtan_forecaster.py`
- **Input construction** (`_build_inputs`, lines 107–203): same sorted-union-plus-one-hot-scatter pattern as the ContiFormer wrapper — `(B, V, L_max)` → `(B, T_max, 2V)`:
  - History positions: `x_val = value, x_msk = 1`
  - Forecast positions: `x_val = 0, x_msk = 0` (encoder-hidden)
- **Reference points** (fair_defaults): uniform linspace across `[0, 10.0]`, 64 points.
- **Deterministic use** (line 217): `z_mean = z_out[:, :, :latent_dim]` — we take only the mean and discard log-variance. No KL regularizer; no sampling.
- **Decoder**: called with the same `time_steps` used for encoder (sorted union). Output `[B, T_max, V]`.
- **Loss**: MSE restricted to `pred_mask_full == True` positions.
- **Device patch** (`_patch_device`, line 207): upstream hardcodes `self.device` at init; we monkey-patch at each forward to respect Lightning's device placement.

## Diff table

| Upstream arg/behavior | Wrapper value | Notes |
|---|---|---|
| `x [B, T, 2V]` (values + mask concat) | `[B, T_max, 2V]`, per-variate one-hot | Matches upstream shape; forecast hidden |
| `time_steps [B, T]` | Sorted union of valid timestamps | Matches upstream semantics |
| `query` (reference points) | `torch.linspace(0.0, t_max, 64)` | Reasonable; paper didn't specify count |
| Encoder output `[B, 64, 2·latent_dim]` (mean+logvar) | Take mean only, drop logvar | Deterministic use; appropriate for regression |
| KL loss (VAE) | Not applied | Point-prediction task, no sampling needed |
| MSE on unobserved positions | MSE on `pred_mask_full` positions | Same semantic (interpolation = forecasting hidden future) |

## Per-dimension verdicts

| Dimension | Status | Notes |
|---|---|---|
| Task framing (interpolation→forecasting) | GREEN | Paper's primary task is interpolation; forecasting is "interpolating the future" — natural fit |
| Encoder/decoder signatures | GREEN | Shapes correct; ref-point query/key roles correctly swapped between encoder and decoder |
| Reference-point placement | GREEN | Uniform linspace across `[0, t_max]` is sensible |
| Mask convention (1=observed, 0=missing) | GREEN | Matches upstream (`scores.masked_fill(mask == 0, -1e9)` at upstream line 49) |
| Time embedding (`learn_emb=True`, embed_time=128) | GREEN | Standard config; handles absolute times directly |
| Loss (MSE on forecast) | GREEN | Identical to paper's interpolation loss |
| Deterministic use (no KL) | GREEN | Appropriate for point-prediction regression |
| Hyperparameters | YELLOW | `rec_hidden=gen_hidden=576, latent_dim=64, num_ref=64` — tuned for 7.8M param budget, not validated vs paper config |
| Device patching | GREEN | Intentional, documented workaround for Lightning distributed placement |

## Hyperparameters

Our defaults (`rec_hidden=576, gen_hidden=576, latent_dim=64, embed_time=128, num_ref_points=64, num_heads=1, learn_emb=True`). **Param count**: 7,727,805 (−0.93% of 7.8M target).

Paper doesn't publish explicit configs in the vendored README; our choices are driven by the fair-comparison param budget. Acceptable for intra-paper comparison.

## Bottom-line

mTAN is being used correctly for forecasting. The wrapper faithfully implements the paper's encoder-decoder architecture with proper one-hot multivariate scattering, masking-based future hiding, deterministic latent extraction, and interpolation-style MSE loss. No blocking concerns; launch Phase-2 training as-is.

**Optional polish** (not blocking):
- If overfitting appears in smoke (val loss stagnates while train loss drops), consider adding a small KL regularizer (`β · KL(z_mean, N(0,1))`) to restore some of the VAE's regularization effect. Unlikely to be needed on synthetic data but cheap to add.
