# RoMAE Baseline Spec — Upstream-Contract Audit

**Date**: 2026-04-21
**Verdict**: **GREEN** — no concrete issues to fix before Phase-2 training launch.

## Upstream summary

- **Paper**: RoMAE (arXiv:2505.20535) — masked autoencoder for irregular time series; pretraining + fine-tuning on classification/interpolation. No explicit forecasting benchmark in original paper.
- **Main class**: `RoMAEForPreTraining` at `_romae_repo/romae/model.py:284–406`
- **Forward signature**: `forward(values, mask, positions, pad_mask=None, label=None)`
  - `values`: `[B, T, C, H, W]` 5D (tubelet-patchified)
  - `mask`: `[B, N]` bool, **True = masked (to reconstruct)**
  - `positions`: `[B, n_pos_dims, N]` float (per-token coordinates for RoPE)
  - `pad_mask`: `[B, N]` bool, True = padding
- **Loss**: `nn.MSELoss(reduction='none')` pre-configured in `__init__` (line 308); applied only to masked reconstructed positions; padding zeroed and averaged.
- **Architecture**: Encoder strips masked positions (`x = x[~mask]`), processes visible tokens, decoder fills in masked positions using mask tokens + positional RoPE.

## Our wrapper's adaptation for forecasting

- **File**: `romae_forecaster/romae_forecaster.py`
- **Token convention**: each observation is one token `(value, timestamp, variate_id)`.
- **Values**: reshaped `[B, N] → [B, N, 1, 1, 1]` (tubelet_size=(1,1,1)).
- **Positions**: `torch.stack([timestamps, variate_id], dim=1)` → `[B, 2, N]` — unnormalized timestamps (0–10 range) consumed directly by RoPE.
- **Mask**: `pred_mask` (True = forecast target) passed straight through — semantically matches upstream's "mask = to reconstruct".
- **Pad mask**: flipped to upstream convention (`~our_pad_mask`) where True = padding.
- **Loss**: identical to upstream (`nn.MSELoss(reduction='none')` set via `model.set_loss_fn`).
- **Normalization**: `normalize_targets=False` (correct for tubelet_size=(1,1,1) per upstream README).

## Diff table

| Upstream arg | Wrapper value | Notes |
|---|---|---|
| `values` | `[B, N, 1, 1, 1]` | Reshaped from `[B, N]`; no zeroing at masked positions (correct — upstream extracts masked values separately) |
| `mask` | `pred_mask` | Same semantic: True = to predict |
| `positions` | `[B, 2, N]` of `(t, variate_id)` | Unnormalized timestamps (RoPE is scale-invariant) |
| `pad_mask` | `~our_pad_mask` | Correctly flipped to upstream's True=pad |
| `label` | Not passed | Loss computed via pre-configured `loss_fn`; `label=None` is upstream default |

## Per-dimension verdicts

| Dimension | Status | Notes |
|---|---|---|
| Task adaptation (MAE→forecasting) | GREEN | Forecasting = masking the future and reconstructing; natural extension of MAE pretraining objective |
| Forward signature | GREEN | All required args passed; no renames that break semantics |
| Mask semantics | GREEN | True=masked matches upstream; encoder strips masked positions as designed |
| Time encoding (RoPE) | GREEN | Unnormalized timestamps acceptable; RoPE is scale-invariant within reasonable ranges |
| Loss | GREEN | Identical to upstream default |
| Hyperparameters | GREEN | 288/7/6 is a reasonable interpolation between RoMAE-tiny (180/12/3) and RoMAE-small (432/12/6) |

## Hyperparameters (our defaults vs published sizes)

| Param | Ours | RoMAE-tiny | RoMAE-small |
|---|---|---|---|
| enc_d_model | 288 | 180 | 432 |
| enc_nhead | 6 | 3 | 6 |
| enc_depth | 7 | 12 | 12 |
| dec_d_model | 180 | — | — |
| dec_nhead | 3 | — | — |
| dec_depth | 2 | — | — |
| p_rope_val | 0.75 | — | — |

**Param count**: 7,803,073 (+0.04% of 7.8M target).

## Bottom-line

RoMAE is being used correctly for forecasting. The wrapper is an informed, faithful adaptation of RoMAE's masked-autoencoder paradigm to the forecasting task: the "future region" is the "mask" in MAE terminology, and the decoder reconstructs it. No code changes required before Phase-2 training.

**Optional polish** (not blocking):
- Add a comment documenting the expected timestamp range (e.g., `[0, t_max]`) near the `positions` construction, for future maintainers.
