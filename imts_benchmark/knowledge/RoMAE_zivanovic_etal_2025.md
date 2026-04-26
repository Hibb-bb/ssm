# RoMAE — Rotary Masked Autoencoders are Versatile Learners
Zivanovic, Di Gioia, Scaffidi, de los Rios, Contardo, Trotta. 2025. arXiv:2505.20535.

## Mechanism summary
RoMAE = MAE-style masked autoencoder + N-dimensional continuous p-RoPE (NDPRope). Each observation is one flat token with N-D continuous position. The encoder-decoder reconstructs masked tokens. The paper demonstrates this works across images, audio, irregular time-series classification, and irregular time-series interpolation — all with no task-specific architectural specialization.

## Sections we cite in this project

### §3.1 — N-D patchification and continuous positions
Each observation becomes a token with a continuous-coordinate position vector ∈ ℝᴺ. For async multivariate time-series, N=2 is used: (timestamp, variate_id).

**Proposition 4.1**: for any irregular dimension, the patch size must be 1. Our wrapper uses `tubelet_size=(1,1,1)` which honors this ([romae_forecaster.py:62](../../romae_forecaster/romae_forecaster.py#L62)).

### §4.2 — Dealing with large numbers of irregular dimensions (THE key section)
> "We optionally reserve a dimension in Axial RoPE that is used to store the dimensional index i... This method is primarily used when working with multi-variate time series where each variate is irregular. When using this approach we also include the learned [CLS] token, which allows the model to know what dimension each sample belongs to despite RoPE being relative."

This is exactly what our wrapper does:
- Position vector = [timestamp, variate_id_as_float] ([romae_forecaster.py:107](../../romae_forecaster/romae_forecaster.py#L107))
- `[CLS]` token included (`use_cls=True`, line ~60)

Verified compliant in [Audit D](../audit/).

### §3.1 (second half) — NDPRope
For an embedding of per-head dim `head_dim` and N positional dimensions, head_dim is split into N equal axial groups of size `axis_dim = head_dim/N`. Each axial group applies p-RoPE on its own position coordinate. Paper uses `p=0.75` (75% of the 2D sub-space rotates, 25% is left unrotated for length extrapolation).

### §4.3 — Effects of relative position + Proposition 4.2
- Without `[CLS]` token → Encoder is translation-invariant in position space.
- With `[CLS]` token → Encoder can reconstruct absolute positions from the `[CLS]` "anchor".

Consequence for our wrapper: since we include `[CLS]`, the variate_id encoding (even as a float) provides absolute-variate-identity to the model. Reconstructing "which variate is this token?" is feasible.

### §5.3 — Irregular time-series classification benchmark
The paper evaluates on DESC ELAsTiCC + UEA datasets + Pendulum. It beats specialized baselines (ATAT, ContiFormer, mTAN, S5) on classification.

**What the RoMAE paper does NOT benchmark**:
- Forecasting with future-horizon masking (our task).
- The paper's masking pattern is interpolation (random middle tokens hidden). Our wrapper masks all tokens with timestamp ≥ history (a contiguous future window). The wrapper handles this correctly via `pred_mask`, but the paper does not test this setting.

### §5.1 — Pretraining is always used in paper
The paper's Tiny ImageNet setup: 200 epochs of self-supervised MAE pretraining on unlabeled data, THEN 15 epochs of labeled fine-tuning. Our wrapper does NOT pretrain — we train from scratch on 1000 samples per regime.

**This is the likely cause** of RoMAE's huge seed variance in our benchmark (std ≈ mean on Phase 2/3/4). Per Part 1 §6 of [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md), a follow-up ablation that adds 100 epochs of pretraining on unlabeled synthetic data should close the variance gap.

## Architectural choices tied to our wrapper

| Knob | Paper default | Our wrapper | Match? |
|---|---|---|---|
| Encoder d_model | 432 (small) / 720 (base) | 288 | close to "small" variant |
| Encoder depth | 12 (small / base) | 7 | reduced |
| Decoder d_model | 180 (asymmetric) | 180 | ✓ |
| p-RoPE p | 0.75 | 0.75 | ✓ |
| `[CLS]` | True | True | ✓ |
| Tubelet size for irregular dim | 1 | 1 | ✓ |
| Pretraining | 200+ epochs on unlabeled | NONE | ✗ (potential source of variance) |

## Non-obvious implementation detail: cache
`NDPRope.cache` is populated on first call within a forward pass and reset at end of each forward ([model.py:404, 501](../../romae_forecaster/_romae_repo/romae/model.py)). This is an optimization, not a bug. See [Audit D](../audit/).
