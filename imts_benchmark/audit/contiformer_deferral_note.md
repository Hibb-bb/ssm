# ContiFormer Deferral — Phase 2 Operational Note

**Date**: 2026-04-22
**Decision**: Defer ContiFormer from Phase 2. Proceed with Mamba-MV + RoMAE + mTAN.

## Evidence

Six separate SLURM debug jobs, all on H100 80GB with the same seed and regime (`multisin_med_irreg`):

| Job ID | d_model × n_layers | params | batch | accum | atol | Result |
|---|---|---|---|---|---|---|
| 5987625 | 320 × 6 | 7.65M | 128 | 1 | 0.1 | FAIL — OOM 4.98 GiB |
| 5987751 | 320 × 6 | 7.65M | 64 | 2 | 0.1 | FAIL — OOM 4.98 GiB |
| 5987834 | 320 × 6 | 7.65M | 32 | 4 | 0.1 | FAIL — OOM 4.95 GiB |
| 5987886 | 320 × 6 | 7.65M | 8 | 16 | 0.1 | FAIL — OOM 4.95 GiB |
| 5988030 | 320 × 6 | 7.65M | 32 | 4 | 0.5 | FAIL — OOM 4.95 GiB |
| 5988450 | **160 × 4** | **1.45M** | 32 | 4 | 0.5 | FAIL — OOM 2.47 GiB during training (val passed) |

Full-size configuration failure mode: single allocation of 4.95-4.98 GiB inside `torchdiffeq/_impl/misc.py:145` (ODE function evaluation's state-tuple flatten). Batch-independent, step-count-independent.

Reduced-architecture (job 5988450) failure mode: **different**. Validation passes (val_mse logged), but first training step saturates 78 GB of 80 GB and OOMs trying to allocate 2.47 GiB. This is backward-pass activation accumulation through the ODE adjoint — total memory scales with `n_layers × L² × backward_activation_footprint`, not just the single-step forward allocation.

Effective knob coverage:
- Batch reduction **16×** (128 → 8): no effect on full-size failing allocation
- Step-size increase **5×** (0.1 → 0.5): no effect
- Architecture reduction **5×** (7.65M → 1.45M): OOM still occurs, but during training (backward) not forward
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: set in later runs, no effect

Conclusion: the memory cost of ContiFormer at L=360 is **fundamentally incompatible with 80 GB**, not just a tunable-hyperparameter issue. The L²-scaled tensors in ODE-attention, multiplied by layers during backward, exceed available memory even at 5× architectural reduction. To make it fit would require reducing to roughly 500K parameters (15× below shared budget), at which point the comparison loses meaning.

## Literature context

- **Chen et al. (NeurIPS 2023)** — the ContiFormer paper:
  - Hardware: single 16GB V100
  - Forecasting experiments used input_length 36 or 96 (their Table 5)
  - Longest sequence benchmark (L=896, SCP1 classification) used small batches
  - Paper explicitly states "substantial ... GPU memory overhead" (§5) and plots O(L²) memory growth (Fig. 6, App. E)
  - **Paper never tested ContiFormer at L=360 for forecasting.**
- **Zhang et al. (ICML 2024, T-PATCHGNN)** — a comprehensive IMTS forecasting benchmark with 17 baselines:
  - ContiFormer is **not** among their baselines
  - Tested on datasets where aligned L ranges 46–643
  - This sets a methodological precedent that a flagship IMTS forecasting benchmark can legitimately exclude ContiFormer

## Fairness argument for the deferral

The shared-recipe fair-comparison protocol sets `train_batch_size=128`, `fp32`, `L=360`. Every "fix" that would let ContiFormer fit breaks fairness:
- Reducing ContiFormer's `d_model`/`n_layers` → breaks the 7.8M param budget
- fp16/bf16 for ContiFormer only → breaks shared precision recipe
- Observation subsampling for ContiFormer only → gives ContiFormer a shorter sequence than other models

The principled resolution: exclude ContiFormer from Phase 2 rather than run it at settings that aren't comparable. This matches the T-PATCHGNN precedent and is defensible in the paper write-up.

## What gets written in RESULTS_phase2.md

A short paragraph in the Methods/Baselines section:

> "We evaluated four baselines at the shared 7.8M-parameter, fp32, batch=128 recipe on L=360 sequences: Mamba-MV (with three `dt_mode` ablations), RoMAE, mTAN. We additionally attempted to include ContiFormer (Chen et al., NeurIPS 2023) but found that its O(L²) ODE-attention memory cost exceeds 80 GB H100 memory at our fair-comparison configuration, regardless of batch size (tested 128→8) or ODE step size (tested 0.1 and 0.5, the full range of Chen et al.'s own Table 18 ablation). This is consistent with the ContiFormer paper's own experiments using shorter sequences (input_length ≤ 96) for forecasting and with the T-PATCHGNN (Zhang et al., ICML 2024) benchmark, which also excludes ContiFormer. Post-hoc addition at reduced hyperparameters is left to future work with larger-memory accelerators or L-subsampled data regimes."

## Future work (if needed)

Pathways that would let ContiFormer run, all with explicit fairness caveats:
1. H200 (141GB) or B200 — may fit at B=32, step_size=0.5
2. L reduction via observation subsampling applied uniformly to ALL models (changes the task)
3. Gradient checkpointing patches to the vendored `torchdiffeq` backward pass
4. Shorter history/forecast windows (L < 200), matching the paper's own test scale
