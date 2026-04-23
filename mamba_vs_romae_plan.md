# Fair Comparison: Multivariate Mamba vs RoMAE on Irregular Synthetic Forecasting

## Context

The user has a Mamba-based multivariate forecaster described in a paper draft (irregular-step SSM + shared-grid alignment + variable-axis attention + temporal Mamba + query-time readout) and a synthetic dataset generator at [src/train/moirai/uni2ts_hongyu/ssm_dk/generate_longgap_multisin.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/generate_longgap_multisin.py) that produces multivariate (K=3) irregular time series with long gaps, in two regimes:

- `sparse_independent` — negative control (no cross-variate dependence)
- `sparse_dependent` — positive control (shared latent sinusoid drives all variates, so the other 2 variates carry useful info across the gap of the 3rd)

Goal: show Mamba-based architecture outperforms the SOTA transformer baseline **RoMAE** (https://github.com/Chromeilion/RoMAE — transformer MAE with RoPEND on raw-float timestamps) on irregular multivariate forecasting, especially where cross-variate reasoning is needed. The comparison must be **fair**: identical data, identical task, matched parameter counts and training compute, identical optimizer/schedule, multi-seed evaluation.

**User decisions (confirmed):**
- Implement the full paper architecture (Mamba MV with variable-axis attention + temporal Mamba), not a per-variate stub
- Dataset scale: 1000 train / 200 val / 200 test per regime
- Prediction target: all K=3 variates at `t >= history (7.0)`
- RoMAE in a separate conda env (Python ≥3.11)
- Shared internal grid size K = 128 (swept in ablation as a secondary concern)

**Current code that this plan changes or reuses:**
- [mamba_forecaster.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mamba_forecaster.py) — existing **univariate** PL module (reuse optimizer/schedule, replace forward)
- [mamba_block.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mamba_block.py) — reuse `MambaIrregularBlock` for the per-variate backbone; reuse `MambaBlock` (learned Δ) for temporal-on-grid layer
- [sinusoidal_datamodule.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/sinusoidal_datamodule.py) — **bug: ignores `n_obs_per_var`**, recomputes `delta_t` across variate boundaries incorrectly. Must be replaced for multivariate.

## Fairness Protocol

| Axis | How it's controlled |
|---|---|
| Data | Regenerate both regimes at `1000/200/200` with fixed seeds. Normalization stats from train+val only. Both models read the **same** HuggingFace arrow splits. |
| Task | Forecast at positions with `timestamps >= history (7.0)`. Loss = MSE on those positions only. Identical for both models. |
| Per-variate slicing | Shared `MultivariateSinusoidalDataModule` parses `target`/`timestamp`/`past_feat_dynamic_real` via `np.split(..., np.cumsum(n_obs_per_var)[:-1])`. Both trainers import it. |
| Optimizer/schedule | AdamW, `lr=5e-4`, `weight_decay=0.01`, warmup=100 steps, cosine decay, grad clip 1.0, `max_epochs=200`, `patience=20`, `train_batch_size=128`, `val_batch_size=32`. Centralized in `shared_config/fair_defaults.py`. |
| Parameter budget | Target ~8M trainable for both. Mamba MV: `d_model=384, n_layer_perv=3, n_layer_temporal=3, varattn_layers=3, d_state=16`. RoMAE: sweep `d_model∈{288,384}, depth∈{6,8}, nhead=6` until within ±5% of Mamba count. Log `sum(p.numel() for p in m.parameters() if p.requires_grad)`. |
| Compute | H100 via SLURM, `bf16-mixed`, single GPU, `num_workers=2`, same 4h walltime. Log wall-clock and peak memory. |
| Seeds | `{1,2,3,4,5}` per (model × regime). Mamba also varies `dt_mode∈{learned,replace,additive}` as intra-model ablation. |
| Metrics | MSE, MAE, R², Pearson, per-sample then averaged — identical to the existing `_test_agg` aggregation. Also compute **in-gap MSE** (positions inside `_gap_intervals`) vs **rest-of-forecast MSE** separately. |

## Implementation

### Step 1 — Regenerate data at scale

Run for both regimes with `gap_variates_per_sample=1` (one variate blanked per sample → cross-variate reasoning is decisive):

```bash
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk
python generate_longgap_multisin.py --regime sparse_independent --n_train 1000 --n_val 200 --n_test 200
python generate_longgap_multisin.py --regime sparse_dependent --n_train 1000 --n_val 200 --n_test 200
```

Output: `ssm_dk/data/{sparse_independent,sparse_dependent}/{train,val,test}/` + `norm_stats.json`.

### Step 2 — Shared multivariate data module

New file: [ssm_model/mamba_forecaster/mv/multivariate_datamodule.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/multivariate_datamodule.py)

Per-sample output dict:
```python
{
  "values_per_var":    list of V tensors [N_d]          # variable-length per variate
  "timestamps_per_var": list of V tensors [N_d]
  "deltat_per_var":     list of V tensors [N_d]         # from past_feat_dynamic_real
  "n_obs_per_var":      tensor [V]
  "history":            scalar
}
```

Collate: batch with variable `N_d` via length-sorted padding, emit per-variate `attention_mask`. Provides two export modes: `"per_variate"` (Mamba MV) and `"flat_tokens"` (RoMAE — concatenated `(value, timestamp, variate_id, mask)` tokens). Prediction mask = `timestamps >= history` on every variate.

### Step 3 — Multivariate Mamba model

New subpackage: `ssm_model/mamba_forecaster/mv/`

- [irregular_ssm.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/irregular_ssm.py) — stacks `MambaIrregularBlock` (reuse existing, dt_mode selectable) for per-variate encoding. `forward(x_d: [B,N_d,D], dt_d: [B,N_d]) -> h_d: [B,N_d,D]`.

- [shared_grid.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/shared_grid.py) — builds uniform grid `s_1=0 < ... < s_K=t_max` with **K=128**. For each `(b, d, k)` computes `π_d(k) = max{i : t_i^{(d)} ≤ s_k}` and `avail_mask_{b,d,k} = 1[π_d(k) exists]`. Aligns latent: `z_{b,d,k} = exp((s_k - t_π)·A_d) · h_{b,d,π}` when available, learned null-state `h_∅^{(d)}` otherwise. Staleness `ρ_{b,d,k} = s_k - t_π` (or `s_k - t_0` when unavailable). Output tensor `H^{(0)}: [B,K,V,D]`.

- [variable_axis_attention.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/variable_axis_attention.py) — batch→(B·K), attend over V variates. Per-slot QKV includes additive embeddings for variate id `v^{(n)}`, availability `m`, and sinusoidal staleness `φ(ρ)`. Availability bias `log(m)` on keys/values so unavailable variates can query but not provide evidence. Pre-norm + residual + LayerNorm.

- [temporal_mamba.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/temporal_mamba.py) — for each variate, regular-grid `MambaBlock` (dt_mode=learned since grid is uniform) along K. Reuses `δ_k = s_k − s_{k−1}` as additive input to the learned Δ via an `additive` block.

- [query_readout.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/query_readout.py) — for each variate `n` and query time `a_j` with `κ(j)=max{k: s_k≤a_j}`: propagate latent by one extra SSM step with `Δ = a_j − s_{κ(j)}`, concat `φ(a_j − s_{κ(j)})`, per-variate linear head → prediction. Batched by building the `(batch, query_idx)` gather.

- [multivariate_forecaster.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/multivariate_forecaster.py) — PL module orchestrating `per_var_ssm → shared_grid_align → [VarAttn + TemporalMamba] × L → query_readout`. Copies the existing training_step / validation_step / test_step / configure_optimizers from [mamba_forecaster.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mamba_forecaster.py) verbatim for fairness; only the forward and the batch-unpack differ.

### Step 4 — RoMAE forecasting wrapper

New subpackage: `ssm_model/romae_forecaster/`

- Vendor RoMAE into `ssm_model/romae_forecaster/_romae/` (git clone the repo, copy `romae/` subfolder).
- Create conda env: `/projects/b1094/StarEmbed/pythonenvs/romae` (Python 3.11, torch ≥2.2, `pip install einops pydantic-settings accelerate safetensors wandb pytorch-lightning datasets`, then `pip install -e ssm_model/romae_forecaster/_romae`).
- [romae_forecaster.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/romae_forecaster/romae_forecaster.py) — PL module wrapping `RoMAEForPreTraining`:
  - Tokenization: one token per observation. Value → `[B, N_total, 1]`. Positions `[B, 2, N_total]` = `[timestamp, variate_index]` (both float). `pad_mask` marks padding.
  - **Forecast mask** (replaces MAE random mask): `mask = (timestamps >= history)`. Encoder sees only `~mask` tokens (plus decoder mask tokens in their place). Reuse RoMAE's `set_loss_fn(nn.MSELoss(reduction='none'))` and gather loss on `mask` positions only.
  - Config at ~8M params: `encoder_config__d_model=288, encoder_config__nhead=6, encoder_config__depth=8, tubelet_size=(1,1,1)` (no patchify — each observation is a token), `n_pos_dims=2`.
  - Same optimizer/scheduler/early-stopping as Mamba via `shared_config/fair_defaults.py`.
- [romae_datamodule.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/romae_forecaster/romae_datamodule.py) — subclasses the shared MV datamodule, uses `format="flat_tokens"`.

### Step 5 — Training entry points

Both scripts import shared defaults from `ssm_model/shared_config/fair_defaults.py`:

- [mv/train_mv.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/mv/train_mv.py) — `--dt_mode {learned,replace,additive} --regime {sparse_independent,sparse_dependent} --seed <int> --grid_K 128`. Mirror [train.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/mamba_forecaster/train.py) structure (callbacks, CSV output, `_test_agg` persistence).
- [romae_forecaster/train_romae.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/romae_forecaster/train_romae.py) — `--regime --seed`. Same CSV schema so downstream aggregation is trivial.

### Step 6 — Evaluation harness

[ssm_model/eval/aggregate_results.py](/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/eval/aggregate_results.py)

- Walks `output/log/mvcompare_v1/<model>/<regime>/<variant>/seed<S>/test_metrics.csv`.
- Emits long-format table `(model, regime, variant, seed, metric, in_gap, value)`.
- Per-cell stats: mean ± std over 5 seeds.
- **Significance**: paired per-sample Wilcoxon signed-rank on test-sample MSE (Mamba vs RoMAE, matched by `item_id`), reported per regime. Also paired bootstrap 10k resamples → 95% CI on MSE difference.
- Headline comparison: in-gap MSE on `sparse_dependent` (cross-variate hypothesis test). Negative control: in-gap MSE on `sparse_independent` (both models similar).

### Step 7 — SLURM layout

Template from existing [run_mamba_sinusoidal.sh](/projects/b1094/StarEmbed/skai_universal_forecaster/bash_script/train/run_mamba_cuda_fixed.sh).

- [scripts/mv/run_mamba_mv.sbatch](/projects/b1094/StarEmbed/skai_universal_forecaster/bash_script/train/mv/run_mamba_mv.sbatch) — `--array=1-30`, indexes `(regime, dt_mode, seed) ∈ {2}×{3}×{5}`, uses env `pythonenvs/mamba`.
- [scripts/romae/run_romae.sbatch](/projects/b1094/StarEmbed/skai_universal_forecaster/bash_script/train/romae/run_romae.sbatch) — `--array=1-10`, indexes `(regime, seed) ∈ {2}×{5}`, uses env `pythonenvs/romae`.
- Both: `--account=p32626 --partition=gengpu --gres=gpu:h100:1 --time=4:00:00 --mem=64G --cpus-per-task=16`. Logs to `output/log/mvcompare_v1/<model>/<regime>/<variant>/seed<S>/`.
- Aggregation: CPU job with `dependency=afterany:<mamba_array_id>:<romae_array_id>`.

## Ablations

| Ablation | Grid |
|---|---|
| Mamba `dt_mode` | `{learned, replace, additive}` × 2 regimes × 5 seeds — primary |
| Regime | `{sparse_independent, sparse_dependent}` — primary |
| Parameter-count Pareto | Mamba `{4M, 8M, 16M}` vs RoMAE `{4M, 8M, 16M}` — secondary |
| Gap length | regenerate with `gap_len_range ∈ {(1.0,1.5), (1.5,3.0), (2.5,3.5)}` — secondary |
| Grid K | `{64, 128, 256}` Mamba-MV only — secondary |

## Verification

1. **Sanity**: run Mamba MV on 20/5/5 with 10 epochs → training loss must decrease, no shape errors, `preds` identical shape to `values_per_var` on prediction positions. Same for RoMAE wrapper.
2. **Parameter parity**: both model configs print `Total trainable params: X` within ±5% of 8M before training.
3. **Data parity**: `item_id` order and prediction-mask positions match byte-for-byte across Mamba and RoMAE trainers (add assertion at DataLoader boundary).
4. **Optimizer parity**: log `optimizer.state_dict()['param_groups']` first 5 keys from both trainers — must match (lr, weight_decay, betas).
5. **End-to-end**: run both on `sparse_dependent` seed=1 for 30 epochs. Confirm test_metrics.csv is emitted. Eyeball in-gap MSE is meaningful (not trivial 0 or NaN).
6. **Full experiment**: submit the SLURM arrays. Aggregate. Check that `sparse_independent` → Mamba ≈ RoMAE (no cross-variate info advantage), `sparse_dependent` → Mamba < RoMAE on in-gap MSE (the hypothesis).
