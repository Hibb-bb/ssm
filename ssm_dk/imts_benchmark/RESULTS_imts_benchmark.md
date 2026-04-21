# Results — Multivariate IMTS Forecasting Benchmark (5 models × 4 regimes)

Run tag: `imts_benchmark_v1`. Status: **stub — populated as Phases 4–5 complete.**

## Protocol

- **Forecast window**: `t ∈ [history=8.0, t_max=10.0]`. Forecast loss = MSE over positions where `timestamps >= history`.
- **Same-recipe fairness** (no HPO): all 5 models share `lr=5e-4`, `patience=20`, `max_epochs=200`, `AdamW wd=0.01`, warmup=100 linear, cosine to 1600 steps, batch 128 / 32, fp32, grad clip 1.0. See [`shared_config/fair_defaults.py`](shared_config/fair_defaults.py).
- **Param budget**: each model sized to ±10% of 7.8M trainable.
- **Seeds**: {1, 2, 3, 4, 5}.
- **Regimes**: `sparse_independent`, `sparse_dependent`, `nogap_independent`, `nogap_dependent`.

## Models

| Model | Variants | Backend | Source |
|---|---|---|---|
| Mamba-MV | `dt_mode ∈ {learned, replace, additive}` | PyTorch | [`mamba_mv/`](mamba_mv/) |
| RoMAE | default | PyTorch | [`romae_forecaster/`](romae_forecaster/) |
| mTAN | default | PyTorch | `mtan_forecaster/` (Phase 3a — pending) |
| ContiFormer | default | PyTorch | `contiformer_forecaster/` (Phase 3b — pending) |
| S5 | default | JAX (separate env) | `s5_forecaster/` (Phase 3c — pending) |

## Run count

5 models × 4 regimes × 5 seeds = 100 runs (Mamba-MV adds 2 extra dt_modes × 4 × 5 = 40, total 140).

## Results

_To be populated by `eval/aggregate_results.py` after Phase 4 completes._

## Predecessor

The frozen 2-model results at `history=7.0` are documented in
[`RESULTS_mamba_vs_romae.md`](RESULTS_mamba_vs_romae.md) (mvcompare_v1 tree
under `output/log/mvcompare_v1/` is untouched and citable).
