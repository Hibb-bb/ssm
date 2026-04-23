# IMTS Benchmark — Multivariate Mamba vs SSM / Transformer Baselines

A controlled benchmark for **irregular multivariate time-series (IMTS)
forecasting** on synthetic sinusoidal data. The central question:

> When variates are sampled asynchronously (and/or have long missing gaps),
> does a true-Δt-discretized Mamba model (Mamba-MV) forecast more accurately
> than strong SSM / transformer baselines?

The benchmark grows across four data phases, each testing one additional
axis of difficulty while keeping the training recipe fixed:

| Phase | Generator | Output dir | What's new |
|---|---|---|---|
| 2 | `generate_multivariate_sinusodial_data.py` | `data_correct/` | dense, sync timestamps across variates |
| 3 | `generate_multivariate_sinusodial_data_async.py` | `data_correct_async/` | per-variate independent timestamps (async dense, no gaps) |
| 4-1 | `generate_multivariate_sinusodial_data_gap.py --gap_variate_mode fixed_v3` | `data_correct_gap/` | one long gap, always on variate v3 (forward-mixture test) |
| 4-2 | `generate_multivariate_sinusodial_data_gap.py --gap_variate_mode random_uniform` | `data_correct_gap_random/` | gap on a variate drawn uniformly from {v0, v1, v2} |

Every phase keeps the same signal/convex-mix construction, the same forecast
horizon (`history=8.0`, `t_max=10.0`), and the same four irregularity regimes
(see below), so phase-to-phase deltas isolate exactly one factor.

---

## Signal model (same for every phase)

Per sample, `V=3` variates, `N_OBS=120` observations each, on `[0, t_max=10]`:

```
v0(t) = sin(2π f0 t + φ0) + noise
v1(t) = sin(2π f1 t + φ1) + noise
v3(t) = w · v0(t) + (1 − w) · v1(t) + noise          # convex mixture
                                                      # w ~ U(0.2, 0.8)
```

`v3` is algebraically a weighted sum of the other two — **cross-variate
information is load-bearing**, so a model that ignores the other variates
(or can't align them across their asynchronous timestamps) pays a
measurable error.

### Irregularity regimes (four, by `frac_regular`)

Timestamps are produced by `generate_timestamps(n_obs, t_max, frac_regular)`
which interpolates between a uniform grid and a fully-irregular sample:

| Regime | `frac_regular` | Interpretation |
|---|---|---|
| `multisin_regular`    | 1.0 | all observation times on the regular grid |
| `multisin_low_irreg`  | 0.8 | 80% regular, 20% jittered |
| `multisin_med_irreg`  | 0.3 | 30% regular, 70% jittered (the "main" setting) |
| `multisin_high_irreg` | 0.0 | fully irregular |

The same regime names are reused across all four phases; directories live
under `data_correct*/multisin_{regime}/{train,val,test}/`.

### Phase-4 gap injection

On top of Phase 3, `generate_multivariate_sinusodial_data_gap.py` draws a
single forbidden interval (`gap_start ∈ [2.5, 6.0]`, `gap_len ∈ [1.5, 3.0]`)
and drops every observation of the gapped variate(s) inside it. Each
sample stores the gap location in `gap_starts`, `gap_ends`, and the gapped
variate index in `gapped_variate_index`, which the aggregator uses to slice
per-sample metrics into `in_gap` vs `out_gap` regions.

Two variants test opposite directions of the mixture:

- **Phase 4-1 (`fixed_v3`)** — the gapped variate is always `v3`. Model
  must infer the sum from the two parts (v0, v1 visible across the gap).
- **Phase 4-2 (`random_uniform`)** — the gapped variate is `v0`, `v1`, or
  `v3` with equal probability. When `v0` or `v1` is gapped, the model must
  invert the mixture using `v3` and the other part.

### Regenerating data

```bash
# Phase 2 (sync dense)
python generate_multivariate_sinusodial_data.py

# Phase 3 (async dense)
python generate_multivariate_sinusodial_data_async.py

# Phase 4-1 (gap on v3)
python generate_multivariate_sinusodial_data_gap.py \
    --gap_variate_mode fixed_v3 \
    --output_root ./data_correct_gap

# Phase 4-2 (gap on random variate)
python generate_multivariate_sinusodial_data_gap.py \
    --gap_variate_mode random_uniform \
    --output_root ./data_correct_gap_random
```

All generators use fixed seeds → byte-reproducible outputs. Defaults:
`n_train=1000`, `n_val=200`, `n_test=200`. Full design notes in
[`BUILD_NOTES.md`](BUILD_NOTES.md).

### Storage schema (HuggingFace Arrow)

| field | type | note |
|---|---|---|
| `item_id` | string | `sin_00042` etc. |
| `target` | list[float] | per-variate values concatenated |
| `timestamp` | list[float] | per-variate timestamps concatenated |
| `past_feat_dynamic_real` | list[float] | per-variate Δt (first Δt per variate = 0) |
| `n_obs_per_var` | list[int] | length V = 3 |
| `history` | float | 8.0 for every sample |
| `gap_starts` / `gap_ends` / `gapped_variate_index` | lists[float/int] | Phase 4 only (empty lists for Phase 2/3) |

---

## Models

Three baselines at time of this commit (mTAN dropped — persistent
collapse across all sizing/LR configurations and timestamp normalizations):

| Model | File | Config | Notes |
|---|---|---|---|
| **Mamba-MV** | [`imts_benchmark/mamba_mv/multivariate_forecaster.py`](imts_benchmark/mamba_mv/multivariate_forecaster.py) | `d_model=256`, `dt_mode ∈ {learned, replace}` | `additive` dropped; true-Δt discretization is the paper's scientific claim |
| **S5** | [`imts_benchmark/s5_forecaster/s5_forecaster.py`](imts_benchmark/s5_forecaster/s5_forecaster.py) | paper-native (`d_model=128`, `state_dim=256`, `lr=1e-3`, `wd=0.05`, `batch=32`×accum=4) | SSM-spectral params (`Lambda`, `log_step`, `D`, `B`, `C`) excluded from weight decay |
| **RoMAE** | [`imts_benchmark/romae_forecaster/romae_forecaster.py`](imts_benchmark/romae_forecaster/romae_forecaster.py) | paper-native Rotary-PE masked autoencoder | 2D positions `(timestamp, variate_id)` as two float RoPEND coords |

Forecast objective: MSE on `pred_mask = valid & (timestamp ≥ history)`
for every model. Metrics reported target-weighted (`Σss_res / Σcount`) +
per-variate breakdowns; for Phase 4, additionally stratified
`in_gap` / `out_gap` for each variate `v ∈ {0, 1, 2}`.

Shared training knobs live in
[`imts_benchmark/shared_config/fair_defaults.py`](imts_benchmark/shared_config/fair_defaults.py):
`AdamW lr=5e-4, wd=0.01` (baseline; S5 overrides), warmup=100 steps,
cosine to 1600 steps, `max_epochs=200`, `patience=50`, `batch=128`,
seeds `{1..5}`.

---

## Running an experiment

Each phase has its own sbatch trio. Example for Phase 3:

```bash
cd imts_benchmark/scripts
sbatch run_mamba_mv_phase3.sbatch    # 4 regimes × 2 dt_modes × 5 seeds = 40 runs
sbatch run_s5_phase3.sbatch          # 4 × 5 = 20 runs
sbatch run_romae_phase3.sbatch       # 4 × 5 = 20 runs
```

Replace `phase3` with `phase4_1` / `phase4_2` for the gap phases. Logs
land under `output/log/imts_benchmark_v2/phase{3,4_1,4_2}/{model}/{regime}/{dt_mode}/seed{s}/`
with `test_metrics.csv` and `per_sample.jsonl`.

After all jobs finish, aggregate:

```bash
python imts_benchmark/eval/aggregate_results.py \
    --log_root /path/to/output/log/imts_benchmark_v2/phase3
```

This produces wide/long CSVs per regime, paired Wilcoxon (Mamba-MV best
vs each baseline), and the per-variate in-gap / out-gap breakdown for
Phase 4.

### Weights & Biases (optional)

All fair trainers accept `--use_wandb` plus optional `--wandb_project`,
`--wandb_entity`, and `--wandb_run_name`. Install `wandb` in the training
environment, export `WANDB_API_KEY`, and for Slurm arrays pick a run name
that matches the job grid (same variables as in e.g.
[`run_mamba_mv_phase4_1.sbatch`](imts_benchmark/scripts/run_mamba_mv_phase4_1.sbatch)):
`--wandb_run_name "${REGIME}_${DT_MODE}_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"`.
Baseline Slurm scripts have no `DT_MODE`; use `${REGIME}_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}` or add your own variant tag. Offline runs:
set `WANDB_MODE=offline` and sync later with `wandb sync`. See
[`imts_benchmark/shared_config/fair_defaults.py`](imts_benchmark/shared_config/fair_defaults.py)
module docstring for the same notes.

---

## Environment

- Python 3.12 conda env at `/projects/b1094/StarEmbed/pythonenvs/mamba`
  (PyTorch 2.5 + CUDA 12.4, `mamba-ssm` built from source with the CUDA
  selective-scan kernel). Mamba / RoMAE run from this env. Add
  `pip install wandb` if you use `--use_wandb`.
- JAX/Flax env at `/projects/b1094/StarEmbed/pythonenvs/s5-jax` for S5.
  Built via [`imts_benchmark/scripts/build_s5_env.sh`](imts_benchmark/scripts/build_s5_env.sh).

Vendored upstream baseline code lives under `imts_benchmark/*/_upstream/`
and is read-only.

---

## Documentation index

- [`imts_benchmark/docs/REPORT_consolidated_2026-04-22.md`](imts_benchmark/docs/REPORT_consolidated_2026-04-22.md)
  — full build report and diagnosis log (supersedes earlier BUILD_REPORT*)
- [`imts_benchmark/docs/RESULTS_phase2.md`](imts_benchmark/docs/RESULTS_phase2.md)
  — Phase 2 (sync-dense) results across the four regimes
- [`imts_benchmark/docs/HPO_PLAN.md`](imts_benchmark/docs/HPO_PLAN.md)
  — rigorous HPO design, gated on Phase 3/4 Mamba vs S5 outcome
- [`BUILD_NOTES.md`](BUILD_NOTES.md) — synthetic-data design notes
- [`imts_benchmark_plan.md`](imts_benchmark_plan.md) — running benchmark plan
- [`mamba_vs_romae_plan.md`](mamba_vs_romae_plan.md) — original 2-model plan (superseded, kept for history)
- [`synthetic_sin_plan.md`](synthetic_sin_plan.md) — original dataset plan (superseded by Phase 2+)

Files prefixed `_` (`_smoke_*.py`, `_smoke_plot.py`) are design-time
smoke checks, not part of the benchmark itself.



# WANDB Command

```bash
srun ${MAMBA_ENV}/bin/python -m imts_benchmark.mamba_mv.train_mv \
    --regime $REGIME \
    --dt_mode $DT_MODE \
    --seed $SEED \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --use_wandb \
    --wandb_project TSKing \
    --wandb_run_name "${REGIME}_${DT_MODE}_seed${SEED}"
```