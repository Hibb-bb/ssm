# `ssm_dk/` — Synthetic Long-Gap Multivariate Forecasting Experiments

Working directory for the **multivariate IMTS forecasting benchmark**:
the paper's multivariate Mamba architecture (irregular-step SSM → shared-grid
alignment → variable-axis attention → temporal Mamba → query-time readout)
compared against transformer MAE (RoMAE) and three additional IMTS baselines
(mTAN, ContiFormer, S5) on irregular multivariate time series, with and
without long missing gaps.

> **Phase note**: the original 2-model `mv_vs_romae` work (history=7.0,
> Mamba-MV vs RoMAE only) is frozen under
> [`imts_benchmark/RESULTS_mamba_vs_romae.md`](imts_benchmark/RESULTS_mamba_vs_romae.md)
> and [`imts_benchmark/BUILD_REPORT_mamba_vs_romae.md`](imts_benchmark/BUILD_REPORT_mamba_vs_romae.md).
> The current phase extends this to 5 models × 4 regimes (long-gap +
> no-gap) at history=8.0; see
> [`imts_benchmark_plan.md`](imts_benchmark_plan.md) for the full plan,
> and `imts_benchmark/RESULTS_imts_benchmark.md` once results land.

---

## Folder map

```
ssm_dk/
├── README.md                       # you are here
├── generate_longgap_multisin.py    # synthetic data generator
├── BUILD_NOTES.md                  # detailed synthetic-data build notes
├── synthetic_sin_plan.md           # original design doc for the dataset
├── mamba_vs_romae_plan.md          # plan for the model comparison
├── _smoke_plot.py, _smoke_plot.png # 1-sample sanity plot used during design
└── imts_benchmark/                    # model code + experiment scripts
    ├── BUILD_REPORT_mamba_vs_romae.md             # what we built and how (read this first)
    ├── SETUP.md                    # env + RoMAE install + data regeneration
    ├── shared_config/              # fair-comparison knobs (single source of truth)
    ├── shared_data/                # multivariate datamodule (per-variate / flat-tokens)
    ├── mamba_mv/                   # paper's full multivariate Mamba architecture
    ├── romae_forecaster/           # RoMAE adapter (MAE → forecaster)
    ├── eval/                       # result aggregation + paired Wilcoxon
    └── scripts/                    # SLURM sbatch files
```

`data/` and `imts_benchmark/romae_forecaster/_romae_repo/` are **not** committed
(see `.gitignore`). Both are regenerated / installed per the instructions
below.

---

## Synthetic data

The dataset generator is `generate_longgap_multisin.py`. It produces
multivariate (V=3) irregular time series with explicit long missing gaps,
in two regimes:

- **`sparse_independent`** — each variate drawn from its own
  `(freq, amp, phase, noise)`. No shared latent. Negative control for
  cross-variate reasoning.
- **`sparse_dependent`** — all variates driven by a shared latent
  sinusoid `s(t) = A0·sin(2π f0 t + φ0)`. Each variate d observes
  `a_d·s(t − τ_d) + b_d·priv_d(t) + noise`. Positive control — when a
  long gap blanks one variate, the other two carry useful cross-variate
  information across the gap.

Per-sample: V=3 variates, ~120 observations each over `[0, t_max=10]`,
with independent random irregular timestamps per variate. One forbidden
interval of length ~1.5-3.0 is placed in `[2.5, 6.0)`; by default exactly
**one** of the three variates has all its observations inside the gap dropped
(configurable via `--gap_variates_per_sample`). The forecast horizon is set
at training time via `--history` (default `8.0` in the current
`imts_benchmark` runs; the frozen `mv_vs_romae` runs used `7.0`). Each
sample stores the full trajectory on `[0, t_max]`, so `history` can shift
without regenerating data. The no-gap variants set
`--gap_variates_per_sample 0 --n_gaps 0`.

Storage: sparse-flat HuggingFace arrow dataset. Each row has

| field | type | note |
|---|---|---|
| `item_id` | string | `sin_00042` etc. |
| `target` | list[float] | all variates' values concatenated |
| `timestamp` | list[float] | per-variate timestamps, concatenated |
| `past_feat_dynamic_real` | list[float] | per-variate Δt (first per-variate entry = 0) |
| `n_obs_per_var` | list[int] | per-variate observation count, length V |
| `history` | float | scalar, same for every sample (was `7.0` in original generation; current runs override at training time) |

### Regenerating

```bash
cd ssm_dk
python generate_longgap_multisin.py \
    --regime sparse_independent --n_train 1000 --n_val 200 --n_test 200
python generate_longgap_multisin.py \
    --regime sparse_dependent   --n_train 1000 --n_val 200 --n_test 200
```

Outputs `ssm_dk/data/{sparse_independent,sparse_dependent}/{train,val,test}/`
and `norm_stats.json`. Fixed seeds make the generation byte-reproducible
across machines. Full design details in
[`BUILD_NOTES.md`](BUILD_NOTES.md).

---

## Model comparison (`imts_benchmark/`)

See [`imts_benchmark/BUILD_REPORT_mamba_vs_romae.md`](imts_benchmark/BUILD_REPORT_mamba_vs_romae.md) for the
full build report (what was reused from the canonical
`ssm_model/ssm/mamba_experiments/`, what was modified, the pitfalls and
their fixes). See [`imts_benchmark/SETUP.md`](imts_benchmark/SETUP.md) for
installation.

Two models, parameter-matched to **~7.80M trainable**:

- **Mamba-MV** — [`imts_benchmark/mamba_mv/multivariate_forecaster.py`](imts_benchmark/mamba_mv/multivariate_forecaster.py).
  Entry file to understand the architecture top-to-bottom; the five
  helper modules in the same folder are the pieces it calls in order.
  Reuses `MambaBlock` / `MambaIrregularBlock` from
  `ssm_model/ssm/mamba_experiments/forecaster/mamba_block.py`.
- **RoMAE** — [`imts_benchmark/romae_forecaster/romae_forecaster.py`](imts_benchmark/romae_forecaster/romae_forecaster.py).
  Wraps `RoMAEForPreTraining` (from github.com/Chromeilion/RoMAE, pinned
  to commit `480cfaf`) with forecast-style masking and `(timestamp,
  variate_id)` feeding RoPEND as two float coordinate dims.

Both use the same optimizer / schedule / batch size / precision / seeds —
controlled centrally in `imts_benchmark/shared_config/fair_defaults.py`.

### Running the full experiment

```bash
cd ssm_dk/imts_benchmark/scripts
MAMBA_JID=$(sbatch --parsable run_debug_mamba.sbatch)      # fp32 sanity
ROMAE_JID=$(sbatch --parsable run_debug_romae.sbatch)      # fp32 sanity
MAMBA_FULL=$(sbatch --parsable --dependency=afterok:$MAMBA_JID run_mamba_mv.sbatch)
ROMAE_FULL=$(sbatch --parsable --dependency=afterok:$ROMAE_JID run_romae.sbatch)
sbatch --dependency=afterany:${MAMBA_FULL}:${ROMAE_FULL} run_aggregate.sbatch
```

This runs 2 debug jobs (fp32, 2 epochs, ~1 min each) as a safety gate,
then the full 30-task Mamba array × 10-task RoMAE array, then a CPU
aggregator. Results land under
`/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v1/`:
per-seed `test_metrics.csv` + `per_sample.jsonl`, plus top-level
`summary.csv` and paired Wilcoxon significance (Mamba-MV best variant
vs RoMAE) per regime.

---

## Environment

Python 3.12 conda env at `/projects/b1094/StarEmbed/pythonenvs/mamba` on
Quest. PyTorch 2.5 + CUDA 12.4, `mamba-ssm` built from source (CUDA
selective-scan kernel available). Built via
`ssm_model/ssm/mamba_experiments/environment/build_mamba_env.sh`.

RoMAE dependencies (`pydantic-settings`, `accelerate`, `safetensors`,
`nvidia-ml-py`, `wandb`) were added to the same env. Full install recipe
in [`imts_benchmark/SETUP.md`](imts_benchmark/SETUP.md).

---

## Status / quick reference

- **Paper hypothesis**: Mamba-MV beats RoMAE on `sparse_dependent` (long
  gap + cross-variate needed); parity on `sparse_independent` (negative
  control).
- **Fairness**: params matched to within 0.2% (7.79M vs 7.80M); identical
  optimizer (`AdamW lr=5e-4, wd=0.01`), schedule (warmup=100, cosine to
  1600 steps), batch size (128), precision (fp32), seeds ({1..5}),
  metrics (MSE / MAE / R² / Pearson, plus per-variate MSE for the
  gapped-variate breakdown).
- **Headline comparison**: in-gap MSE on `sparse_dependent`, derivable
  from per-sample json by identifying the gapped variate (the one with
  fewer observations in that sample).
