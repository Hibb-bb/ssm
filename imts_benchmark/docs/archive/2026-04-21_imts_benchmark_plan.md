# Extend IMTS Benchmark: mTAN, ContiFormer, S5 + No-Gap Dataset Variant + history=8.0

> **On approval**: mirror this plan to `/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/imts_benchmark_plan.md` so it lives alongside `synthetic_sin_plan.md` and `mamba_vs_romae_plan.md` in the project tree.

## Context

The Mamba-MV vs RoMAE benchmark (`mv_vs_romae`, submitted ICLR) is complete at `history=7.0` with one-variate long gaps. To strengthen the paper's claim:

1. **Add three IMTS baselines** — mTAN, ContiFormer, S5 — standard in t-PatchGNN (ICML 2024).
2. **Add a no-gap, async-only dataset variant** — same two regimes, same generator, but with `--gap_variates_per_sample 0 --n_gaps 0`. Isolates the effect of long gaps vs irregular sampling.
3. **Shift forecast window to history=8.0** — aligns with the paper's "context=8 / pred=2" spec. This invalidates reuse of the existing `history=7.0` runs, so all 5 models re-run in a new log tree.
4. **Folder and report rename** — `mv_vs_romae/` → `imts_benchmark/`; `.md` reports renamed so the frozen `history=7.0` results remain citable as `*_mamba_vs_romae.md` while new results live in `*_imts_benchmark.md`.

User-confirmed this session:
- **Same-recipe fairness** (no HPO): all 5 models use the exact existing `mv_vs_romae` training recipe.
- **All 3 Mamba dt_modes** on every regime (`learned`, `replace`, `additive`).
- **history=8.0** for all new runs.
- **S5 in JAX env** (Option 1 from prior discussion — vendor `lindermanlab/S5`, separate JAX trainer emitting same output schema).
- **Data-gen rules unchanged** (shared-latent for dependent, independent for negative control). No convex-mix regime.

## Goal

Unified benchmark table across **5 models × 4 regimes × 5 seeds**, identical training recipe, architectural HPs sized to ~7.8M params per model. Final table shows whether Mamba's advantage persists on no-gap data (testing the "long-gap bridging" vs "general IMTS quality" hypothesis).

---

## Shared training recipe (fixed for all 5 models)

| Knob | Value | Source |
|---|---|---|
| Optimizer | AdamW | `fair_defaults.py:LR=5e-4` |
| lr | 5e-4 | existing |
| weight_decay | 0.01 | existing |
| warmup | 100 linear steps | existing |
| schedule | cosine, 1600 steps | existing |
| max_epochs | 200 | existing |
| patience | 20 | existing |
| gradient_clip | 1.0 | existing |
| batch (train/val) | 128 / 32 | existing |
| precision | fp32 (`32-true`) | existing |
| seeds | {1, 2, 3, 4, 5} | existing |
| **history** | **8.0** (new; was 7.0) | user decision |
| t_max | 10.0 | existing |
| n_vars | 3 | existing |

No per-model overrides. If any baseline collapses (R² ≤ 0 on `sparse_dependent` seed=1), **report honestly — no escape valve**. The paper narrative is "identical recipe."

---

## Regimes (4 total, stored under `ssm_dk/data/`)

| Regime | Signal | Gaps | Source |
|---|---|---|---|
| `sparse_independent` | per-variate IID sinusoids | 1 variate gapped per sample | existing |
| `sparse_dependent` | shared latent + per-variate lag | 1 variate gapped per sample | existing |
| `nogap_independent` | per-variate IID sinusoids | none | **NEW** Phase 2 |
| `nogap_dependent` | shared latent + per-variate lag | none | **NEW** Phase 2 |

---

## Run count

| Model | Variants | Regimes | Seeds | Runs |
|---|---|---|---|---|
| Mamba-MV | 3 dt_modes | 4 | 5 | 60 |
| RoMAE | 1 | 4 | 5 | 20 |
| mTAN | 1 | 4 | 5 | 20 |
| ContiFormer | 1 | 4 | 5 | 20 |
| S5 | 1 | 4 | 5 | 20 |
| **Total** | | | | **140** |

All under `output/log/imts_benchmark_v1/<model>/<regime>/<variant>/seed<S>/`. The existing `mvcompare_v1` tree is untouched and stays as the frozen `history=7.0` precursor citable in the paper.

---

## Implementation

### Phase 1 — Folder + file rename

```
ssm_dk/mv_vs_romae/                           ssm_dk/imts_benchmark/
├── mamba_mv/                                 ├── mamba_mv/                       (unchanged code)
├── romae_forecaster/                         ├── romae_forecaster/               (unchanged code)
├── shared_data/               →              ├── shared_data/                    (unchanged)
├── shared_config/                            ├── shared_config/                  (HISTORY=8.0, BENCHMARK_TAG bumped)
├── eval/                                     ├── eval/                          (model+regime lists extended)
├── RESULTS.md                                ├── RESULTS_mamba_vs_romae.md       (frozen history=7.0 result)
└── BUILD_REPORT.md                           ├── RESULTS_imts_benchmark.md       (new, Phase 5 populates)
                                              ├── BUILD_REPORT_mamba_vs_romae.md  (frozen)
                                              ├── BUILD_REPORT_imts_benchmark.md  (new, built incrementally)
                                              ├── mtan_forecaster/                (NEW Phase 3)
                                              ├── contiformer_forecaster/         (NEW Phase 3)
                                              └── s5_forecaster/                  (NEW Phase 3, JAX env)
```

Steps:
1. `grep -rn "mv_vs_romae" ssm_dk/ bash_script/` → list all references.
2. Move directory; update all imports + slurm scripts.
3. In `shared_config/fair_defaults.py`: change `HISTORY = 7.0 → 8.0`; change `LOG_ROOT_DEFAULT` to `output/log/imts_benchmark_v1`; add `nogap_independent`, `nogap_dependent` to `REGIMES`; leave all other knobs unchanged.
4. Rename `RESULTS.md → RESULTS_mamba_vs_romae.md` and `BUILD_REPORT.md → BUILD_REPORT_mamba_vs_romae.md` (freezes the history=7.0 work).
5. Create stubs `RESULTS_imts_benchmark.md`, `BUILD_REPORT_imts_benchmark.md` with a header noting "history=8.0, patience=20, same-recipe protocol, 5 models × 4 regimes × 5 seeds."
6. Smoke-run one Mamba seed (dt_mode=learned, regime=sparse_dependent) on the renamed tree → confirms imports resolve and new log path populates.

### Phase 2 — No-gap dataset variant

```bash
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

python generate_longgap_multisin.py --regime sparse_independent \
    --n_train 1000 --n_val 200 --n_test 200 \
    --gap_variates_per_sample 0 --n_gaps 0 \
    --out_subdir nogap_independent

python generate_longgap_multisin.py --regime sparse_dependent \
    --n_train 1000 --n_val 200 --n_test 200 \
    --gap_variates_per_sample 0 --n_gaps 0 \
    --out_subdir nogap_dependent
```

- Read `generate_longgap_multisin.py` first. If `--gap_variates_per_sample 0` doesn't short-circuit `apply_forbidden_mask`, add a 2-line guard in `build_sample`.
- Add `--out_subdir` CLI flag if missing so the two variants coexist.
- Verify with `_smoke_plot.py` on one sample of each new regime: dense async irregular sampling, zero forbidden intervals.
- `norm_stats.json` records `n_gaps=0`; normalization stats are computed per-regime (don't share with sparse_* regimes).

**Note on history=8.0 and existing long-gap data**: gap generator uses `gap_start ∈ [2.5, 6.0]`, `gap_len ∈ [1.5, 3.0]`, so gaps can end as late as t≈9.0. With history=8, ~half the samples will have gaps that partially overlap the forecast window `[8, 10]`. This is acceptable — cross-variate reasoning across partial gap-overlap is still the test — but document it clearly in `BUILD_REPORT_imts_benchmark.md`.

### Phase 3 — Baseline model adapters

Pattern for each new baseline (mirror of `romae_forecaster/`):
```
<model>_forecaster/
├── _upstream/                    # vendored upstream code + LICENSE
├── <model>_forecaster.py         # PL module (PyTorch) OR JAX trainer (S5)
├── <model>_datamodule.py         # thin wrapper on MultivariateSinusoidalDataModule
└── train_<model>.py              # entry point, imports add_fair_args()
```

Architectural HPs sized to **±10% of 7.8M** trainable params (matching Mamba-MV 7.79M / RoMAE 7.80M) — not tuned. Log param count at train start.

All adapters emit identical `test_metrics.csv` + `per_sample.jsonl` schema → `aggregate_results.py` stays model-agnostic.

#### Phase 3a — mTAN (`mtan_forecaster/`) — PyTorch
- Upstream: `github.com/reml-lab/mTAN`, MIT. Vendor `src/models.py` with LICENSE.
- Use `enc_mtan_rnn` + a forecast head. Input format: `flat_tokens` from shared datamodule.
- Starting config (paper default): `rec-hidden=64`, `latent-dim=16`, `num-ref-points=64`. Scale `rec-hidden` and `num-ref-points` upward to hit ~7.8M params. Confirm final count before training.

#### Phase 3b — ContiFormer (`contiformer_forecaster/`) — PyTorch
- Upstream: `github.com/microsoft/SeqML/tree/main/ContiFormer`, MIT. Vendor the `ContiFormer/` subfolder.
- Use the irregular-TS regression wrapper. Input format: `flat_tokens`.
- Scale `d_model` / `depth` to ~7.8M params.

#### Phase 3c — S5 (`s5_forecaster/`) — JAX
- Upstream: `github.com/lindermanlab/S5`, `pendulum` branch (irregular-TS variant). Vendor with LICENSE.
- New env: `/projects/b1094/StarEmbed/pythonenvs/s5-jax` — `jax[cuda12]`, `flax`, `optax`, `datasets` (for HF Arrow reading), `numpy`, `pyyaml`.
- Write a JAX trainer (no PL) that:
  - Reads HF Arrow via `datasets.load_from_disk`.
  - Uses same `(train+val)` normalization stats from `norm_stats.json`.
  - Applies `pred_mask = timestamps >= 8.0`.
  - Uses AdamW (optax), lr=5e-4, cosine schedule, warmup=100, max_epochs=200, patience=20 — matches the PyTorch recipe exactly.
  - Emits `test_metrics.csv` + `per_sample.jsonl` in identical schema.
- Size via `d_model`, `n_layers`, `d_state` to ~7.8M params.

### Phase 4 — Training entry points + SLURM

- Each `train_<model>.py` calls `add_fair_args()` from `shared_config/fair_defaults.py`, adds its own architectural knobs, emits the shared schema.
- **SLURM**: one `.sbatch` per model under `bash_script/train/imts_benchmark/`:
  - `run_mamba_imts.sbatch`: `--array=1-60` = `(3 dt_modes) × (4 regimes) × (5 seeds)`.
  - `run_romae_imts.sbatch`: `--array=1-20` = `(4 regimes) × (5 seeds)`.
  - `run_mtan_imts.sbatch`, `run_contiformer_imts.sbatch`: same `--array=1-20` as RoMAE.
  - `run_s5_imts.sbatch`: `--array=1-20`, uses `s5-jax` env.
- Accounting: `p33049 --partition=gengpu --gres=gpu:h100:1 --time=4:00:00 --mem=64G --cpus-per-task=16`.
- Aggregation: one CPU job with `dependency=afterany:<each_array_id>`.

### Phase 5 — Evaluation extension

`eval/aggregate_results.py`:
- Extend model enumeration to `{mamba_mv, romae, mtan, contiformer, s5}`.
- Extend regime enumeration to 4 regimes.
- Emit `summary.csv`, `long.csv`. Paired Wilcoxon (Mamba learned vs each baseline) per regime.

`eval/make_plots.py`, `plot_comparison_v2.py`, `plot_training_curves.py`, `plot_forecast_examples.py`:
- Extend color map to 5 models.
- **New plot**: `plot_gap_vs_nogap_delta.py` — per model, compare `(sparse_dependent MSE) − (nogap_dependent MSE)` to measure how much of each model's performance is explained by gap-bridging specifically.

`RESULTS_imts_benchmark.md`:
- Headline 5×4 table with mean ± std across 5 seeds.
- Paired Wilcoxon significance table.
- Gap-vs-no-gap delta table + interpretation.
- Link back to `RESULTS_mamba_vs_romae.md` as the frozen history=7.0 precursor.

---

## Critical files / entry points

| File | Action |
|---|---|
| `ssm_dk/generate_longgap_multisin.py` | Add `--out_subdir`, verify `--n_gaps 0` short-circuits |
| `ssm_dk/imts_benchmark/shared_config/fair_defaults.py` | `HISTORY=8.0`, add nogap regimes, `BENCHMARK_TAG="imts_benchmark_v1"`, update `LOG_ROOT_DEFAULT` |
| `ssm_dk/imts_benchmark/shared_data/multivariate_datamodule.py` | Unchanged (already reads `history` as a parameter) |
| `ssm_dk/imts_benchmark/eval/aggregate_results.py` | Extend model+regime lists |
| NEW `ssm_dk/imts_benchmark/{mtan,contiformer,s5}_forecaster/` | Phase 3 |
| NEW `ssm_dk/imts_benchmark/RESULTS_imts_benchmark.md` | Populated in Phase 5 |
| `ssm_dk/imts_benchmark/RESULTS_mamba_vs_romae.md` | Frozen historical (renamed) |
| NEW `bash_script/train/imts_benchmark/run_*.sbatch` | Phase 4 |

---

## Verification

1. **Rename sanity** (Phase 1): one existing Mamba seed re-runs cleanly in the new tree at `history=8.0`; writes to `imts_benchmark_v1/`; old `mvcompare_v1` untouched.
2. **No-gap data sanity** (Phase 2): `_smoke_plot.py` shows dense async irregular samples with no forbidden intervals.
3. **Per-model smoke run** (Phase 3): each new adapter trains 10 epochs on 20/5/5 subset of `nogap_dependent`. Loss decreases. Param count within ±10% of 7.8M.
4. **End-to-end per model** (Phase 4): one seed, one regime per model, full training. `test_metrics.csv` emitted; MSE/MAE/R²/Pearson in plausible ranges.
5. **Full run** (Phase 4): 140 runs complete, all seeds reach early-stop or max_epochs.
6. **Aggregation** (Phase 5): 5 models × 4 regimes × 5 seeds read into one CSV; plots render; paired Wilcoxon produces p-values.
7. **Final**: `RESULTS_imts_benchmark.md` has 5-model comparison + gap-vs-no-gap analysis; figures reproducible from `eval/`.

---

## Known documented caveats

- **history=8.0 + existing gap range** means ~half the long-gap samples have a gap that overlaps the forecast window `[8, 10]`. Acceptable under the current hypothesis test but should be noted in the paper's data-description section.
- **No escape valve for failing baselines** — if any baseline collapses, it is reported as-is under the same-recipe fairness protocol.
- **Parameter budget** — ±10% of 7.8M per model. mTAN's paper defaults produce far smaller models; scaling up `rec-hidden`/`num-ref-points` to reach budget is not a published configuration and should be validated on a small run before the full array launches.
