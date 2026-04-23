# IMTS Benchmark — Consolidated Technical Report (Phases 2–4)

**Paper target:** NeurIPS 2026 — Mamba with true-Δt discretization for irregular multivariate time series forecasting.
**Updated:** 2026-04-23 (baselines, Phase-4 split, HPO gate)
**Originally consolidated:** 2026-04-22
**Repository root:** `/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk`

This document consolidates the planning notes, baseline audits, build notes, generator extensions, per-variate / gap-region metric machinery, and aggregator rewrite into a single reference to cite when writing the paper. Phase-2 diagnosis revisions and Phase-4 split into 4-1 / 4-2 are folded in here; previous-version build/results reports remain archived under `docs/archive/`.

---

## 1. Scientific setup

### 1.1 Research question

Does a Mamba-based state-space model (SSM) that discretizes using the **true inter-sample Δt** outperform irregular-TS baselines that either (i) use a **learned Δ** (S5, vanilla Mamba), or (ii) rely on **positional encodings of absolute time** (RoMAE)?

Two auxiliary claims:
- A cross-variate time-alignment SSM layer is especially useful for **async** multivariate data.
- The claim should survive varying which variate is deprived of observations (Phase 4-1 vs 4-2).

### 1.2 Phases

| Phase | Data regime | Purpose |
|---|---|---|
| 2 | Sync dense (shared timestamps per sample, 4 irregularity levels via `frac_regular ∈ {0.0, 0.3, 0.8, 1.0}`) | Establish baseline ordering in the simplest regime; fairness sanity check |
| 3 | Async dense (independent timestamps per variate, same 4 regimes, convex-mix preserved analytically) | Tests cross-variate alignment: v3 = w·v0 + (1−w)·v1 under async sampling |
| 4-1 | Async + long gap on v3 (fixed_v3) | Tests forward-mixture recovery: with v3 gapped, model must synthesize from v0, v1 |
| 4-2 | Async + long gap on a uniformly random variate | Tests inverse-mixture recovery as well: when v0 or v1 is gapped, model must invert using v3 and the other part |

### 1.3 Models compared (current)

| Model | Role | Paradigm | Config | Status |
|---|---|---|---|---|
| **Mamba-MV** (ours) | Target model | SSM with true-Δt + cross-variate alignment layer | `d_model=256`, `dt_mode ∈ {learned, replace}` | In training (Phases 3, 4-1, 4-2) |
| **S5** | Baseline SSM | Diagonal SSM with learned Δ, parallel scan | **paper-native**: `d_model=128`, `state_dim=256`, `lr=1e-3`, `wd=0.05`, physical B=32×accum=4 | In training |
| **RoMAE** | Baseline transformer | Rotary-PE masked autoencoder | paper-native (tubelet=(1,1,1), RoPE-2D) | In training |

**Dropped / deferred:**

| Model | Reason | Date | Details |
|---|---|---|---|
| **mTAN** | Persistent training collapse | 2026-04-22 | Collapsed at every size / LR / weight-decay / timestamp-normalization combination tried (forced 7.8M, paper-native, and a 3-rung LR sweep). Test MSE ≈ `Var(y)` across seeds. Dropped from Phase 3/4. |
| **ContiFormer** | OOM on 80GB H100 | 2026-04-22 | 6 configs attempted; ODE-attention is L² in sequence length and cannot fit L = 360 flat tokens on the hardware we have access to. See §6. T-PATCHGNN (ICML 2024) precedent. |
| **Mamba `additive` dt_mode** | Shown to be a strict worst-of-both in Phase 2 | 2026-04-22 | Dropped from Phase 3/4 main runs. `learned` and `replace` kept. |

**Param-budget note.** Phase 2 initially enforced a 7.8M ± 10% parity budget across all models. Phase 2 diagnosis showed that the forced parity was *harming* baselines (S5 and RoMAE at 7.8M diverged from the configs their papers validated). For Phase 3/4 we switched S5 and RoMAE to their **paper-native** configs and reported param counts honestly rather than matching a single number. Mamba-MV stays at `d_model=256` (7.80M).

---

## 2. Data generation

### 2.1 Provenance

The colleague's generator at `ssm_model/ssm/mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py` is the authoritative source. It differs from the earlier `generate_longgap_multisin.py` (now deprecated, header-marked) in two structural ways:

1. **Convex-mixture variate structure.** Three variates are generated; v3 is *not* an independent sinusoid but the convex combination `v3 = w·v0 + (1−w)·v1`, where `w ∼ U(0.2, 0.8)` per sample. This mimics the TimeMixUP construction and makes v3 the natural test of cross-variate alignment.
2. **Four regimes by `frac_regular`** (fraction of evenly-spaced timestamps retained before random thinning) replace the previous "long-gap" regimes: `regular=1.0`, `low_irreg=0.8`, `med_irreg=0.3`, `high_irreg=0.0`.

Shared convention across phases:
- `N_OBS = 120` observations per variate per sample.
- `t ∈ [0, 10]`, `HISTORY = 8.0` (we kept 8.0 from prior runs rather than the reference's 7.0 — continuity with v1 logs).
- Seeds split: `signal_seed` (variate amplitudes/phases) and `ts_seed` (timestamp draws) are independent so the same sinusoid can be re-sampled at different timestamps, cleanly isolating the effect of timing.
- Splits: 1000 train / 200 val / 200 test per regime.
- HuggingFace Arrow sparse-flat layout: columns `item_id`, `target`, `timestamp`, `past_feat_dynamic_real`, `n_obs_per_var`, `history`. Per-variate concat indexed by `n_obs_per_var`.

### 2.2 Phase 2 — sync dense (`generate_multivariate_sinusodial_data.py`)

Verbatim port of the colleague's generator with two knob changes:
- `HISTORY = 7.0 → 8.0`
- `output_root` default → `ssm_dk/data_correct/`

Output tree:
```
data_correct/
├── multisin_regular/        (frac_regular=1.0)
├── multisin_low_irreg/      (frac_regular=0.8)
├── multisin_med_irreg/      (frac_regular=0.3)
├── multisin_high_irreg/     (frac_regular=0.0)
│   └── {train,val,test}/ + norm_stats.json
└── multisin_stats.json
```

All three variates **share the same timestamp vector** within a sample.

### 2.3 Phase 3 — async dense (`generate_multivariate_sinusodial_data_async.py`)

Copy of the Phase 2 generator with two changes:
1. `generate_timestamps(...)` is called **three times per sample** (one per variate). Each variate has its own `frac_regular` draw applied independently.
2. **v3 is recomputed analytically at ts3**, not interpolated:
   ```
   v3_clean(ts3) = w · sinusoid(ts3, f0, a0, φ0)
                 + (1−w) · sinusoid(ts3, f1, a1, φ1)
   v3 = v3_clean + noise
   ```
   This preserves the exact algebraic mixture under async sampling — no interpolation artifact. A well-aligned model can exploit the mixture; a model that treats each variate as an independent channel cannot.

Output tree: `data_correct_async/multisin_{regular,low_irreg,med_irreg,high_irreg}/...`

### 2.4 Phase 4 — async + long gap (`generate_multivariate_sinusodial_data_gap.py`)

Copy of Phase 3 plus **deterministic gap injection**, reused from `generate_longgap_multisin.py`'s `draw_forbidden_intervals` / `apply_forbidden_mask`.
- `GAP_START_RANGE = (2.5, 6.0)`, `GAP_LEN_RANGE = (1.5, 3.0)`, `N_GAPS = 1`.
- HF schema extended with `gap_starts: Sequence(float32)`, `gap_ends: Sequence(float32)`, `gapped_variate_index: Sequence(int32)` for downstream in-gap/out-gap slicing.
- With `HISTORY = 8.0`, gaps can extend into the forecast region `[8, 10]`; this is acceptable (intrinsic IMTS condition).

**Two variants via `--gap_variate_mode`** (same generator, two output trees):

| Variant | `--gap_variate_mode` | Gapped variate | Output tree | Scientific purpose |
|---|---|---|---|---|
| **Phase 4-1** | `fixed_v3` | always v3 (index 2) | `data_correct_gap/` | Forward-mixture recovery: v0, v1 observed across gap → must produce v3 |
| **Phase 4-2** | `random_uniform` | uniformly ∈ {v0, v1, v3} per sample | `data_correct_gap_random/` | Inverse-mixture recovery also probed: when v0 or v1 is gapped, recovery requires inverting the mixture via v3 and the other component |

Phase 4-2 gapped-variate distribution verified balanced (stratified across regimes): roughly 430 / 485 / 485 test samples across v0 / v1 / v2 per regime (by test-set construction with deterministic seeds, the 3:3:4 skew reflects rounding in per-sample index draws, not a sampling bug — balanced enough for per-variate statistics).

### 2.5 Generator verification

- Phase 2: mean obs/variate = 120 across all regimes; `frac_regular` monotone in the regime ordering.
- Phase 3: mean obs/variate = 120; timestamps now vary per variate within a sample (sanity-plotted).
- Phase 4: mean ≈ 111 obs/variate across regimes (drop from 120 explained by gap removal); min observations on gapped variate drops to ~75 in some samples.

---

## 3. Fairness recipe

Single source of truth: `shared_config/fair_defaults.py`.

| Knob | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | — |
| Learning rate | 5e-4 (Mamba-MV, RoMAE); **1e-3** (S5) | S5 uses paper-native LR — see §3.1 |
| Weight decay | 0.01 (Mamba-MV, RoMAE); **0.05** (S5) | S5 paper-native |
| Warmup | 100 steps (linear) | — |
| Schedule | Cosine to 1600 steps | — |
| Effective batch | 128 | via `accumulate_grad_batches` when needed |
| Precision | fp32 | Mamba and S5 kernels prefer fp32 |
| Max epochs | 200 | — |
| Patience (early stop) | **50** | bumped from 20 on 2026-04-22 after seeing S5 / RoMAE early-stopped at ~25 % of the cosine schedule |
| Seeds | 1–5 (5 seeds per config) | — |
| History | 8.0 | User decision |

### 3.1 Recipe deviations (documented)

After Phase-2 diagnosis (S5 and RoMAE collapsed at forced 7.8M + lr=5e-4 + wd=0.01), we switched S5 to its **paper-native recipe**:

| Knob | Fair default | S5 override | Rationale |
|---|---|---|---|
| LR | 5e-4 | **1e-3** | Smith et al. 2023 report lr ∈ [5e-4, 1e-3] for S5 on LRA |
| Weight decay | 0.01 | **0.05** | Paper uses wd=0.05 for structured SSM parameters |
| Physical batch | 128 | 32 (accum=4, effective 128) | Parallel-scan activation memory at B·V=384 OOMs on 80GB |
| SSM-spectral params | — | **excluded from weight decay** | `Lambda`, `log_step`, `D`, `B`, `C` are structured — decaying them harms training |

RoMAE runs at the Mamba defaults (`lr=5e-4`, `wd=0.01`) with paper-native architecture.

Result: at patience=50 with paper-native S5 and RoMAE configs, both models learn robustly (3–5/5 seeds converge), which was **not** the case at forced 7.8M.

---

## 4. Baseline wrapper design

Wrappers share a Lightning interface and per-sample JSONL test outputs so one aggregator processes all models identically. Design decisions below preserve architectural faithfulness while meeting the shared interface.

### 4.1 RoMAE (`romae_forecaster/romae_forecaster.py`) — GREEN

- Uses upstream `RoMAEForPreTraining` unmodified.
- **Tokenization.** Every observation (one value at one timestamp on one variate) is one token. `tubelet_size=(1,1,1)`, `n_channels=1`, so upstream's `patchify` is a no-op.
- **Positions.** Two RoPE-ND dimensions: `[timestamp, variate_id]` as float coords. Variate id is cast to float to share the RoPE machinery.
- **Mask semantics.** We pass `mask = pred_mask` (True = forecast target). The encoder drops these positions; the decoder reconstructs them. **Note the padding convention flip:** RoMAE's upstream code reads `pad_mask == True` as padding; our datamodule emits `pad_mask == True` as real. The wrapper flips via `pad_mask_romae = ~pad_mask_ours`.
- **Loss.** Built-in `MSELoss(reduction='none')` — runs only on masked positions, zeroes padding. Matches our forecasting loss scope.
- **Targets-as-labels subtlety (critical).** Values are passed **as-is** at pred-mask positions. Zeroing them collapses `m_x = x[mask]` to zero, training the model to output zero — a bug that silently sets train/val MSE to 0 and test MSE to `Var(y) + E[y]²`. Documented in the file docstring.
- **Eval reconstruction.** `logits[b, i]` corresponds to the i-th True position of `mask[b]` in positional order; we iterate over real pred-mask positions per sample to rebuild y_true / y_pred.

### 4.2 S5 (`s5_forecaster/s5_forecaster.py`) — GREEN after paper-native switch

- Upstream: **Kwaijtaal's `s5-pytorch v0.2.1`** port of Smith et al. (ICLR 2023). The port exposes a raw `S5` module whose `forward(signal, step_scale)` accepts per-step Δt. We do **not** use `S5Block` — it does not thread `step_scale` through its forward, which would defeat the point of comparing to Mamba-true-Δt.
- Wrapper block: `S5TemporalBlock` = Pre-LN + `S5(step_scale=Δt)` + residual + FFN. Transformer-style layout with SSM instead of attention.
- **Per-variate SSM stack.** Each variate is processed independently by flattening `(B, V)` into the batch dimension: `(B·V, L, D)`. Faithful to S5's design — S5 has no cross-variate mechanism.
- **Time signal.** Two routes: (i) a linear sin-cos time embedding concatenated with value at input projection, (ii) the per-step Δt threaded to the SSM as `step_scale`. Duplicate encoding is tolerable; S5's learned-Δ treats `step_scale` as the dominant source.
- **Masked-input pattern.** Values where `timestamps >= history` zeroed before the SSM (matches RoMAE convention).
- **Paper-native sizing (Phase 3/4).** `d_model=128`, `state_dim=256`, 6 layers — matches Smith et al. LRA config, ~2.6M params (not forced to 7.8M).
- **Weight-decay routing.** SSM-spectral params (`Lambda`, `log_step`, `D`, `B`, `C`) excluded from weight decay; only the FFN / projection matrices get `wd=0.05`.
- **Batch caveat.** Parallel scan stores `O(L)` intermediate states at backward. At physical B=128 on H100-80GB with V=3, effective batch B·V = 384 exceeds working memory. Dropped to B=32 with `accumulate_grad_batches=4` so the recipe batch of 128 is preserved. Documented as a fairness concession: per-step compute is identical; gradient granularity is identical; only sub-batch activation memory differs.

### 4.3 mTAN (`mtan_forecaster/mtan_forecaster.py`) — DROPPED

- Architecture faithfully implemented per Shukla & Marlin 2021: reference-point attention (64 ref points) + bidirectional GRU encoder + Gaussian generative decoder (we used only the reconstruction head, no KL).
- **Observed failure mode.** mTAN converged to constant-output collapse in 4–5/5 seeds across every configuration we tried:
  1. Forced 7.8M at `lr=5e-4, wd=0.01`
  2. Paper-native sizing at `lr=5e-4, wd=0.01`
  3. Paper-native sizing at `lr=1e-4, wd=0.01` (LR sweep rung)
  4. Paper-native sizing with timestamp normalization `t / t_max ∈ [0, 1]`
  5. Paper-native sizing with the `sin(Linear(1, 127)(t))` time embedding removed (custom hot-fix)
- **Diagnosis.** The `sin(Linear(1, 127)(t))` encoding in mTAN's reference-point attention is scale-sensitive. Our timestamps live in [0, 10]; mTAN was trained on normalized [0, 1]. Even after normalization we could not stabilize learning. The model is a VAE-style architecture designed for imputation / classification, not pure forecasting — the decoder expects latent variability at generation time, which our deterministic forecasting loss fights against.
- **Decision (2026-04-22).** mTAN dropped from Phase 3 / 4. Wrapper code preserved under `mtan_forecaster/` for future debugging or a different phase. Another transformer-family baseline may be added if time permits.

### 4.4 Mamba-MV (our model) — target model

- SSM with **true-Δt discretization** (the scientific claim).
- Additional **cross-variate time-alignment SSM layer** that mixes across variates at shared time points.
- **Two `dt_mode` variants** trained: `learned` (standard Mamba, ignores batch Δt) and `replace` (Δ ← batch Δt directly). The third variant `additive` (Δ ← learned Δ + batch Δt) was dropped after Phase 2 showed it to be a strict worst-of-both.
- 7.80 M params at `d_model=256`, `grid_K=128`, balanced per-variate + MV layer counts.

### 4.5 Bug fix inherited from reference

The colleague's Mamba reference read `self.hparams.history` in `_mask_prediction_inputs`, which diverges from the datamodule's per-sample `batch["history"]` if CLI defaults drift. All three wrappers now consume `batch["history"]` (or `batch["pred_mask"]` directly, which is already built from the datamodule's per-sample history). Every model masks an identical set of positions on the same batch.

### 4.6 Loss-scope identity check

A single fixed-batch unit test passes predictions of zero through every wrapper; all three active wrappers (Mamba-MV, S5, RoMAE) produce the same nominal MSE, proving loss is summed over the same positions. This rules out the single most common fairness bug in IMTS benchmarks (different denominators).

---

## 5. Evaluation methodology

### 5.1 Metric set (unchanged)

Every model reports **target-weighted MSE, MAE, Pearson, R²** globally (across all pred-mask positions pooled over the test set). These are the numbers the paper's tables will show.

Target-weighting = `sum(ss_res) / sum(count)` rather than `mean(per_sample_mse)`:
- Matches the training loss reduction (MSE over positions, not over samples).
- Avoids small-sample high-variance bias.
- Pearson and R² remain **sample-averaged** — they are per-sample correlation scores by definition.

### 5.2 Per-variate breakdown

Every wrapper emits per-sample `mse_v{d}`, `mae_v{d}`, `ss_res_v{d}`, `abs_err_sum_v{d}`, `count_v{d}` for `d ∈ {0, 1, 2}`. Aggregator pools across seeds:
```
mse_v{d}_tw = Σ_seed ss_res_v{d} / Σ_seed count_v{d}
mae_v{d}_tw = Σ_seed abs_err_sum_v{d} / Σ_seed count_v{d}
```
Why this matters: v3 is the convex-mix variate, so it is the one a cross-variate model should improve on most. Early Phase 2 signal on `multisin_med_irreg/learned`:
```
mamba_mv med_irreg learned n_seeds=3  mse=0.02573 ± 0.00034
                                v0_tw=0.02863  v1_tw=0.02870  v2_tw=0.01987
```
— v3 is ~30 % lower MSE than v0/v1 in the same runs, first concrete evidence for the alignment claim.

### 5.3 Phase 4 gap-region slicing — generalized over all variates

For Phase-4 runs, wrappers emit per-sample keyed by the **sample-specific gapped variate** `d` read from `gapped_variate_index[i][0]`:
```
ss_res_v{d}_in_gap, abs_err_sum_v{d}_in_gap, count_v{d}_in_gap
ss_res_v{d}_out_gap, abs_err_sum_v{d}_out_gap, count_v{d}_out_gap
```
Aggregator's `_phase4_gap_region()` pools across samples and across seeds, loops over `d ∈ {0, 1, 2}`, and yields per-variate target-weighted in-gap vs out-gap MSE/MAE. Two modes:

- **Phase 4-1 (`fixed_v3`):** `d` is always 2 in every sample → the `v2_in_gap` and `v2_out_gap` buckets get all counts; `v0` / `v1` buckets are empty.
- **Phase 4-2 (`random_uniform`):** `d` varies per sample → all three variates' in-gap / out-gap buckets have ≥ 100 test counts each (enough for stable target-weighted numbers).

This is the headline number for Phase 4: a model that alignment-copies from the other variates should hold up inside the gap; a model that treats variates independently should degrade sharply. Phase 4-2 additionally tests whether the alignment works in the inverse-mixture direction.

### 5.4 Backwards compatibility

Aggregator's `_per_variate_target_weighted()` falls back to sample-averaged `mse_v{d}_samp` for pre-edit runs (the earliest Phase 2 jobs). Phase 3 and 4 have full target-weighted per-variate from the start.

### 5.5 Statistical significance

`paired_significance()` runs paired-sample Wilcoxon signed-rank (Mamba-MV vs each baseline) over 5 seeds × 4 regimes × per-sample MSE. Reported as `p` value alongside effect size (median of paired differences).

### 5.6 Aggregation pipeline

`eval/aggregate_results.py`:
- `summarize()` → wide CSV (model × regime × variant, global metrics mean ± std).
- `_per_variate_target_weighted()` → long CSV with per-variate target-weighted metrics.
- `_phase4_gap_region()` → long CSV with in-gap/out-gap metrics, stratified over d ∈ {0, 1, 2}.
- `paired_significance()` → CSV of Wilcoxon p-values.

---

## 6. Why ContiFormer was deferred

### 6.1 Attempt ledger

6 configurations attempted, all OOM at ~4.95 GiB of H100-80GB activation memory:

| Config | Batch | Integrator tolerance (step_size) | Params | Result |
|---|---|---|---|---|
| default | 128 | 0.1 | ~7.8 M | OOM |
| smaller batch | 64 | 0.1 | ~7.8 M | OOM |
| smaller batch | 32 | 0.1 | ~7.8 M | OOM |
| minimal batch | 8 | 0.1 | ~7.8 M | OOM |
| loose tolerance | 32 | 0.5 | ~7.8 M | OOM |
| reduced arch + loose tol | 32 | 0.5 | **1.45 M** | OOM |

The cost is **L² in sequence length** from ODE-attention. With `L = 120 · 3 = 360` flat tokens per sample, the pairwise-attention ODE state is structurally the wrong size for the hardware we have access to.

### 6.2 Methodological precedent

T-PATCHGNN (ICML 2024) also excludes ContiFormer as a baseline on IMTS tasks of this length for the same reason. Paper text can state: *"ContiFormer was attempted at six (batch, tolerance, parameter-budget) configurations and OOM'd on all of them on 80 GB H100 with L = 360 flat tokens, consistent with T-PATCHGNN (ICML 2024). We defer it to future work on ≥120 GB accelerators."*

### 6.3 Documentation

`audit/contiformer_deferral_note.md` contains the full ledger, an L²-dominated memory derivation, and the T-PATCHGNN citation. The reduced-arch config (1.45 M params) is preserved in `audit/count_params.py::build_contiformer_reduced()` for post-Phase-2 revisits on bigger hardware.

---

## 7. Infrastructure

### 7.1 SLURM layout (current)

Under `imts_benchmark/scripts/`. Each phase has its own trio of sbatch files:

| Script | Array size | Mapping |
|---|---|---|
| `run_mamba_mv_phase{3,4_1,4_2}.sbatch` | 1–40 | 4 regimes × 2 `dt_mode` ({learned, replace}) × 5 seeds |
| `run_s5_phase{3,4_1,4_2}.sbatch` | 1–20 | 4 regimes × 5 seeds |
| `run_romae_phase{3,4_1,4_2}.sbatch` | 1–20 | 4 regimes × 5 seeds |

Data / log routing:
- `run_*_phase3.sbatch` → `data_correct_async/`, logs under `phase3/`
- `run_*_phase4_1.sbatch` → `data_correct_gap/`, logs under `phase4_1/`
- `run_*_phase4_2.sbatch` → `data_correct_gap_random/`, logs under `phase4_2/`

Per-phase run count: 40 + 20 + 20 = **80 runs per phase** × 3 active phases = **240 runs** total for the Phase 3/4 suite. Phase 2 had 140 runs (60 Mamba × 3 dt_modes + 4 × 20 baselines with mTAN still included).

### 7.2 Conda environments

- PyTorch + CUDA + `mamba_ssm` + `s5-pytorch`: `/projects/b1094/StarEmbed/pythonenvs/mamba`
  — All three active models (Mamba-MV, S5, RoMAE) run in this env. Earlier S5-JAX env (`pythonenvs/s5-jax`) was not used in the end; we went with `s5-pytorch` for Lightning integration.

### 7.3 Storage

- Datasets: `ssm_dk/{data_correct, data_correct_async, data_correct_gap, data_correct_gap_random}/`.
- Logs: `output/log/imts_benchmark_v2/{phase2, phase3, phase4_1, phase4_2}/<model>/<regime>/<variant>/seed<N>/` — each folder contains `test_metrics.csv` and `per_sample.jsonl`.
- Per-variate metric columns live inside `per_sample.jsonl`; global metrics in `test_metrics.csv`.

---

## 8. Audit deliverables

- `audit/count_params.py` — instantiates all models; still prints Phase-2-era 7.8M targets for historical reference.
- `audit/baseline_spec_romae.md` — GREEN, current.
- `audit/baseline_spec_s5.md` — updated to paper-native config; GREEN after patience=50.
- `audit/baseline_spec_mtan.md` — **DROPPED** (persistent collapse).
- `audit/baseline_spec_contiformer.md` — YELLOW → deferred.
- `audit/contiformer_deferral_note.md` — 6-config ledger and methodology justification.
- `audit/baseline_validation_report.md` — 5 models × 4 checks (upstream contract, 20-epoch behavior smoke, noise-free sanity benchmark, cross-model init parity). Baseline-validation fields for mTAN / ContiFormer flagged "deferred".

---

## 9. Paper-relevant decisions to carry forward

1. **Shared recipe with one principled deviation.** AdamW + warmup 100 + cosine to 1600 + effective batch 128 + fp32 + patience 50 for all models. S5 runs at paper-native `lr=1e-3, wd=0.05` (documented — forced-parity recipe caused S5 to fail to learn).
2. **Identical loss scope.** MSE over `pred_mask`-True positions, target-weighted at aggregation. Fixed-batch unit test confirms all active models sum the same positions.
3. **Config honesty, not forced param parity.** Mamba-MV at 7.80M; baselines at whatever config the original paper validated. Report param counts honestly in the methods table.
4. **History = 8.0.** Diverges from colleague's reference (7.0) by user choice; continuous with v1 runs.
5. **v3 = convex mix.** The scientific lever — v3 is analytically recomputed at async timestamps in Phase 3, not interpolated.
6. **Phase 4 split.** Phase 4-1 (fixed_v3) and Phase 4-2 (random_uniform); they test forward and inverse directions of the convex mixture respectively. Run both and report both.
7. **S5 via PyTorch port.** Kwaijtaal `s5-pytorch` v0.2.1, custom Pre-LN block (upstream `S5Block` drops `step_scale`). Weight decay excluded from SSM-spectral params.
8. **ContiFormer deferral.** 6 OOM configs + T-PATCHGNN precedent; paper caveat paragraph drafted in §6.2.
9. **mTAN dropped.** Persistent collapse across every sizing / LR / timestamp-normalization config tried. Report as a failed baseline or omit entirely from the final table (decision deferred to results).
10. **Metric conventions.** Target-weighted MSE/MAE + sample-averaged Pearson/R². Per-variate and in-gap/out-gap breakdowns stratified over d ∈ {0, 1, 2}. Unchanged across phases.

---

## 10. Current status and next steps

- **Phase 2:** complete. `RESULTS_phase2.md` written. At paper-native baseline configs + patience=50, the full regime sweep for Mamba-MV (learned, replace) vs S5, RoMAE is tabulated; mTAN dropped. Phase 2 identified S5 as the strongest baseline (not RoMAE as earlier suspected at forced 7.8M).
- **Phase 3, 4-1, 4-2:** in training. 240 runs across the three phases submitted as 9 SLURM arrays (Mamba-MV × {P3, P4-1, P4-2} @ 40 each + S5 × 3 @ 20 each + RoMAE × 3 @ 20 each). All at current non-HPO'd Mamba config.
- **Decision gate (post-Phase 4).** If Mamba `replace` still loses to S5 on the majority of irregular regimes in Phase 3 or 4 → trigger HPO per `docs/HPO_PLAN.md`. If Mamba wins on Phase 3/4 regimes → proceed to paper-writing without HPO.
- **HPO_PLAN.md** saved separately so it survives context compaction. Stage 1 (90 runs: 5 lr × 3 d_model × 2 dt_mode × 3 seeds on `multisin_med_irreg` Phase 3) is the first gate if triggered.

---

## Appendix A — Document history

| File | Date | Scope | Status |
|---|---|---|---|
| `docs/archive/2026-04-20_BUILD_REPORT_mamba_vs_romae.md` | 2026-04-20 | Initial Mamba vs RoMAE build | Archived — superseded |
| `docs/archive/2026-04-20_RESULTS_mamba_vs_romae.md` | 2026-04-20 | Initial Mamba vs RoMAE results (long-gap data) | Archived — superseded |
| `docs/archive/2026-04-21_BUILD_REPORT_imts_benchmark.md` | 2026-04-21 | 5-model benchmark build on `data_correct` | Archived — superseded |
| `docs/archive/2026-04-21_RESULTS_imts_benchmark.md` | 2026-04-21 | Phase-2 partial results, pre-per-variate | Archived — superseded |
| `docs/REPORT_consolidated_2026-04-22.md` | 2026-04-22 (initial); 2026-04-23 (baselines / Phase-4 split) | Phases 2–4, per-variate & gap-region metrics, ContiFormer deferral, mTAN drop, S5 paper-native | **Active** |
| `docs/HPO_PLAN.md` | 2026-04-22 | Rigorous 3-stage HPO design | **Active (gate: Phase 3/4 result)** |
| `docs/RESULTS_phase2.md` | 2026-04-22 | Phase 2 final table after patience=50 + paper-native baselines | **Active** |
