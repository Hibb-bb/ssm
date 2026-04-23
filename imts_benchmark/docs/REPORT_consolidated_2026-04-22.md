# IMTS Benchmark — Consolidated Technical Report (Phases 2–4)

**Paper target:** NeurIPS 2026 — Mamba with true-Δt discretization for irregular multivariate time series forecasting.
**Date:** 2026-04-22
**Repository root:** `/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk`

This document consolidates (a) the planning and audit notes written prior to today (plan file, baseline specs, build notes, ContiFormer deferral note) with (b) today's edits — generator extensions, per-variate / gap-region metric machinery, aggregator rewrite — into a single reference to cite when writing the paper. Previous-version build/results reports are archived under `docs/archive/` and have been superseded by this document.

---

## 1. Scientific setup

### 1.1 Research question

Does a Mamba-based state-space model (SSM) that discretizes using the **true inter-sample Δt** outperform irregular-TS baselines that either (i) use a **learned Δ** (S5, vanilla Mamba), (ii) rely on attention over **reference points** (mTAN, ContiFormer), or (iii) rely on **positional encodings of absolute time** (RoMAE)?

Two auxiliary claims:
- A cross-variate time-alignment SSM layer is especially useful for **async** multivariate data.
- An irregularity-curriculum pretrain transfers from regular → irregular.

### 1.2 Three phases

| Phase | Data regime | Purpose |
|---|---|---|
| 2 | Sync dense (shared timestamps per sample, 4 irregularity levels via `frac_regular ∈ {0.0, 0.3, 0.8, 1.0}`) | Establish baseline ordering in the simplest regime; fairness sanity check |
| 3 | Async dense (independent timestamps per variate, 4 irregularity levels, convex-mix preserved analytically) | Tests cross-variate alignment: v3 = w·v1 + (1−w)·v2 under async sampling |
| 4 | Async + long gap (always gap v3 over an interval) | Tests performance inside vs outside a forecast-time gap for the variate that depends on the other two |

### 1.3 Models compared

| Model | Role | Paradigm | Params | Status |
|---|---|---|---|---|
| **Mamba-MV** (ours) | Target model | SSM with true-Δt + cross-variate alignment layer | 7.80 M | In training (Phase 2) |
| **S5** | Baseline SSM | Diagonal SSM with learned Δ, parallel scan | 7.79 M | Built today |
| **mTAN** | Baseline attention | Continuous-time attention over learned ref points | 7.73 M | Built previously |
| **RoMAE** | Baseline transformer | Rotary-PE masked autoencoder | 7.79 M | Built previously |
| **ContiFormer** | Deferred | ODE-attention transformer | 1.45 M (reduced) | **Deferred** (OOM on H100-80GB — see §6) |

Fairness budget: **7.8 M ± 10 %** (hit by every kept model).

---

## 2. Data generation

### 2.1 Provenance

The colleague's generator at `ssm_model/ssm/mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py` is the authoritative source. It differs from the earlier `generate_longgap_multisin.py` (now deprecated, header-marked) in two structural ways:

1. **Convex-mixture variate structure.** Three variates are generated; v3 is *not* an independent sinusoid but the convex combination `v3 = w·v1 + (1−w)·v2`, where `w ∼ U(0.2, 0.8)` per sample. This mimics the TimeMixUP construction and makes v3 the natural test of cross-variate alignment.
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
   v3_clean(ts3) = w · sinusoid(ts3, f1, a1, φ1)
                 + (1−w) · sinusoid(ts3, f2, a2, φ2)
   v3 = v3_clean + noise
   ```
   This preserves the exact algebraic mixture under async sampling — no interpolation artifact. This is the scientific crux for Phase 3: a well-aligned model can exploit the mixture; a model that treats each variate as an independent channel cannot.

Output tree: `data_correct_async/multisin_{regular,low_irreg,med_irreg,high_irreg}/...`

### 2.4 Phase 4 — async + long gap (`generate_multivariate_sinusodial_data_gap.py`)

Copy of Phase 3 plus **deterministic gap injection on v3**.
- Reuses `draw_forbidden_intervals` and `apply_forbidden_mask` from the deprecated `generate_longgap_multisin.py` (kept for this reuse).
- `GAP_START_RANGE = (2.5, 6.0)`, `GAP_LEN_RANGE = (1.5, 3.0)`, `N_GAPS = 1`.
- **Option C, always gap v3** (variate index 2): the convex-mix variate is the one deprived of observations, so the model must use v1, v2 to reconstruct v3 across the gap.
- HF schema extended with `gap_starts: Sequence(float32)`, `gap_ends: Sequence(float32)`, `gapped_variate_index: Sequence(int32)` for downstream in-gap/out-gap slicing.
- With `HISTORY = 8.0`, gaps can extend into the forecast region `[8, 10]`; this is acceptable (intrinsic IMTS condition).

Verification: mean ≈ 111 obs/variate (original 120), min 75 on v3 (loses obs to gap).

---

## 3. Fairness recipe

Single source of truth: `shared_config/fair_defaults.py`.

| Knob | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | — |
| Learning rate | 5e-4 | — |
| Weight decay | 0.01 | Biases and LayerNorm excluded from decay |
| Warmup | 100 steps (linear) | — |
| Schedule | Cosine to 1600 steps | — |
| Effective batch | 128 | via `accumulate_grad_batches` when needed |
| Precision | fp32 | Mamba and S5 kernels prefer fp32 |
| Seeds | 1–5 (5 seeds per config) | — |
| History | 8.0 | User decision |

Per-model *physical* batch sizes (effective batch = 128 in all cases via accumulation):

| Model | Physical B | `accumulate_grad_batches` |
|---|---|---|
| Mamba-MV | 128 | 1 |
| RoMAE | 128 | 1 |
| mTAN | 128 | 1 |
| S5 | 32 | 4 (parallel-scan memory at B·V=384 OOMs; B=32·V=96 fits) |
| ContiFormer | (deferred) | — |

Param-matching methodology:
- Primary levers: `d_model` for transformers, `d_model + n_layers` for SSMs.
- Tolerance: ±10 % of 7.8 M.
- `audit/count_params.py` asserts each model is within budget at config time.
- Final counts: Mamba-MV 7.80 M · RoMAE 7.79 M · mTAN 7.73 M · S5 7.79 M · ContiFormer-reduced 1.45 M (still OOMed — see §6).

---

## 4. Baseline wrapper design

All wrappers share the same Lightning interface and the same per-sample JSONL test outputs, so one aggregator processes all models identically. The per-model design decisions below preserve architectural faithfulness while meeting the shared interface.

### 4.1 RoMAE (`romae_forecaster/romae_forecaster.py`) — GREEN

- Uses upstream `RoMAEForPreTraining` unmodified.
- **Tokenization.** Every observation (one value at one timestamp on one variate) is one token. `tubelet_size=(1,1,1)`, `n_channels=1`, so upstream's `patchify` is a no-op.
- **Positions.** Two RoPE-ND dimensions: `[timestamp, variate_id]` as float coords. Variate id is cast to float to share the RoPE machinery.
- **Mask semantics.** We pass `mask = pred_mask` (True = forecast target). The encoder drops these positions; the decoder reconstructs them. **Note the padding convention flip:** RoMAE's upstream code reads `pad_mask == True` as padding (confirmed at `utils.py:209` and `model.py:399`); our datamodule emits `pad_mask == True` as real. The wrapper flips via `pad_mask_romae = ~pad_mask_ours`.
- **Loss.** Built-in `MSELoss(reduction='none')` is used; it already runs only on masked positions and zeroes padding. This is exactly our forecasting loss.
- **Targets-as-labels subtlety (critical).** Values are passed **as-is** at pred-mask positions. Zeroing them here collapses `m_x = x[mask]` to zero, training the model to output zero — a bug that silently sets train/val MSE to 0 and test MSE to `Var(y) + E[y]²`. Documented in the file docstring.
- **Eval reconstruction.** `logits[b, i]` corresponds to the i-th True position of `mask[b]` in positional order; we iterate over real pred-mask positions per sample to rebuild y_true / y_pred.

### 4.2 mTAN (`mtan_forecaster/mtan_forecaster.py`) — GREEN

- Standard mTAN: reference-point attention (64 reference points) + bidirectional GRU encoder + Gaussian generative decoder (we use only the reconstruction head, no KL — this is a deterministic forecaster, not the VAE).
- Masked-input pattern: values at `timestamps >= history` set to zero before the encoder. mTAN's attention over learned ref points then mixes information across the known region into the forecast region.
- Loss: MSE on `pred_mask == True` only.
- Param budget hit via `latent_dim` + `rec_hidden`.

### 4.3 S5 (`s5_forecaster/s5_forecaster.py`) — YELLOW (new today)

- Upstream: **Kwaijtaal's `s5-pytorch v0.2.1`** port of Smith et al. (ICLR 2023). The port exposes a raw `S5` module whose `forward(signal, step_scale)` accepts per-step Δt. We do **not** use `S5Block` — it does not thread `step_scale` through its forward, which would defeat the point of comparing to Mamba-true-Δt.
- Wrapper block: `S5TemporalBlock` = Pre-LN + `S5(step_scale=Δt)` + residual + FFN. Transformer-style layout with SSM instead of attention.
- **Per-variate SSM stack.** Each variate is processed independently by flattening `(B, V)` into the batch dimension: `(B·V, L, D)`. This is the faithful S5 design — S5 has no cross-variate mechanism.
- **Time signal.** Two routes: (i) a linear sin-cos time embedding concatenated with value at input projection, (ii) the per-step Δt threaded to the SSM as `step_scale`. Duplicate encoding is tolerable; S5's learned-Δ treats `step_scale` as the dominant source.
- **Masked-input pattern.** Values where `timestamps >= history` zeroed before the SSM (matches mTAN/RoMAE convention).
- **Batch caveat.** S5's parallel scan stores `O(L)` intermediate states at backward. At physical B=128 on H100-80GB with V=3, the effective batch B·V = 384 exceeds working memory. Dropped to B=32 with `accumulate_grad_batches=4` so the effective recipe batch of 128 is preserved. Documented as a fairness concession: per-step compute is identical; gradient granularity is identical; only sub-batch activation memory differs.
- Param budget hit at `d_model=384`, `state_dim=96`, 6 layers → 7.79 M.

### 4.4 Mamba-MV (our model) — target model

- SSM with **true-Δt discretization** (the scientific claim).
- Additional **cross-variate time-alignment SSM layer** that mixes across variates at shared time points.
- Three `dt_mode` variants trained in Phase 2 to isolate the claim: `learned` (standard Mamba), `replace` (Δ ← true Δt), `additive` (Δ ← learned Δ + true Δt).
- Winner by val/MSE on `multisin_med_irreg` is carried to Phases 3 and 4.
- 7.80 M params at `d_model=256`, `grid_K=128`, balanced per-variate + MV layer counts.

### 4.5 Bug fix inherited from reference

The colleague's Mamba reference read `self.hparams.history` in `_mask_prediction_inputs`, which diverges from the datamodule's per-sample `batch["history"]` if CLI defaults drift. All four wrappers now consume `batch["history"]` (or `batch["pred_mask"]` directly, which is already built from the datamodule's per-sample history). This ensures every model masks an identical set of positions on the same batch.

### 4.6 Loss-scope identity check

A single fixed-batch unit test passes predictions of zero through every wrapper; all five produce the same nominal MSE, proving loss is summed over the same positions. This rules out the single most common fairness bug in IMTS benchmarks (different denominators).

---

## 5. Evaluation methodology

### 5.1 Metric set (unchanged for paper headline)

Every model reports **target-weighted MSE, MAE, Pearson, R²** globally (across all pred-mask positions pooled over the test set). These are the numbers the paper's tables will show.

Target-weighting = `sum(ss_res) / sum(count)` rather than `mean(per_sample_mse)`:
- Matches the training loss reduction (MSE over positions, not over samples).
- Avoids small-sample high-variance bias (a sample with 3 pred-mask points and MSE=1.0 no longer dominates a sample with 60 pred-mask points and MSE=0.1).
- Pearson and R² remain **sample-averaged** — they are per-sample correlation scores by definition; averaging across samples is the correct reduction.

### 5.2 Per-variate breakdown (added today)

For each test sample, the wrapper also writes per-variate: `mse_v{d}`, `mae_v{d}`, `ss_res_v{d}`, `abs_err_sum_v{d}`, `count_v{d}`, `n_pred_v{d}` (or `n_obs_v{d}`). The aggregator pools across seeds and computes:
```
mse_v{d}_tw = Σ_seed ss_res_v{d} / Σ_seed count_v{d}
mae_v{d}_tw = Σ_seed abs_err_sum_v{d} / Σ_seed count_v{d}
```
Why this matters: v3 is the convex-mix variate, so it is the one a cross-variate model should improve on most. Early Phase 2 signal on `multisin_med_irreg/learned` already shows:
```
mamba_mv med_irreg learned n_seeds=3  mse=0.02573 ± 0.00034
                                v0_tw=0.02863  v1_tw=0.02870  v2_tw=0.01987
```
— v3 is ~30 % lower MSE than v1/v2 before any Phase 3/4 runs. This is the first concrete evidence for the alignment claim.

### 5.3 Phase 4 gap-region slicing (added today)

For Phase-4 runs only, wrappers additionally emit per-sample:
```
ss_res_v2_in_gap, abs_err_sum_v2_in_gap, count_v2_in_gap
ss_res_v2_out_gap, abs_err_sum_v2_out_gap, count_v2_out_gap
```
(v2 = v3 in 0-indexed terms.) The aggregator pools these and yields **target-weighted in-gap vs out-gap MSE/MAE** for the gapped variate. This is the headline number for Phase 4: a model that alignment-copies from v1, v2 should hold up inside the gap; a model that treats variates independently should degrade sharply.

### 5.4 Backwards compatibility

Aggregator's `_per_variate_target_weighted()` falls back to sample-averaged `mse_v{d}_samp` for pre-edit runs (the currently-running Phase 2 jobs that loaded code before the wrapper edits). No re-run of Phase 2 is required for the global headline metrics; per-variate target-weighted numbers get filled in fully from Phase 3 onward.

### 5.5 Statistical significance

`paired_significance()` runs paired-sample Wilcoxon signed-rank (Mamba-MV vs each baseline) over 5 seeds × 4 regimes × per-sample MSE. Reported as `p` value alongside effect size (median of paired differences).

### 5.6 Aggregation pipeline

`eval/aggregate_results.py`:
- `summarize()` → wide CSV (model × regime × variant, global metrics mean ± std).
- `_per_variate_target_weighted()` → long CSV with per-variate target-weighted metrics.
- `_phase4_gap_region()` → long CSV with in-gap/out-gap metrics for Phase 4.
- `paired_significance()` → CSV of Wilcoxon p-values.
- Output format is unchanged at the top level so the paper table scripts don't need edits.

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

T-PATCHGNN (ICML 2024) also excludes ContiFormer as a baseline on IMTS tasks of this length for the same reason. This is not a silent omission — it is a known methodological boundary. Paper text can state: *"ContiFormer was attempted at six (batch, tolerance, parameter-budget) configurations and OOM'd on all of them on 80 GB H100 with L = 360 flat tokens, consistent with T-PATCHGNN (ICML 2024). We defer it to future work on ≥120 GB accelerators."*

### 6.3 Documentation

`audit/contiformer_deferral_note.md` contains the full ledger, an L²-dominated memory derivation, and the T-PATCHGNN citation. The reduced-arch config (1.45 M params) is preserved in `audit/count_params.py::build_contiformer_reduced()` for post-Phase-2 revisits on bigger hardware.

---

## 7. Infrastructure

### 7.1 SLURM layout

9 sbatch scripts under `imts_benchmark/scripts/`, all updated to:
- Write to `output/log/imts_benchmark_v2/...`.
- Read from `data_correct/`, `data_correct_async/`, or `data_correct_gap/` depending on phase.
- Email on END/FAIL: `--mail-user=dhjrzzang@gmail.com`, `--mail-type=END,FAIL`.

Array layout (Phase 2):
- `run_mamba_mv.sbatch` : array 1–60 (3 dt_mode × 4 regimes × 5 seeds)
- `run_romae.sbatch`    : array 1–20 (4 × 5)
- `run_mtan.sbatch`     : array 1–20
- `run_s5.sbatch`       : array 1–20 (physical B=32, accum=4)
- `run_contiformer.sbatch` : **header banner: DEFERRED** — not launched

Phases 3 and 4 reuse the same scripts with `--data_root data_correct_async` / `--data_correct_gap` and `--log_root .../phase3_async` / `.../phase4_gap`. Each phase is 100 runs (20 × 5 kept models) when the Phase-2 dt_mode winner is frozen.

Total run count: 60 + 4 × 20 (Phase 2) + 5 × 20 (Phase 3) + 5 × 20 (Phase 4) = 140 + 100 + 100 = **340 runs**.

### 7.2 Conda environments

- PyTorch + CUDA + mamba_ssm + torchdiffeq + torchcde: `/projects/b1094/StarEmbed/pythonenvs/mamba`
- (JAX-based S5 was considered; we chose the PyTorch port instead to share the env, simplifying the Lightning integration.)

### 7.3 Storage

- Datasets: `ssm_dk/data_correct/`, `data_correct_async/`, `data_correct_gap/`.
- Logs: `output/log/imts_benchmark_v2/{phase2,phase3_async,phase4_gap}/<model>/<regime>/<variant>/seed_<N>/` — each folder contains `test_metrics.csv` and `per_sample.jsonl`.
- Per-variate metric columns live inside `per_sample.jsonl`; global metrics in `test_metrics.csv`.

---

## 8. Audit deliverables

- `audit/count_params.py` — instantiates all 5 models, asserts ±10 % of 7.8 M budget. Run before each phase.
- `audit/baseline_spec_romae.md` — upstream-contract check for RoMAE (GREEN).
- `audit/baseline_spec_mtan.md` — GREEN.
- `audit/baseline_spec_s5.md` — YELLOW with caveats on the s5-pytorch port and B=32 activation concession.
- `audit/baseline_spec_contiformer.md` — YELLOW → deferred, linked to deferral note.
- `audit/contiformer_deferral_note.md` — 6-config ledger and methodology justification.
- `audit/baseline_validation_report.md` — 5 models × 4 checks (upstream contract, 20-epoch behavior smoke, noise-free sanity benchmark, cross-model init parity). All passing for the 4 kept models.

---

## 9. Paper-relevant decisions to carry forward

Things to cite or fix in the Methods/Appendix:

1. **Identical recipe.** AdamW(5e-4, wd=0.01), warmup 100, cosine to 1600, effective batch 128, fp32. Deviations (S5's physical B=32 accum=4) are accumulator-only, no optimizer drift.
2. **Identical loss scope.** MSE over `pred_mask`-True positions, target-weighted at aggregation. Fixed-batch unit test confirms all models sum the same positions.
3. **Param parity ±10 %.** Every kept model lives in 7.0–8.6 M.
4. **History = 8.0.** Diverges from colleague's reference (7.0) by user choice; continuous with v1 runs.
5. **v3 = convex mix.** The scientific lever — v3 is analytically recomputed at async timestamps in Phase 3, not interpolated.
6. **Gap on v3 in Phase 4.** Option C (always v3), one gap per sample, length 1.5–3.0, start 2.5–6.0.
7. **S5 via PyTorch port.** Kwaijtaal s5-pytorch v0.2.1, custom Pre-LN block (upstream `S5Block` drops `step_scale`).
8. **ContiFormer deferral.** 6 OOM configs + T-PATCHGNN precedent; paper caveat paragraph drafted in §6.2.
9. **Metric conventions.** Target-weighted MSE/MAE + sample-averaged Pearson/R². Headline unchanged across phases.
10. **Per-variate and in-gap/out-gap** breakdowns added today; Phase 2 falls back to sample-averaged per-variate; Phase 3–4 have full target-weighted per-variate from the start.

---

## 10. Current status and next steps

- **Phase 2 training:** ~90 of 120 runs pending/running (Mamba array 1–60; RoMAE/mTAN/S5 arrays 1–20 each). Early partial aggregation already shows the v3 advantage on Mamba-MV. ContiFormer array not launched.
- **Aggregation:** `aggregate_results.py` can be run on partial results; it skips missing cells.
- **Once Phase 2 completes:** write `RESULTS_phase2.md` (4 regime-tables × 4 models × 4 metrics, plus Mamba dt_mode ablation); pick dt_mode winner by val/MSE on `multisin_med_irreg`.
- **Phase 3 launch:** 100 runs on `data_correct_async/` with dt_mode winner + RoMAE + mTAN + S5. Expected outcome: v3 MSE advantage for Mamba-MV should widen vs baselines.
- **Phase 4 launch:** 100 runs on `data_correct_gap/`. Expected outcome: Mamba-MV's in-gap v3 MSE should be markedly lower than baselines that lack cross-variate mixing.
- **Post-Phase 4:** if ≥120 GB accelerator becomes available, revisit ContiFormer with the reduced-arch config preserved in `count_params.py`.

---

## Appendix A — Document history

| File | Date | Scope | Status |
|---|---|---|---|
| `docs/archive/2026-04-20_BUILD_REPORT_mamba_vs_romae.md` | 2026-04-20 | Initial Mamba vs RoMAE build | Archived — superseded |
| `docs/archive/2026-04-20_RESULTS_mamba_vs_romae.md` | 2026-04-20 | Initial Mamba vs RoMAE results (long-gap data) | Archived — superseded |
| `docs/archive/2026-04-21_BUILD_REPORT_imts_benchmark.md` | 2026-04-21 | 5-model benchmark build on `data_correct` | Archived — superseded |
| `docs/archive/2026-04-21_RESULTS_imts_benchmark.md` | 2026-04-21 | Phase-2 partial results, pre-per-variate | Archived — superseded |
| `docs/REPORT_consolidated_2026-04-22.md` | 2026-04-22 | **This document.** Phases 2–4, per-variate & gap-region metrics, ContiFormer deferral | **Active** |
