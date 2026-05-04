# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A controlled benchmark for **irregular multivariate time-series (IMTS) forecasting**, built around a single scientific question: does a Mamba SSM that discretizes with the true inter-sample Δt outperform IMTS baselines (S5, RoMAE) that use a learned Δ or absolute-time positional encodings? Synthetic phases (2 / 3 / 4-1 / 4-2) isolate one factor at a time (sync→async→long-gap); Phase 5 ports the same setup to T-PatchGNN real datasets (PhysioNet, Activity, USHCN) imported under [tpatchgnn_data/](tpatchgnn_data/); Phase 6 ports it to the **TIME-IMM** suite (8 public datasets, e.g. `imm_fnspid`, `imm_gdelt`, …) imported under [time_imm_data/](time_imm_data/), plus **MIMIC** as a separate real-clinical run. A parallel **UEA classification add-on** (NeurIPS 2026 Table) reuses the Mamba-MV encoder against RoMAE Table 4 baselines on five (and now eighteen) UEA datasets under the Kidger 30% sync-drop irregularization protocol; UEA splits live under [data_uea/](data_uea/).

[README.md](README.md) covers the synthetic data design + signal model in depth. [BUILD_NOTES.md](BUILD_NOTES.md) is the data-generation provenance log (deprecated `generate_longgap_multisin.py` regime; current generators are the `generate_multivariate_sinusodial_data*.py` family). Active design docs live under [imts_benchmark/docs/](imts_benchmark/docs/) (`HPO_PLAN.md`, `REPORT_consolidated_2026-04-22.md`, `RESULTS_phase{2,3,4_1,4_2,5_real}.md`).

## High-level architecture

Three things live side by side:

1. **Synthetic data generators** at the repo root — `generate_multivariate_sinusodial_data*.py`. These produce HF Arrow datasets under `data_correct/`, `data_correct_async/`, `data_correct_gap/`, `data_correct_gap_random/`. Schema is documented in the README; key invariant is the **sparse-flat layout** (`target` / `timestamp` / `past_feat_dynamic_real` / `n_obs_per_var` / `history`) shared with [tpatchgnn_data/](tpatchgnn_data/) so real and synthetic data flow through the same datamodule.

2. **Models** under [imts_benchmark/](imts_benchmark/), each in its own subpackage with a `train_*.py` entrypoint:
   - [mamba_mv/](imts_benchmark/mamba_mv/) — the paper's target model. Pipeline is `PerVariateIrregularSSM → SharedGridAligner → L × (VariableAxisAttention + TemporalMambaOnGrid) → QueryReadout`. `dt_mode ∈ {learned, replace, concat}`; `additive` was dropped after Phase 2. Has both `train_mv.py` (forecasting) and `train_cls.py` (UEA classification — replaces query readout with HAN-style hierarchical attention pool).
   - [mamba_pretrain/](imts_benchmark/mamba_pretrain/) — **vendored pretrain-branch sibling** of `mamba_mv`. Same encoder, different downstream wiring (`multivariate_forecaster.py`, `multivariate_classifier.py`, `train_mv.py`, `train_cls.py`). Used for the pretrain-arch PhysioNet HPO/confirm chain and the pretrain UEA classification chain. Treat as a frozen mirror — when `mamba_mv` is updated, decide explicitly whether to mirror the change here.
   - [mamba_rope/](imts_benchmark/mamba_rope/) — hybrid: per-variate Mamba SSM front-end + axial-RoPE attention back-end. Same `add_fair_args` CLI; sbatches live alongside the others (`run_mamba_rope_hpo_{physionet,ushcn}.sbatch`, `run_mamba_rope_p10_ushcn.sbatch`).
   - [s5_forecaster/](imts_benchmark/s5_forecaster/) — vendored upstream S5 (`_upstream/`, gitignored, read-only).
   - [romae_forecaster/](imts_benchmark/romae_forecaster/) — uses an external editable install of `Chromeilion/RoMAE` (cloned to `${ROMAE_DIR}` by the build scripts; `_romae_repo/` is gitignored).
   - mTAN and ContiFormer are present in tree but **dropped** from active runs (collapse / OOM — see [docs/REPORT_consolidated_2026-04-22.md](imts_benchmark/docs/REPORT_consolidated_2026-04-22.md) §1.3).

3. **Shared infrastructure** all three trainers import:
   - [shared_config/fair_defaults.py](imts_benchmark/shared_config/fair_defaults.py) — `add_fair_args(parser)` defines the canonical CLI surface (`--regime`, `--seed`, `--output_dir`, `--data_root`, plus all training knobs). **Any change here affects all 3 trainers.** Also exposes `apply_auto_meta(args)` and `derive_phase_tags(args)`.
   - [shared_config/global_metrics.py](imts_benchmark/shared_config/global_metrics.py) — target-weighted aggregation primitives the aggregator depends on.
   - [shared_config/wandb_lightning.py](imts_benchmark/shared_config/wandb_lightning.py) — `--use_wandb` plumbing.
   - [shared_data/multivariate_datamodule.py](imts_benchmark/shared_data/multivariate_datamodule.py) — single Lightning DataModule. `format="per_variate"` for Mamba-MV; `format="flat_tokens"` for RoMAE. Pred mask is `valid & (timestamp >= history)`.
   - [shared_data/uea_classification_datamodule.py](imts_benchmark/shared_data/uea_classification_datamodule.py) — UEA classification DataModule. Reads from `data_uea/<Dataset>/`, applies per-sample z-score, then deterministic Kidger 30% sync drop keyed by `(sample_idx, drop_seed)`. Stratified 80/20 val split with `split_seed=42` fixed across HPO and final eval so val identity is stable. Class weights are inverse-frequency on the kept training split, normalized to sum to `C`.

The eval surface ([imts_benchmark/eval/](imts_benchmark/eval/)) is large (~30 files); the canonical entry points:
- `aggregate_results.py` — walks `<root>/<model>/<regime>/<variant>/seed<S>/test_metrics.csv` + `per_sample.jsonl`. Prefers target-weighted per-variate fields (`ss_res_v{d}`, `abs_err_sum_v{d}`, `count_v{d}`) added 2026-04-22 — Phase-2 runs predate these, so the aggregator falls back to sample-averaged `mse_v{d}` / `mae_v{d}`.
- `aggregate_phase5_v2.py` — real-dataset variant; emits both target-weighted *and* T-PatchGNN-style variable-averaged metrics from `mse_tpg`/`mae_tpg` columns. Compares against T-PatchGNN paper anchors.
- `aggregate_phase6_imm.py` — TIME-IMM variant; aggregates per-dataset confirm results across the 8 IMM datasets and compares against IMM-TSF paper anchors.
- `aggregate_hpo_winners_physionet.py` / `aggregate_hpo_winners_real.py` / `aggregate_hpo_winners_imm.py` / `aggregate_immtsf_hpo_winners.py` — pick val/MSE-best `(lr, eff_bs)` per `dt_mode` from the 27-cell sweep and write the JSON consumed by the corresponding confirm sbatch. The IMM variants pick winners per-dataset.
- `aggregate_uea_hpo.py` — UEA classification variant; picks val-acc (or val-macro-F1) best `(lr, batch_size)` per `dt_mode` per dataset and writes winners JSON consumed by `run_uea_cls_final*.sbatch`. Plus `plot_uea_cls_curves.py` / `uea_class_balance.py` / `uea_confusion.py` for diagnostics.

## Common commands

### Training (single run)

```bash
python -m imts_benchmark.mamba_mv.train_mv \
    --data_root <data_root> --regime <regime> --seed <s> --output_dir <dir> \
    --dt_mode {learned,replace,concat}
python -m imts_benchmark.s5_forecaster.train_s5    --data_root ... --regime ... --seed ... --output_dir ...
python -m imts_benchmark.romae_forecaster.train_romae --data_root ... --regime ... --seed ... --output_dir ...
```

For real datasets (`physionet` / `activity` / `ushcn`), pass `--auto_meta`. This reads `{data_root}/{regime}/norm_stats.json` and overrides `n_vars` / `history` / `t_max` (mapped from `time_max`). Without it, the trainer uses the synthetic defaults (V=3, history=8, t_max=10).

### Aggregation

```bash
python imts_benchmark/eval/aggregate_results.py --log_root <output_root>
python imts_benchmark/eval/aggregate_phase5_v2.py            # paths hardcoded
python -m imts_benchmark.eval.aggregate_hpo_winners_physionet --hpo_log_dir <hpo_dir>
```

### SLURM submission (real-data chains on Delta)

Submit scripts encode the canonical dependency chain — `smoke → sweep (afterok) → aggregator (afterany) → confirm (afterok)`. The aggregator uses `afterany` deliberately so partial-failure sweeps still produce winners as long as each `dt_mode` has at least one usable cell.

- **PhysioNet (Phase 5)**:
  - `bash imts_benchmark/scripts/submit_physionet_delta.sh` — smoke (1) → S5 (5) + RoMAE (5) + Mamba-MV (15), all `afterok:smoke`.
  - `SMOKE_ID=<jobid> bash imts_benchmark/scripts/submit_hpo_physionet_delta.sh` — sweep (27) → aggregator → confirm (15).
- **MIMIC**: `bash imts_benchmark/scripts/submit_hpo_mimic_delta.sh` — same shape as PhysioNet HPO; `SKIP_SMOKE=1` to skip the smoke gate after the first successful run.
- **TIME-IMM (Phase 6)** — per-dataset, takes a list of dataset names, defaults to all 8 public IMM datasets:
  - `bash imts_benchmark/scripts/submit_imm_delta.sh [imm_xxx ...]` — full HPO chain per dataset (sweep → aggregator → confirm).
  - `bash imts_benchmark/scripts/submit_imm_matched_delta.sh [imm_xxx ...]` — single-seed run per dataset under the **IMM-TSF baseline protocol** (no HPO, matches `_imm_tsf_repo/main_all.py` exactly so numbers compare directly to the IMM-TSF paper Tables 3–11 "Without Textual Data" rows). Override via `DT_MODE=` (default `concat`) and `SEED=` (default 1). See [feedback memory on baseline-protocol matching](../../u/seojininus/.claude/projects/-projects-bfrf-seojininus-ssm/memory/feedback_baseline_protocol_match.md) — matched-protocol runs are the head-to-head comparison; HPO-of-baselines is deferred.
- **UEA classification add-on** — versioned chain submitters (v2 / v3 / v3_1 / pretrain) under both supervised (`mamba_mv`) and pretrain-arch (`mamba_pretrain`) entrypoints. The `delta_x86` variants are the active ones; H200 is preferred:
  - `bash imts_benchmark/scripts/submit_uea_cls_pipeline_v3_delta_x86.sh` — supervised chain (smoke → HPO → aggregator → final).
  - `bash imts_benchmark/scripts/submit_uea_cls_pretrain_pipeline_delta_x86.sh` / `submit_uea_cls_pretrain_70d_pipeline_delta_x86.sh` — pretrain-arch chain (30-dataset and 70-dataset variants). The `30d_size` script ([submit_uea_cls_30d_size_pipeline_delta_x86.sh](imts_benchmark/scripts/submit_uea_cls_30d_size_pipeline_delta_x86.sh)) runs the size-axis ablation (small vs base, 90 cells under RoMAE Table 4 protocol).
  - Smoke gates: `run_uea_cls_30d_highv_smoke_delta_x86.sbatch`, `run_uea_cls_pretrain_30d_size_smoke_delta_x86.sbatch`, `run_uea_cls_pretrain_70d_smoke_delta_x86.sbatch`. Pass before submitting full chains.
- **PhysioNet pretrain-arch (alternate to Phase 5 supervised)** — `bash imts_benchmark/scripts/submit_hpo_pretrain_physionet_delta.sh` (or `_delta_ai.sh` for DeltaAI) — same smoke→sweep→aggregator→confirm shape, but routes through `mamba_pretrain` instead of `mamba_mv`.

All submit scripts support `DRY_RUN=1`. Match sbatches by suffix: `_delta` = DeltaAI (aarch64/GH200), `_delta_x86` = Delta (x86_64/A100/A40/H200); plain (no suffix) = Quest. Many `*_delta_x86.sbatch.bak` files are last-known-good before edits — keep them.

### Smoke tests before long arrays

- `sbatch smoke_test_delta_x86.sbatch` — 2-epoch Mamba-MV on synthetic async, Delta x86 env.
- `sbatch imts_benchmark/scripts/run_physionet_smoke_delta.sbatch` — 2 epochs of all 3 models on PhysioNet, validates `test_mse_tpg` column emission. Exit 0 = pass; chain real-data arrays on this jobid.
- `sbatch imts_benchmark/scripts/run_mimic_smoke_delta_x86.sbatch` — same idea for MIMIC.
- `sbatch imts_benchmark/scripts/smoke_test_imm_fnspid_delta_x86.sbatch` — 2-epoch IMM smoke (uses `imm_fnspid` as the canary dataset); pass before submitting the 8-dataset Phase-6 chain.

### Environments

Three env-build scripts at the repo root, one per cluster:
- `bash build_env.sh` — Quest (synthetic phases). Env: `/projects/b1094/StarEmbed/pythonenvs/mamba`.
- `bash build_env_delta.sh` — DeltaAI (GH200, aarch64, sm_90). Env: `/projects/bfrf/seojininus/envs/mamba`.
- `bash build_env_delta_x86.sh` — Delta non-AI (x86_64). Env: `/projects/bfrf/seojininus/envs/mamba_x86`. Build script targets `TORCH_CUDA_ARCH_LIST=8.0;8.6` (A100/A40), but the resulting env *also runs on the x86 H200 partition* (`gpuH200x8`): PyTorch's prebuilt wheel covers sm_90 and the source-built `causal-conv1d`/`mamba-ssm` JIT-PTX onto sm_90 at first launch. Active PhysioNet/IMM training and HPO chains all run on `gpuH200x8`; A100 is fine but tends to have longer queues, so prefer H200 unless an sbatch has a specific reason not to.

Both build PyTorch 2.5.1+cu124, then `causal-conv1d 1.6.1` and `mamba-ssm 2.3.1` from source (this is the slow part — `MAX_JOBS=8` by default), then `pip install -r requirements_delta.txt`, then editable RoMAE pinned at SHA `480cfaf80cf1f0630774998cb5c595ee684007e2`. Set `RECREATE_ENV=1` to wipe; `RUN_GPU_VERIFY=1` runs an inline Mamba forward pass (only inside a GPU allocation). Use `delta_x86_preflight.sh` on a Delta login node before first build to sanity-check accounts/modules/CUDA. The `s5-jax` env referenced in older docs is **defunct** — current S5 runs in PyTorch out of the same env as Mamba/RoMAE.

## Critical conventions

- **`add_fair_args` is the contract.** All three trainers parse the same flags. When you add a knob meant to be uniform across models, add it there, not to one trainer.
- **Output dir layout drives the aggregator.** Keep `<root>/<model>/<regime>/<variant>/seed<S>/` exactly. The aggregator infers the schema from the last 5 path parts.
- **Vendored upstream paths are gitignored and read-only**: `imts_benchmark/*/_upstream/` (S5), `_romae_repo/` (RoMAE clone), and `_imm_tsf_repo/` + `_time_imm_repo/` (IMM-TSF baseline + TIME-IMM data tooling for Phase 6). Don't edit; rebuild / re-clone via the env scripts. The matched-protocol IMM sbatch reads its hyperparameters out of `_imm_tsf_repo/main_all.py` — when that upstream changes, re-derive the matched-protocol args rather than hand-editing the sbatch.
- **Phase 4 datasets carry `gap_starts` / `gap_ends` / `gapped_variate_index`** as Arrow columns; Phase 2/3 datasets carry empty lists for the same fields. The aggregator only computes `in_gap` / `out_gap` splits when those columns are non-empty.
- **WANDB_DIR + per-run output_dir are tied.** Sbatches set `export WANDB_DIR="${OUTPUT_DIR}"` so offline-mode artifacts land beside the run's checkpoints. If you change one, change the other.
- **Effective batch is what's compared, not physical batch.** Models that OOM at the shared `train_batch_size=128` drop physical batch and bump `--accumulate_grad_batches`. PhysioNet eff-bs grid in the HPO sweep: `64=(16,4)`, `128=(32,4)`, `256=(32,8)`.
- **Two log roots, two clusters.** Quest writes under `/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/` (synthetic phases). Delta writes under `/projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/` (real datasets). The synthetic Quest tree is the historical record cited in the paper; the Delta tree is current work.
- **Don't reactivate `additive` `dt_mode`** — it's a strict worst-of-both per Phase-2 diagnosis. `learned` / `replace` / `concat` are the supported set.
- **UEA per-dataset defaults are sentinel-driven.** `train_cls.py` carries a `ROMAE_DATASET_DEFAULTS` table inheriting RoMAE Table 12 / App. C.2 values for `batch_size`, `label_smoothing`, `grad_clip`, `grid_K`, `head_dropout`. Sbatches pass `-1` as a sentinel to mean "use the dataset default"; any non-`-1` CLI value overrides. RoMAE's `confidence_c` convention is converted to PyTorch's `label_smoothing = 1 - c` here — don't double-convert.
- **V-axis attention is O(V²).** High-V UEA datasets (Heartbeat V=61, etc.) need physical `batch_size=1` plus `--accumulate_grad_batches 16` to stay within memory while preserving effective batch. Effective-batch is what's compared.
- **SGD-tuned LRs don't transfer to AdamW.** RoMAE Table 12 LRs are SGD-magnitudes; using them directly under AdamW caused divergence in UEA round 1. See [feedback memory](../../u/seojininus/.claude/projects/-projects-bfrf-seojininus-ssm/memory/feedback_optimizer_lr_dont_transfer.md) — re-derive the LR grid for the optimizer actually in use.
