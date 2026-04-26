# IMTS Benchmark — Docs map

## Current (active)

- **[`REPORT_consolidated_2026-04-22.md`](REPORT_consolidated_2026-04-22.md)** — single source of truth for paper writing. Covers Phases 2 / 3 / 4-1 / 4-2, baseline wrappers (Mamba-MV, S5, RoMAE), mTAN drop, ContiFormer deferral, fairness recipe, evaluation methodology, SLURM infrastructure. See the **Addendum 2026-04-24** at the bottom for Audit C, HPO submission, `concat` dt_mode, and per-phase winner strategy.
- **[`HPO_PLAN.md`](HPO_PLAN.md)** — v2 (mechanism-informed, 2026-04-23). Supersedes v1 (archived). **HPO pipeline submitted 2026-04-24** (6-job SLURM dependency chain, results pending).
- **[`../audit/AUDIT_C_gamma_rho.md`](../audit/AUDIT_C_gamma_rho.md)** — Empirical refutation of the "shared-grid decay is the bottleneck" hypothesis. Trained `γ` stays at init; implied decay ≈ 0.997. Bottleneck attribution shifts to per-variate SSM capacity.
- **[`../knowledge/README.md`](../knowledge/README.md)** — reference papers + summary stubs (S5, RoMAE, Mamba, our draft). Use these to ground paper citations.

## Baseline specs (active)

Under `../audit/`, tracking the wrapper code that's still in use:

- `../audit/baseline_spec_romae.md` — **GREEN** (paper-native RoPE-2D MAE; ~7.8M params)
- `../audit/baseline_spec_s5.md` — **GREEN** (paper-native d_model=128, state_dim=256; ~1.4M params). **Updated 2026-04-25** with: kernel-vs-S4 verification, channel-independence justification, IMTS adaptation rationale, and a paper-ready citation block.

Specs for dropped baselines moved to `../audit/archive/` (mTAN, ContiFormer).

## Phase-result docs

- `RESULTS_phase2.md` — **done** (sync dense; 120 runs)
- `RESULTS_phase3.md` — **done** (async dense; 80 runs; HPO gate triggered)
- `RESULTS_phase4_1.md` — **done** (gap on v3 fixed; 80 runs; HPO gate triggered 2/3)
- `RESULTS_phase4_2.md` — **done** (gap on random variate; 80 runs; Mamba-MV `learned` wins high_irreg)
- **[`RESULTS_phase5_real.md`](RESULTS_phase5_real.md)** — **active** (real T-PatchGNN datasets: Activity, USHCN; p10 reruns submitted 2026-04-25 as jobs 6384307/6384308/6384309). Includes T-PatchGNN Table 1 comparison context, parameter ratios (~47×), and decision branches (shrink Mamba-MV vs other dt_modes vs ship as-is).

## Rendered assets

- `phase2_final/` — Phase 2 final plots (all 4 models)
- `phase3_results/` — Phase 3 plots (3 active models)
- `phase4_1_results/` — Phase 4-1 plots (+ gap-variate out-of-gap MSE)
- `phase4_2_results/` — Phase 4-2 plots (+ gap-variate out-of-gap MSE per variate)

## HPO pipeline (submitted 2026-04-24)

Status: running. Expected output artifacts:

- `HPO_COMPARISON_phase3_high_irreg.md` — tuned Mamba-MV vs S5/RoMAE baseline (pending)
- `HPO_COMPARISON_phase4_2_high_irreg.md` — same for Phase 4-2 (pending)
- `output/log/.../hpo_mamba_mv/phase{3,4_2}/winner_config.json` — chosen `(dt_mode, lr, bs)` per phase
- New rows `variant=hpo_tuned_{dt_mode}` appended to `phase{3,4_2}_summary_wide.csv`

See `imts_benchmark/scripts/submit_full_hpo_pipeline.sh` for the submission graph.

## Archive (superseded)

Old reports kept for provenance but not cited in the paper. All content still relevant has been merged into `REPORT_consolidated_2026-04-22.md` (+ its 2026-04-24 addendum).

**Report-level archive** (`archive/`):
- `archive/2026-04-20_BUILD_REPORT_mamba_vs_romae.md`
- `archive/2026-04-20_RESULTS_mamba_vs_romae.md`
- `archive/2026-04-20_mamba_vs_romae_plan.md` — pre-benchmark planning
- `archive/2026-04-20_synthetic_sin_plan.md` — data-gen planning (data now built)
- `archive/2026-04-21_BUILD_REPORT_imts_benchmark.md`
- `archive/2026-04-21_RESULTS_imts_benchmark.md`
- `archive/2026-04-21_imts_benchmark_plan.md` — original benchmark plan
- `archive/HPO_PLAN_v1_2026-04-22.md` — superseded by current HPO_PLAN.md (v2)

**Audit-level archive** (`../audit/archive/`):
- `../audit/archive/baseline_spec_mtan.md` — mTAN dropped after Phase-2 collapse
- `../audit/archive/baseline_spec_contiformer.md` — ContiFormer deferred after 6-config OOM
- `../audit/archive/contiformer_deferral_note.md` — decision ledger
