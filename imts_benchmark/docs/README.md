# IMTS Benchmark — Docs map

## Current (active)

- **[`REPORT_consolidated_2026-04-22.md`](REPORT_consolidated_2026-04-22.md)** — single source of truth for paper writing. Covers Phases 2 / 3 / 4-1 / 4-2, baseline wrappers (Mamba-MV, S5, RoMAE), mTAN drop, ContiFormer deferral, fairness recipe, evaluation methodology, SLURM infrastructure. Last revised 2026-04-23.
- **[`RESULTS_phase2.md`](RESULTS_phase2.md)** — Phase 2 (sync dense) final table. Mamba dt_mode ablation, paper-native baselines, patience=50.
- **[`HPO_PLAN.md`](HPO_PLAN.md)** — rigorous 3-stage HPO design. **Gated on Phase 3/4 outcome** — only triggered if Mamba `replace` loses to S5 on the majority of irregular regimes in Phase 3/4.

## Baseline specs (active — live alongside wrapper code)

Under `../audit/`, not here — per-model spec sheets that track the wrapper code:

- `../audit/baseline_spec_romae.md` — **GREEN** (paper-native RoPE-2D MAE)
- `../audit/baseline_spec_s5.md` — **GREEN** after paper-native switch (d_model=128, state_dim=256, lr=1e-3, wd=0.05, SSM-spectral params excluded from WD)
- `../audit/baseline_spec_mtan.md` — **DROPPED** (persistent collapse across 5 configs; wrapper code preserved)
- `../audit/baseline_spec_contiformer.md` — **DEFERRED** (6-config OOM)
- `../audit/contiformer_deferral_note.md` — 6-config OOM ledger + T-PATCHGNN precedent
- `../audit/baseline_validation_report.md` — upstream-contract / smoke / cross-init parity checks

## Phase-result docs

- `RESULTS_phase2.md` — done
- `RESULTS_phase3.md` — pending (async dense training)
- `RESULTS_phase4_1.md` — pending (gap on v3)
- `RESULTS_phase4_2.md` — pending (gap on random variate)

## Archive (superseded)

Old reports kept for provenance but not cited in the paper. All content still relevant has been merged into `REPORT_consolidated_2026-04-22.md`:

- `archive/2026-04-20_BUILD_REPORT_mamba_vs_romae.md`
- `archive/2026-04-20_RESULTS_mamba_vs_romae.md`
- `archive/2026-04-21_BUILD_REPORT_imts_benchmark.md`
- `archive/2026-04-21_RESULTS_imts_benchmark.md`
