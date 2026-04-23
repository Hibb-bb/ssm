# IMTS Benchmark — Docs map

## Current (active)

- **[`REPORT_consolidated_2026-04-22.md`](REPORT_consolidated_2026-04-22.md)** — the single source of truth for paper writing. Covers Phases 2–4, baseline wrappers, fairness recipe, evaluation methodology, ContiFormer deferral, SLURM infrastructure.

## Baseline specs (active — live alongside wrapper code)

These are under `../audit/`, not here, because they're per-model spec sheets that track the wrapper code:

- `../audit/baseline_spec_romae.md` — GREEN
- `../audit/baseline_spec_mtan.md` — GREEN
- `../audit/baseline_spec_s5.md` — YELLOW (port caveats)
- `../audit/baseline_spec_contiformer.md` — YELLOW → deferred
- `../audit/contiformer_deferral_note.md` — 6-config OOM ledger
- `../audit/baseline_validation_report.md` — 5 models × 4 checks

## Archive (superseded)

Old reports kept for provenance but not cited in the paper. All content that's still relevant has been merged into `REPORT_consolidated_2026-04-22.md`:

- `archive/2026-04-20_BUILD_REPORT_mamba_vs_romae.md`
- `archive/2026-04-20_RESULTS_mamba_vs_romae.md`
- `archive/2026-04-21_BUILD_REPORT_imts_benchmark.md`
- `archive/2026-04-21_RESULTS_imts_benchmark.md`

## Phase-result docs (to be written)

Planned once each phase finishes its full run:

- `RESULTS_phase2.md` — sync dense
- `RESULTS_phase3.md` — async dense
- `RESULTS_phase4.md` — async + gap
