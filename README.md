# Sweep Results — Mamba-MV (IMM-TSF + Activity/USHCN)

Backup of HPO and confirm-stage numerical metrics from the IMM-TSF / Activity /
USHCN benchmarking sweeps, plus the pretrain branch's IMM-TSF re-runs at
`d_model = d_hidden ∈ {384, 64}`.

## Layout

All files live under `output/<sweep_name>/...`. The relevant file types are:

- `test_metrics.csv` — per-run test MSE/MAE.
- `*summary*.csv` — aggregated per-(dataset, dt_mode, lr, eff_bs, seed) tables.
- `winners_*.json` — HPO winners selected by val MSE.
- `*hpo*.csv` / `*hpo*.json` — HPO ranking tables and configs.

## Sweep groups

- `mamba_mv_confirm_imm_oldmodel/` — original mamba-sinusoidal-multivariate
  branch model on the 7 IMM-TSF datasets (3-seed confirm).
- `mamba_mv_confirm_imm_d64/` — pretrain-branch architecture (any-variate
  attention + asinh) at `d=64` (~259K params), 3-seed confirm on IMM-TSF.
- `mamba_mv_confirm_imm/` — pretrain-branch arch at `d=384`.
- `mamba_mv_confirm_real_d64/` — pretrain-branch arch at `d=64` on
  Activity/USHCN, 3-seed confirm.
- `mamba_mv_confirm_real_d384/` — pretrain-branch arch at `d=384` on
  Activity/USHCN.
- `mamba_mv_best_confirm/` — earlier matched-protocol confirm runs.

Other top-level dirs (logs, screen logs, etc.) are dropped from this branch;
this branch carries metrics only.

## Companion artifacts

Model checkpoints (best-* and last.ckpt) are stored separately in the private
Hugging Face repo `123anonymous123/mamba-mv-checkpoints`.
