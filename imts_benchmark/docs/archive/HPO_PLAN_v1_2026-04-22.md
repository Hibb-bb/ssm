# Mamba-MV HPO Plan — Contingent (run only if Phase 3/4 at current config doesn't beat S5)

**Status (2026-04-22):** **POSTPONED.** Decision gate: launch Phase 3 + Phase 4 at the current Mamba config first. Only run this HPO if Mamba-MV `replace` still loses to S5 on a majority of irregular regimes after Phase 3/4 evaluation.

**Current Mamba-MV config (frozen until HPO triggered):**
- `d_model=256, d_hidden=256, n_perv_layer=3, n_fusion_blocks=3, n_heads_varattn=4`
- `d_state=16, d_conv=4, expand=2, grid_K=128`
- `lr=5e-4, weight_decay=0.01, dropout=0`
- `warmup_steps=100, num_training_steps=1600, patience=50, batch_size=128`
- `dt_mode ∈ {learned, replace}` only (`additive` dropped — scientifically muddies the binary hypothesis test)
- 7.79 M trainable params

**Scientific question driving HPO**: if `replace` still loses to S5 at this config, is it because (a) our architecture is fundamentally weaker for this task, or (b) our chosen HPs are suboptimal? HPO isolates (b).

---

## HPO protocol (if triggered)

### Stage 1 — Primary HPO grid (90 runs)

Joint sweep on `data_correct_async/multisin_med_irreg` (Phase 3 canonical IMTS regime). Selection on val/MSE only; test set never touched.

| Axis | Values | Justification |
|---|---|---|
| `lr` | {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} | Log-spaced over 1.5 orders of magnitude. Covers Mamba paper default (5e-4) and S5's winning LR (1e-3), plus safety margins. LR is the single knob most likely to close the Mamba–S5 gap. |
| `d_model` | {128, 192, 256} | 128 ≈ 1.4M params (matches S5), 256 ≈ 7.8M (current), 192 ≈ 4M (midpoint). Addresses the "Mamba only wins because it's bigger" reviewer concern. |
| `dt_mode` | {learned, replace} | Joint-sweep means each dt_mode gets its own best HPs. Stronger statement than "we tuned replace and applied same HPs to learned." |

Fixed (Mamba paper defaults): `d_state=16, d_conv=4, expand=2, dropout=0`. Shared recipe: `wd=0.01, warmup=100, cosine=1600, patience=50, batch=128, fp32`.

Grid: 5 × 3 × 2 × 3 seeds = **90 runs** (≈3 hours SLURM parallel).

**Selection**: per dt_mode, find `(lr*, d_model*)` with lowest mean val/MSE over 3 seeds.
**Tiebreakers** (within 1σ of winner): prefer smaller `d_model` → higher `lr` → lower `wd`.

### Stage 2 — Architectural sensitivity (27 runs)

One-axis-at-a-time at Stage 1 winner `(lr*, d_model*, dt_mode=replace)`. Used to confirm winner is robust; update only if non-default wins by ≥5% val/MSE.

| Axis | Values | Why |
|---|---|---|
| `grid_K` | {64, 128, 256} | Cross-variate alignment layer resolution — core to our architecture |
| `n_perv_layer` | {2, 3, 4} | Per-variate SSM depth |
| `n_fusion_blocks` | {2, 3, 4} | Cross-variate fusion depth — novel component |

Grid: 3 + 3 + 3 = 9 cells × 3 seeds = **27 runs**.

### Stage 3 — Phase 3 main grid at frozen config (40 runs)

`dt_mode ∈ {learned, replace}` × 4 regimes × 5 seeds, each dt_mode using its own Stage-1+2 winning `(lr, d_model)`. On `data_correct_async/`.

### Stage 4 — Phase 4 main grid (40 runs)

Same structure, on `data_correct_gap/`.

### Baselines (no HPO, unchanged from Phase 2)

S5 + RoMAE at paper-native configs × 4 regimes × 5 seeds × 2 phases = 80 runs. mTAN excluded (consistent mean-prediction collapse — see Methods paragraph below).

### Total if HPO triggered

- Stage 1 + 2: 117 runs (HPO itself)
- Stages 3 + 4: 80 runs (Mamba main, both phases)
- Baselines: 80 runs
- **Grand total: 277 runs** (~8 hours SLURM parallel)

---

## What we explicitly decided NOT to sweep and why

| Cut axis | Fixed value | Justification |
|---|---|---|
| `weight_decay` {0.01, 0.05} | `0.01` | Mamba paper default. S5's `0.05` is specific to S5's HiPPO init (which we correctly exclude from decay). No paper or empirical signal it helps Mamba. |
| `dropout` {0.0, 0.1} | `0.0` | Mamba paper uses `0.0` everywhere (Section E.2.1). Dataset small (1000 train), overfitting risk mild. |
| `warmup_steps` {100, 500} | `100` | Shared recipe convention; no signal it's the bottleneck. |
| `expand` {1, 2, 4} | `2` | Mamba paper Section 3.4: "We always fix E=2 in our experiments." Non-negotiable. |
| `additive` dt_mode | dropped | Scientifically muddies the `learned vs replace` binary test. The paper's claim is cleaner as a 2-variant comparison. |
| Phase 2 re-run at post-HPO config | skipped | Phase 2 results at old config already reported in `phase2_final/`. Re-running adds cost without changing scientific conclusion. |
| Appendix scaling ablation (finer `d_model` sweep) | covered by Stage 1 | Stage 1 already has 3 `d_model` values. |
| Baseline HPO | none | S5, RoMAE at paper-native per T-PATCHGNN (ICML 2024) convention. Doing baseline HPO would violate the "each model at its paper config" fairness principle. |

---

## Methods paragraph draft (for paper, if HPO is run)

> *"Mamba-MV hyperparameters were selected by a joint grid search on the Phase 3 (async dense) validation set at the medium-irregularity regime. Primary grid: learning rate ∈ {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} × model dimension ∈ {128, 192, 256} × dt_mode ∈ {learned, replace}, with 3 seeds per cell (90 configurations). Secondary architectural sweeps on `grid_K`, `n_perv_layer`, and `n_fusion_blocks` were performed at the primary winner (27 runs). Per-dt_mode winners were applied to all Phase 3 and Phase 4 main-grid experiments (5 seeds, all 4 irregularity regimes). Mamba paper defaults were retained for `d_state=16, d_conv=4, expand=2, dropout=0`. Baselines (S5, RoMAE) used their paper-native configurations per the T-PATCHGNN (ICML 2024) evaluation convention; no baseline HPO was performed. mTAN was evaluated under its paper-native config but exhibited consistent mean-prediction collapse across all seeds and regimes, and is excluded from the main comparison (see Appendix X)."*

---

## Decision gate: run this HPO if…

After Phase 3 main at current Mamba config completes, check `phase3_summary_wide.csv` for Mamba-MV (both dt_modes) vs S5:

- **Skip HPO** if: Mamba `replace` mean MSE ≤ S5 mean MSE on ≥3 of 4 Phase 3 regimes, AND the one losing regime has ≤20% gap.
- **Trigger HPO** if: Mamba `replace` loses on ≥2 Phase 3 regimes by >20% MSE, OR Mamba `learned` beats `replace` on ≥2 regimes (suggesting true-Δt doesn't help at current HPs).

Rationale: if our method is already competitive at an untuned config, HPO is a defensive addition. If it's losing decisively, HPO is necessary to distinguish "architecture problem" from "HP problem."
