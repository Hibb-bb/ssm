# Mamba-MV HPO Plan — v2 (mechanism-informed, 2026-04-23)

**Status**: ACTIVE. Supersedes `archive/HPO_PLAN_v1_2026-04-22.md`.

**What changed vs v1 and why**:

- v1 assumed the shared-grid `exp(−γ·ρ)` decay was the main bottleneck and proposed sweeping `grid_K ∈ {64, 128, 256, 512}` as the primary axis. **[Audit C](../audit/AUDIT_C_gamma_rho.md) refutes this**: trained `γ` stays at init in all 5 seeds, implied decay ≈ 0.997 (near-identity), 0% of slot-channel pairs show severe decay. Grid resolution is not the mechanism.
- v1's scientific target was "beat S5 on Phase 3". The cross-phase results show this is the wrong target: Mamba-MV already wins Phase 4-2 × high_irreg (0.02865 vs S5 0.03186), and loses Phase 3 by a narrowing margin as irregularity increases. The right target is a **Pareto-dominant config**: competitive on Phase 3 AND preserving the Phase 4-2 advantage.
- v1 did not address the 7.8M vs 1.4M parameter disparity with S5. Reviewers will flag this. v2 adds a size-matched diagnostic stage.
- [Audit B](../audit/) finding: `dt_mode=replace` silently disables Mamba's selective mechanism. v2 keeps the `learned` vs `replace` comparison but adds a note about reframing `replace` in the paper.

**Current Mamba-MV config (baseline, frozen at what produced the 140-run benchmark)**:
- `d_model=256, d_hidden=256, n_perv_layer=3, n_fusion_blocks=3, n_heads_varattn=4`
- `d_state=16, d_conv=4, expand=2, grid_K=128`
- `lr=5e-4, weight_decay=0.01, dropout=0`
- 7.79M trainable params
- `dt_mode ∈ {learned, replace}` (`additive` dropped, see v1)

**Target regime**: all HPO runs on `data_correct_async/multisin_high_irreg` (user-confirmed scope). Extensions to other regimes only at Stage 3 winners.

---

## Scientific question hierarchy (drives stage structure)

1. **Q1**: Is Mamba-MV's Phase 3 × high_irreg loss an HP problem or an architecture problem? → Stage 1 HPO grid.
2. **Q2**: Does any single config Pareto-dominate on (Phase 3, Phase 4-2)? → Stage 2 Pareto test.
3. **Q3**: Is the param-count gap (7.8M vs 1.4M) confounding the comparison? → Stage 3 size-matched diagnostic.
4. **Q4**: Would replacing Mamba's Stage-1 SSM with an S5 block close the Phase 3 gap? → Stage 4 mechanism isolation. (Directly tests Audit C's conclusion.)
5. **Q5** (contingent): Does a `multiplicative` dt_mode unify the Phase 3 and Phase 4 regimes? → Stage 5, only if Stage 2 has no Pareto winner.

---

## Stage 1 — Primary HPO grid (54 runs, ≈2h SLURM parallel)

Joint sweep on `data_correct_async/multisin_high_irreg`. Selection on val MSE only; test set never touched.

| Axis | Values | Justification |
|---|---|---|
| `lr` | {3e-4, 1e-3, 3e-3} | Covers Mamba default (5e-4, approximated by 3e-4) and S5's winning LR (1e-3). Log-spaced. |
| `d_state` | {16, 64, 256} | **New primary axis.** 16 = current, 256 = S5-matched. Audit C points to per-variate SSM capacity as the real bottleneck. |
| `dt_mode` | {learned, replace} | Keeps the core architectural test. Joint-sweep means each dt_mode gets its own best (lr, d_state). |

Fixed (Mamba paper defaults): `d_conv=4, expand=2, dropout=0`. Fixed architecturally: `d_model=256, n_perv_layer=3, n_fusion_blocks=3, grid_K=128`. Shared recipe: `wd=0.01, warmup=100, cosine=1600, patience=50, batch=128, fp32`. Seeds: 3.

Grid: 3 × 3 × 2 × 3 seeds = **54 runs**.

**Selection**: per `dt_mode`, pick `(lr*, d_state*)` with lowest mean val MSE across 3 seeds. Tiebreakers within 1σ of winner: smaller `d_state` → higher `lr`.

**What Q1 "HP vs architecture" looks like after Stage 1**:
- If best Stage-1 config reduces Phase 3 MSE from 0.0227 toward S5's 0.0176 by ≥30%: HP problem, HPO was necessary.
- If best Stage-1 config plateaus above 0.020: architecture problem, Stage 4 becomes the key experiment.

---

## Stage 2 — Pareto dominance test (10 runs, ≈1h)

Take the 2 Stage-1 winners (one per `dt_mode`). Run each on `data_correct_gap_random/multisin_high_irreg` × 5 seeds = **10 runs**.

**Decision matrix**:

| Outcome | Interpretation |
|---|---|
| Both configs: Phase 3 MSE ≤ 0.020 AND Phase 4-2 MSE ≤ 0.029 | Clean Pareto winner; use the `dt_mode` with the better median across both phases. |
| `learned` Pareto-dominates | Paper should drop `replace` and report only `learned` (per Audit B). |
| `replace` Pareto-dominates | Paper can use `replace`, but reframe as "continuous-time SSM ablation". |
| Neither Pareto-dominates (each wins one phase) | Trigger Stage 5 (`multiplicative`). |

---

## Stage 3 — Size-matched diagnostic (20 runs, ≈1h)

Addresses the 7.8M vs 1.4M reviewer concern. Shrink Mamba-MV to ~1.4M params:
- `d_model=128` (halves d_model)
- `n_perv_layer=2` (−1 layer from baseline 3)
- `n_fusion_blocks=2` (−1 block)
- all other HPs = Stage 1 winner (per `dt_mode`)

Run on all 4 phases × high_irreg × 5 seeds per `dt_mode` winner = **20 runs** (or 40 if both `dt_modes` are retained).

**Interpretation**:
- Phase ordering preserved (size-matched Mamba-MV loses Phase 3 but wins Phase 4-2): param count is not confounding; the architectural story holds at matched scale.
- Phase ordering breaks (size-matched loses Phase 4-2 too): the Phase 4-2 win is partially from capacity, not just architecture. Requires careful paper language.

---

## Stage 4 — Mechanism isolation: replace Stage-1 with an S5 block (5 runs, ≈3h due to S5's 3.5× wall-clock)

This directly tests Audit C's conclusion that the per-variate SSM is the bottleneck. Replace the `MambaIrregularBlock` stack in [irregular_ssm.py](../mamba_mv/irregular_ssm.py) with a 6-layer S5 stack (same weights shared across variates), matching the S5 wrapper's paper-native config (`d_model=128 internal, state_dim=256`). Keep Stages 2-5 (grid alignment, var-axis attention, temporal Mamba, query readout) unchanged.

On `data_correct_async/multisin_high_irreg` × 5 seeds = **5 runs**.

**Prediction**: if Stage 1's best Mamba-MV Phase 3 MSE is ≥ 0.019 and Stage 4 drops below 0.018, this cleanly confirms Audit C: "the Phase 3 gap is from the per-variate SSM, not the alignment". This lets the paper position Mamba-MV as "S5-like per-variate dynamics + our novel cross-variate fusion" — a sharper contribution.

---

## Stage 5 — `multiplicative` dt_mode (contingent, 10 runs)

**Run only if Stage 2 has no Pareto winner** (i.e., `learned` wins one phase and `replace` wins the other).

Add a third `dt_mode` to [mamba_block.py](../../../ssm_model/ssm/mamba_experiments/forecaster/mamba_block.py) `MambaIrregularBlock`:
```python
elif self.dt_mode == "multiplicative":
    dt_raw = x_dbl[:, :self.dt_rank]
    dt_learned = F.softplus(self.dt_proj.weight @ dt_raw.t() + self.dt_proj.bias[:, None])
    dt_learned = rearrange(dt_learned, 'd (b l) -> b d l', l=seqlen)
    dt = dt_learned * (dt_real / dt_real.median())     # gate × scaled physical dt
```

**Rationale**: keeps Mamba's selective mechanism (via `dt_learned`) while injecting time-awareness (via `dt_real`). Neither `learned` nor `replace` does both. Needs confirmation against Mamba paper §3.2 on the semantics of Δ; blocked on the Mamba PDF until user attaches it.

Run on `multisin_high_irreg` Phase 3 + Phase 4-2 × 5 seeds = **10 runs**.

---

## Explicit non-axes and why

| Cut axis | Fixed value | Justification |
|---|---|---|
| `grid_K` | 128 | **Audit C**: decay is identity, grid resolution is second-order. Would be added as a Stage-2B sub-sweep only if Stage 1 is inconclusive. |
| `d_model` (primary sweep) | 256 | Replaced by `d_state` as the more mechanism-relevant capacity axis. Stage 3 size-matched diagnostic covers the d_model concern separately. |
| `weight_decay` | 0.01 | Mamba paper default. S5's 0.05 is specific to its HiPPO init (which it correctly excludes from decay). |
| `warmup_steps` | 100 | Shared-recipe convention; no signal it matters. |
| `expand` | 2 | Mamba paper §3.4 "always fixed at E=2". Non-negotiable. |
| `additive` dt_mode | dropped | Scientifically muddies the core comparison per v1. |
| Baseline HPO (S5, RoMAE) | none | T-PATCHGNN convention. "Same recipe, not per-model HPO" preference holds for baselines. |

---

## Budget summary

| Stage | Runs | Wall time (parallel) |
|---|---|---|
| 1 — Primary grid | 54 | ~2h |
| 2 — Pareto test | 10 | ~1h |
| 3 — Size-matched | 20 | ~1h |
| 4 — Mechanism isolation | 5 | ~3h (S5 is slow) |
| 5 — Multiplicative (contingent) | 10 | ~1h |
| **Total if Stage 5 triggered** | **99** | ~8h |
| **Total if Stage 5 not triggered** | **89** | ~7h |

Down from v1's 277 baseline / 357 conditional.

---

## Methods-paragraph draft (for paper)

> *"We ran a mechanism-informed hyperparameter search on Mamba-MV, prompted by a post-hoc audit showing that the shared-grid decay mechanism is effectively unused after training (the learnable decay rate stays at initialization, giving an implied decay factor of 0.997 across all seeds). Consequently we focused HPO on per-variate SSM capacity (`d_state ∈ {16, 64, 256}`) and the time-injection variant (`dt_mode ∈ {learned, replace}`) rather than on the grid resolution. We swept learning rate over three log-spaced values ({3e-4, 1e-3, 3e-3}) on the Phase-3 high-irregularity validation set, 3 seeds per cell (54 configurations). A Pareto test on the Phase-4-2 validation set selected the `(lr, d_state, dt_mode)` triple that balances the two regimes. A separate size-matched diagnostic row (~1.4M params) and a mechanism-isolation experiment (replacing Stage 1 with an S5 block) confirm that the architectural story does not reduce to parameter count or to the per-variate SSM choice alone. Baselines (S5, RoMAE) used their paper-native configurations."*

---

## Decision flow after Stage 1

```
Stage 1 best Phase 3 MSE
   |
   ├─ ≤ 0.020 (near S5) ──► Stage 2 (Pareto test)
   |                            |
   |                            ├─ Pareto winner ──► Stage 3 (size-matched) + write paper
   |                            └─ No Pareto winner ──► Stage 5 (multiplicative)
   |
   └─ > 0.020 (still losing) ──► Stage 4 (S5 block substitution)
                                    |
                                    ├─ S5-block variant wins ──► paper story: "our fusion, S5's dynamics"
                                    └─ S5-block variant also loses ──► fundamental arch issue; rethink contribution
```
