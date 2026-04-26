# Data-centric pretraining on TSFMs for irregular time series (our NeurIPS 2026 draft)

## Status
Early draft, active. Located in this conversation as an attached PDF.

## What the draft currently claims
1. "Irregular Time Series Foundation Models (TSFMs) based on Mamba architecture" — positions Mamba-MV as a TSFM contribution.
2. Three desiderata the draft states a good model must satisfy:
   - Capture latent continuous dynamics.
   - Capture cross-variate correlation with async observations (implicit interpolation).
   - [Placeholder third desideratum].
3. Pretraining as a key contribution (the "data-centric" part) — not yet executed in the benchmark.
4. Formal problem setup (§3.2) defines irregular multivariate forecasting with a query set `q_j = (n_j, a_j)`.

## What the empirical results in [INTERPRETATION_async_imts.md](../docs/INTERPRETATION_async_imts.md) suggest about the draft's framing

### Strong claims to keep
- The query-time readout (§3.4 of the draft) is the mechanism that lets Mamba-MV predict at arbitrary async query times — a feature S5 doesn't have (S5 predicts at observation timestamps only).
- The cross-variate fusion via shared-grid + variable-axis attention is the mechanism that drives the Phase 4-2 win over S5.

### Claims to reframe
- **"Handles async IMTS" as a blanket claim** → narrow to "handles inverse-mixture recovery when variates become unobservable". The Phase 3 results show S5 is sufficient for async-but-observable. The paper's unique value is gapped.
- **`dt_mode=replace`** → per [Audit B](../audit/), `replace` bypasses Mamba's selective mechanism entirely. Either drop `replace` from the main claim, or rename it as "continuous-SSM ablation" to be transparent.
- **"Shared-grid alignment with exponential decay"** → per [Audit C](../audit/AUDIT_C_gamma_rho.md), the decay is never used by trained models. The alignment is effectively nearest-neighbor latent carry-over + staleness features; the decay term can be removed without loss.

### Claims to add
- After HPO Stage 4 (mechanism isolation): "per-variate SSM choice dominates Phase 3 performance; our fusion architecture is orthogonal to and composable with stronger per-variate SSMs." This repositions the contribution as modular rather than monolithic.
- Honest limitations:
   - S5 is already sufficient on continuous-irregular async data.
   - RoMAE without pretraining is unstable; fair comparison requires the same pretraining budget.
   - The 7.8M-param Mamba-MV loses to 1.4M-param S5 on Phase 2/3; the Phase 4-2 win is architecturally attributable (size-matched Stage 3 diagnostic).

## Sections that need content once HPO runs

| Draft section | Content needed | Source |
|---|---|---|
| §3.x (dt_mode ablation) | Full table across Phase 3 + Phase 4-2 × dt_modes | HPO Stages 1 + 2 |
| §4.x (parameter-count ablation) | Size-matched row | HPO Stage 3 |
| §4.x (mechanism isolation) | Stage-1-with-S5-block row | HPO Stage 4 |
| §4.x (multiplicative dt) | Contingent; may or may not appear | HPO Stage 5 |

## Section of the draft that likely needs to be rewritten
§1 Introduction — currently flagged "[HY: following draft need to be rewrite]". Given the narrowed framing above, the rewrite should emphasize:
1. There is a qualitative difference between async-but-observable IMTS (where S5 suffices) and async-with-gaps IMTS (where cross-variate inference is required).
2. Our Mamba-MV architecture is the first to explicitly provide structured cross-variate fusion on async IMTS, and we show it is necessary for the gapped / inverse-mixture regime.
3. Pretraining data-centric methods are future work that will extend the benchmark; current results are without pretraining.
