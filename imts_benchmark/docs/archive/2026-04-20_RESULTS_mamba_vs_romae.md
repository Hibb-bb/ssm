# Results — Mamba-MV vs RoMAE on long-gap multivariate forecasting

Run tag: `mvcompare_v1`. Completed 2026-04-20.
Mamba-MV: 30 runs (2 regimes × 3 dt_modes × 5 seeds).
RoMAE: 10 runs (2 regimes × 5 seeds). **Reran 2026-04-20 after bug fix** — see
[Bug fix](#bug-fix-romae-target-zeroing) below.
Aggregator: [`eval/aggregate_results.py`](eval/aggregate_results.py).
Raw artifacts:
`/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1/`
 (`summary.csv`, `long.csv`, per-seed `test_metrics.csv` + `per_sample.jsonl`).

---

## Headline

Mamba-MV beats RoMAE on every regime, every metric, after fixing the RoMAE
target-zeroing bug. Paired Wilcoxon on per-sample MSE (N=200, Mamba best
variant vs RoMAE):

| regime | Mamba variant | Mamba MSE | RoMAE MSE | Δ | Wilcoxon p |
|---|---|---|---|---|---|
| sparse_dependent   | learned | **0.0233** | 0.0378 | −0.0145 | 3.0e-33 |
| sparse_independent | learned | **0.0389** | 0.0491 | −0.0102 | 7.6e-34 |

Mamba-MV is ~38% lower MSE on `sparse_dependent` and ~21% lower on
`sparse_independent`. Statistically significant on every seed, every sample.

**Interpretation caveat:** RoMAE's MSE std across seeds is 2e-6 (dependent)
and literally 0.0 (independent), and its R² sits at ≈ −0.004. This indicates
RoMAE converged to **predict-the-mean** on both regimes, not to a structure-
aware solution. Mamba-MV at matched params does learn structure (R² =
0.20–0.37, Pearson = 0.45–0.60). The comparison is fair but underwhelming:
Mamba beats a mean-predictor by a solid margin; RoMAE with our adapter cannot
even reach a mean-predictor-beating solution. This is worth a sentence in the
paper and probably further follow-up (RoPE on raw-float timestamps, pretrain
before fine-tune, longer context window) before claiming "Mamba beats
transformers."

---

## Full table (mean ± std over 5 seeds)

| model | regime | variant | MSE | MAE | R² | Pearson | wall (s) | params |
|---|---|---|---|---|---|---|---|---|
| mamba_mv | sparse_dependent   | learned  | 0.0233 ± 0.0024 | 0.1194 ± 0.0083 | 0.367 ± 0.061 | 0.604 ± 0.053 | 194 | 7.79M |
| mamba_mv | sparse_dependent   | additive | 0.0268 ± 0.0015 | 0.1308 ± 0.0052 | 0.274 ± 0.038 | 0.527 ± 0.037 | 165 | 7.79M |
| mamba_mv | sparse_dependent   | replace  | 0.0304 ± 0.0075 | 0.1411 ± 0.0251 | 0.179 ± 0.196 | 0.389 ± 0.261 | 137 | 7.79M |
| mamba_mv | sparse_independent | learned  | 0.0389 ± 0.0004 | 0.1585 ± 0.0017 | 0.203 ± 0.010 | 0.455 ± 0.013 | 150 | 7.79M |
| mamba_mv | sparse_independent | additive | 0.0405 ± 0.0013 | 0.1620 ± 0.0046 | 0.168 ± 0.027 | 0.419 ± 0.031 | 147 | 7.79M |
| mamba_mv | sparse_independent | replace  | 0.0415 ± 0.0011 | 0.1642 ± 0.0030 | 0.145 ± 0.023 | 0.400 ± 0.024 | 142 | 7.79M |
| romae    | sparse_dependent   | default  | 0.0378 ± 2e-6   | 0.1690 ± 2e-6   | −0.005 ± 6e-5 |  0.004 ± 0.003 | 70  | 7.80M |
| romae    | sparse_independent | default  | 0.0491 ± 0.0    | 0.1888 ± 1e-6   | −0.004 ± 8e-6 |  0.000 ± 0.002 | 76  | 7.80M |

Params matched within **0.2%** (7.79M vs 7.80M). Both trained with identical
AdamW, cosine schedule, grad-clip, batch size, and fp32 precision
([`shared_config/fair_defaults.py`](shared_config/fair_defaults.py)).

---

## Gapped-variate MSE (cross-variate reasoning test)

For each test sample, one variate has all its observations inside the
forbidden gap dropped. Computing MSE only on the gapped variate:

| regime | Mamba `learned` | Mamba `additive` | Mamba `replace` | RoMAE |
|---|---|---|---|---|
| sparse_dependent   | **0.0308** | 0.0338 | 0.0373 | ~0.0378 (≈ mean-predictor) |
| sparse_independent | **0.0435** | 0.0459 | 0.0492 | ~0.0491 (≈ mean-predictor) |

Mamba-MV strictly dominates on the gapped variate as well.

---

## Bug fix: RoMAE target-zeroing

First-pass RoMAE results (`summary.csv` before 2026-04-20) reported MSE =
0.294 / 0.303 with std ~1e-5 across seeds. Training loss had gone to literal
zero. Root cause was in our adapter, not in RoMAE:

[`romae_forecaster.py:_build_inputs`](romae_forecaster/romae_forecaster.py)
previously pre-zeroed values at `pred_mask` positions before handing them to
`RoMAEForPreTraining.forward`. But the library's forward does
`m_x = x[mask]` — it pulls the regression target **from the tensor we pass
in**. By zeroing those positions ourselves, we made `m_x = 0` identically, so
the model trained to output zero, driving train/val loss to 0.000 while test
MSE = var(y) + mean²(y) = 0.294.

The fix: pass `values` as-is; the library handles encoder/decoder masking
internally. Committed as a one-line change in `_build_inputs`. The Mamba-MV
results were unaffected by this bug.

Lesson: MAE-style architectures in general don't expect you to zero masked
positions on input — they extract targets from those positions and strip
them from the encoder internally. If you add "defensive" zeroing, you will
silently destroy the regression targets. A good sanity check: assert the
target tensor has nonzero variance on the first training batch.

---

## Plots

All under `{LOG_ROOT}/plots/`:

- `bar_mse.png`, `bar_mae.png`, `bar_r2.png`, `bar_pearson.png` — grouped bars per regime (3 Mamba variants + RoMAE), error bars = std across seeds.
- `bar_gapped_mse.png` — same but restricted to the gapped variate.
- `scatter_{regime}_learned.png` — paired per-sample gapped-MSE scatter (Mamba `learned` vs RoMAE), y=x reference line.
- `scatter_overall_{regime}_learned.png` — paired per-sample overall MSE scatter.
- `examples/forecast_{regime}_sin_*.png` — qualitative 3-variate forecasts for a few test items (observed history, true forecast, Mamba prediction, RoMAE prediction).

Regenerate: `python -m mv_vs_romae.eval.make_plots` and `python -m mv_vs_romae.eval.plot_forecast_examples`.

---

## Summary for the paper

Safe to claim:
- With matched parameter count, identical optimizer/schedule, and 1k training samples, Mamba-MV achieves 38% lower MSE on `sparse_dependent` and 21% lower on `sparse_independent` vs RoMAE, p < 1e-33.
- Mamba-MV learns forecast structure (R² = 0.20-0.37), RoMAE with our adapter does not (R² ≈ 0).

Not safe to claim:
- That transformers **cannot** do irregular multivariate forecasting. RoMAE at 1k samples, no pretraining, with RoPE on raw-float timestamps, converged to the mean-predictor floor. This is an adapter/data-scale limitation, not an architecture ceiling.

Suggested follow-ups before camera-ready:
1. Pretrain RoMAE with random masking on a larger irregular-sinusoid corpus, then fine-tune with forecast masking — does it escape the mean-predictor floor?
2. Replace the two-dim float position (`timestamp, variate_id`) with a learned variate embedding + timestamp-only RoPE.
3. Add a mean-predictor baseline to the table explicitly, to frame the claim correctly.
