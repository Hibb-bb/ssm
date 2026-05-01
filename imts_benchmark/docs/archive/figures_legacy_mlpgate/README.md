# Legacy figure — superseded 2026-04-30

These files render the **old MLP-gate + Dropout** classification head:

- per-variate temporal gate via `MLP_t` over `H^(L)`
- variate-axis gate via `MLP_v` over `H^(v)`
- final `LayerNorm + Dropout + Linear`

That design was replaced before the UEA classification pipeline was submitted.
The locked head (see
[`docs/REPORT_uea_classification_design_2026-04-30.md`](../../REPORT_uea_classification_design_2026-04-30.md)
and [`mamba_mv/classification_head.py`](../../../mamba_mv/classification_head.py))
uses two **single-query attention pools** (HAN-style; Yang et al. 2016) with
learnable queries `q_t, q_v` and **no dropout**, plus a `LayerNorm` before the
stage-2 input and another before the final `Linear`.

Files kept for provenance only — do not cite in the paper.
