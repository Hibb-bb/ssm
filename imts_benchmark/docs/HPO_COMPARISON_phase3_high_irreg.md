# HPO winners vs baseline — phase3 × high_irreg

Per-dt_mode winners (picked by min val/mse on HPO sweep, confirmed over 5 seeds):

- `learned`: lr=2e-3, bs=64
- `replace`: lr=2e-3, bs=64
- `concat`: lr=2e-3, bs=64

| Rank | Model | Variant | n_seeds | MSE (mean ± std) | R² | Pearson | Params |
|------|-------|---------|---------|------------------|-----|---------|--------|
| 1 | mamba_mv | hpo_tuned_replace | 5 | 0.01316 ± 0.00231 | 0.698 | 0.836 | 7,789,395 |
| 2 | mamba_mv | hpo_tuned_concat | 5 | 0.01655 ± 0.00397 | 0.620 | 0.785 | 7,791,699 |
| 3 | s5 | default | 5 | 0.01760 ± 0.00758 | 0.584 | 0.763 | 1,395,969 |
| 4 | mamba_mv | replace | 5 | 0.02275 ± 0.00215 | 0.481 | 0.692 | 7,789,395 |
| 5 | mamba_mv | hpo_tuned_learned | 5 | 0.02375 ± 0.01144 | 0.457 | 0.627 | 7,789,395 |
| 6 | mamba_mv | learned | 5 | 0.02434 ± 0.00073 | 0.442 | 0.665 | 7,789,395 |
| 7 | romae | default | 5 | 0.03274 ± 0.01729 | 0.266 | 0.370 | 7,803,073 |
