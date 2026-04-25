# HPO winners vs baseline — phase4_2 × high_irreg

Per-dt_mode winners (picked by min val/mse on HPO sweep, confirmed over 5 seeds):

- `learned`: lr=5e-4, bs=128
- `replace`: lr=2e-3, bs=64
- `concat`: lr=1e-4, bs=64

| Rank | Model | Variant | n_seeds | MSE (mean ± std) | R² | Pearson | Params |
|------|-------|---------|---------|------------------|-----|---------|--------|
| 1 | mamba_mv | hpo_tuned_replace | 5 | 0.02425 ± 0.00303 | 0.445 | 0.670 | 7,789,395 |
| 2 | mamba_mv | hpo_tuned_learned | 5 | 0.02824 ± 0.00137 | 0.356 | 0.599 | 7,789,395 |
| 3 | mamba_mv | learned | 5 | 0.02865 ± 0.00067 | 0.344 | 0.594 | 7,789,395 |
| 4 | mamba_mv | hpo_tuned_concat | 5 | 0.02921 ± 0.00186 | 0.328 | 0.578 | 7,791,699 |
| 5 | romae | default | 5 | 0.02927 ± 0.01487 | 0.339 | 0.480 | 7,803,073 |
| 6 | mamba_mv | replace | 5 | 0.03107 ± 0.00234 | 0.290 | 0.556 | 7,789,395 |
| 7 | s5 | default | 5 | 0.03186 ± 0.00469 | 0.266 | 0.524 | 1,395,969 |
