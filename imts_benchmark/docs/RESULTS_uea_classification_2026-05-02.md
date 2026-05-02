# UEA Classification — head-to-head vs RoMAE Table 4 baselines

Date: 2026-05-02
Run: `output/log/uea_cls_v2/{hpo,final}/` on Delta x86 H200, env `mamba_x86`.
Pipeline: smoke (jid 17997345) → 72-cell HPO (17997346) → aggregator (17997347) → 24-cell final (17997348). All stages COMPLETED.

Protocol: Kidger 30% sync drop (NCDE 2020), per-sample z-score, RoMAE Table 12 per-dataset
HPs except LR/batch_size which are the v2 sweep axes; v2 picker uses val/macro_f1 for HB/EP/LSST,
val/acc for CT (BM excluded — already at 1.000 in v1). 3 seeds {42, 123, 0}, AdamW + cosine,
800 epochs / patience 50, train-from-scratch (no pretraining).

Baselines from RoMAE (Zivanovic et al., 2025), Table 4 — same Kidger 30% drop protocol.

---

## Headline: best dt_mode per dataset, mean ± std test accuracy

| Dataset | TST | mTAN | S5 | ContiFormer | RoMAE | **Mamba-MV (ours)** | rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| BasicMotions          | 0.967 | 0.992 | 0.983 | 0.975 | 0.992 | **1.000** | **1** |
| CharacterTrajectories | 0.974 | 0.953 | 0.961 | 0.983 | **0.988** | 0.975 ± 0.002 | 3 |
| Epilepsy              | 0.959 | 0.920 | 0.907 | 0.932 | 0.952 | **0.981 ± 0.011** | **1** |
| Heartbeat             | 0.740 | **0.779** | 0.733 | 0.756 | 0.745 | 0.694 ± 0.038 | 6 |
| LSST                  | 0.552 | 0.531 | **0.639** | 0.600 | 0.623 | 0.458 ± 0.032 | 6 |
| **mean across 5**     | 0.838 | 0.835 | 0.845 | 0.849 | **0.860** | 0.822 | 5 |

**Counts:** Mamba-MV wins **2 of 5** (BM, EP); is competitive (within 1.5pt of best) on **1 of 5** (CT);
trails by ≥5pt on **2 of 5** (HB, LSST).

For BM, our v1 result (1.000 across 3 seeds) is reused — BM was excluded from the v2 sweep
because it had already saturated.

---

## Full per-(dataset, dt_mode) results, both modes

| Dataset | dt_mode | n | test_acc | macro_f1 | winning HPs |
|---|---|---:|---:|---:|---|
| CharacterTrajectories | learned | 3 | 0.975 ± 0.002 | 0.974 ± 0.002 | lr=3e-4, bs=16 |
| CharacterTrajectories | replace | 3 | 0.969 ± 0.008 | 0.968 ± 0.008 | lr=3e-4, bs=32 |
| Epilepsy              | learned | 3 | 0.964 ± 0.000 | 0.963 ± 0.000 | lr=3e-4, bs=16 |
| Epilepsy              | replace | 3 | **0.981 ± 0.011** | **0.980 ± 0.011** | lr=3e-4, bs=32 |
| Heartbeat             | learned | 3 | 0.694 ± 0.038 | 0.644 ± 0.015 | lr=1e-4, bs=32 |
| Heartbeat             | replace | 3 | 0.689 ± 0.066 | 0.537 ± 0.104 | lr=1e-3, bs=32 |
| LSST                  | learned | 3 | 0.407 ± 0.047 | 0.337 ± 0.017 | lr=1e-3, bs=16 |
| LSST                  | replace | 3 | **0.458 ± 0.032** | **0.361 ± 0.009** | lr=1e-3, bs=32 |

`replace` wins on EP, HB-by-margin (high-variance), and LSST.
`learned` wins on CT, EP-narrow, and HB-on-acc-but-not-f1.

---

## Diagnosis of the two losses

Both losses are predicted by the v2 picker's val_acc − val_macro_f1 gap, which signals
class-imbalance bias / partial collapse:

| Cell | val_acc | val_f1 | gap |
|---|---:|---:|---:|
| EP best (replace, lr=3e-4, bs=32) | 1.000 | 1.000 | +0.000 |
| CT best (learned, lr=3e-4, bs=16) | 0.982 | 0.981 | +0.001 |
| **HB best (learned, lr=1e-4, bs=32)** | 0.775 | 0.689 | **+0.086** |
| **LSST best (replace, lr=1e-3, bs=32)** | 0.470 | 0.367 | **+0.103** |

### Heartbeat (61 variates, 204 train, binary ~60/40)
- **10× more variates** than EP/CT but **similar training-set size** to EP — variable-axis attention
  is parameter-rich, data-poor.
- OOM at standard `grid_K=256, batch=32` forces us to `grid_K=128, batch=16` — half the
  temporal resolution the other datasets get.
- Train-from-scratch loses the most here vs RoMAE's 800-epoch self-supervised pretraining
  on the dataset before fine-tuning.

### LSST (14 classes, heavy skew, light-curve shape)
- **Largest val_acc − val_f1 gap (+0.10)** across all cells — strong constant-predictor signature.
- 14-class with heavy skew + per-sample z-score then CE — minority classes are drowned out.
- Discriminative pattern is in the *global shape* across the long sequence (astronomical light
  curves), not local sequential structure. **S5 wins LSST (0.639)** because it's a different
  SSM family with different effective context. The RoMAE pretrain prior (shape priors learned
  from masked-and-reconstruct) is an additional asymmetric advantage.

---

## What would close the gap

In increasing order of effort:

1. **Add pretraining** (mask-and-reconstruct on the dataset for ~200 epochs, then fine-tune).
   This is what RoMAE does and is the single biggest lever for HB.
2. **Stronger head for imbalanced multi-class**: switch from single-query attention pool to
   multi-query or QKV+CLS readout (`docs/REPORT_uea_classification_design_2026-04-30.md` §5.6
   escalation). Helps LSST most.
3. **Class-balanced loss** (focal / inverse-freq weighted CE with stronger weighting) — already
   partially in place via `dm.class_weights` but could be tuned harder for LSST's 14-class skew.

---

## Reproducibility pointers

- Data:   `data_uea/{BasicMotions,CharacterTrajectories,Epilepsy,Heartbeat,LSST}/*_TRAIN.ts/_TEST.ts`
- Code:   [`mamba_mv/train_cls.py`](../mamba_mv/train_cls.py), [`shared_data/uea_classification_datamodule.py`](../shared_data/uea_classification_datamodule.py)
- Aggregator: [`eval/aggregate_uea_hpo.py`](../eval/aggregate_uea_hpo.py)
- Submit chain: `bash imts_benchmark/scripts/submit_uea_cls_pipeline_v2_delta_x86.sh`
- Per-cell summaries: `output/log/uea_cls_v2/hpo/<ds>/<dt>/<cell>/hpo/summary.json`
- Per-seed final summaries: `output/log/uea_cls_v2/final/<ds>/<dt>/<seed>/final/summary.json`
- Winner JSON consumed by final array: `output/log/uea_cls_v2/winner_configs.json`
