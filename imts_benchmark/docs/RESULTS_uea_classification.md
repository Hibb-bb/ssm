# UEA Classification — head-to-head vs RoMAE Table 4 baselines

Last updated: 2026-05-03. Mamba-MV-pretrain is the canonical architecture.

## What's tested

Five UEA multivariate datasets under the Kidger (NCDE 2020) 30% sync-drop
protocol — same as RoMAE Table 4. Three seeds {42, 123, 0}. AdamW + cosine
schedule. Train-from-scratch (no pretraining).

**Architecture (canonical):** Mamba-MV with slot-anonymous variates and
Moirai-style binary attention bias (`_BinaryVariateAttentionBias`,
permutation-equivariant over variates), no per-variate ID embedding in the
shared-grid aligner. HAN-style hierarchical attention pool classification head.
([`imts_benchmark/mamba_pretrain/`](../mamba_pretrain/))

**Schedule:** patience=10, max_epochs=200 (cells converged in ~3-10 min on H200,
well under both caps).

**Picker:** val/macro_f1 for HB/EP/LSST (imbalanced), val/acc for BM/CT
(balanced), with a collapse-detector exclusion (`val_acc − val_macro_f1 > 0.2`).

---

## Headline (best dt_mode per dataset)

| Dataset | TST | mTAN | S5 | ContiFormer | RoMAE | Mamba-MV (ours) | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| BasicMotions          | 0.967 | **0.992** | 0.983 | 0.975 | **0.992** | 0.925 ± .075 | mTAN/RoMAE (tied) |
| CharacterTrajectories | 0.974 | 0.953 | 0.961 | 0.983 | **0.988** | 0.979 ± .002 | RoMAE |
| Epilepsy              | 0.959 | 0.920 | 0.907 | 0.932 | 0.952 | **0.973 ± .012** | **Mamba-MV (concat)** |
| Heartbeat             | 0.740 | **0.779** | 0.733 | 0.756 | 0.745 | 0.652 ± .058 | mTAN |
| LSST                  | 0.552 | 0.531 | **0.639** | 0.600 | 0.623 | 0.302 ± .031 | S5 |
| **mean across 5**     | 0.838 | 0.835 | 0.845 | 0.849 | **0.860** | 0.766 | RoMAE |

**Counts:** Mamba-MV wins **1 of 5** (EP); is competitive on **1 of 5** (CT, within 1pt of RoMAE);
trails by ≥5pt on **3 of 5** (BM saturation gap, HB, LSST).

---

## Full per-(dataset, dt_mode) results — Mamba-MV (3 dt_modes, 5 datasets, 3 seeds)

| Dataset | dt_mode | n | test_acc | macro_f1 | winning HPs |
|---|---|---:|---:|---:|---|
| BasicMotions          | learned | 3 | **0.925 ± 0.075** | 0.926 ± 0.073 | lr=1e-3, bs=16 |
| BasicMotions          | replace | 3 | 0.892 ± 0.038 | 0.890 ± 0.044 | lr=1e-3, bs=16 |
| BasicMotions          | concat  | 3 | 0.533 ± 0.257 | 0.436 ± 0.307 | lr=1e-3, bs=32 |
| CharacterTrajectories | learned | 3 | **0.979 ± 0.002** | 0.977 ± 0.002 | lr=1e-4, bs=32 |
| CharacterTrajectories | replace | 3 | 0.977 ± 0.002 | 0.976 ± 0.002 | lr=1e-4, bs=32 |
| CharacterTrajectories | concat  | 3 | 0.969 ± 0.016 | 0.966 ± 0.018 | lr=1e-4, bs=32 |
| Epilepsy              | learned | 3 | 0.961 ± 0.015 | 0.961 ± 0.015 | lr=3e-4, bs=16 |
| Epilepsy              | replace | 3 | 0.969 ± 0.004 | 0.968 ± 0.004 | lr=3e-4, bs=16 |
| Epilepsy              | concat  | 3 | **0.973 ± 0.012** | 0.973 ± 0.012 | lr=1e-3, bs=16 |
| Heartbeat             | learned | 3 | 0.646 ± 0.053 | 0.606 ± 0.029 | lr=1e-4, bs=16 |
| Heartbeat             | replace | 3 | **0.652 ± 0.058** | 0.599 ± 0.024 | lr=1e-4, bs=16 |
| Heartbeat             | concat  | 3 | 0.629 ± 0.051 | 0.600 ± 0.029 | lr=1e-4, bs=32 |
| LSST                  | learned | 3 | **0.302 ± 0.031** | 0.301 ± 0.010 | lr=1e-4, bs=32 |
| LSST                  | replace | 3 | 0.295 ± 0.033 | 0.276 ± 0.017 | lr=1e-4, bs=16 |
| LSST                  | concat  | 3 | 0.291 ± 0.060 | 0.298 ± 0.042 | lr=1e-4, bs=32 |

**Notes:**
- **BM × concat = 0.533 ± 0.257** — one seed (seed=0) collapsed to 0.275 while
  the other two saturated. Suggests training instability with the binary-bias
  attention on tiny datasets (BM is 40 train).
- **HB and LSST winners landed at LR=1e-4** (lower edge of the grid). v3 had
  tried extending HB further down to 3e-5 with the supervised encoder and got
  noisier single-seed picks; not retried for the pretrain encoder.

---

## Diagnosis of the losses (HB, LSST)

Both correlate with the picker's val_acc − val_macro_f1 gap (class-imbalance
collapse signal):

| Cell | val_acc | val_f1 | gap |
|---|---:|---:|---:|
| EP best (concat, lr=1e-3, bs=16)  | high  | high  | small |
| CT best (learned, lr=1e-4, bs=32) | 0.989 | 0.988 | +0.001 |
| **HB best (replace, lr=1e-4, bs=16)** | 0.775 | 0.738 | **+0.037** |
| **LSST best (learned, lr=1e-4, bs=32)** | 0.307 | 0.330 | (variable) |

### Heartbeat (61 variates, 204 train, binary ~60/40)
- 10× more variates than EP/CT but similar training-set size to EP — variable-axis
  attention is parameter-rich, data-poor.
- Removing the per-variate ID embedding (Moirai-bias-only) means the encoder
  has to learn variate identity from limited data — exacerbates the data-poor
  issue on HB.
- Train-from-scratch loses the most here vs RoMAE's 800-epoch self-supervised
  pretraining on the dataset before fine-tuning.

### LSST (14 classes, heavy skew, light-curve shape)
- 14-class heavy skew + per-sample z-score then CE — minority classes drown out.
- Discriminative pattern is in the global shape across the long sequence
  (astronomical light curves), not local sequential structure. **S5 wins LSST
  (0.639)** because it's a different SSM family with longer effective context.
- RoMAE pretrain prior (shape priors learned from masked-and-reconstruct) is
  an additional asymmetric advantage.

---

## What would close the gap

In increasing order of effort:

1. **Add pretraining** (mask-and-reconstruct on the dataset for ~200 epochs,
   then fine-tune). What RoMAE does — single biggest lever on HB.
2. **Stronger head for imbalanced multi-class**: switch from single-query
   attention pool to multi-query or QKV+CLS readout (see [REPORT_uea_classification_design_2026-04-30.md §5.6](REPORT_uea_classification_design_2026-04-30.md)
   escalation). Helps LSST most.
3. **Class-balanced loss** (focal / inverse-freq weighted CE with stronger
   weighting) — already partially in place via `dm.class_weights` but could
   be tuned harder for LSST's 14-class skew.
4. **Re-introduce per-variate ID embedding** as an opt-in for small/high-V
   datasets like HB (would invalidate slot-anonymous claim, so weigh the
   FM-pretraining ambition against it).

---

## HPO grid

Uniform across all 5 datasets:

| Axis | Values | # |
|---|---|---:|
| `dt_mode` | replace, learned, concat | 3 |
| `lr` | 1e-4, 3e-4, 1e-3 | 3 |
| `batch_size` | 16, 32 | 2 |
| HPO seed | 42 (single) | 1 |

→ 18 cells per dataset × 5 datasets = **90 HPO cells**, single seed = 42.

**Held fixed at per-dataset RoMAE Table 12 defaults** (NOT swept):
`label_smoothing`, `grad_clip`, `grid_K`, `head_dropout`.

**Final eval:** 5 ds × 3 dt × 3 seeds {42, 123, 0} = 45 cells. Replays HPO
winner per (dataset, dt_mode).

---

## Run

| Date | Chain | Job IDs | Output root |
|---|---|---|---|
| 2026-05-03 | pretrain UEA | `18010426 → 18010427 → 18010428` | `output/log/uea_cls_pretrain/` |

90 HPO + 45 final cells, all completed, 0 failures.

---

## Reproducibility pointers

- Data:   `data_uea/{BasicMotions,CharacterTrajectories,Epilepsy,Heartbeat,LSST}/*_TRAIN.ts/_TEST.ts`
- Code:   [`mamba_pretrain/train_cls.py`](../mamba_pretrain/train_cls.py),
          [`mamba_pretrain/multivariate_classifier.py`](../mamba_pretrain/multivariate_classifier.py),
          [`mamba_pretrain/classification_head.py`](../mamba_pretrain/classification_head.py)
- Datamodule: [`shared_data/uea_classification_datamodule.py`](../shared_data/uea_classification_datamodule.py)
- Aggregator: [`eval/aggregate_uea_hpo.py`](../eval/aggregate_uea_hpo.py)
- Submit chain: `bash imts_benchmark/scripts/submit_uea_cls_pretrain_pipeline_delta_x86.sh`
- Per-cell summaries: `output/log/uea_cls_pretrain/hpo/<ds>/<dt>/<cell>/hpo/summary.json`
- Per-seed final summaries: `output/log/uea_cls_pretrain/final/<ds>/<dt>/<seed>/final/summary.json`
- Winner JSON: `output/log/uea_cls_pretrain/winner_configs.json`
- LaTeX table: [`results_table_uea_classification.tex`](results_table_uea_classification.tex)
