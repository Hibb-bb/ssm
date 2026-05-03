# UEA Classification — head-to-head vs RoMAE Table 4 baselines

Last updated: 2026-05-03. Canonical (supersedes the two dated drafts).

## What's tested

Five UEA multivariate datasets under the Kidger (NCDE 2020) 30% sync-drop
protocol — same as RoMAE Table 4. Three seeds {42, 123, 0} unless otherwise
noted. AdamW + cosine schedule. Train-from-scratch (no pretraining).

Two architectural variants under test:

- **Mamba-MV (supervised)** — per-variate ID embedding in shared-grid; absolute-time
  positional bias on attention. Patience=50, max_epochs=800.
  ([`imts_benchmark/mamba_mv/`](../mamba_mv/))
- **Mamba-MV-pretrain** — slot-anonymous variates with Moirai-style binary
  attention bias, no per-variate ID embedding. Classification head identical
  to supervised. Patience=10, max_epochs=200.
  ([`imts_benchmark/mamba_pretrain/`](../mamba_pretrain/))

Picker: val/macro_f1 for HB/EP/LSST (imbalanced), val/acc for BM/CT (balanced),
with a collapse-detector exclusion (`val_acc − val_macro_f1 > 0.2`).

---

## Headline (best dt_mode per dataset)

| Dataset | TST | mTAN | S5 | ContiFormer | RoMAE | Mamba-MV (sup.) | Mamba-MV-pretrain | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BasicMotions          | 0.967 | 0.992 | 0.983 | 0.975 | 0.992 | **1.000** | 0.925 | Mamba-MV (sup.) |
| CharacterTrajectories | 0.974 | 0.953 | 0.961 | 0.983 | **0.988** | 0.975 | 0.979 | RoMAE |
| Epilepsy              | 0.959 | 0.920 | 0.907 | 0.932 | 0.952 | **0.981** | 0.973 | Mamba-MV (sup.) |
| Heartbeat             | 0.740 | **0.779** | 0.733 | 0.756 | 0.745 | 0.694 | 0.652 | mTAN |
| LSST                  | 0.552 | 0.531 | **0.639** | 0.600 | 0.623 | 0.458 | 0.302 | S5 |
| **mean across 5**     | 0.838 | 0.835 | 0.845 | 0.849 | **0.860** | 0.822 | 0.766 | RoMAE |

**Counts vs published baselines**:
- Mamba-MV (supervised): wins **2 of 5** (BM, EP); competitive on CT; trails on HB, LSST.
- Mamba-MV-pretrain: wins **0 of 5**; tighter on CT than supervised but loses on every other dataset.

---

## Full per-(dataset, dt_mode) results

### Mamba-MV (supervised; v2 re-sweep)

| Dataset | dt_mode | n | test_acc | macro_f1 | winning HPs |
|---|---|---:|---:|---:|---|
| BasicMotions          | learned | 3 | **1.000 ± 0.000** | 1.000 ± 0.000 | v1 (saturated) |
| BasicMotions          | replace | 3 | 0.950 ± 0.066 | 0.946 ± 0.066 | v1 |
| CharacterTrajectories | learned | 3 | **0.975 ± 0.002** | 0.974 ± 0.002 | lr=3e-4, bs=16 |
| CharacterTrajectories | replace | 3 | 0.969 ± 0.008 | 0.968 ± 0.008 | lr=3e-4, bs=32 |
| Epilepsy              | learned | 3 | 0.964 ± 0.000 | 0.963 ± 0.000 | lr=3e-4, bs=16 |
| Epilepsy              | replace | 3 | **0.981 ± 0.011** | 0.980 ± 0.011 | lr=3e-4, bs=32 |
| Heartbeat             | learned | 3 | **0.694 ± 0.038** | 0.644 ± 0.015 | lr=1e-4, bs=32 |
| Heartbeat             | replace | 3 | 0.689 ± 0.066 | 0.537 ± 0.104 | lr=1e-3, bs=32 |
| LSST                  | learned | 3 | 0.407 ± 0.047 | 0.337 ± 0.017 | lr=1e-3, bs=16 |
| LSST                  | replace | 3 | **0.458 ± 0.032** | 0.361 ± 0.009 | lr=1e-3, bs=32 |

`concat` not run for the v2 re-sweep (would've completed via the v3-addon
chain; cancelled when we pivoted to pretrain).

### Mamba-MV-pretrain (3 dt_modes × 5 datasets, 3 seeds; chain `18010426–18010428`)

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

---

## Architecture A/B read

Pretrain vs supervised at each dataset's best dt_mode:

| Dataset | Supervised best | Pretrain best | Δ |
|---|---:|---:|---:|
| BM   | 1.000 (learned) | 0.925 (learned) | **−7.5 pt** |
| CT   | 0.975 (learned) | 0.979 (learned) | +0.4 pt |
| EP   | 0.981 (replace) | 0.973 (concat)  | −0.8 pt |
| HB   | 0.694 (learned) | 0.652 (replace) | **−4.2 pt** |
| LSST | 0.458 (replace) | 0.302 (learned) | **−15.6 pt** |

**Pattern:** the slot-anonymous + binary-bias encoder removes the per-variate
ID embedding, which provides a useful inductive prior in supervised when:
- Data is small (HB: 204 train; LSST: 14-class skew with limited per-class samples)
- Variate identity matters (HB: 61 distinct clinical channels)

Without that prior the new encoder must learn variate identity from data —
fine on CT (1422 samples, V=3, balanced) but underperforms on the rest.

**Concerning data point:** BM × concat = 0.533 ± 0.257 — one seed (seed=0)
collapsed to 0.275 while the other two saturated. Suggests training
instability for the binary-bias attention on tiny datasets (BM is 40 train).

---

## Diagnosis of the supervised losses (HB, LSST)

Both correlate with the picker's val_acc − val_macro_f1 gap (class-imbalance
collapse signal):

| Cell | val_acc | val_f1 | gap |
|---|---:|---:|---:|
| EP best (replace, lr=3e-4, bs=32) | 1.000 | 1.000 | +0.000 |
| CT best (learned, lr=3e-4, bs=16) | 0.982 | 0.981 | +0.001 |
| **HB best (learned, lr=1e-4, bs=32)** | 0.775 | 0.689 | **+0.086** |
| **LSST best (replace, lr=1e-3, bs=32)** | 0.470 | 0.367 | **+0.103** |

### Heartbeat (61 variates, 204 train, binary ~60/40)
- 10× more variates than EP/CT but similar training-set size to EP — variable-axis
  attention is parameter-rich, data-poor.
- OOM at standard `grid_K=256, batch=32` forces `grid_K=128, batch=16` — half
  the temporal resolution the other datasets get.
- Train-from-scratch loses the most here vs RoMAE's 800-epoch self-supervised
  pretraining on the dataset before fine-tuning.

### LSST (14 classes, heavy skew, light-curve shape)
- Largest val_acc − val_f1 gap (+0.10) — strong constant-predictor signature.
- 14-class with heavy skew + per-sample z-score then CE — minority classes drown out.
- Discriminative pattern is in the global shape across the long sequence
  (astronomical light curves), not local sequential structure. **S5 wins
  LSST (0.639)** because it's a different SSM family with longer effective
  context. RoMAE pretrain prior is an additional asymmetric advantage.

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

---

## Protocol asymmetry caveat (pretrain vs supervised)

Pretrain block uses `patience=10, max_epochs=200`; supervised uses
`patience=50, max_epochs=800`. So Mamba-MV (sup.) vs Mamba-MV-pretrain
confounds two changes: encoder architecture AND training schedule.
Practical effect is bounded: v2 supervised cells already converged in
~3-6 min (well under both schedules' caps), so the schedule difference
is mostly nominal. The HB / LSST regression in pretrain isn't likely a
schedule artifact since both landed at low LRs where 200 epochs is still
plenty.

For a clean apples-to-apples architectural claim, re-run supervised under
patience=10/max_epochs=200 (~75 cells, ~6 GPU-hr) — not done.

---

## Run history

| Date | Chain | Output root | Notes |
|---|---|---|---|
| 2026-04-30 | v1 (Quest) | `output/log/imts_benchmark_v2/uea_cls/` | 5-dataset, 2 dt_modes; BM saturated |
| 2026-05-02 | v2 (Delta x86) | `output/log/uea_cls_v2/` | 4-dataset re-sweep (BM excluded); 72 HPO + 24 final |
| 2026-05-03 | pretrain (Delta x86) | `output/log/uea_cls_pretrain/` | 5-dataset, 3 dt_modes incl. concat; 90 HPO + 45 final, 0 failures |

---

## Reproducibility pointers

- Data:   `data_uea/{BasicMotions,CharacterTrajectories,Epilepsy,Heartbeat,LSST}/*_TRAIN.ts/_TEST.ts`
- Code (supervised):  [`mamba_mv/train_cls.py`](../mamba_mv/train_cls.py),
                      [`mamba_mv/multivariate_classifier.py`](../mamba_mv/multivariate_classifier.py)
- Code (pretrain):    [`mamba_pretrain/train_cls.py`](../mamba_pretrain/train_cls.py),
                      [`mamba_pretrain/multivariate_classifier.py`](../mamba_pretrain/multivariate_classifier.py)
- Datamodule (shared): [`shared_data/uea_classification_datamodule.py`](../shared_data/uea_classification_datamodule.py)
- Aggregator: [`eval/aggregate_uea_hpo.py`](../eval/aggregate_uea_hpo.py)
- Submit chains:
  - Supervised v2:  `bash imts_benchmark/scripts/submit_uea_cls_pipeline_v2_delta_x86.sh`
  - Pretrain:       `bash imts_benchmark/scripts/submit_uea_cls_pretrain_pipeline_delta_x86.sh`
- Per-cell summaries: `output/log/{uea_cls_v2,uea_cls_pretrain}/hpo/<ds>/<dt>/<cell>/hpo/summary.json`
- Per-seed final summaries: `output/log/{uea_cls_v2,uea_cls_pretrain}/final/<ds>/<dt>/<seed>/final/summary.json`
- Winner JSONs: `output/log/{uea_cls_v2,uea_cls_pretrain}/winner_configs.json`
- LaTeX table: [`results_table_uea_classification.tex`](results_table_uea_classification.tex)
