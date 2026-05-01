# UEA Classification Design & HPO Plan (Mamba-MV)

Date: 2026-04-30 (v1) / 2026-05-01 (v2 addendum)
Scope: NeurIPS 2026 add-on table — Mamba-MV vs. RoMAE Table 4 baselines on five
irregularized UEA classification datasets.

**Status.** v1 pipeline executed end-to-end. Headline: BM=1.0 across 3 seeds
(beats every RoMAE Table 4 baseline). HB and LSST trailed; HB exhibited
**majority-class collapse** (val_acc identical across 3 seeds, macro_f1=0.42).
Diagnosis traced to two interacting bugs: *picker-metric mismatch* on imbalanced
data + *LR grid* not data-aware. v2 addresses both. See §6.

---

## 1. Datasets and irregularization

Five UEA datasets, matching RoMAE Table 4 \citep{zivanovic2025rotary}:

| Dataset                | $V$ | classes | train | test |
|------------------------|-----|---------|-------|------|
| BasicMotions           | 6   | 4       | 40    | 40   |
| CharacterTrajectories  | 3   | 20      | 1422  | 1436 |
| Epilepsy               | 3   | 4       | 137   | 138  |
| Heartbeat              | 61  | 2       | 204   | 205  |
| LSST                   | 6   | 14      | 2459  | 2466 |

We use the official splits as-is. Validation is a stratified 80/20 split of the
official training set (`split_seed=42`, fixed across HPO and final eval so the
val identity is stable).

Irregularity is induced via the Kidger (NCDE 2020) protocol: drop a fraction
$\rho = 0.3$ of timestamps **synchronously across all variates**, with a
deterministic per-(sample, drop\_seed) drop mask. Datamodule:
`imts_benchmark/shared_data/uea_classification_datamodule.py`.

Per-sample z-score normalization is applied before drop. Class weights are
inverse-frequency on the (kept) training split, normalized to sum to $C$.

---

## 2. Architecture — final design (locked)

The encoder of Section 3.4 of the draft is reused without modification:
per-variate irregular SSM → shared-grid alignment → $L$ alternating
(variable-axis attention + temporal Mamba) blocks. Output is
$H^{(L)} \in \mathbb{R}^{K \times V \times d}$ plus the availability mask
$m_k^{(n)} \in \{0,1\}$ marking grid bins where variate $n$ has at least one
real observation by grid-time $s_k$.

The forecasting query readout is **replaced** by a two-stage hierarchical
attention pool (HAN-style; \citet{yang2016hierarchical}):

**Stage 1 — per-variate temporal pool.** A single learnable query
$q_t \in \mathbb{R}^d$ (shared across variates) scores each $(k, n)$ slot:
- score: $s^t_{k,n} = q_t^\top H^{(L)}_{k,n} / \sqrt{d}$ (Bahdanau scale)
- mask: bins with $m_k^{(n)} = 0$ (null-state) are softmaxed to $-\infty$
  (degenerate $V$ where no slot is available falls back to uniform to avoid NaN)
- pool: $\alpha^t = \mathrm{softmax}_k$, $H^{(v)}_n = \sum_k \alpha^t_{k,n} H^{(L)}_{k,n}$

**Stage 2 — variate-axis pool (after LayerNorm).**
- normalize: $\tilde H^{(v)}_n = \mathrm{LayerNorm}(H^{(v)}_n)$
- score: $s^v_n = q_v^\top \tilde H^{(v)}_n / \sqrt{d}$ with learnable $q_v$
- pool: $\alpha^v = \mathrm{softmax}_n$, $z = \sum_n \alpha^v_n \tilde H^{(v)}_n$

**Classifier.** $y = W \cdot \mathrm{LayerNorm}(z) + b$, $W \in \mathbb{R}^{C \times d}$.

**No dropout in the head**, matching the dropout-free classifier convention of
S5 \citep{smith2023simplified} and RoMAE \citep{zivanovic2025rotary}; weight
decay is the sole regularizer. Code:
[mamba_mv/classification_head.py](../mamba_mv/classification_head.py).

Why this head, not MLP gates / mean pool / CLS token:
- Hierarchical attention is the standard mask-friendly pool when both the time
  axis and the variate axis are heterogeneously informative; a flat mean over
  $K \cdot V$ ignores per-variate prominence and is hurt by the null-state mask
  noise.
- Single-query attention pools (no extra MLPs) keep parameter count negligible
  ($\sim 3\text{K}$ at $d=256$) and are the form used by HAN, EnvBERT, and
  most "attention pool" implementations in time-series classification.
- LayerNorm before stage 2 stabilizes the pooled magnitudes from variates with
  very different sparsity (e.g. Heartbeat where most channels are dense vs.
  variates that are mostly null-state).

---

## 3. Optimization

| | |
|---|---|
| optimizer | AdamW |
| weight decay | 0.05 |
| schedule | cosine decay, 10% linear warmup |
| max\_epochs | 800 |
| early stopping | val/acc, patience 50 |
| precision | `32-true` (Mamba CUDA kernel asserts `delta.dtype == u.dtype`; bf16-mixed breaks this) |
| seeds (final) | 42, 123, 0 |
| HPO seed | 42 |

Per-dataset hyperparameters inherited from RoMAE Table 12:

| dataset | batch | label\_smoothing (PyTorch $p$) | grad\_clip | grid\_K |
|---|---|---|---|---|
| BasicMotions | 8 | 0.0 | 1.0 | 128 |
| CharacterTrajectories | 16 | 0.1 | 1.0 | 128 |
| Epilepsy | 16 | 0.2 | 1.0 | 128 |
| Heartbeat | 16 | 0.0 | 2.0 | 128 |
| LSST | 16 | 0.1 | 10.0 | 64 |

**Note on label\_smoothing convention.** RoMAE App. A.1 reports a *confidence*
$c$ ("reducing each correct class label from $1$ to a confidence value $c$");
PyTorch `F.cross_entropy(label_smoothing=p)` takes the *smoothing amount* $p$.
Mapping: $p = 1 - c$. RoMAE $c \in \{1.0, 0.9, 0.8, 1.0, 0.9\}$ →
PyTorch $p \in \{0.0, 0.1, 0.2, 0.0, 0.1\}$. We caught this when an early
smoke run on BasicMotions stalled at val\_loss $= \ln(4)$ (uniform-target floor)
with $p = 1.0$ (full smoothing = no signal).

**Heartbeat memory.** $V = 61$ × the variable-axis attention forced an OOM at
`grid_K=256, batch=32`. Reduced to `grid_K=128, batch=16` — fits in 32G with
H100, and matches the other "ample-budget" datasets.

---

## 4. HPO design

**Sweep axis.** Learning rate only, $\{3\times10^{-4},\,10^{-3},\,3\times10^{-3}\}$.
All other hyperparameters are inherited from RoMAE Table 12 to make this a
head-to-head comparison rather than per-method tuning. Justification: the user
explicitly preferred the same-recipe-for-all-models discipline (see memory:
"Fair-comparison preference").

**Grid size.** $5 \text{ datasets} \times 2 \text{ dt\_modes} \times 3 \text{ LRs}
\times 1 \text{ HPO seed} = 30$ runs.

**Selection.** Per (dataset, dt\_mode), pick the LR with highest validation
accuracy on the HPO seed. Aggregator:
[eval/aggregate\_uea\_hpo.py](../eval/aggregate_uea_hpo.py) writes
`winner_configs.json` and a markdown summary.

**Final eval.** $5 \times 2 \times 3 \text{ seeds} = 30$ runs at the picked LR.
Report mean $\pm$ std test accuracy, plus macro-F1 for the imbalanced datasets
(LSST, Epilepsy, Heartbeat).

**Why both `dt_mode = learned` and `replace`.** These are two hard-coded
encodings of the irregular $\Delta t$ in our Section 3.4: `replace` substitutes
the missing-bin token with a learned null state $h_\emptyset^{(n)}$, while
`learned` learns a positional embedding for the inter-arrival gap. They are
the only meaningful architectural axis that lives inside our model and is not
shared with all baselines, so reporting both keeps the ablation honest.

**Compute envelope.** ~22 GPU-hours total on H100, ~1 day end-to-end with
parallel queue.

---

## 5. v2 addendum (2026-05-01) — protocol fixes

### 5.1 What v1 got wrong

1. **Picker metric on imbalanced data.** Aggregator selected by best val/acc.
   For binary HB (~60/40 imbalance), a model collapsed to majority-class
   prediction scores val_acc ≈ 0.72 (the majority fraction) — *higher* than a
   slightly-wrong actual learner. Result: aggregator picked the **collapsed**
   `replace × LR=3e-3` cell as the HB winner. All 3 final seeds inherited it
   and produced identical test_acc=0.722 with macro_f1=0.42 (the constant-
   predictor signature).
2. **LR grid not data-aware.** `{3e-4, 1e-3, 3e-3}` was wrong on both ends:
   too high for HB (caused the collapse), too low for LSST (RoMAE used 3e-2;
   our cap was an order of magnitude lower).
3. **batch_size not swept.** Inherited from RoMAE Table 12 (BS values tuned for
   SGD+momentum); unjustified import under our AdamW recipe.
4. **No head regularization.** RoMAE Table 12 prescribes `dropout=0.2 +
   stochastic_depth=0.2` on EP and LSST (the overfit-prone cells); we used
   neither.

### 5.2 What v2 changes

**Aggregator** ([eval/aggregate_uea_hpo.py](../eval/aggregate_uea_hpo.py)):
- Per-dataset picker metric: `val/macro_f1` for HB, EP, LSST (imbalanced);
  `val/acc` for BM, CT (balanced).
- **Collapse detector**: any cell with
  `val_acc − val_macro_f1 > 0.2` is logged to stderr and excluded from the
  selection pool. Catches majority-class collapse regardless of which metric
  is the primary picker.
- Records `batch_size` and `head_dropout` in `winner_configs.json` so the
  final-eval array can replay the full picked config.

**Classification head** ([mamba_mv/classification_head.py](../mamba_mv/classification_head.py)):
- New `head_dropout: float = 0.0` arg. A `nn.Dropout(p)` layer sits between
  `LayerNorm(z)` and `Linear` when `p > 0`. No other architectural change —
  pooling order (time-within-variate, then variate, then linear) is unchanged.

**Validation logging** ([mamba_mv/multivariate_classifier.py](../mamba_mv/multivariate_classifier.py)):
- `validation_step` now stashes `(labels, preds)`; `on_validation_epoch_end`
  computes and logs `val/macro_f1`. `train_cls.py` captures it into
  `summary.json`.

**Per-dataset config** ([mamba_mv/train_cls.py](../mamba_mv/train_cls.py),
`ROMAE_DATASET_DEFAULTS`):

| Dataset | head_dropout | rationale |
|---|---|---|
| BasicMotions | 0.0 | already saturated; v2 excludes BM from rerun |
| CharacterTrajectories | 0.0 | RoMAE Table 12 also uses 0 for CT |
| Epilepsy | **0.2** | matches RoMAE Table 12 |
| Heartbeat | 0.0 | RoMAE Table 12 uses 0 for HB |
| LSST | **0.2** | matches RoMAE Table 12 |

### 5.3 v2 HPO grid (4 datasets, BM excluded)

| Dataset | LR grid (v2) | BS grid (v2) | picker (v2) |
|---|---|---|---|
| CharacterTrajectories | {3e-4, 1e-3, 3e-3}    | {8, 16, 32} | val/acc |
| Epilepsy              | {3e-4, 1e-3, 3e-3}    | {8, 16, 32} | **val/macro_f1** |
| Heartbeat             | **{1e-4, 3e-4, 1e-3}**  | {8, 16, 32} | **val/macro_f1** |
| LSST                  | **{1e-3, 3e-3, 1e-2}** | {8, 16, 32} | **val/macro_f1** |

Bold = changed from v1. HB shifted **down** (avoid collapse-prone region);
LSST shifted **up** (RoMAE used 3e-2 with SGD; under AdamW we map roughly to
1e-2 ceiling).

**Grid size.** 4 ds × 2 dt × 3 LR × 3 BS × 1 HPO seed = **72 cells**.
**Final eval.** 4 ds × 2 dt × 3 seeds = **24 cells**.
**Compute envelope.** ~73 GPU-hours total, ~2 days end-to-end with parallel
queue.

### 5.4 v2 SLURM artifacts

```
imts_benchmark/scripts/run_uea_cls_hpo_v2.sbatch         # 72-cell HPO array
imts_benchmark/scripts/run_aggregate_uea_hpo_v2.sbatch   # CPU aggregator
imts_benchmark/scripts/run_uea_cls_final_v2.sbatch       # 24-cell final array
imts_benchmark/scripts/submit_uea_cls_pipeline_v2.sh     # chained submitter
```

Output root for v2: `output/log/imts_benchmark_v2/uea_cls_v2/` (separate from
v1 `uea_cls/` so results don't collide).

### 5.5 What v2 does NOT change (deliberately)

- **Pooling order** in the head. Still time-within-variate → variate → linear.
  v1 results show this is sufficient for BM/CT/EP; the trouble cells are
  protocol bugs, not capacity bugs. Head-architecture redesign is gated on
  whether v2 closes the HB/LSST gap.
- **No pre-training.** Still train-from-scratch with AdamW. RoMAE pre-trains;
  we do not. The paper write-up will continue to flag this protocol asymmetry.
- **`max_epochs = 800, patience = 50`** unchanged. RoMAE uses per-dataset
  epochs (15–150). The longer schedule is more conservative; with EarlyStopping
  the realized epochs are usually well below 800.

### 5.6 Decision tree after v2

If v2 closes HB/LSST gap → ship the table, write up the protocol-asymmetry
caveat. No more head changes for the paper.

If v2 still trails on HB/LSST → escalate to head-architecture work:
- multi-query attention pool (cheapest upgrade)
- joint K·V single-query pool (test whether the two-stage decomposition is
  the limit)
- full QKV self-attention with [CLS] readout (RoMAE-style head; most
  expensive, only if cheaper variants plateau)

---

## 6. SLURM pipeline (v1)

Three chained array jobs under `imts_benchmark/scripts/`:

1. `run_uea_cls_hpo.sbatch` — array `1-30`, `gpu:h100:1`, 2h walltime.
2. `run_aggregate_uea_hpo.sbatch` — CPU job, `afterok` on (1).
3. `run_uea_cls_final.sbatch` — array `1-30`, same GPU spec, `afterok` on (2).

`submit_uea_cls_pipeline.sh` wires the dependency chain. Index decoding inside
each array job:
- HPO: `dataset_idx = k // 6; dt_idx = (k % 6) // 3; lr_idx = k % 3`
- Final: `dataset_idx = k // 6; dt_idx = (k % 6) // 3; seed_idx = k % 3`

Outputs land under
`/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/uea_cls/`
in `hpo/<dataset>/<dt_mode>/lr<LR>/hpo/summary.json` and
`final/<dataset>/<dt_mode>/seed<S>/final/summary.json`.

---

## 7. Status (2026-04-30, v1 results)

- Smoke runs done: BasicMotions val=1.0/test=0.95, Heartbeat val=0.725/test=0.61.
- Pipeline submitted: HPO=6829456, Aggregator=6829457, Final=6829459.
- First two HPO results (BM × replace):
  - lr=3e-4: val=1.0, **test=1.0**, macro-F1=1.0
  - lr=1e-3: val=1.0, test=0.875, macro-F1=0.875
- Remaining 28 HPO tasks pending (`Priority` queue reason).

When the final array completes, results land in `final/.../summary.json` and
will be summarized in `RESULTS_uea_classification_<date>.md` alongside the
RoMAE Table 4 head-to-head.
