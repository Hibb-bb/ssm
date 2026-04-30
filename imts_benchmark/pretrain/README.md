# Mamba IMTS Pretraining — Design Notes & Concerns

This document is the **decision log** for the multivariate Mamba pretraining
pipeline that lives in `hongyu/ssm/imts_benchmark/pretrain/`.  It is meant to
be (a) a reference for everyone touching this code, and (b) raw material for
the eventual paper.  Every non-obvious choice we make is recorded here with
its rationale, the alternatives we considered, citations to prior work, and
the open questions we still need to resolve.

If you change a knob, **also update this file**.

> Status: 2026-04-29.  Single-phase Stage A (`single_phase`) run completed
> early (step 44K / 80K) due to a bounded eval blow-up on GDELT/ClusterTrace.
> See **§7.K** for results, **§7.L** for the eval fix, **§7.M** for the
> LOTSA stall root cause (89% of LOTSA weight is on 2 univariate datasets),
> and **§7.N** for the Moirai-style sampling fix landed today (solar/wind
> 89% → 2.3% of mass; V=9..16 model-seen 9% → 41.6%; cosine 30K LR replaces
> multistep 80K).  Next launch: `single_moirai_v2_synth30_lotsa70_regimeMix40_d384_cosine30000_warmup500_s42`,
> awaiting user approval.

------------------------------------------------------------------------

## 1. Goals & scope

Build an iteratively-changeable IMTS pretraining infrastructure that can:

1. Pretrain `MultivariateMambaForecaster` (and its `Sandwich` variant)
   on a mixture of:
   - LOTSA real time-series data (`/home/ubuntu/lotsa_data`),
   - synthetic multivariate data (`chronos2_synth`, `kernelsynth_irregular`).
2. Support a Moirai-style task distribution (Woo et al., 2024,
   §3.1: "instead of a single context length, multiple context lengths
   are sampled from a task distribution"; `arXiv:2402.02592`) and a
   Chronos-2-style heterogeneous-task curriculum (Stella et al., 2025).
3. Stay swap-friendly: changing a stage YAML (or a single CLI flag)
   should let us re-run an ablation with no code edits.
4. Validate / iterate against the same downstream IMTS benchmarks we
   already evaluate the model on (`activity`, `ushcn` — same protocol
   as `scripts/hongyu_lambda_script/run_mamba_mv_real_pipeline_local.sh`).

------------------------------------------------------------------------

## 2. Pipeline at a glance

```
sources.py           ─┐
  LOTSAHFSource       │
  ChronosSynthSource  ├─►  mixed_dataset.py  ─►  collate.py  ─►  train_pretrain.py
  KernelSynthSource   │   (logical-source     (pad to V_max=20,    (Lightning)
                     ─┘    weighted mixing,    instance-norm,
                           per-source priors,  randomize var slots)
                           Moirai-style cap)
                                  ▲
                                  │
                       task_sampler.py + degradation.py
                       (per-example V, W, history, regime;
                        on-the-fly degradation for LOTSA)
```

The model expects the dense batch produced by `collate.py`:

| Tensor                | Shape                | Notes                                                  |
| --------------------- | -------------------- | ------------------------------------------------------ |
| `values`              | `[B, V_pad, L_max]`  | per-(b,v) z-scored using context-only stats            |
| `timestamps`          | `[B, V_pad, L_max]`  | per-sample normalized to `[0, 1]`                       |
| `deltat`              | `[B, V_pad, L_max]`  | per-sample normalized; first per-variate entry = 0     |
| `valid_mask`          | `[B, V_pad, L_max]`  | true at observed (non-padded) positions                |
| `pred_mask`           | `[B, V_pad, L_max]`  | `valid ∧ ts ≥ history ∧ target_variate_mask`           |
| `valid_variate_mask`  | `[B, V_pad]`         | true for non-padded variate slots                      |
| `target_variate_mask` | `[B, V_pad]`         | true for variates we supervise                         |
| `time_scale`, `freq`  | `[B]`, list[str]     | preserved for future scale-conditioning                |

------------------------------------------------------------------------

## 3. Design choices and discussion (the six concerns)

### 3.1 Per-sample timestamp normalization to `[0, 1]`

**What we do.**  Inside `degradation.py::LOTSAToIrregular`, every per-variate
timestamp array is mapped to `[0, 1]` *per sample* by dividing by the window
duration.  `delta_t` is rescaled by the same factor.  `time_scale` (the
original duration in source units) and `freq` (e.g. `"5T"`, `"D"`, `"Y"`) are
preserved in the batch dict for optional model conditioning.

**Why.**  Three reasons:

1. **Architectural constraint.** `mamba_mv/shared_grid.py::SharedGridAligner`
   discretizes continuous time onto a fixed grid `linspace(0, t_max, K)`.
   With `K` fixed at construction, the *grid spacing relative to obs density*
   is the only thing that matters for the model.  If we feed raw absolute
   times, the same `(K, t_max)` would mean very different things for a 5-min
   PEMS sample vs a yearly CMIP6 sample.  Per-sample `[0,1]` normalization
   makes "obs density per grid cell" invariant to source frequency.

2. **It matches the closest IMTS-FM precedent.**  mTAN
   (Shukla & Marlin, 2021, ICLR; `arXiv:2101.10318`, §3.1) literally does
   this: "we rescale time to lie in $[0, 1]$ … the reference time points
   $r_k$ are chosen as a uniform grid on $[0, 1]$".  Our `SharedGridAligner`
   is a learned analogue of mTAN's reference-grid attention; the
   normalization choice carries over for the same reason.

3. **Cross-frequency mixing.**  LOTSA contains datasets at frequencies
   from sub-second (`100ms`) to yearly.  Without per-sample normalization
   any single learned positional / continuous-time encoding has to span 12
   orders of magnitude.

**What major TSFMs do (literature scan).**

| Model      | Time treatment                                                        | Source                        |
| ---------- | --------------------------------------------------------------------- | ----------------------------- |
| Moirai-1/2 | No continuous time axis.  Patches are uniform on a regular grid; frequency is a learned categorical embedding ("Any-Variate" attention; §3.2). | Woo et al. 2024, `arXiv:2402.02592` |
| Chronos-1  | No time axis.  Series is tokenized (mean-scaled then quantized).      | Ansari et al. 2024, `arXiv:2403.07815` |
| Chronos-2  | Same: mean-scale + quantize values; horizon implicit via task token.  | Stella et al. 2025            |
| Time-MoE   | Token-level transformer with RevIN; no continuous time axis.          | Shi et al. 2024, `arXiv:2409.16040` |
| TimeGPT    | Time encoded as relative position; no `[0,1]`.                        | Garza et al. 2024             |
| **mTAN**   | **Reference times in `[0,1]`; continuous-time attention.**            | **Shukla & Marlin 2021, `arXiv:2101.10318`** |
| Latent-ODE | Raw absolute time (in dataset's natural unit).                        | Rubanova et al. 2019          |
| GRU-D / NCDE | Raw absolute time.                                                  | Che et al. 2018; Kidger et al. 2020 |
| tPatchGNN  | Time normalized within each window by the prediction-horizon length.  | Wang et al. 2024 (ICML)       |

**Take-away.** Major *value-tokenizing* TSFMs (Moirai, Chronos, Time-MoE)
do not face this question because they do not consume continuous time.
Among IMTS-specific models that *do*, normalized time on `[0,1]` is the
standard choice (mTAN, tPatchGNN dataset-relative).  Our design is in
the IMTS lineage, not the value-tokenizer lineage.

**What we lose, and how we plan to recover it.**  Absolute scale info
(is "0.5" half a year or half a minute?) is no longer in `timestamps`.
We mitigate this in two ways:

1. The collate already returns `time_scale` and `freq` per sample.
2. **OPEN**: we currently **do not feed those into the model** — that is
   a TODO listed in §6.  Doing so requires a small `freq_embedding +
   log_time_scale` head fused into `multivariate_forecaster.py`.  Without
   it, the model has no way to distinguish "half-day" from "half-year".
   This is the single biggest weakness of the current `[0,1]` choice.

### 3.2 On-the-fly degradation: GPU-idle risk?

**What we do.**  `LOTSAToIrregular` and the regime sampler run inside the
DataLoader worker process (CPU).  Each batch is degraded fresh.

**Why.**  Iteration speed.  We can change `regime_dist`, drop fractions,
jitter, etc. in YAML and re-launch with no data prep — critical for
ablations.

**Risk.**  CPU-bound preprocessing can starve the GPU.

**Empirical evidence so far.**

- Stage A smoke (synthetic only, `num_workers=4`, `bf16-mixed`,
  H100 80 GB): GPU ran at **~98 %** utilization (see
  `terminals/3.txt`).  Loss decreased from ~1.7 → ~0.9 in 200 steps.
- Stage B smoke (mixed synthetic + LOTSA, same workers): step time
  was ~10× slower, dominated by **HF dataset disk reads**, not by
  `degradation.py`.  GPU utilization dropped (we did not record
  exact numbers; this is the bottleneck to plan around).

**Diagnosis.**  Augmentations are small NumPy ops on a window of at most
`~512 × 20 ≈ 10⁴` floats — single-digit milliseconds per sample.  The
real bottleneck is HF datasets indexing into LOTSA Arrow shards;
Hugging Face streams 1 row at a time and does not batch reads.

**Mitigations (in order of recommended adoption).**

1. **More workers.**  At 4 workers we are leaving CPU on the table.
   Bumping to `num_workers=12–16` with `prefetch_factor=4` should
   close the gap on the H100 box.  This is a config change only.
2. **Dataset capping by IO**, not just by sample count.  The Moirai
   weight map already deprioritizes the large 1B-sample LOTSA sets;
   `lotsa_exclude` in `sources.yaml` already drops the largest ones.
3. **Materialize a pre-degraded LOTSA cache** (one-time build, then
   read sequentially from Arrow).  We deliberately do not do this
   yet because it removes the on-the-fly variability we want for
   exploration; we'll only do it once we have settled on regime
   priors.
4. **Smaller LOTSA effective subset** for ablations.  Stage A is
   synthetic-only by design; CPU is not a concern there.

**Decision.**  Keep on-the-fly for now; track GPU util in W&B; switch
to a cached path later if GPU drops below ~70 %.

### 3.3 `TimestampJitter` — does it make sense?

**Your concern.**  Real-world observation times are usually unambiguous;
jittering them might distort the empirical distribution the model sees.

**Where it currently fires.**

| Source           | Stage-A regime mass on (sync, mixed, async) | Jitter prob inside that regime |
| ---------------- | ------------------------------------------- | ------------------------------ |
| `chronos2_synth` | `0.10 / 0.45 / 0.40`                        | `0.30 / 0.50` in mixed/async   |
| `kernelsynth`    | `0.10 / 0.40 / 0.45`                        | `0.30 / 0.50` in mixed/async   |
| `lotsa_degraded` | `0.20 / 0.55 / 0.25`                        | `0.30 / 0.50` in mixed/async   |
| `lotsa_regular`  | `1.0 / 0 / 0`                               | never                          |

So today, **~28 % of `lotsa_degraded` samples have their timestamps
jittered**.  That is plausibly noise the model will not see at test time
on `activity` / `ushcn`, where timestamps are exact (study seconds /
months).

**Discussion.**

- For *synthetic* sources we have full control over the latent
  generative process; jitter is a free augmentation that only affects
  observation-time, not the underlying signal.  It makes the model
  robust to tiny mis-recordings (which exist in some real datasets,
  e.g. ICU vitals can drift by minutes).
- For *real* LOTSA we are claiming "time = $t_i$" and then telling the
  model "actually time = $t_i + \varepsilon$".  This is, in effect,
  **adding observation-time noise that does not exist downstream**.

**Proposed change (OPEN, asking).**  Restrict `TimestampJitter` to
synthetic sources only.  Concretely, set `jitter = None` in the
LOTSA paths inside `degradation.REGIMES`, or — cleaner — make jitter
a per-source-overridable knob in `task_sampler.py` and turn it off in
the LOTSA overrides.  See §7 "Pending change requests".

**Counter-argument (why we might still want a *little* jitter on LOTSA).**
A handful of real datasets do have measurement-time uncertainty
(MIMIC vitals, sensor stamping under clock drift).  We could keep jitter
on at very low magnitude (`jitter_fraction=(0.0, 0.05)` and
`apply_prob ≤ 0.10`) for `mixed` only, off for `async`.  Listed as
"option B" below.

### 3.4 Validation / testing during pretraining

**What we do today.**  An optional small *synthetic-only* val loop
(`val_steps > 0` in `PretrainDataModuleArgs`) that tracks loss on a
held-out chronos2/kernelsynth stream.  No real-data validation.

**Why this is not enough.**

- Pretraining loss can drop while downstream IMTS test MSE goes up
  (overfit to synthetic statistics).  Chronos-2's own ablation
  (Stella et al. 2025, "Synthetic-only" row) shows that synthetic-only
  pretraining is competitive on GIFT-Eval but visibly worse on
  fev-bench; we want to catch that gap during training, not after.
- Our fundamental measure of success is downstream forecasting MSE on
  the IMTS benchmarks we already use.  We should track exactly that.

**Proposed addition (OPEN, asking).**

1. Add a `DownstreamIMTSValDataset` that wraps the existing
   `tpatchgnn_data/{activity,ushcn}/{val,test}` HF datasets (already
   present, in the same sparse-flat schema we use in `train_mv.py`),
   pads to `V_pad = 20`, normalizes time per-window to `[0,1]`, and
   uses the IMTS-benchmark `history` field unchanged.  Code reuse
   with `shared_data/multivariate_datamodule.py::collate_per_variate`
   is straightforward; we just need a thin adapter that:
   - sets `valid_variate_mask = [True]*V_real + [False]*(20-V_real)`,
   - rescales `timestamps` to `timestamps / time_max`,
   - rescales `history` the same way,
   - leaves *values* in the dataset's natural units (no
     instance-norm here so val MSE is comparable to existing
     benchmarks).
2. Run val every `N` training steps (e.g. 2 000) with
   `Trainer(val_check_interval=N, num_sanity_val_steps=0)`.  Log
   `val/mse_<ds>` and `val/mae_<ds>` for `ds in {activity, ushcn}`.
3. Use `val/mse_activity + val/mse_ushcn` (sum or mean) as the
   ModelCheckpoint monitor → "best so far during pretraining".

**PhysioNet caveat.**  `tpatchgnn_data/physionet/norm_stats.json`
reports `n_vars=41`, which exceeds our `max_dim=20`.  Two options:

- **(a)** Skip physionet during pretraining val (consistent with the
  HPO pipeline that already skips physionet — see `hongyu_lambda_script
  /run_mamba_mv_real_pipeline_local.sh:14`).
- **(b)** Bump `max_dim` to a value that fits everything we want to
  validate on (e.g. 50).  This adds ~1 % of model params (variate
  embedding table grows from `[20, d_model]` to `[50, d_model]`); the
  cost is negligible.

**IMM-TSF / Time-IMM benchmark.**  The user mentions
[IMM-TSF](https://github.com/blacksnail789521/IMM-TSF) (Chang et al.
2025, NeurIPS D&B) which is a benchmark *library* whose data is
distributed via two equivalent channels:

- canonical:
  [Time-IMM GitHub](https://github.com/blacksnail789521/Time-IMM)
  (~77 MB, plain CSVs, no Git LFS, no auth);
- mirror:
  [Kaggle dataset](https://www.kaggle.com/datasets/blacksnail789521/time-imm)
  (same files, but requires Kaggle account/API token).

**Decision: pull from GitHub.**  No login, no rate-limit, no licensing
surprise from Kaggle's TOS, and the GitHub layout is exactly what
`IMM-TSF/lib/parse_datasets.py` consumes — so reproducing their numbers
is a matter of byte-equivalent input.  The IMM-TSF README itself lists
both as interchangeable; GitHub is the canonical author distribution.

**What ships from Time-IMM.**  Eight datasets out of nine: `CESNET`,
`ClusterTrace`, `EPA-Air`, `FNSPID`, `GDELT`, `ILINet`, `RepoHealth`,
`StudentLife`.  `MIMIC` is excluded by the data-use agreement, as
noted in the Time-IMM README, and we skip it.

**Splits — match the paper exactly.**  IMM-TSF's `parse_datasets.py`
exposes two split modes; the default used in the paper's batch
pipeline (`main_all.py:128`) and the `main.py` "regular run" branch
(`main.py:1227–1228`) is `split_method="sample"`:

> per-record temporal split, 60 % train / 20 % val / 20 % test,
> chunk-by-chunk in chronological order

The other mode (`"instance"`) is annotated *"only for in-domain
transfer learning"* in `main.py:1228`.  We mirror their default —
`sample` — so our test-MSE is directly comparable to their reported
numbers.  Concretely (`parse_datasets.py:715–730`):

```python
elif args.split_method == "sample":
    grouped = defaultdict(list)
    for i, (cid, *_) in enumerate(all_chunks):
        rec_id, idx_str = cid.rsplit("_chunk", 1)
        grouped[rec_id].append((int(idx_str), i))
    for rec_id, lst in grouped.items():
        lst.sort(key=lambda x: x[0])
        N = len(lst)
        t_end = int(N * 0.6)
        v_end = int(N * 0.8)
        train_idx += [i for _, i in lst[:t_end]]
        val_idx   += [i for _, i in lst[t_end:v_end]]
        test_idx  += [i for _, i in lst[v_end:]]
```

**Per-dataset chunk parameters — copy from `update_args_for_dataset`
(`main.py:788–836`).**  These determine which `(history, pred_window,
stride, time_unit)` defines a chunk; we must use the same to be
table-comparable:

| Dataset      | history | pred_window | stride | time_unit |
|--------------|---------|-------------|--------|-----------|
| GDELT        | 14      | 14          | 14     | days      |
| RepoHealth   | 31      | 31          | 31     | days      |
| FNSPID       | 31      | 31          | 31     | days      |
| ClusterTrace | 12      | 12          | 12     | hours     |
| StudentLife  | 31      | 31          | 31     | days      |
| ILINet       | 36      | 36          | 4      | weeks     |
| CESNET       |  7      |  7          |  7     | days      |
| EPA-Air      |  7      |  7          |  7     | days      |
| MIMIC*       | 24      | 24          | 24     | hours     |

\* MIMIC excluded.  ILINet is a single-record dataset; the temporal
`sample` split is the only sensible one (matches the paper).

**Three reproduction-fidelity caveats** (from `parse_datasets.py` and
`evaluation.py`):

1. *Their per-record normalization is global*: `(x − x.mean()) /
   x.std()` over the **entire** record before chunking
   (lines 103–111).  This is *not* the context-only instance-norm we
   use during pretraining.  For paper-comparable IMM-TSF eval we
   apply the **same** global normalization on the offline-converted
   data; our pretraining val on `activity` / `ushcn` keeps
   context-only norm because that's what tPatchGNN's loader does for
   those.

2. *Chunks with no text in the history window are dropped even when
   `enable_text=False`* (lines 217–221).  This is a quirk of their
   loader: setting `enable_text=False` returns empty texts but the
   filter still runs.  We must reproduce this drop in the converter
   (i.e. drop chunks whose record has no text overlap with the
   history span), otherwise our test set is strictly larger than
   theirs and the numbers stop being comparable.  The text *content*
   is irrelevant to us; only its presence is checked.

3. *Their reported MSE / MAE are in z-scored space, not original
   units.*  `IMM-TSF/lib/evaluation.py::compute_error` (lines 27–30)
   computes `(truth_repeated − pred_y) ** 2 * mask` directly on the
   z-scored tensors that come out of the loader — no inverse
   transform.  Original-units numbers across IMM-TSF datasets would
   span ~12 orders of magnitude (EPA-Air ~10², ClusterTrace ~10⁶,
   FNSPID volume ~10⁷), which is one signal that the paper Table is
   in z-scored space.  Our val loop therefore logs **both**:
   - `val/mse_<ds>` — denormalized, in original units (matches the
     existing `train_mv.py` reporting style; useful to eyeball model
     health and easier to interpret per-dataset).
   - `val/mse_z_<ds>` — z-scored, no denormalization (paper-Table
     comparable; this is the column to copy into the paper's MSE
     row).
   `val/mae_*` mirrors the same split.

**Implementation plan (after sign-off, see §7.B′).**  One-shot
converter `scripts/convert_imm_tsf_to_sparse.py` that, for each of the
eight datasets:

1. Reads `data/{ds}/processed/{rec}/time_series.csv` and
   `text.csv` exactly the way `IMM-TSF/lib/parse_datasets.py` does;
2. Builds chunks with the per-dataset `(history, pred_window,
   stride, time_unit)`, applies the *no-text-in-history → drop*
   filter to match their chunk count;
3. Applies their *per-record global* z-score on values (storing
   `(mu, std)` per `(record, feature)` for later denormalization
   when reporting metrics);
4. Splits chunks per record with `sample` (60/20/20 chronological);
5. Emits sparse-flat HF parquet under
   `imts_benchmark/data/imm_tsf_sparse/{ds}/{train,val,test}/` with
   the same schema as `tpatchgnn_data/`, plus `time_max` and
   per-record norm stats in `norm_stats.json`;
6. We then add the **test split of each dataset** to the IMTS
   downstream val DataLoader (skipping val/train splits — we don't
   train on IMM-TSF, only evaluate).

Skipping multimodal text is intentional; it's out of scope for the
first FM iteration, and the IMM-TSF code path supports it at runtime
via `--enable_text` so a multimodal extension is a separate
follow-up.

### 3.5 Value normalization (instance / RevIN-style)

**What we do.**  In `collate.py`, for every `(sample, variate)`, values
are standardized using **context-only mean and std** (so future-leak is
impossible).  Variates with fewer than 3 context observations fall back
to all-window stats; below 1 obs we use `(0, 1)`.

This is the IMTS analogue of:

- Chronos-2 mean-scaling: "each input series is mean-scaled by dividing
  by the mean of $|x_t|$ in the context window before tokenization"
  (Stella et al. 2025).
- Moirai standardization: "instance scaling … is applied within each
  series" (Woo et al. 2024, §3).
- RevIN (Kim et al. 2022, ICLR): explicit reversible per-instance
  normalization applied at input, undone at output.

**Why per-variate (not whole-sample).**  IMTS series can have wildly
different scales across variates (e.g. activity sensors: gyroscope ~ ±2
rad/s, accelerometer ~ ±10 m/s²; ushcn: temperature ~ tens of °C,
precipitation ~ mm).  A single shared scale would bury the small-scale
variates.  Per-variate avoids this.

**Why context-only.**  Standardizing on the full window leaks
target-side statistics into the input.  Standard practice in time-series
forecasting with normalization.

**Loss in normalized space.**  We compute Huber/MSE on normalized
values.  Reporting downstream metrics will denormalize predictions
(multiply by the per-(b,v) std, add the per-(b,v) mean) before
comparing to ground truth in original units.  Currently the val loop
does *not* denormalize because we have no val loop on real data yet —
adding the IMTS val loop (§3.4) will require us to *not* re-normalize
real-data values during validation, so val MSE is in dataset units and
directly comparable to numbers from `train_mv.py`.

**Open.**  Should we *also* expose a "no normalization" ablation
switch?  Yes — `standardize_per_var: bool` already exists in
`make_collate`; adding a `--no_instance_norm` flag in `train_pretrain.py`
gives us the ablation for free.

### 3.6 Strategy for systematically adjusting components

This is the most general concern.  Our current plan:

#### 3.6.1 Decision principles

1. **Always evaluate on downstream IMTS test MSE**, not on pretraining
   loss alone.  The model can drive pretraining Huber to 0.05 by
   memorizing synthetic noise; downstream activity MSE is the real
   target.
2. **Change one knob at a time.**  Combinatorial sweeps are
   intractable on this compute budget.
3. **Stage A is the ablation playground; Stage B/C are confirmation.**
   Stage A is small (~50 K steps, synthetic-only, no LOTSA disk IO),
   so each ablation is cheap.  Use Stage A to pick winners before
   spending Stage B/C compute.
4. **Always seed everything**, including DataLoader workers.  Lightning
   `seed_everything(seed, workers=True)` already covers this.
5. **Track the same six metrics for every run**: `train/loss_huber`,
   `val/mse_activity`, `val/mse_ushcn`, `val/mae_activity`,
   `val/mae_ushcn`, GPU util.

#### 3.6.2 Recommended ablation axes (each ~ 50 K Stage-A steps, 1×H100 ≈ 6 h)

| # | Axis                           | Levels                                | Rationale                                              |
| - | ------------------------------ | ------------------------------------- | ------------------------------------------------------ |
| 1 | Loss                           | `huber` (default), `mse`              | Time-MoE §3.3: "Huber for robustness to outliers".     |
| 2 | Time normalization             | `[0,1]` (default), absolute (with `t_max=time_max` per source) | Sanity-check our biggest design call.                  |
| 3 | Instance normalization         | on (default), off                     | Match Chronos-2 ablation pattern.                      |
| 4 | Source mix in Stage A          | 100 % synth (default), 50/50 with regular LOTSA | Test: does any LOTSA from step 0 help or hurt? |
| 5 | Variate slot randomization     | on (default), off                     | Moirai claim that fixed slots → position overfit.      |
| 6 | `regime_dist` (irregularity)   | default, `(0.5, 0.3, 0.15, 0.05)` (regular-heavy), `(0.05, 0.05, 0.4, 0.5)` (async-heavy) | Sensitivity of FM to the augmentation prior.           |
| 7 | TimestampJitter on LOTSA       | off (proposed), on (current)          | Tests whether real-world time-noise hurts.             |
| 8 | `max_dim`                      | 20 (default), 50 (covers PhysioNet)   | Cost is small; might unlock physionet val.             |

For each axis we keep all other knobs at their default.  Pick winners
by `val/mse_activity + val/mse_ushcn`.  Promote the winning Stage-A
config into Stage B (with LOTSA) and re-run only the *new* mix knob.

#### 3.6.3 Compute discipline

- One Stage-A ablation per H100 day.
- Always run the same `seed=42` for ranking; only switch to multi-seed
  (1..5) for the final winner of each axis.
- Persist every run's config + git SHA + W&B run-id under
  `pretrain/runs/<axis>_<level>/`.
- Use Lightning's `ModelCheckpoint` with the downstream val metric as
  monitor; load the best checkpoint into the IMTS benchmark
  (`hongyu_lambda_script`) for the final fine-tune / linear-probe
  evaluation.

#### 3.6.4 What to NOT touch in the first ablation pass

To keep the search budget tractable, we freeze these for now:

- Architecture (`d_model=384`, `d_hidden=384`, `n_perv_layer=3`,
  `n_fusion_blocks=3`, `grid_K=128`, `d_state=16`).
- Optimizer (AdamW + cosine, `lr=5e-4`, `wd=0.01`,
  `gradient_clip_val=1.0`).
- Mixed precision (`bf16-mixed`).
- `max_dim=20` (only revisited under axis 8).

Once axes 1–7 are settled, we will revisit architecture + optimizer in
a second pass with the chosen data/loss/normalization recipe fixed.

------------------------------------------------------------------------

## 4. Pretraining stage curriculum (recap)

Stages defined in `pretrain/configs/stage_{a,b,c}.yaml`.  Plan §5
verbatim, paraphrased here for paper-writing convenience:

| Stage | Mix                                                      | Window range (per-variate steps) | Goal                                                                 |
| ----- | -------------------------------------------------------- | -------------------------------- | -------------------------------------------------------------------- |
| A     | 70 % chronos2_synth, 30 % kernelsynth                    | 64 – 256                         | Warm-start on async dynamics, masks, staleness; cheap to ablate on.  |
| B     | 35 % chronos2, 20 % kernel, 30 % LOTSA-deg, 15 % LOTSA-reg | 128 – 512                        | Inject real distributions; main pretraining.                         |
| C     | 25 % / 15 % / 40 % / 20 %                                 | 256 – 1024                       | Long-context post-training; mirror Chronos-2 stage 2 idea.            |

Per-variate window cap (1024 in stage C) is far below the longest
LOTSA series; we sample windows, not whole series.  This matches Moirai
(§3.1, "We sample a window of length $\ell$ uniformly on a per-task
distribution" — paraphrasing).

------------------------------------------------------------------------

## 5. Known limitations / caveats

1. **`max_dim = 20` excludes PhysioNet (n_vars=41).**  Already
   excluded from the existing HPO pipeline.  If we ever pretrain
   for downstream PhysioNet use, bump `max_dim`.
2. **`time_scale` and `freq` are batched but not yet consumed by the
   model.**  This is the *correct* move only if absolute scale
   doesn't matter; we should validate that with a real run, and add
   conditioning if it does (TODO in §6).
3. **Disk IO bottleneck on LOTSA.**  Mitigated by `num_workers`;
   eventually fixed by a degraded-cache.
4. **No multimodal text (IMM-TSF).**  Out of scope for this iteration.
5. **No ablation tracking infrastructure.**  We log per-run, but no
   centralized "axis-i level-j" sweep file.  Plan to add a tiny
   `runs/axis_summary.csv` once we start the ablation grid.

------------------------------------------------------------------------

## 6. TODO (independent of the launch)

- [ ] Feed `time_scale` and `freq` into the model (small embedding
  fused with the first per-variate token).  Required for the `[0,1]`
  normalization to be lossless.  Probably 50 LoC in
  `multivariate_forecaster.py`.
- [ ] Add `--no_instance_norm` flag to `train_pretrain.py`
  (collate already supports it via `standardize_per_var=False`).
- [ ] Add `--no_var_slot_randomize` flag to `train_pretrain.py`
  (collate already supports it via `randomize_slots=False`).
- [ ] Add a tiny `pretrain/runs/axis_summary.csv` updater that
  ingests the W&B run summary at end of fit.

------------------------------------------------------------------------

## 7. Pending change requests

This section is the running checklist for code/config changes between
the design draft (§§1–6) and the first real pretraining run.  As of
2026-04-26, items 7.A–7.C and 7.B′ are landed; 7.D and the first
ablation are pending.

### 7.A  Restrict `TimestampJitter` to synthetic sources only  (✅ implemented)

Your concern in §3.3.  Threaded `apply_jitter: bool = True` through
`StageSpec` and `SourceTaskOverride`, default `True`, and set
`False` in `task_sampler.default_overrides()` for both
`lotsa_regular` and `lotsa_degraded`.  `_stage_to_kwargs` now passes
this flag into `LOTSAToIrregular`, which gates the per-regime
`TimestampJitter` lookup.  Synthetic sources retain jitter.

Verification: 256-window smoke test per source; `lotsa_degraded`
non-lattice timestamp rate dropped from ~28 % to 0 %, while
`chronos2_synth` stayed > 95 %.

### 7.B  Downstream IMTS val on `activity` and `ushcn`  (✅ implemented)

Your concern in §3.4.  New file `pretrain/val_imts.py`
(`IMTSValDataset`, `make_imts_val_collate`, `build_imts_val_loaders`),
plus wiring in `datamodule.py::PretrainDataModule.val_dataloader` and
`train_pretrain.py`.  Logs `val/mse_<ds>` / `val/mae_<ds>` in
**original units** (denormalized using context-only stats) plus
`val/mse_z_<ds>` / `val/mae_z_<ds>` in **z-scored space**.  Cadence
controlled by `--val_check_steps` (default 2000), subset by
`--val_imts_subset` (default 1024).  The model's `validation_step`
detects IMTS batches via the presence of `value_mu` / `value_std` /
`values_orig` keys and skips the synthetic-style `val/huber` /
`val/mse` aggregates for those batches (those metrics are noisy on
unnormalized real-world data and would dilute the synthetic val
loss).

### 7.B′  IMM-TSF / Time-IMM downstream val  (✅ implemented)

Paper-comparable test-set evaluation on the eight Time-IMM datasets
(CESNET, ClusterTrace, EPA-Air, FNSPID, GDELT, ILINet, RepoHealth,
StudentLife).  As of 2026-04-26 this is wired end-to-end; what was
landed:

- **Data on disk**: `git clone https://github.com/blacksnail789521/Time-IMM`
  → `/home/ubuntu/imm_tsf_data` (~77 MB, no Kaggle account).
- **Converter**: `pretrain/scripts/convert_imm_tsf_to_sparse.py`.
  Run once to populate `imts_benchmark/data/imm_tsf_sparse/{ds}/{train,
  val,test}/` (≈39 MB).  Reproduces:
  - per-record temporal 60/20/20 split (`split_method="sample"`,
    `parse_datasets.py:715-730`);
  - per-dataset `(history, pred_window, stride, time_unit)` from
    `update_args_for_dataset` (table in §3.4);
  - per-record GLOBAL z-score, mu/std stored on each row
    (`parse_datasets.py:103-111`);
  - the *no-text-in-history → drop* filter
    (`parse_datasets.py:217-221`).
  Schema is identical to `tpatchgnn_data/{regime}/{split}/`, plus the
  extra `value_mu_per_var` / `value_std_per_var` columns.
- **`pretrain/val_imts.py`**: added `norm_mode={context_only,
  precomputed_per_record}` to `IMTSValDataset` so the same loader
  serves both activity/ushcn (context-only) and IMM-TSF
  (precomputed).  `build_imts_val_loaders` gained a `name_prefix`
  arg so IMM-TSF metrics log as `val/mse_imm_<ds>` (no name collision
  with the existing `val/mse_activity` / `val/mse_ushcn`).
- **`mamba_mv/multivariate_forecaster.py::_log_imts_val`**: now logs
  **both** `val/mse_<ds>` (original units) and `val/mse_z_<ds>`
  (z-scored, paper-comparable; see §3.4 caveat 3).
- **`pretrain/datamodule.py` + `pretrain/train_pretrain.py`**: new
  CLI flags `--val_imm_tsf_data_root`, `--val_imm_tsf_datasets`,
  `--val_imm_tsf_split`, `--val_imm_tsf_subset`,
  `--val_imm_tsf_batch_size`, `--val_imm_tsf_num_workers`.
  Recorded in `ablation.json` for each run.
- **No training on IMM-TSF.**  Pretraining mix unchanged; only the
  test split is added to the live val DataLoader list.

Smoke-tested against all 8 datasets simultaneously on the existing
stage-A smoke config (30 steps, IMTS val every 15 steps): the logged
`val/mse_z_*` metrics fall in `[1.33, 3.05]` for an untrained model,
which is the expected order of magnitude for a randomly-initialized
predictor on z-scored targets.  Original-unit
`val/mse_imm_*` ranges from ~370 (EPA-Air, AQI²) to ~1.9e13
(FNSPID, stock-price²) — exactly why we report both: the original-unit
column is uninterpretable across datasets, and the z-scored column
is what the paper reports.

### 7.C  Ablation tracking: `--ablation_axis` / `--ablation_level`  (✅ implemented)

Helper for §3.6.  `train_pretrain.py` now accepts
`--ablation_axis <name>` and `--ablation_level <name>` flags and dumps
`<output_dir>/ablation.json` containing the full args dict + git SHA
on every run.  Banner echoes the axis/level at start so it's visible
in the SLURM stdout.  If W&B is enabled, the same payload is logged
as hyperparameters for filtering.

### 7.D  GPU / DataLoader profiling pass before launch  (script ready, launch pending)

Run a 5-minute Stage-A and a 5-minute Stage-B with Lightning's
`SimpleProfiler` plus an `nvidia-smi` sampler to confirm the §7.D
thresholds:

- Stage A: GPU util ≥ 95 %, no augmentation hot path > 5 ms/sample.
- Stage B: GPU util ≥ 70 % at `num_workers=12`; if not, raise to 16
  or pre-cache LOTSA.

Implementation landed (2026-04-26):

- `train_pretrain.py` gained `--profiler {simple,advanced,pytorch}`
  (wires Lightning's profilers, output saved next to the run logs)
  and `--max_minutes <float>` (uses `pl.Trainer(max_time=...)` so the
  profiler flushes cleanly at the wall-clock cap; SIGTERM via
  `timeout` would lose the buffered profile output).
- **Architecture aligned with downstream `train_mv.py`**: bumped
  `train_pretrain.py` defaults from `d_model=256` to **`d_model=384`**
  (and `d_hidden=384`).  This was already documented as the pretrain
  arch in §3.6.4, but the CLI defaults silently kept the historical
  256.  Effects:
  - Pretrained checkpoint can be loaded into the fine-tune script
    (`imts_benchmark/mamba_mv/train_mv.py:63-64`) with **no
    architectural surgery** — same param shapes — so the foundation
    model is genuinely *the* model we'll evaluate against the
    activity / ushcn / physionet benchmarks.
  - Param count goes 3.6 M → 7.9 M for the standard arch, matching
    the colleague's PhysioNet HPO checkpoint we observed in
    `output/mamba_mv_hpo_real_replace/`.  The (384/256)² ≈ 2.25×
    factor is exactly what the model summary table prints.
  - Precision stays `bf16-mixed` (NOT `32-true` like
    `fair_defaults.py`).  The fp32 requirement in `fair_defaults.py:102-104`
    was working around a CUDA selective-scan kernel dtype mismatch
    that we already fixed in `mamba_mv/multivariate_forecaster.py`
    (`if dt.dtype != x.dtype: dt = dt.to(x.dtype)`).  bf16-mixed is
    ~2-3× faster on H100 with no accuracy regression; we don't want
    to leave that throughput on the table during the (much longer)
    pretraining stage.
- `pretrain/scripts/profile_stages.sh [stage_a|stage_b|both]` runs
  the actual 5-minute pass.  Defaults: Stage A bs=64 nw=8 5 min;
  Stage B bs=32 nw=12 5 min.  Val loaders disabled to isolate the
  steady-state fwd/bwd/data path.  Spins an nvidia-smi sampler in
  parallel (2 s polling).  Since the profile script invokes
  `train_pretrain.py`, it inherits the new `d_model=384` default
  automatically — no per-script override needed.
- `pretrain/scripts/summarize_profile.py --stage_dir <run_dir>`
  parses both artifacts, prints a per-bucket breakdown
  (data / fwd / bwd / optim wall time), the top-10 actions by total
  time, and a PASS/FAIL against the §7.D threshold.

To launch (once the GPU is free)::

    bash imts_benchmark/pretrain/scripts/profile_stages.sh both

    # then for each stage:
    python imts_benchmark/pretrain/scripts/summarize_profile.py \
        --stage_dir imts_benchmark/pretrain/runs/profile/stage_a
    python imts_benchmark/pretrain/scripts/summarize_profile.py \
        --stage_dir imts_benchmark/pretrain/runs/profile/stage_b

Cost: ~10 min wall-clock.  Should be run **before** the axis-4
ablation so we have a comparable throughput number per axis.

#### 7.D.1  First-pass results (2026-04-26, `d_model=384`, bf16-mixed)

**Stage A (synthetic only):**

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| GPU util mean | **97.9 %** | ≥ 95 % | PASS |
| GPU mem peak | 64.9 GB | < 75 GB | PASS |
| DataLoader % of fit | **0.0 %** | < 30 % | PASS |
| Throughput | 2.69 it/s @ bs=64 | — | ~5.2 h to 50 K steps |

The pre-baked synthetic Arrow files load fast enough that the GPU is
saturated.  Stage A is good to go for axis-4 ablation.

**Stage B (chronos2 + kernelsynth + LOTSA-deg + LOTSA-reg):**

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| GPU util mean | **5.6 %** | ≥ 70 % | **FAIL** |
| GPU mem peak | 55.8 GB | < 75 GB | PASS |
| DataLoader % of fit | 26.0 % | < 30 % | borderline |
| `train_dataloader_next` mean | **3.32 s** / call | should be ≪ step time (~110 ms) | **FAIL** |
| Throughput | 0.28 it/s @ bs=32, nw=12 | — | unworkable |

The "DataLoader %" rule of thumb (< 30 %) is **misleading** here: the
profiler buckets are wall-clock-on-main-thread, but most of the LOTSA
pipeline runs on background workers and *blocks* the main thread inside
`train_dataloader_next`.  The sharper signal is GPU util — the GPU is
idle 94 % of the time waiting for LOTSA windows.  We need to move the
LOTSA loading off the critical path before launching axis-4.

**Why this happens.**  Each LOTSA window pays for, in order:

1. HF Arrow record load from a random sub-dataset (disk + decompression);
2. Random-window crop with min-observation enforcement;
3. Live degradation (`RandomDrops`, `RegularGaps`, `ThresholdCutoff`,
   `StartDelay`, etc., with regime/source-conditional priors);
4. Sparse-flat conversion + variate-slot randomization in `collate.py`.

At `bs=32, num_workers=12`, this comes out to ~3.3 s wall-clock per
batch (each worker processes ~2.7 s of work for one window in serial,
since the pipeline is mostly Python and the inter-step interval gives
each worker 12 × bs × ~0.27 s ≈ 100 s of work to do).  Budget: at
2.69 it/s on the GPU side from Stage A, we'd want each batch ready in
≤ 370 ms; we're 9× over.

**Fix candidates (cheapest first):**

1. **Bump `num_workers` 12 → 24** (we have 26 CPUs, 213 GB free RAM).
   Cheap, no code change.  Likely buys ~2× → still won't hit 70 %.
2. **Pre-cache the degraded LOTSA stream** to Arrow files (one-time
   pass, e.g. 100 K windows per stage-mix bucket at the actual
   regime distribution; degradation is then **off the hot path** and
   we re-shuffle / re-augment lightly at load time).  Most expensive
   to implement, biggest win.
3. **Profile the degradation pipeline in isolation** (run a 1-min
   pass of just `LOTSAToIrregular.__getitem__`) to find which
   transform is slow — the priors say `RandomDrops` and `RegularGaps`
   are O(N), but `ThresholdCutoff` may have hidden cost from
   per-variate quantile computation.  Cheap to do, focuses fix #2.

#### 7.D.2  Standalone LOTSA pipeline diagnostic (CPU-only, 2026-04-26)

Script: `pretrain/scripts/profile_lotsa_pipeline.py`. Iterates the
full LOTSA pipeline (HF random access → `_to_2d` → degradation) on
the CPU with no model and no GPU, sampling `--max_per_source` records
from every discovered source.  Output JSON: `runs/profile/lotsa_pipeline.json`.

Result (1640 samples × 164 sources, wall 142 s):

| Stage | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| `hf_random_get` | **79 ms** | 18 ms | 109 ms | **2179 ms** |
| `to_2d` | 5 ms | 0.7 ms | 8 ms | 135 ms |
| `degrade` | **0.5 ms** | 0.5 ms | 0.9 ms | 1.3 ms |
| `yield_total` | **87 ms** | 20 ms | 120 ms | **2370 ms** |

**Headline:** the augmentation / degradation pipeline is essentially
free (0.5 ms/sample, ≈ 0.6 % of yield wall).  All 91 % of the wall
clock is HF Arrow random-access — `ds[i]` decompresses an entire
multivariate record before we crop a small window out of it.

**Distribution is bimodal:** the median LOTSA record loads in 20 ms
(workable), but the long tail is catastrophic:

| Source | ms/sample | Notes |
|---|---:|---|
| `lotsa:wind_power` | **2390** | 22× longer than one GPU step (110 ms) |
| `lotsa:solar_power` | 2375 | same |
| `lotsa:residential_pv_power` | 532 | |
| `lotsa:residential_load_power` | 386 | |
| `lotsa:wind_farms_with_missing` | 160 | |
| ... 159 others | 0.3 - 30 ms | normal |

Stage B's observed `train_dataloader_next = 3.3 s/batch` matches this
exactly: with `bs=32, nw=12`, each worker prepares one full batch
sequentially, `32 × 86.9 ms ≈ 2.78 s` plus dispatch overhead → 3.3 s.
Workers parallelism *cannot* help when the per-sample latency in the
tail (2.4 s) already exceeds the entire GPU step time (110 ms) —
those tail samples block one worker for >20 GPU-steps.

**Why is `ds[i]` slow?** HF Datasets' default `__getitem__` for
`Sequence(float)` features goes through the Python "lists" formatter:
the Arrow `Float32Array` is converted into a Python `list[float]` of
the **full record length** *first*, then numpy will convert that
list back into a `np.ndarray`.  Each step is O(N) Python-level
allocation.  For `wind_power` (one univariate record of 7,397,147
floats) this is two ≈2.3 s passes per access — even with the OS page
cache fully warm.  This is independent of disk I/O.

#### 7.D.3  Moirai's solution: zero-copy PyArrow-direct indexer

Investigated `uni2ts/src/uni2ts/data/indexer/hf_dataset_indexer.py`
(Apache-2 by Salesforce).  Moirai bypasses HF Datasets' Python-list
formatter for sequence columns and goes directly through PyArrow:

```python
pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
chunk = pa_subtable.column("target").chunks[0]
flat = chunk.slice(i, 1).flatten()           # zero-copy; ListArray
arr  = flat.flatten().to_numpy(False)        # zero-copy on Float32 buffer
arr  = arr.reshape(V, -1)                    # for nested Sequence(Sequence(float))
```

Moirai also pairs this with two complementary tricks (loader.py):
proportional sampling (longer series sampled more often, equal
window-time per series) and `PackCollate` bin-packing variable-length
samples into fixed buckets so GPU work is uniform per batch.

**A/B test on this box** (script: `scripts/profile_lotsa_indexer_compare.py`,
sidecar: `runs/profile/lotsa_indexer_compare.json`, cold cache via
`drop_caches`):

| Dataset | Records | Slow (ms) | Fast (ms) | Speedup | Shape OK |
|---|---:|---:|---:|---:|:---:|
| `wind_power` | 1 | 2395 | 1.9 | **1246×** | ✓ |
| `solar_power` | 1 | 2348 | 0.8 | **3091×** | ✓ |
| `residential_pv_power` | 233 | 456 | 0.3 | **1500×** | ✓ |
| `wind_farms_with_missing` | 337 | 141 | 0.3 | **507×** | ✓ |
| `bull` | 41 | 20.4 | 0.3 | 69× | ✓ |
| `spain` | 1 | 20.2 | 0.6 | 31× | ✓ |
| `covid_deaths` | 266 | 0.14 | 0.22 | 0.6× (wash) | ✓ |
| `m4_yearly` | 22,739 | 0.11 | 0.23 | 0.5× (wash) | ✓ |
| **OVERALL** | 28 | **280.4** | **0.35** | **791×** | — |

Throughput projection (bs=32, nw=12, GPU step 110 ms):
- slow path:  0.748 s/batch → **14.7 % GPU util** (matches our
  Stage-B observed 5.6 %; the remaining gap is collate / dispatch)
- fast path:  0.001 s/batch → **100 % GPU util** (compute-bound)

**Conclusion:** the I/O bottleneck is *not* disk, *not* `num_workers`,
*not* degradation cost.  It is HF Datasets' Python-list materialization
of `Sequence(float)` columns.  Moirai's fix is the canonical answer;
porting their ~80-line indexer pattern eliminates 99.9 % of the wall
on the heavy records.  We do not install `uni2ts` (incompatible
dependency pins: `torch<2.5`, `numpy~=1.26`); instead the relevant
Apache-2 code is ported in-tree.

#### 7.D.4  Fix choice — adopt the indexer pattern

| Fix | Mean ms/sample | Batch wait (bs=32, nw=12) | Effort | GPU util est. |
|---|---:|---:|---|---:|
| (a) `num_workers` 12 → 24 | 87 (no change) | 1.39 s | trivial | ~8 % (still bad) |
| (b) Exclude top-5 heaviest from `lotsa_exclude` | ~30 | ~1 s | 5 min | ~10 % |
| (c) (a) + (b) | ~30 | ~0.5 s | 5 min | ~22 % |
| (d) Pre-cache LOTSA windows | ~0.5 | <20 ms | 1-2 days + hrs to build | ~95 % |
| **(e) Adopt Moirai indexer** | **~0.4** | **<20 ms** | **~half a day** | **~95-100 %** |

Selected: **(e)**.  Same GPU util ceiling as (d) but no offline
build step, no extra disk, and we keep the on-the-fly degradation
contract (essential for the experiment-velocity argument in §3.2).

Implementation plan:
1. Port `HuggingFaceDatasetIndexer._pa_column_to_numpy` and
   `_getitem_int` into `imts_benchmark/pretrain/sources.py` as a
   private helper or a thin replacement of the current `_to_2d` +
   `__iter__` path on `LOTSAHFSource`.  Keep the existing public
   API (yields `dict` with `target/freq/item_id/source_name`) so
   the rest of the pipeline (`MixedDataset`, `LOTSAToIrregular`,
   collate, …) is unchanged.
2. Add a regression test that asserts shape + dtype parity with the
   old path on five sources covering univariate / multivariate /
   variable-length / `wind_power`-style giant single-record cases.
3. Re-run `scripts/profile_stages.sh` for Stage B and confirm GPU
   util ≥ 90 %.  Persist the updated profile under
   `runs/profile/stage_b_post_indexer_fix/` so reviewers can replay.
4. Optional follow-up (deferred): port `get_proportional_probabilities`
   to weight LOTSA records by length, and `PackCollate` to bin-pack
   variable-length samples — both improve throughput further but
   don't unblock the bottleneck so they're out of scope for the
   first axis-4 ablation.

#### 7.D.5  Post-fix verification (2026-04-26)

Code change: `pretrain/sources.py` — added `_HFArrowIndexer`
(in-tree port of Moirai's `HuggingFaceDatasetIndexer`, Apache-2,
Salesforce, Inc.) and rewired `LOTSAHFSource.__post_init__` /
`__iter__` to read through it.  Public API of `LOTSAHFSource`
(yielded dict shape) is unchanged.

Two regression tests in `pretrain/scripts/`:

1. `test_lotsa_indexer_parity.py` — for each of 8 representative
   sources (covering univariate/multivariate, single-huge-record
   and many-records, frequencies from 4 s to yearly), draws 5
   random indices and asserts:

       new path shape == old path shape
       new path dtype == old path dtype == float32
       np.allclose(new, old, equal_nan=True, rtol=0, atol=0)

   Result: **28/28 passed**.  Per-record yield speedup ranged
   from 1× (covid_deaths, m4_yearly — both already <1 ms) to
   1130× (solar_power: 2384 ms → 2.1 ms).

2. `profile_lotsa_pipeline.py` — re-run with same shape as the
   pre-fix sweep (1640 samples × 164 sources, cold OS page cache):

   | Stage | pre-fix | post-fix | speedup |
   |---|---:|---:|---:|
   | `hf_random_get` mean | 79.3 ms | **0.32 ms** | **248×** |
   | `hf_random_get` p99  | 2178 ms | **0.87 ms** | **2503×** |
   | `to_2d` mean         | 5.3 ms  | ~0 ms       | (now no-op) |
   | `degrade` mean       | 0.5 ms  | 1.1 ms      | +0.6 ms (pure CPU; fine) |
   | **`yield_total` mean** | **86.9 ms** | **1.46 ms** | **59×** |
   | **`yield_total` p99**  | **2370 ms** | **5.64 ms** | **420×** |

   Slowest sources post-fix:
   `residential_load_power` 6.18 ms, `cmip6_1955` 4.24 ms,
   `residential_pv_power` 4.05 ms, `era5_2017` 3.76 ms,
   `era5_2018` 3.71 ms — well below the 110 ms/step GPU budget.

   Throughput projection (bs=32, nw=12):
       per-worker  :  686 samples/sec
       total       : 8229 samples/sec → 257 batches/sec
       GPU target  :  9.09 batches/sec (110 ms/step)

   Data pipeline is now ≈27× faster than the GPU consumer, i.e.
   strictly GPU-bound.  GPU starvation factor reported as ≈0×.

Sidecars:
- `pretrain/runs/profile/lotsa_indexer_compare.json`  (A/B)
- `pretrain/runs/profile/lotsa_pipeline_post_indexer.json`  (full sweep)

End-to-end Stage-B confirmation (5-min wall-clock, same config and
architecture as §7.D.1, GPU on the same H100; only the indexer
patch is different):

| Metric | pre-fix | post-fix | Δ |
|---|---:|---:|---:|
| GPU util mean | 5.6 % (FAIL) | **97.7 % (PASS)** | 17.4× |
| GPU util p10 / p50 / p90 | 0 / 0 / 0 % | 97 / 98 / 100 % | — |
| Steps completed in 5 min | 84 | **1023** | **12.2×** |
| `train_dataloader_next` total | 279 s (26 % of fit) | **0.3 s (0.0 % of fit)** | 930× |
| `train_dataloader_next` mean/call | 3322 ms | sub-ms | ~10⁴× |
| Forward bucket total | 7 s | 81 s | (12× more useful work) |
| Backward bucket total | 11 s | 114 s | (10× more useful work) |
| Optimizer bucket total | 9 s | 99 s | (11× more useful work) |

Pre-fix Stage-B output preserved at
`pretrain/runs/profile/stage_b_pre_indexer_fix/` so reviewers can
re-run `summarize_profile.py` against either tree.

Stage B is now firmly **compute-bound**.  The remaining wall-clock
during the profile is ~196 ms/call of `RichProgressBar` UI overhead
at `--log_every_n_steps 10`; this drops by 5× under the production
`--log_every_n_steps 50` cadence.

Conclusion: §7.D ticks closed.  We're ready to wire W&B and launch
the axis-4 ablation.

#### 7.D.6  Float32 overflow in per-(b,v) standardization (fixed)

The post-fix Stage-B re-profile ran cleanly on the GPU side but emitted
a `RuntimeWarning: overflow encountered in square` from
`numpy._core._methods.py` while computing per-variate `.std()` inside
`collate.py`.  The data path itself is fine — the warning is purely a
numerics artifact of computing
`var = ((x - mu) ** 2).mean()` in **float32**:

- `vals_padded` lives in float32 (collate-time pad buffer).
- `np.ndarray.std()` accumulates `(x - mu) ** 2` in the array's own
  dtype unless told otherwise, so on records whose deviation magnitude
  ever crosses ~1.84e19 (the float32 sqrt of the overflow boundary
  3.4e38), the squared term saturates to `inf` and the resulting std
  is `nan` / `inf`.  `min_std` would then clamp it to the floor and
  silently divide by 1e-3, blowing the standardized signal up to
  ~1e6 — a mode that would *quietly* poison gradients across the
  affected batches.

We could not isolate the exact offending source in 800 batches with
`num_workers=0`, so the trigger is rare and probably tied to a
particular synthetic outlier.  Fix is defensive in two places:

- `pretrain/collate.py` (per-(b,v) context-only standardization)
- `pretrain/val_imts.py` (per-(b,v) standardization on the val side)

Both now do `x.astype(np.float64, copy=False).{mean, std}()` before
squashing back to a Python `float`.  Cost is negligible (a few stats
per variate per sample, on already-tiny gathered arrays) and the
fix turns the entire failure class off in one place, so we don't have
to chase whichever record / regime / window combination tripped it
on the first profile.

Verification: re-ran 800 Stage-B batches with
`-W "error::RuntimeWarning"` (any RuntimeWarning becomes a hard
error); 800/800 batches pass, no overflow event.  Logged here so we
remember not to revert the float64 promotion if someone later
"optimizes" the inner loop.

### 7.E  Optional: bump `max_dim` to 50 to allow PhysioNet val

Trade-off discussed in §3.4.  We do **not** propose this by default;
it's listed for completeness.

### 7.G  W&B logging  (✅ implemented)

Per the user's call ("only when we start actual training or stage a
training"), W&B is **opt-in** and runs **alongside** `CSVLogger`,
never instead of it.  The CSV file is the source of truth for all
post-hoc parsing (e.g. `summarize_profile.py`); W&B is for live
dashboards and sweep visualisation.

CLI surface (mirrors the rest of the imts_benchmark trainers via
`shared_config/wandb_lightning.py`):

| Flag                  | Default                                          | Purpose                                                                   |
| --------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| `--use_wandb`         | off                                              | Master switch.  Profiling / smoke runs leave it off.                      |
| `--wandb_project`     | `WANDB_PROJECT` env, else `mamba-imts-pretrain`  | Stable project name; one project for the whole pretraining sweep.         |
| `--wandb_entity`      | `WANDB_ENTITY` env                               | Team / org.                                                               |
| `--wandb_run_name`    | `stage_<s>_axis-<a>_lvl-<l>_seed<seed>`          | Auto-generated from stage + ablation tag for legible sweep dashboards.    |
| `--wandb_mode`        | `online` (or env `WANDB_MODE`)                   | Set `offline` on machines without `WANDB_API_KEY`; the run still streams to `./wandb/` and can be `wandb sync`-ed later. |
| `--wandb_tags`        | `[]`                                             | Extra tags; merged with auto-tags `stage_<s>`, `loss_<l>`, `axis_<a>`, `level_<l>`. |

What gets logged:

- All argparse args plus extras: git SHA, resolved stage_cfg path,
  resolved sources_cfg path, **n_params / n_params_trainable**, and
  the per-stage mix dict.
- All `self.log(...)` calls from the LightningModule (`train/huber`,
  `train/mse`, `lr`, etc.) and from the IMTS / IMM-TSF val loops
  (`val/mse_imts_<ds>`, `val/mse_imm_<ds>`, `val/mse_z_imm_<ds>`).
- The auto-tags above so the project view groups runs by stage and by
  ablation axis at a glance.

Smoke verification (2026-04-26): 30-step `--stage a --use_wandb
--wandb_mode offline` run completes in ~25 s, the local run dir has a
fully-populated `run-*.wandb` file with all the config keys
(`d_model`, `huber_delta`, `ablation_axis`, `stage_mix`, `n_params`,
…) and the `train/huber` history stream.  The integration is silent
on the not-using-wandb path: a stage-a profiling run with
`--use_wandb` omitted still goes through `CSVLogger` only.

### 7.F  First real ablation — axis 4 (Stage-A composition)  (running)

Per §3.6 the first axis to vary is the source mix.  Plan:

| Run                | Stage cfg                                    | Stage-A mix                                            |
| ------------------ | -------------------------------------------- | ------------------------------------------------------ |
| `axis4_synth_only` | `configs/stage_a.yaml`                       | 70 % chronos2 + 30 % kernelsynth (no LOTSA)            |
| `axis4_synth_lotsa`| `configs/stage_a_with_lotsa.yaml`            | 55 % chronos2 + 25 % kernelsynth + 20 % regular LOTSA  |

Each 50K steps, identical model (`d_model=384`, 7.84 M params),
optimizer (AdamW, lr=5e-4, 200-step warmup), val cadence (every 2k
steps), and val set (`activity` + `ushcn` from tpatchgnn at 1024
windows each, plus the 8 IMM-TSF datasets at 1024 windows each).
Loss is Huber (δ=1.0) following Time-MoE.

Runner: `scripts/run_axis4_ablation.sh`.  Outputs land under
`runs/axis4/axis4_<arm>/`; W&B runs are written offline to
`runs/axis4/axis4_<arm>/wandb/offline-run-*` (this machine has no
`WANDB_API_KEY`); push them with
`wandb sync runs/axis4/axis4_<arm>/wandb/offline-run-*` once authed.

**Decision rule.**  Promote the arm with lower
`val/mse_z_imm_<ds>` averaged over the 8 IMM-TSF datasets at
step 50K.  Tiebreaker: `val/mse_activity + val/mse_ushcn` at step
50K.  We use the **z-scored** IMM-TSF metric (not the raw-units one)
so per-dataset scale doesn't dominate the average — see §3.4 caveat
3 and §7.B′ for the dual-metric design.

Decision on axis 2 (absolute scale conditioning) is deferred until
after this finishes — if the synthetic-only run wins without
absolute-scale info, axis 2 likely doesn't matter.

Status (2026-04-26 18:04 UTC): only the synth_only arm is queued; we
read its result and decide whether to go to Stage B or stay on Stage
A first.  Running in screen session `axis4_synth_only`, W&B project
**TSKing**, run name **`stageA_chronos70_kernel30_d384_huber_50k_s42`**.
~6 steps/s on the H100, ETA ≈ 2.3 h to step 50K.
W&B online: https://wandb.ai/magicslabnorthwestern/TSKing/runs/8mcds09x

Note: the first launch attempt (`axis4_synth_only_v0_died_at_step6749/`)
died at step 6749 because the bash wrapper was a child of the agent
shell session, which got SIGHUP'd when the user's local SSH dropped.
**Lesson learned: always use `screen` (or `tmux` / `setsid`) for
multi-hour runs** — the screen session is owned by systemd, not the
SSH session, so it survives any local-side disconnect.

Operator notes:
- Reattach the live run: `screen -r axis4_synth_only`
- Detach again from inside screen: `Ctrl-A D`
- Cancel cleanly: reattach + Ctrl-C (Lightning will save a final ckpt)
- Hard kill: `screen -S axis4_synth_only -X quit`
- Val cycle timing was measured at ~30 s with the full 10-dataset val
  set (activity, ushcn, 8 IMM-TSF) at 1024 windows each → 25 cycles
  over the run = ~12 min total val overhead, well within budget.

`axis4_synth_lotsa` is **not** scheduled for now per user direction;
re-enable by passing `ARMS="synth_only synth_lotsa"` to the runner.

**What gets persisted for stage A → stage B handoff** (added §7.H):
- Periodic step-keyed ckpts: `ckpt-step=NNNNNNNN.ckpt` every 5K steps
  + always-current `last.ckpt` for crash recovery.
- "Best by val/mse_z_imm_avg" ckpts: `best-step{...}-mse{...}.ckpt`,
  top 3 retained; the average is over the 8 IMM-TSF datasets in
  z-scored space (paper-comparable).  Driven by the new
  `_AggregateValMetricsCallback`, which also logs
  `val/mse_z_imts_avg`, `val/mse_z_avg_all`, and the MAE counterparts.
- `ablation.json` snapshots all CLI args (now also `init_from`,
  `init_from_strict`, `ckpt_resume`).
- `transition.json` (only on stage transitions): records the source
  ckpt path, source `global_step`, source ablation config, and the
  new run's stage_cfg / max_steps / lr / warmup.

**Stage A → B usage** (`scripts/run_stage_b_from_a.sh`):

```bash
INIT_FROM=/home/ubuntu/hongyu/ssm/imts_benchmark/pretrain/runs/axis4/axis4_synth_only/best-step00050000-mseX.XXXX.ckpt \
    bash imts_benchmark/pretrain/scripts/run_stage_b_from_a.sh
```

`--init_from` loads model `state_dict` only (optimizer / scheduler /
step counter / RNG / data-sampler cursor all start fresh).  `--ckpt`
remains the full Lightning resume, used **only for crash recovery
within the same stage**.  Mixing both is rejected at parse time.

------------------------------------------------------------------------

### 7.I  Results — Stage A (synth-only) and Stage B (mixed) on the original architecture

**Stage A — `axis4_synth_only`** (50K steps, ~3.5 h, 1×H100):

- Run dir: `runs/axis4/axis4_synth_only/`
- Init: from scratch.  Mix: 70 % `chronos2_synth` + 30 % `kernelsynth_irregular`.
- Best `val/mse_z_imm_avg = 1.2722` at **step 15K**, then drifts up — clear overfitting on synthetic-only.
- Best ckpt promoted to Stage B: `best-step00015000-mse1.2722.ckpt`.
- W&B: https://wandb.ai/magicslabnorthwestern/TSKing/runs/8mcds09x

**Stage B — `stageB_chronos35_kernel20_lotsaDeg30_lotsaReg15_d384_huber_200k_fromA-15K_s42`**
(200K steps, ~16 h, 1×H100):

- Init: `--init_from runs/axis4/axis4_synth_only/best-step00015000-mse1.2722.ckpt` (weights only; fresh optimizer / scheduler / RNG).
- Mix: 35 % chronos2 + 20 % kernelsynth + 30 % degraded LOTSA + 15 % regular LOTSA.
- 80 val cycles total (every 2.5 K steps), full 10-dataset val set each cycle.
- `transition.json` records full provenance (source ckpt, source step, source ablation config).

Stage B Z-MSE trajectory (selected steps):

| step    | imm_avg | activity | ushcn (raw) |
| ------- | ------- | -------- | ----------- |
| 2.5K    | 1.2441  | 0.0033   | 0.6249      |
| 25K     | **1.1420** ← best | 0.0032 | 0.6253 |
| 50K     | 1.3311  | 0.0032   | 0.6248      |
| 100K    | 1.2471  | 0.0034   | 0.6242      |
| 150K    | 1.2763  | 0.0033   | 0.6240      |
| 200K    | 1.2677  | 0.0033   | 0.6243      |

(USHCN raw `val/mse_ushcn` ≈ 0.62 throughout — the per-record
z-scoring on USHCN's near-constant precipitation columns inflates
its z-MSE to ~15 K, which is the §3.4 caveat 3 instability we
flagged at design time.  We rely on `imm_avg` and the raw IMTS
metrics for decision-making.)

Per-IMM-dataset best step (final - best):

| dataset       | best_step | best_mse_z | final_mse_z | Δ       |
| ------------- | --------- | ---------- | ----------- | ------- |
| CESNET        | 27.5K     | 1.0229     | 1.0561      | +0.0332 |
| ClusterTrace  | 137.5K    | 0.9479     | 0.9920      | +0.0441 |
| EPA-Air       | 77.5K     | 0.6870     | 0.9442      | +0.2572 |
| FNSPID        | 100K      | 0.9240     | 1.3411      | +0.4172 |
| GDELT         | 32.5K     | 1.2077     | 1.2247      | +0.0170 |
| ILINet        | 25K       | 1.7944     | 2.5224      | +0.7279 |
| RepoHealth    | 25K       | 1.0025     | 1.2195      | +0.2169 |
| StudentLife   | 75K       | 0.8174     | 0.8419      | +0.0245 |

**Stage A → Stage B delta on `mse_z_imm_avg`:** 1.2722 → 1.1420
(−10.2 %).  The improvement comes early (within the first 25K
Stage B steps) and then plateaus.  ILINet and FNSPID are the small
datasets where overfitting is most pronounced past step 50K.

**Best Stage B ckpt for downstream use** (under the original
architecture; **see §7.J** for why the ckpt cannot be reused after the
any-variate-attention switch):
`runs/stage_b/stageB_chronos35_kernel20_lotsaDeg30_lotsaReg15_d384_huber_200k_fromA-15K_s42/best-step00025000-mse1.1420.ckpt`.

**Implications.**
1. The mixed-source curriculum **does** transfer signal from the
   Stage-A weights into a stronger model — this validates the
   Stage A → Stage B handoff design (`--init_from` + best-ckpt
   promotion + `transition.json`).
2. Both stages overfit relatively early.  At 50K Stage A and
   25K Stage B, val IMM stops improving but train Huber keeps
   dropping (e.g. final 0.34 at step 200K vs 1.10 at step 1K).
   This argues for either (a) more data diversity (Stage C with
   more LOTSA), (b) stronger regularization, (c) early stopping
   integrated into the run, or (d) a fundamentally different
   capacity / objective — see §7.J on the architecture switch
   we are about to A/B for hypothesis (d).
3. The `axis4_synth_lotsa` arm was never run; the synth-only arm
   was sufficient evidence to commit to the synth-then-mixed
   curriculum.

------------------------------------------------------------------------

### 7.J  Any-variate attention switch (collaborator change + bug fix)

**Origin.**  Collaborator (Dawei) committed two architecture changes to
branch `pretrain` (commits `28362ad`, `9c5bf7d`) on 2026-04-27,
identifying two issues with the original design:

1. **Per-slot variate embeddings don't generalize.**  The original
   `nn.Embedding(n_vars, d_model)` indexed by `torch.arange(V)` binds
   meaning to slot index.  Combined with our collator's per-batch slot
   randomization (intentional, so the model treats variate IDs as
   anonymous), this means each embedding row is being trained to
   represent a *random average* over all variate types in all source
   datasets.  At downstream time the slot-index embeddings carry no
   meaningful signal.
2. **Per-variate readout heads are similarly slot-keyed.**
   `QueryReadout.head_weight: (n_vars, d_in, 1)` and per-variate decay
   `gamma_raw: (n_vars, d_model)` learn slot-conditional scale/offset,
   which is doubly redundant given (a) we already do per-(b,v) value
   standardization in the collator and (b) slot indices are randomized.

**Files touched (collaborator):**
- `imts_benchmark/mamba_mv/variable_axis_attention.py`: removed
  `variate_embed: nn.Embedding(n_vars, d_model)`, added Moirai-style
  `_BinaryVariateAttentionBias(n_heads=H)` (a 2 × H learned tensor).
- `imts_benchmark/mamba_mv/shared_grid.py`: removed
  `variate_embed: nn.Embedding(n_vars, d_model//4)`; `proj` input width
  shrinks from `d_h + d_m//4 + 1 + 2*n_freq` (497) to `d_h + 1 + 2*n_freq` (401).
- `imts_benchmark/mamba_mv/query_readout.py`: replaced per-variate
  `(head_weight, head_bias, gamma_raw)` with shared `nn.Linear(D+2*n_freq, 1)`
  and shared `gamma_raw: (D,)`.

**My independent assessment & two changes I applied on top.**

*Direction:* +1.  Removing slot-keyed parameters is correct — they were
fitting the slot-randomization noise, not signal.  Replacing with
Moirai's any-variate bias (Woo et al., 2024, §3.2) is the right
permutation-equivariant primitive.

*Issue 1 (FIXED): the new attention bias was dead code.*
The collaborator's `VariableAxisAttention.forward(...)` accepts an
optional `var_id` argument, but the only call site
(`MultivariateMambaForecaster.forward` at lines 159 and 515) **never
passed it**.  Result: `var_id=None` always, `bias=None` always,
`self.var_bias` allocated but never trained.  Fix:
- Added `MultivariateMambaForecaster._build_var_id(B, V, device)`
  returning `arange(V).expand(B, V)`.
- Both forward call sites now pass `var_id=var_id` to `attn(...)`.
- Verified by autograd: `fusion_attn.{0,1,2}.var_bias.weight.weight`
  receives non-zero gradients on a smoke-tested forward+backward.

Why `arange(V)` and not random IDs?  In Moirai they pack multiple
series into one batch element, so per-series random IDs are needed
for the bias to distinguish "same series" from "different series."
We do not pack — each batch element is one window with V *distinct*
variates — so slot indices are unique within the element by
construction.  Under this choice the bias reduces to a learned
self-vs-other bias (diagonal vs off-diagonal of the V x V attention
matrix), which is a known-good inductive prior for a tiny parameter
cost (`2 * n_heads = 8` params per fusion block).

*Issue 2 (REVERTED): MQA was conflated with the variate-ID change.*
The collaborator's diff also changed K/V projections from
`d_model → d_model` to `d_model → d_model // n_heads`.  This is
multi-query attention — a separate optimization aimed at autoregressive
KV-cache memory.  Reasons to revert:
- We attend over **V ≤ 20 tokens per slot**, so attention is already
  `O(V²) ≈ 400` ops per slot — **trivial**; KV memory is not a
  bottleneck.
- We **don't decode autoregressively** along V, so the KV-cache
  motivation doesn't apply.
- **Moirai-1's any-variate attention itself uses standard MHA**, not
  MQA.  The colleague's "match Moirai" framing actually argues
  *against* MQA here.
- MQA is less expressive — heads can no longer specialize their
  key/value subspaces.  Combining "remove variate signal" + "compress
  KV expressiveness" stacks two reductions in one diff, conflating
  effects.

Reverted in `variable_axis_attention.py`: K/V projections back to
`d_model → d_model`, expand-from-1 hack removed.  If we want MQA
later, ablate it in isolation against MHA on the same val set.

*Issue 3 (open / minor): slot-keyed pieces still remain elsewhere.*
- `SharedGridAligner.gamma_raw: (n_vars, d_hidden)` and `null_state:
  (n_vars, d_hidden)` are still per-slot.  Same generalization argument
  applies — these would benefit from the same any-variate treatment in
  a follow-up pass.  Not a blocker for the upcoming run.
- `n_vars` argument is now unused in `QueryReadout` and `SharedGridAligner`
  (modulo the tables above).  Cosmetic cleanup nit.

**Old checkpoint compatibility — verified incompatible.**
The two best Stage A / Stage B ckpts under the original architecture
**cannot be reused** under the new one:
- 6 learned params have no destination
  (`{fusion_attn.0,1,2}.variate_embed.weight`, `grid.variate_embed.weight`,
  `readout.head_weight`, `readout.head_bias`).
- 2 params have shape mismatch (`grid.proj.weight: (384, 497) → (384, 401)`,
  `readout.gamma_raw: (20, 384) → (384,)`).
- 7 new params would be randomly initialized (`var_bias` x 3, shared
  `readout.head.{weight,bias}`, plus rebuilt buffers).

So the new-arch run is necessarily from-scratch.  No `--init_from` is
possible — the A/B genuinely starts from random weights.

**A/B plan.**
- Run `stageA_chronos70_kernel30_anyvariate_d384_huber_50k_s42` with
  the same data, optimizer, val cadence, and seed as the original
  `stageA_chronos70_kernel30_d384_huber_50k_s42`.
- Headline metric: `val/mse_z_imm_avg` at the best step.  Original
  baseline = **1.2722 @ step 15K**.
- Tiebreaker: `val/mse_activity + val/mse_ushcn` at the same step.
- Param count after the change: **7,760,297** (vs old arch ~7.84 M);
  the small reduction is because we removed several per-variate tables
  but added the tiny bias.  Step time and memory should be within
  rounding noise of the original.

**Status (2026-04-27, 21:46 UTC).**  Architecture changes landed
locally on branch `pretrain` (`git_sha=9c5bf7d` for the colleague's
diff; my `var_id` wiring + MQA revert are uncommitted at launch).
Stage A any-variate run is **live** in `screen` session
`anyvariate_stage_a`:

- Out dir: `runs/anyvariate/stage_a_synth_only/`
- W&B: https://wandb.ai/magicslabnorthwestern/TSKing/runs/09h7kbw1
- Throughput: ~6.63 it/s (matches the original baseline of ~6 it/s)
- ETA to step 50K: ~2.1 h
- Reattach: `screen -r anyvariate_stage_a`
- Hard kill: `screen -S anyvariate_stage_a -X quit`

Launch script: `scripts/run_anyvariate_stage_a.sh`.  Same data, same
hparams, same val cadence as the baseline `axis4_synth_only` run
(`runs/axis4/axis4_synth_only/`); the only differences are the four
architecture changes above (per-slot embeddings out, any-variate bias
in + wired, MQA reverted, shared readout head).

### 7.K  Single-phase Stage A (`single_phase`, 80K planned, ES at 44K)

**Motivation.**  The two-phase curriculum (`aligned_a_constlr`) bought us
0.97 → 0.97 on `val/mse_z_imm_avg` for ~3 h of compute — the easy→main
phase shift was paying for itself only on EPA-Air and GDELT (2/8 of the
IMM-TSF datasets); the other six picked phase-1 checkpoints in the
honest val→test analysis.  We dropped the curriculum and ran a single
**non-stationary** mix with the regime distribution applied uniformly
to every source, plus a **multi-step LR schedule** (DeepSeek-LLM
80/90 recipe) so the run can be extended later with no replay.

**Config** (`configs/stage_a_single_phase.yaml`,
`scripts/run_single_phase.sh`):

```
mix:                           regime_dist:                task_type_probs:
  chronos2_synth: 0.20           regular: 0.30               univ:    0.05
  kernelsynth:    0.10           sync:    0.10               all-tg:  0.95
  lotsa_degraded: 0.70           mixed:   0.40               part-tg: 0.00
                                 async:   0.20

variate_count_dist:  [0.05, 0.40, 0.50, 0.05]   # IMM-TSF Table 1 alignment
per_var_max_length:  384                         # was 256, fits ILINet/CESNET
min_ctx_obs_per_target: 8                        # was 3, prevents starvation

max_steps:        80000     warmup:     1000  schedule: multistep
batch_size:           32    lr:         5e-4  precision: bf16-mixed
gradient_clip_val:   1.0    early_stop_patience: 10
synth_val_n_windows: 512    val_check_steps: 2000
```

`lotsa_degraded` carries `clean_target_mode=false` so the supervision
matches IMM-TSF eval (degrade context AND future, predict only at
surviving timestamps).  Despite the name "lotsa_degraded", the 30%
regular regime within it produces *clean* LOTSA windows — net effective
exposure: 21% clean LOTSA, 49% degraded LOTSA, 30% synth.

**Outcome.**  Run died at step 44,000 / 80,000 (~2 h 47 m wall clock):

| step | train/huber (median) | val/mse_z_imm_avg | val/synth_r2_overall |
|---|---|---|---|
| 2K  | 0.339 | 1.13 | 0.43 |
| 18K | 0.297 | 1.01 | 0.50 |
| **26K** | 0.298 | **0.967 (best)** | 0.50 |
| 32K | 0.293 | 1.01 | 0.50 |
| 40K | 0.284 | 1.17 | 0.50 |
| 42K | — | 0.977 | 0.51 |
| 44K | 0.337 | **inf** (GDELT, ClusterTrace blow-up → ES tripped) | 0.50 |

Best ckpt: `best-step00026000-mse0.9671.ckpt`.  Improvement vs
`aligned_a_constlr` baseline (0.97 → 0.967): essentially flat, but
the curve was still descending right up to the blowup, so the
underlying optimization was alive — see §7.L for why it crashed.

**Per-IMM-TSF dataset best (cherry-picked along trajectory)**:

| dataset      | best z-MSE | step  | dataset      | best z-MSE | step  |
|--------------|------------|-------|--------------|------------|-------|
| EPA-Air      | 0.44       | 36K   | CESNET       | 1.01       | 14K   |
| FNSPID       | 0.23       | 20K   | StudentLife  | 0.82       | 42K   |
| ClusterTrace | 0.84       | 32K   | RepoHealth   | 0.71       | 26K   |
| GDELT        | 1.21       | 42K   | ILINet       | **2.05**   | 2K    |

ILINet *regressed* from step 2K onwards.  Same regression pattern
visible (less severely) on aligned_a_constlr.  Hypothesis: ILINet
is V=12 weekly — the only 4-bucket dataset where our model hits
hard cross-variate coupling.  See §7.M.

**Synthetic in-distribution validation** is a fixed 512-window
snapshot built once at fit-start with `seed = train_seed + 1000`,
held in CPU RAM, run at every `val_check_steps`.  These windows are
**never seen during training** (different seed → independent stream
from the train dataloader).  Per-source pooled R² in raw-z space:

| step | overall | chronos2 | kernelsynth | lotsa_degraded |
|---|---|---|---|---|
|  2K | 0.43 | 0.55 | 0.13 | 0.19 |
| 18K | 0.50 | 0.65 | 0.13 | 0.21 |
| 32K | 0.50 | 0.65 | 0.18 | 0.20 |
| 42K | 0.51 | 0.65 | 0.18 | 0.21 |

**Reading**: chronos2 is learnable (R² 0.65, the multivariate signal
ceiling we set in the data); kernelsynth and LOTSA both stall around
R² ≈ 0.20, but for very different reasons (see §7.M).

### 7.L  Eval explosion in single_phase (asinh ↔ sinh blow-up)

**Symptom.**  Step 44K eval produced `val/mse_z_imm_GDELT = inf` and
`val/mse_z_imm_ClusterTrace = 2.7e+11`.  EarlyStopping was configured
with `check_finite=True` so it terminated the run.

**Root cause.**  `_log_imts_val` inverts model outputs from asinh-z
back to raw-z with `torch.sinh(preds_asinh)`.  `sinh` is exponential:
`sinh(10) ≈ 1.1e4`, `sinh(15) ≈ 1.6e6`, `sinh(20) ≈ 2.4e8`.  The model
trains on bounded asinh-z targets — across the eight IMM-TSF
datasets the true `|asinh-z|` target tops out at:

| dataset      | max |asinh-z| target | sinh(that) |
|--------------|----------------------|------------|
| ILINet       | **2.01**             | 3.7        |
| ClusterTrace | **2.26**             | 4.7        |
| EPA-Air      | 3.09                 | 11.0       |
| GDELT        | 3.48                 | 16.2       |
| StudentLife  | 3.35                 | 14.2       |
| CESNET       | 3.29                 | 13.3       |
| RepoHealth   | 3.87                 | 24.0       |
| FNSPID       | 4.05                 | 28.6       |

If the model occasionally emits an asinh-z prediction outside its
trained support — which can happen after a single bad gradient step
or in a region of parameter space the optimizer drifts into — sinh
exponentiates the error and the per-batch z-MSE explodes.  ILINet
and ClusterTrace blew first because their target range is tightest.

**Why didn't earlier runs hit `inf`?**  They came close but didn't
quite reach it:

| run | finite max val/mse_z_imm_avg |
|---|---|
| `curriculum/phase2_main` | **90,273** |
| `aligned_a_constlr/phase2_main` | 15.99 |
| `aligned_a/phase2_main` | 1.20 |
| `single_phase` | inf (ES tripped at step 44K) |

What's different about single_phase: (a) longest duration (44K vs
≤16-30K), (b) constant-LR for the entire run (multistep doesn't
decay until 80% of total = 64K, never reached), (c) no per-batch
clipping of the model's *output* (we had `gradient_clip_val=1.0`
which constrains the parameter step but not the activation).

**Fix** (`mamba_mv/multivariate_forecaster.py::_log_imts_val`):

```python
with torch.no_grad():
    preds_asinh = self.forward(batch)
    preds_asinh = preds_asinh.clamp(-10.0, 10.0)   # bound sinh
preds_z = torch.sinh(preds_asinh)
```

`±10` keeps the worst per-element z-error at `sinh(10) ≈ 1.1e4`,
which is "bad" but no longer infinity, so EarlyStopping and
checkpoint selection still see usable numbers.  This does NOT change
training behavior — only eval metric robustness.

Also dropped `mse_z` / `mae_z` logging for activity and USHCN: those
have known fixed scales and the published baselines report
original-unit MSE, so per-(b,v) z is uninformative there *and*
historically inflates to 1e10 because of near-constant series
(USHCN precipitation; see §Q4).  The aggregate callback also now
filters non-finite values from the IMM-TSF mean, and ES uses
`check_finite=False`, so a single transient blowup cannot terminate
a run.

### 7.M  LOTSA stalled — root cause: 89% of weight on 2 univariate datasets

**Observation.**  Despite LOTSA being 70% of training, both the
in-distribution synth val (`val/synth_lotsa_degraded_r2 ≈ 0.20`) and
the IMM-TSF metrics show LOTSA-trained capacity is barely improving
beyond step ~6K.

**Diagnostic** (`scripts/diagnose_single_phase.py`).  We reconstructed
the model from `best-step26000`, replayed a fresh in-distribution
snapshot dataloader (2048 windows, seed = train_seed + 5000 ≠ train
+ 0 ≠ synth_val + 1000), and binned by source.  Per-source aggregates:

| source | samples | active V (median, p10/p90) | ctx obs/var (median) | hor obs/var (median) | MSE (asinh-z) | R²    |
|---|---|---|---|---|---|---|
| chronos2_synth  |  404 |  3  (1 / 9)  | 132 | 44 | 0.61 | **0.51** |
| kernelsynth     |  187 |  1  (1 / 6)  | 132 | 44 | 0.67 | 0.23 |
| **lotsa_degraded** | **1457** | **1  (1 / 1)**  | 107 | 31 | 1.08 | 0.33 |

Every single LOTSA window in the 2048-sample snapshot has **V = 1**.
The variate-count distribution is supposed to be `[V=1: 5%, V=2-8:
40%, V=9-16: 50%, V=17-20: 5%]`, but the active V depends on
`min(total_var, max_variates)`, and `total_var` depends entirely on
which physical LOTSA dataset got sampled.

**Following the weights.**  The weight map comes from Moirai's
`uni2ts/cli/conf/pretrain/data/lotsa_v1_weighted.yaml` (token-count
weighting).  Top 10 of 170 LOTSA datasets:

| rank | dataset                            |   weight  | % of total | V |
|------|------------------------------------|----------:|-----------:|---|
|  1   | solar_power                        | 33,835.57 | **44.98 %**| **1** |
|  2   | wind_power                         | 33,835.22 | **44.98 %**| **1** |
|  3   | australian_electricity_demand      |  1,055.32 | 1.40 %     | 1 |
|  4   | residential_pv_power               |    543.18 | 0.72 %     | 3 |
|  5   | residential_load_power             |    467.01 | 0.62 %     | 3 |
|  6   | oikolab_weather                    |    457.67 | 0.61 %     | 1 |
|  7   | LOOP_SEATTLE                       |    391.83 | 0.52 %     | (multi) |
|  8   | wind_farms_with_missing            |    375.55 | 0.50 %     | 1 |
|  9   | sunspot_with_missing               |    338.00 | 0.45 %     | 1 |
| 10   | PEMS_BAY                           |    238.44 | 0.32 %     | (multi) |

`solar_power` + `wind_power` together are **89.96 %** of LOTSA's
sampling weight, both univariate.  Across all 170 datasets:

- 78 / 170 datasets are multivariate (V ≥ 2): **45.9 %**.
- Multivariate datasets carry 1,534 / 75,225 of the total weight: **2.0 %**.

So with `lotsa_degraded` at 70% of training, ~98% of LOTSA samples are
univariate → **63% of all training is univariate solar/wind**.  Only the
30% synthetic share carries genuine multivariate structure.  This
explains:

- `val/synth_lotsa_degraded_r2` plateaus near 0.20: the LOTSA val
  windows are also univariate, so we're scoring the model on a task
  it does see in training, but it's a hard univariate task (irregular
  electricity / weather / sunspot) without much for the multivariate
  axis-attention to learn from.
- ILINet (V=12 weekly) regressing from step 2K: the model never sees
  a 12-variate window from LOTSA, only from the 30% synth share.
  Once early synth-driven multivariate priors are overwritten by
  univariate LOTSA mass, ILINet performance drifts down.
- Original-units MSE on LOTSA-style sources looks fine (MAE 0.57)
  because predicting the recent value or a smoothed mean gets you
  most of the way for univariate solar/wind/sunspot.  R² is the
  honest score (0.33) and it's gated by V=1.

**Plots** (`runs/single_phase/.../diagnostics/`):

- `single_phase_predictions_lotsa_degraded.png`: 6 LOTSA samples,
  all V=1, model often predicts a near-constant baseline through
  the future when the context is mostly noise.
- `single_phase_predictions_chronos2_synth.png`: model correctly
  extends smooth trends but flatlines through high-frequency
  oscillation — consistent with predicting the conditional mean,
  which is MSE-optimal for unpredictable detail but means the
  visible predictions look "lazy".
- `single_phase_predictions_kernelsynth.png`: same "predict the
  mean" failure mode is more pronounced because kernelsynth often
  has stronger high-frequency content.

**Implications for the next run.**  Three fixes worth considering
before relaunching:

1.  **Cap `solar_power` and `wind_power` weights** — clip top-N
    weights to e.g. 200, renormalize.  Brings effective LOTSA
    multivariate share from 2 % → ~25 %.  This is the smallest
    intervention.
2.  **Stratified LOTSA sampling**: separate LOTSA into "univariate"
    and "multivariate" pools, sample the pools at e.g. 50/50 instead
    of weighted-flat.  Bigger lever.
3.  **Synthetic multivariate from univariate LOTSA**: pick K
    univariate series at random, treat them as a synthetic K-variate
    record (this is essentially what Chronos-2's "covariate stacking"
    does for cross-variate training data).  Highest leverage but
    requires a new data path.

We are *not* relaunching yet (per user instruction): fix
`preds_asinh` clamp + drop activity/USHCN z-metrics now, then
decide on the LOTSA fix path together.

------------------------------------------------------------------------

### 7.N  Moirai-style sampling fix (post-§7.M, pre-relaunch)

Two surgical changes that bring our LOTSA sampling in line with
Moirai's (Woo et al., 2024, §3.2) intent.  Triggered by the §7.M
diagnosis.

#### 7.N.1  Why our pre-fix code was wrong

Moirai's published `lotsa_v1_weighted.yaml`
(`uni2ts/cli/conf/pretrain/data/lotsa_v1_weighted.yaml`) lists
*per-series multipliers*, not direct probabilities.  The numbers are
precomputed so that

\[
  \text{num\_ts}_k \times \text{yaml\_weight}_k
  \;=\; \omega_k
  \;=\; \min\!\Big(\frac{\#\text{obs}_k}{\sum_j \#\text{obs}_j},\; \varepsilon\Big),
  \qquad \varepsilon = 0.001
\]

Moirai's actual sampling probability is therefore
\(p(D_k) \propto \omega_k\), and they realize it via
`ConcatDataset` of `TimeSeriesDataset`s whose `__len__` equals
`num_ts × dataset_weight`.  Sampling indices uniformly across the
concat then *implicitly* weights each dataset by `num_ts × yaml_w`.

Our `make_logical_source` was sampling proportional to `yaml_w`
**only**, ignoring `num_ts`.  Two univariate datasets,
`solar_power` and `wind_power`, each have `num_ts = 1` but
`yaml_w ≈ 33,835`, so they each got ~45 % of LOTSA mass — the §7.M
finding.  Multivariate datasets (`PEMS04`/`PEMS_BAY`/`LOOP_SEATTLE`/
`subseasonal`/...) all sat below 1 %.

Fix (`mixed_dataset.py::make_logical_source`):

```python
w = [_lookup_weight(s.name, weight_map) * max(1, len(s)) for s in physical_sources]
```

Effect (smoke test, 4 096 samples, `STACK_ALL` policy):

| metric                                | pre-fix | post-fix |
|---------------------------------------|--------:|---------:|
| `solar_power + wind_power` mass       | 89.96 % | **2.27 %** |
| top-1 dataset mass                    | 44.98 % | **4.25 %** |
| #datasets with ≥ 1 % mass             |   ~6    | **~30**  |

Top-N now: `LOOP_SEATTLE` 4.25, `Q-TRAFFIC` 4.25, `alibaba_cluster_trace_2018` 4.25,
`azure_vm_traces_2017` 4.25, `borg_cluster_data_2011` 4.25, ...
`solar_power` 1.14, `wind_power` 1.14.  Mass is now spread over 30+
datasets including all the major multivariate ones.

#### 7.N.2  Univariate→multivariate stacking

Moirai's `MultiSampleTimeSeriesDataset` (paper §3.2:
"constructing multivariate time series from sub-datasets with
univariate time series, by randomly concatenating them") stacks
`K` random univariate series from the same dataset into a
`[K, T]` sample.  Without this step, ~half of LOTSA's diversity
is locked behind univariate-on-disk datasets that the model only
ever sees at V=1.

We implemented this as `StackedLOTSASource` in `sources.py`.  Per
iteration:

1. Memoized peek determines if the underlying dataset is
   univariate-on-disk or natively multivariate.
2. Natively-MV: pass through unchanged via parent `__iter__`.
3. Univariate-on-disk: stack
   `K = min(max_variates, num_ts)` random series, truncated to
   `min(T_i)` so all rows have the same length.  No padding.

**Why `K = max_variates` and not sampled from `variate_count_dist`?**
We measured a "double-sampling cap" bug in the smoke test: if both
the source and the downstream `LOTSAToIrregular._sample_n_var`
draw from `variate_count_dist`, the model-seen V follows
`min(K_source, K_transform)`, which biases V low.  Concretely,
sampling K_source from the dist gave bucket `V=9..16` only 35 %
mass vs target 55 %.  Stacking `max_variates` always lets the
downstream transform produce the requested V exactly.

Wiring (`datamodule.py::build_logical_sources`):
the default policy is `STACK_ALL` — every LOTSA dataset is wrapped
in `StackedLOTSASource`; stacking only fires for univariate-on-disk
ones.  Override via `sources.yaml::lotsa_stacking_datasets`:
- `"all"` (default): wrap every dataset
- `"moirai_only"`: use Moirai's curated list (~22 datasets), more
  conservative
- `"none"` / `[]` / `null`: disable stacking
- list of names: stack only those

Empirical V distribution (smoke test, post-transform, model-seen,
N = 2 048 samples, `STACK_ALL`):

| bucket      | target | pre-fix (single_phase) | post-fix |
|-------------|-------:|----------------------:|--------:|
| V = 1       |  5 %   | ~63 %                 | **7.9 %**  |
| V = 2..8    | 35 %   | ~25 %                 | **47.0 %** |
| V = 9..16   | 55 %   |  ~9 %                 | **41.6 %** |
| V = 17..20  |  5 %   |  ~3 %                 | **3.4 %**  |

Bucket 3 (V=9..16) — where 5 of 8 IMM-TSF eval datasets live — went
from ~9 % of training time to 41.6 %.  The remaining gap to the 55 %
target comes from natively-MV LOTSA datasets with V < 20
(`subseasonal` V=4, etc.) which force `_sample_n_var` to its cap;
fixing that would require stacking *across* multivariate datasets,
which we leave for a future iteration.

#### 7.N.3  `variate_count_dist` re-tuned for IMM-TSF

Updated default in `task_sampler.py::StageSpec` and the single-phase
config:

|                | pre-fix              | post-fix              |
|----------------|----------------------|------------------------|
| `variate_count_dist` | [0.10, 0.60, 0.25, 0.05] | **[0.05, 0.35, 0.55, 0.05]** |

The rationale is empirical: the IMM-TSF Table-1 V counts are
`{V=4..16, V=10, V=10, V=11, V=11, V=10, V=11, V=9}` — i.e. 7 of 8
datasets sit at V=9..11 (RepoHealth=10, CESNET=10, ILINet=11,
ClusterTrace=11, StudentLife=9, EPA-Air=4..16, FNSPID=10,
GDELT=10).  Concentrating mass in bucket 3 trains directly on the
regime where evaluation happens.  Bucket 1 is now small because
`StackedLOTSASource` ensures we rarely actually emit V=1 samples.

#### 7.N.4  Synthetic sources unchanged

`ChronosSynthSource` and `KernelSynthSource` already produce
multivariate samples whose V varies per record (chronos2 median V=3,
p90=9; kernelsynth user-controlled).  They feed into the same
`LOTSAToIrregular` transform, so the new `variate_count_dist` already
applies to them via `_sample_n_var`.  No code change there.

#### 7.N.5  LR schedule: cosine 30 K (replacing multistep 80 K)

Per §Q2 / §7.K: the multistep schedule never actually decayed before
ES tripped at 44 K / 80 K.  The replacement is

```
warmup = 500 steps; peak LR = 5e-4; cosine to 0 over 30 K total steps
```

Rationale: validation loss in `single_phase` plateaued from step
~6 K (data-limited, not optimization-limited).  30 K total = ~2.25 h
on 1×H100 at our throughput.

#### 7.N.6  What we did NOT port from Moirai

For honesty, the deltas vs Moirai's pretraining:

| feature                             | Moirai                | us                              |
|-------------------------------------|-----------------------|----------------------------------|
| dataset weighting                   | `num_ts × yaml_w`     | **same** (§7.N.1)               |
| univariate→MV stacking              | curated 35-dataset list, `beta_binomial(2, 5, 128)` | **all univariate, K = max_variates = 20** |
| variate subsampling distribution    | `beta_binomial(2, 5, 128)` capped at total | **bucketed [0.05, 0.35, 0.55, 0.05]** capped at total |
| in-dataset sequence sampling        | proportional to series length | uniform across series (TODO if it matters) |
| window cropping                     | `PatchCrop` in patch units, freq-conditional patch size | `_sample_window` in raw steps, no patching |
| history split                       | `MaskedPrediction` mask_ratio ~ U(0.15, 0.5) | `_sample_history_norm` ~ U(0.5, 0.95) inverted |
| irregularity                        | none (regular grid)    | `LOTSAToIrregular` 4-regime degradation |

The architecture differences are a separate axis (§7.J).

------------------------------------------------------------------------

### 7.Q  H2 result + H1 launch (2026-04-30)

#### 7.Q.1  H2 finished — partial win, big diagnostic signal

`H2_nostack_synth30_lotsa70_d384_cosine30000_warmup500_minstd1e-3_s42`
ran the full 30 K cosine steps (no early stopping fired), W&B run id
`0mu3`, wall-clock ~1 h 55 min.

**Headline:** best `val/mse_z_imm_avg = 0.9359 @ step 10 K` — beats
the 0.965 ceiling of the three prior d=384 runs (30/70, 50/50, 10/90)
by ~3 %.

**But the trajectory is unstable**:

| step  | mse_z_imm_avg |
|-------|---------------|
|  2 K  | 1.016          |
|  4 K  | 1.071          |
|  6 K  | 1.094          |
|  8 K  | 1.686          |
| **10 K** | **0.9359**  |
| 12 K  | 1.232          |
| 14 K  | 1.328          |
| 16 K  | 1.075          |
| 22 K  | 1.055          |
| 30 K  | 1.111          |

Step 10 K reads more like a lucky checkpoint between two random-walk
peaks than a sustained improvement.  The post-22 K plateau (~1.05–
1.11) sits right back on the old ceiling.

#### 7.Q.2  Per-IMM-TSF dataset deltas at the H2 best (step 10 K)

| dataset       | V    | step-10 K z-MSE | vs prior runs                |
|---------------|------|-----------------|-------------------------------|
| **ClusterTrace** | 11 | 0.7366          | clear improvement            |
| **RepoHealth**   | 10 | 0.9057          | clear improvement            |
| **USHCN** (raw)  | 5  | 0.78 (sustained 20–30 K) | better (was ~0.85) |
| StudentLife   | 9    | 0.8306          | small improvement            |
| CESNET        | 10   | 1.0189          | unchanged                    |
| GDELT         | 10   | 1.2141          | unchanged                    |
| EPA-Air       | 4–16 | 0.7830          | unchanged                    |
| FNSPID        | 10   | 0.3515          | unchanged                    |
| **ILINet**    | 11   | **1.6464**      | **still broken**             |
| activity (raw)| 5    | 0.003           | unchanged (already saturated)|

The two genuine multivariate datasets we expected to benefit most from
killing random stacking — ClusterTrace and RepoHealth — *did*
improve.  This is partial confirmation of the wrong-prior hypothesis
(§7.P.2).  ILINet stayed pathological → its issue is not just
stacking.

#### 7.Q.3  The diagnostic signal that actually matters: synth in-dist R²

| step  | chronos2 | kernelsynth | **lotsa_degraded** |
|-------|----------|-------------|---------------------|
|  2 K  | +0.466   | +0.188      | **+0.019**          |
| 10 K  | +0.494   | +0.275      | +0.026              |
| 20 K  | +0.540   | +0.287      | +0.022              |
| 30 K  | +0.530   | +0.298      | **+0.030**          |

Synthetic R² climbs steadily for both synth sources.  LOTSA R² stays
at **+0.03** for the entire run.  This is the loudest signal so far:
the model can fit synthetic dynamics but **cannot fit LOTSA's real
distribution at all** (R²≈0 means barely better than predicting the
mean).  With a healthy decreasing `train/huber` (0.7 → 0.3) this
has to be capacity-bound, not optimization-bound.

#### 7.Q.4  H2 verdict

- Random stacking *was* part of the problem, but only a small part.
- The dominant binding constraint is **model capacity vs LOTSA
  heterogeneity**.
- Decision: launch H1 immediately, inheriting H2's
  `sources_nostack.yaml` so the random-stacking fix carries forward.

#### 7.Q.5  H1 launched

`H1_d512_h512_perv4_fus4_head8_bs24_lr2.5e-4_cosine30000_s42`,
W&B run id `di1k`, screen `H1_bigger`.  Started 2026-04-30 16:44 UTC.

Settings:

| dim                | value | note |
|--------------------|-------|------|
| `d_model`          | 512   | (was 384) |
| `d_hidden`         | 512   | (was 384) |
| `n_perv_layer`     | 4     | (was 3)   |
| `n_fusion_blocks`  | 4     | (was 3)   |
| `n_heads_varattn`  | 8     | (was 4)   |
| `grid_K`           | 128   | unchanged |
| `BATCH_SIZE`       | 24    | (was 32)  |
| `LR`               | 2.5e-4| (was 5e-4 — halved for the 2× model) |
| `MAX_STEPS`        | 30 K  | cosine, warmup 500 |
| `sources_cfg`      | `sources_nostack.yaml` | inherits H2 |

Observed at step ~1 K: **3.69 it/s** (better than the 2.2 it/s
estimate), VRAM **52 GB / 80 GB** (BS=32 may fit next time),
`train/huber ≈ 0.3–0.7`.  ETA ~2 h 15 min.

The three things to read off H1 once it lands:
1. Does `mse_z_imm_avg` push below 0.93 (and *stay* there, unlike H2)?
2. Does `val/synth_lotsa_degraded_r2` finally lift off the +0.03 floor?
3. Does ILINet `mse_z` come below 1.0?

------------------------------------------------------------------------

### 7.P  Three runs, one ceiling — H1/H2 plan (2026-04-30)

#### 7.P.1  Result summary across the three single-phase runs

After §7.N landed, three back-to-back single-phase runs at d=384,
varying only the synthetic↔lotsa mix:

| run                                                              | mix (synth / lotsa) | best `mse_z_imm_avg` | step    | wall-clock | ES at  |
|------------------------------------------------------------------|---------------------|----------------------|---------|------------|--------|
| `single_moirai_v2_synth30_lotsa70_regimeMix40_d384_cosine30000_*`| 30 / 70             | **0.965**            | 16 K    | ~2.3 h     | 30 K   |
| `single_moirai_v2_synth50_lotsa50_d384_cosine60000_*`            | 50 / 50             | **0.997**            |  2 K    | ~2.5 h     | 22 K   |
| `single_moirai_v2_synth10_lotsa90_d384_cosine60000_*`            | 10 / 90             | **0.966**            | 16 K    | ~3.7 h     | 36 K   |

Three things stand out:

1. **Same ceiling.**  All three converge to `mse_z_imm_avg ≈ 0.965`
   within 0.03 of each other.  Mixture proportion is **not** the
   binding constraint.
2. **Same time-to-best.**  Two of three peak around step 16 K; the
   50/50 run peaks earlier (2 K) and aggressively overfits to
   synthetic from then on, but its peak is already at the same level.
3. **Train loss still decreasing.**  At ES time, `train/huber` is
   ~0.27 with healthy slope; the eval metric is what plateaus.

That pattern — eval saturates while train keeps moving — is the
classic signature of either (a) a learning-target / inductive-bias
mismatch, or (b) model capacity exhaustion against the actual eval
distribution.  We have already ruled out raw mix and raw LR schedule.
Two hypotheses remain testable in single 30 K runs.

#### 7.P.2  H2 — kill random univariate stacking

`StackedLOTSASource` (default `STACK_ALL`, §7.N.2) takes K random
univariate LOTSA records (e.g. K solar plants) and stacks them into
a single `[K, T]` "multivariate" sample.  These K series share no
time window, no scale, no domain, and no actual cross-variate
relationship.  Variate-axis attention may be learning the wrong
prior — "treat variates as independent" — because that prior is the
*correct* one for the dominant training distribution but is
catastrophically wrong on genuine multivariate IMM-TSF data
(ILINet V=11, ClusterTrace V=11, …).

H2 turns stacking off via `sources_nostack.yaml` and runs the
otherwise-best mix (30 / 70) for 30 K cosine steps.

- expected effect if hypothesis is right: ILINet R² flips from
  negative to positive; `mse_z_imm_avg` drops below 0.95.
- expected effect if hypothesis is wrong: same ceiling at 0.965, but
  V=1 mass jumps back up so ILINet probably gets *worse*.

Either outcome is informative.  Cost: one 30 K run, ~2.3 h.

Script: `scripts/run_H2_nostack.sh` — launched 2026-04-30 03:13 UTC,
screen `H2_nostack`.

#### 7.P.3  H1 — capacity bump

If H2 doesn't move the ceiling, the next lever is the model itself.
The current 384-dim model is ~37 M params; downstream baselines
(tPatchGNN, CRU) operate at 1–10 M but they are trained per-dataset.
A foundation-style model needs more capacity to absorb LOTSA's
diversity.

H1 doubles capacity:

| dim                | before | after |
|--------------------|--------|-------|
| `d_model`          | 384    | 512   |
| `d_hidden`         | 384    | 512   |
| `n_perv_layer`     | 3      | 4     |
| `n_fusion_blocks`  | 3      | 4     |
| `n_heads_varattn`  | 4      | 8     |

~37 M → ~75 M params.  Throughput drops from 4.5 it/s to ~2.2 it/s
on a single A100 → ~3.8 h for 30 K steps.  Batch size will likely
need to drop from 32 → 24 (configurable via env).

Script: `scripts/run_H1_bigger.sh` — staged, **not yet launched**.
Inherits H2's `sources_nostack.yaml` so a successful H2 + H1 stack
cleanly.

#### 7.P.4  Ordering rationale

Run H2 first because:
- it's a single yaml diff (no model surgery),
- it directly tests the largest deviation we made from Moirai
  (Moirai stacks only ~22 curated datasets and uses
  `beta_binomial(2,5,128)`; we stack everything at K=20),
- if H2 fails, capacity is implicated and H1 is cleanly motivated;
  if H2 succeeds, H1 becomes "stack on top of a working baseline"
  rather than confounded with the stacking change.

------------------------------------------------------------------------

## 8. Citations (short list)

- Woo et al., 2024.  Unified Training of Universal Time Series
  Forecasting Transformers (Moirai).  ICML 2024.  `arXiv:2402.02592`.
- Liu et al., 2024.  Moirai-MoE / Moirai-2 line.
- Ansari et al., 2024.  Chronos: Learning the Language of Time Series.
  TMLR.  `arXiv:2403.07815`.
- Stella et al., 2025.  Chronos-2: from univariate to universal
  forecasting.  Amazon Science technical report.
- Shi et al., 2024.  Time-MoE: Billion-Scale Time Series Foundation
  Models with Mixture of Experts.  ICLR 2025 / `arXiv:2409.16040`.
- Shukla & Marlin, 2021.  Multi-Time Attention Networks for
  Irregularly Sampled Time Series.  ICLR 2021.  `arXiv:2101.10318`.
- Wang et al., 2024.  tPatchGNN.  ICML 2024.
- Kim et al., 2022.  Reversible Instance Normalization (RevIN).
  ICLR 2022.
- Rubanova et al., 2019.  Latent ODEs for Irregularly-Sampled Time
  Series.  NeurIPS 2019.
- Chang et al., 2025.  Time-IMM / IMM-TSF.  NeurIPS 2025 D&B.
- Huber, 1992 (1964 reprint).  Robust Estimation of a Location
  Parameter.

------------------------------------------------------------------------

## 9. Change log

| Date       | Author | Change                                                                        |
| ---------- | ------ | ----------------------------------------------------------------------------- |
| 2026-04-26 | hongyu | Initial design-decisions README; six-concern review; pending-change requests. |
| 2026-04-26 | hongyu | §7.D Stage-B I/O bottleneck localized to HF Datasets Python-list formatter; A/B'd Moirai's PyArrow-direct indexer at 791× speedup; selected fix (e) — port the indexer pattern in-tree. |
| 2026-04-26 | hongyu | §7.D.5 Indexer fix landed in `sources.py::_HFArrowIndexer`; parity tests pass 28/28; Stage-B GPU util 5.6 % → 97.7 % end-to-end (12.2× more steps in same wall-clock). |
| 2026-04-26 | hongyu | §7.D.6 Eliminated `overflow encountered in square` from per-(b,v) standardization by promoting accumulator to float64 in `collate.py` and `val_imts.py`; verified clean over 800 batches with `-W error::RuntimeWarning`. |
| 2026-04-26 | hongyu | §7.G W&B integration in `train_pretrain.py`: `--use_wandb` + project / entity / run_name / mode / tags flags, alongside CSVLogger.  Verified offline-mode 30-step smoke run logs full hparams + train/huber stream. |
| 2026-04-26 | hongyu | §7.F Axis-4 ablation launched: `axis4_synth_only` (`stage_a.yaml`) → `axis4_synth_lotsa` (`stage_a_with_lotsa.yaml`); 50K steps each, 5 h total budget on 1×H100; runner `scripts/run_axis4_ablation.sh`. |
| 2026-04-26 | hongyu | §7.F First launch died at step 6749 (laptop sleep → SSH drop → SIGHUP cascade).  Re-launched `axis4_synth_only` only inside `screen -dmS axis4_synth_only`; W&B online to magicslabnorthwestern/mamba-imts-pretrain. |
| 2026-04-26 | hongyu | §7.F Re-launched (v2) at step 4674 of v1.  Migrated W&B project to **TSKing** and renamed runs to encode mix percentages: `stageA_chronos70_kernel30_d384_huber_50k_s42` (synth_only), `stageA_chronos55_kernel25_lotsa20_d384_huber_50k_s42` (synth_lotsa).  Only the first arm runs by default; we read its result before committing to Stage B. |
| 2026-04-26 | hongyu | §7.H Stage transition support: added `--init_from` (weights-only), kept `--ckpt` (full resume) for crash recovery; the two are mutually exclusive.  Added `_AggregateValMetricsCallback` logging `val/mse_z_imm_avg`, `val/mse_z_imts_avg`, `val/mse_z_avg_all` + a "best by val/mse_z_imm_avg" ModelCheckpoint (top-3).  Added `transition.json` provenance sidecar.  New helper `scripts/run_stage_b_from_a.sh` demonstrates the chain. |
| 2026-04-27 | hongyu | §7.I Stage A (synth-only, 50K) + Stage B (mixed, 200K) results recorded.  Stage A best `val/mse_z_imm_avg = 1.2722 @ 15K`, Stage B best **1.1420 @ 25K** initialized from Stage A best (−10.2 %), then plateau/drift through 200K. |
| 2026-04-27 | dawei  | Architecture: removed `variate_embed` per-slot tables in `VariableAxisAttention` and `SharedGridAligner`; added Moirai-style `_BinaryVariateAttentionBias`; collapsed `QueryReadout` to a shared head + shared `gamma`. |
| 2026-04-27 | hongyu | §7.J Wired `var_id` through `multivariate_forecaster.forward` so the new bias actually trains (was dead code in colleague's diff); reverted the smuggled MQA conversion (K/V projections back to full `d_model`) — Moirai-1 itself uses MHA in any-variate attn and our V ≤ 20 makes MQA's KV-cache motivation moot.  Verified with autograd smoke test; old ckpts confirmed incompatible (6 dropped params, 2 shape mismatches, 7 new). |
| 2026-04-29 | hongyu | §7.K Single-phase Stage A run (`single_phase`, `--lr_schedule multistep`).  ES tripped at step 44K / 80K when GDELT z-MSE went `inf`.  Best ckpt step 26K with `val/mse_z_imm_avg = 0.967`.  Underlying optimization was alive (train/huber 0.34 → 0.28 monotonic, synth in-dist R² 0.43 → 0.51) but eval blew up. |
| 2026-04-29 | hongyu | §7.L Eval-explosion fix: clamp `preds_asinh` to ±10 in `_log_imts_val` before the `sinh` inverse, drop `mse_z`/`mae_z` for activity & USHCN (irrelevant — known fixed scale, paper baselines use original units; per-(b,v) z is also numerically pathological for near-constant USHCN precipitation, see §Q4).  Aggregate callback filters non-finite from the IMM-TSF mean; ES uses `check_finite=False`.  No training-time changes. |
| 2026-04-29 | hongyu | §7.M LOTSA stalled — diagnosed via `scripts/diagnose_single_phase.py` (in-distribution snapshot, 2048 windows, separate seed, ckpt step 26K).  Root cause: 89.96 % of LOTSA sampling weight is on `solar_power` + `wind_power`, both univariate; multivariate datasets (45.9 % of count) carry only 2.0 % of weight.  Net effect: ~63 % of all training is univariate solar/wind, leaving multivariate axis-attention starved and ILINet (V=12) regressing.  Fix candidates listed; no relaunch yet. |
| 2026-04-30 | hongyu | §7.P R1 (`single_moirai_v2_synth10_lotsa90_*`) finished: ES at 36 K, best `val/mse_z_imm_avg=0.966` @ step 16 K — same ceiling as the 30/70 (0.965) and 50/50 (0.997) runs.  Mixture proportion confirmed not the binding constraint.  Three runs collapse to one ceiling → next axis is data structure (H2: stacking) or model capacity (H1). |
| 2026-04-30 | hongyu | §7.P H2 launched: `H2_nostack_synth30_lotsa70_d384_cosine30000_warmup500_minstd1e-3_s42`.  Single yaml diff: `configs/sources_nostack.yaml::lotsa_stacking_datasets="none"` disables `StackedLOTSASource`, leaving univariate-on-disk LOTSA datasets to emit V=1 windows; everything else identical to the 30/70 baseline.  Tests whether random univariate stacking teaches a "treat variates as independent" wrong prior that caps IMM-TSF transfer.  Screen `H2_nostack`, ETA ~2.3 h, W&B project TSKing. |
| 2026-04-30 | hongyu | §7.P H1 staged but **not launched**: `scripts/run_H1_bigger.sh` doubles capacity (d_model 384→512, d_hidden 384→512, n_perv 3→4, n_fusion 3→4, n_heads_varattn 4→8) → ~37 M → ~75 M params.  Default `BATCH_SIZE=24`, `LR=5e-4` (consider `2.5e-4` for the bigger model), inherits `sources_nostack.yaml` so it composes with H2.  Wall-clock estimate ~3.8 h for 30 K steps on 1×A100. |
| 2026-04-30 | hongyu | §7.Q H2 finished (full 30 K, no ES).  Best `val/mse_z_imm_avg = 0.9359 @ step 10 K` (−3 % vs 0.965 ceiling) but trajectory unstable: 1.02 → 1.69 → **0.94** → 1.23 → 1.33 → ~1.07–1.11 plateau.  Per-dataset wins on ClusterTrace (V=11) and RepoHealth (V=10) — partial confirmation of the wrong-stacking-prior hypothesis.  ILINet still broken (1.65), CESNET/GDELT/EPA-Air/FNSPID unchanged.  USHCN sustained improvement to ~0.79.  Loudest signal: `val/synth_lotsa_degraded_r2 = +0.03` for the entire run (vs +0.53/+0.30 for chronos2/kernel) → model cannot fit LOTSA real distribution at d=384.  Decision: launch H1. |
| 2026-04-30 | hongyu | §7.Q H1 launched: `H1_d512_h512_perv4_fus4_head8_bs24_lr2.5e-4_cosine30000_s42` (W&B `di1k`, screen `H1_bigger`).  Doubled arch (~37 M → ~75 M params), `BATCH_SIZE=24`, `LR=2.5e-4` (halved for the bigger model), inherits H2's `sources_nostack.yaml`.  Observed throughput at step ~1 K: 3.69 it/s, VRAM 52 / 80 GB, `train/huber ≈ 0.3–0.7`.  ETA ~2 h 15 min.  Three reads: (a) does `mse_z_imm_avg` push below 0.93 and stay there, (b) does `lotsa_degraded_r2` lift off +0.03, (c) does ILINet drop below 1.0. |
| 2026-04-29 | hongyu | §7.N Moirai-style sampling fix landed.  (1) `make_logical_source` now weights datasets by `num_ts × yaml_w` — solar/wind drop from 89.96 % → 2.27 % of LOTSA mass, top-1 dataset 4.25 %, mass spread over 30+ datasets.  (2) `StackedLOTSASource` wraps univariate-on-disk LOTSA via `MultiSampleTimeSeriesDataset`-style stacking (`K = max_variates = 20`); natively-MV passes through.  (3) Default policy `STACK_ALL` — more aggressive than Moirai's curated list to cover all univariate datasets.  (4) `variate_count_dist` retuned to `[0.05, 0.35, 0.55, 0.05]` to align with IMM-TSF Table-1 V distribution.  Smoke test (`scripts/diagnose_moirai_sampling.py`, N=2048, post-transform): V=9..16 mass 9 % → 41.6 %, V=1 mass 63 % → 7.9 %.  (5) LR schedule switched to cosine 30 K (warmup 500), replacing multistep 80 K which never decayed before ES.  Run name: `single_moirai_v2_synth30_lotsa70_regimeMix40_d384_cosine30000_warmup500_s42`.  No relaunch yet — awaiting user approval. |
