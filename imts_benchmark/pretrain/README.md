# Mamba IMTS Pretraining — Design Notes & Concerns

This document is the **decision log** for the multivariate Mamba pretraining
pipeline that lives in `hongyu/ssm/imts_benchmark/pretrain/`.  It is meant to
be (a) a reference for everyone touching this code, and (b) raw material for
the eventual paper.  Every non-obvious choice we make is recorded here with
its rationale, the alternatives we considered, citations to prior work, and
the open questions we still need to resolve.

If you change a knob, **also update this file**.

> Status: 2026-04-26.  Pre-launch design review; no large pretraining run has
> been launched yet.  Blocking concerns marked **OPEN** below need your call.

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
