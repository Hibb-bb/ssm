# Handoff: Mamba-MV UEA classification

This doc lists **everything** a collaborator needs to (a) reproduce the UEA-5
result set or (b) run Mamba-MV classification on a new dataset of their own.

Anchor: project root in this repo is
`src/train/moirai/uni2ts_hongyu/ssm_dk/imts_benchmark/` (referred to as
`<root>` below).

---

## 1. File manifest — push exactly these

### Core model code
```
<root>/mamba_mv/__init__.py
<root>/mamba_mv/classification_head.py        # HAN-style attention pool head
<root>/mamba_mv/multivariate_classifier.py    # Lightning module (cls)
<root>/mamba_mv/train_cls.py                  # entry point
<root>/mamba_mv/irregular_ssm.py              # per-variate Mamba SSM stage
<root>/mamba_mv/shared_grid.py                # one-time shared-grid alignment
<root>/mamba_mv/temporal_mamba.py             # temporal Mamba on grid
<root>/mamba_mv/variable_axis_attention.py    # variable-axis attention
<root>/mamba_mv/mamba_block.py                # Mamba/MambaIrregular blocks
```

### Data + eval
```
<root>/shared_data/__init__.py
<root>/shared_data/uea_classification_datamodule.py   # UEA .ts loader + Kidger 30% drop
<root>/eval/__init__.py
<root>/eval/aggregate_uea_hpo.py                       # HPO winner picker
```

### SLURM / launchers (Quest-specific paths inside; expect to edit)
```
<root>/scripts/run_uea_cls_hpo.sbatch         # 30-task array (5 ds × 2 dt × 3 LR)
<root>/scripts/run_aggregate_uea_hpo.sbatch   # CPU job, picks per-cell winner LR
<root>/scripts/run_uea_cls_final.sbatch       # 30-task array (5 ds × 2 dt × 3 seeds)
<root>/scripts/submit_uea_cls_pipeline.sh     # chained-dependency submitter
```

### Package marker (must be present at parent so `imts_benchmark.*` resolves)
```
<root>/__init__.py
```

### Design doc (optional but recommended for context)
```
<root>/docs/REPORT_uea_classification_design_2026-04-30.md
```

**Total: ~15 files.** Everything else under `imts_benchmark/` is unrelated to
this task.

### Things NOT to push
- Anything under `<root>/romae_forecaster/`, `s5_forecaster/`, `mtan_forecaster/`,
  `contiformer_forecaster/` — separate baselines, not used by classification.
- Anything in `<root>/mamba_mv/` not listed above (`train_mv.py`,
  `multivariate_forecaster.py`, `query_readout.py` are forecasting-track only).
- `output/log/...` — runtime outputs.
- `data_uea/` — the actual UEA `.ts` files (collaborator already has data).

---

## 2. Quick-start for the collaborator

### 2a. Environment

The training script needs:

```
python>=3.10
torch (CUDA build matching your GPU)
pytorch-lightning>=2.0
mamba-ssm           # only needed if using the Mamba selective-scan kernel; bf16 is NOT supported, run FP32
einops
numpy
```

Quest reference env: `/projects/b1094/StarEmbed/pythonenvs/mamba`. On a different
cluster, recreate the same package set and pin `precision=32-true` (the Mamba
CUDA kernel asserts `delta.dtype == u.dtype` which breaks under bf16-mixed).

### 2b. Run a smoke test on one cell (5 min on H100/A100)

Assuming the collaborator already has a directory `<DATA>/<DatasetName>/`
containing `<DatasetName>_TRAIN.ts` and `<DatasetName>_TEST.ts` in standard
UEA format:

```bash
cd <repo_root>/ssm_dk

python -m imts_benchmark.mamba_mv.train_cls \
    --dataset BasicMotions \
    --data_root <DATA> \
    --dt_mode learned \
    --lr 1e-3 \
    --seed 42 \
    --max_epochs 30 \
    --patience 10 \
    --num_workers 2 \
    --output_dir ./smoke_out \
    --run_name smoke
```

A `summary.json` is written to `./smoke_out/smoke/summary.json` containing
`best_val_acc`, `test_acc`, `macro_f1`, fit time, and the resolved hparams.

### 2c. Adding a new dataset

Two paths, depending on the dataset format:

**Path A (preferred): UEA `.ts` format, all variates present at every step.**
Just call `train_cls` with `--dataset <Name> --data_root <dir>` — the
datamodule handles parsing. Per-dataset defaults (batch_size, label_smoothing,
grad_clip, grid_K) live in
[`mamba_mv/train_cls.py:ROMAE_DATASET_DEFAULTS`](../mamba_mv/train_cls.py).
Add a row there for the new dataset.

**Path B: different format (PhysioNet, USHCN, custom).**
Write a new `LightningDataModule` mirroring
[`shared_data/uea_classification_datamodule.py`](../shared_data/uea_classification_datamodule.py).
Required surface for `train_cls.py` to consume it:

```python
dm.n_vars            # int, number of variates V
dm.n_classes         # int, number of classes C
dm.class_weights     # FloatTensor[C] or None — passed to weighted CE
dm.train_ds          # supports len()
dm.val_ds, dm.test_ds
# Each batch is a dict with keys (matching mamba_mv.multivariate_classifier.forward):
#   x         FloatTensor [B, T, V]   raw values (z-scored or not, your choice)
#   t         FloatTensor [B, T]      timestamps in [0, 1]
#   m         BoolTensor  [B, T, V]   observation mask (True = observed)
#   label     LongTensor  [B]
```

The forecasting datamodule
[`shared_data/multivariate_datamodule.py`](../shared_data/multivariate_datamodule.py)
is another reference for the per-batch tensor layout.

Then point `train_cls.py` at it (swap the `UEAClassificationDataModule` import).

### 2d. Run the full HPO + final-eval pipeline

Edit the four files under `scripts/`:
- the `#SBATCH --account=...` line (currently `p33049`)
- the `#SBATCH --output=...` / `--error=...` paths
- the absolute paths inside the body (`MAMBA_ENV`, `CODE_DIR`, `DATA_ROOT`, `LOG_BASE`)
- the `DATASETS=(…)` list in `run_uea_cls_hpo.sbatch` and `run_uea_cls_final.sbatch`
  (currently the 5 UEA ones)
- `--array=1-N` size where `N = #datasets × 2 dt_modes × 3 LRs` for HPO
  and `N = #datasets × 2 dt_modes × 3 seeds` for final.

Then:
```bash
bash <root>/scripts/submit_uea_cls_pipeline.sh
```

This submits HPO array → aggregator → final array, chained with
`--dependency=afterok:`.

---

## 3. Known footguns (read before running)

1. **Mamba CUDA kernel and precision.** Use `--precision 32-true` (the default).
   bf16-mixed crashes inside `selective_scan_fn` with a dtype assertion.
2. **`enable_checkpointing=False` is set in the Trainer** (in
   `train_cls.py`). Lightning's default ModelCheckpoint writes to a
   CWD-relative `checkpoints/` directory, which collides catastrophically when
   30 array tasks share `cd ${CODE_DIR}`. Don't re-enable it without per-run
   `default_root_dir`.
3. **Label smoothing convention.** RoMAE Table 12 reports a *confidence* `c`
   ("reducing each correct class label from 1 to a confidence value c"); PyTorch
   `F.cross_entropy(label_smoothing=p)` takes the *smoothing amount*. Mapping is
   `p = 1 - c`. The defaults in `ROMAE_DATASET_DEFAULTS` are already in
   PyTorch's convention. If you copy values from a different paper, check the
   convention.
4. **Heartbeat OOM.** At V=61 the variable-axis attention OOMs on 32 GB at
   `grid_K=256, batch=32`. We use `grid_K=128, batch=16` for HB. If the
   collaborator's dataset has high V, expect to reduce `grid_K` or `batch_size`.
5. **`logger=False` is hard-coded in `train_cls.py`.** If the collaborator
   wants W&B, swap it for a `WandbLogger(...)` and remove the comment guard
   above the callback list.

---

## 4. Pointers in the design doc

For context on architecture and HPO choices the collaborator should read
[`docs/REPORT_uea_classification_design_2026-04-30.md`](REPORT_uea_classification_design_2026-04-30.md):
- Section 2 — locked classification head (HAN-style attention pool, no dropout)
- Section 3 — optimizer / schedule / per-dataset HPs
- Section 4 — HPO sweep design (3 LRs only; everything else inherited from
  RoMAE Table 12 to keep the per-method recipe fair)

---

## 5. One-liner to bundle for transfer (optional)

From `<repo_root>/ssm_dk`:
```bash
tar czf mamba_mv_uea_classification.tar.gz \
    imts_benchmark/__init__.py \
    imts_benchmark/mamba_mv/{__init__.py,classification_head.py,multivariate_classifier.py,train_cls.py,irregular_ssm.py,shared_grid.py,temporal_mamba.py,variable_axis_attention.py,mamba_block.py} \
    imts_benchmark/shared_data/{__init__.py,uea_classification_datamodule.py} \
    imts_benchmark/eval/{__init__.py,aggregate_uea_hpo.py} \
    imts_benchmark/scripts/{run_uea_cls_hpo.sbatch,run_aggregate_uea_hpo.sbatch,run_uea_cls_final.sbatch,submit_uea_cls_pipeline.sh} \
    imts_benchmark/docs/{REPORT_uea_classification_design_2026-04-30.md,HANDOFF_uea_classification.md}
```

That's the smallest self-contained bundle. ~15 files, no runtime artifacts, no
unrelated baseline code.
