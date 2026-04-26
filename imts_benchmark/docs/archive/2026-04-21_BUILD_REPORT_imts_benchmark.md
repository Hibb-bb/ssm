# Build Report — Multivariate IMTS Forecasting Benchmark

Companion to [`RESULTS_imts_benchmark.md`](RESULTS_imts_benchmark.md). Documents
what was built, decisions made during construction, and how to reproduce.

Status: **incremental — updated as Phases 1–6 of [`../imts_benchmark_plan.md`](../imts_benchmark_plan.md) complete.**

---

## Phase 1 — Folder rename + history bump (DONE)

Renamed the working tree from `mv_vs_romae/` to `imts_benchmark/`. Updated
all imports (`mv_vs_romae.*` → `imts_benchmark.*`) across:
- `mamba_mv/train_mv.py`
- `romae_forecaster/train_romae.py`
- `eval/{plot_comparison_v2,plot_forecast_examples,plot_training_curves,make_plots}.py`
- `scripts/{run_mamba_mv,run_romae,run_aggregate,run_debug_mamba,run_debug_romae}.sbatch`

Updated log paths (`mvcompare_v1` → `imts_benchmark_v1`) in the same files.

`shared_config/fair_defaults.py` changes:
- `HISTORY = 7.0 → 8.0`
- `LOG_ROOT_DEFAULT` repointed to `output/log/imts_benchmark_v1`
- `REGIMES` extended with `nogap_independent`, `nogap_dependent`

The frozen 2-model report files were renamed:
- `RESULTS.md` → `RESULTS_mamba_vs_romae.md`
- `BUILD_REPORT.md` → `BUILD_REPORT_mamba_vs_romae.md`

The original `output/log/mvcompare_v1/` log tree is **untouched** — it
remains the citable history=7.0 precursor.

**Editable-install gotcha**: the vendored RoMAE package (`romae==0.9.0a0`)
is `pip install -e` registered in the conda env at
`/projects/b1094/StarEmbed/pythonenvs/mamba`. The editable record stored
the OLD path under `mv_vs_romae/romae_forecaster/_romae_repo`, which
broke after the rename. Fix:
```
/projects/b1094/StarEmbed/pythonenvs/mamba/bin/pip install -e \
  /projects/b1094/.../ssm_dk/imts_benchmark/romae_forecaster/_romae_repo
```
After reinstall, `import romae` resolves to the new path. **Verified** via
`python -m imts_benchmark.romae_forecaster.train_romae --help`.

**Verification commands** (both pass):
```
python -m imts_benchmark.mamba_mv.train_mv --help
python -m imts_benchmark.romae_forecaster.train_romae --help
```
Both show all 4 regimes (`sparse_*`, `nogap_*`) in the `--regime` choice
list, `--history` defaulting to 8.0 via fair_defaults.

## Phase 2 — No-gap dataset variants + history=8 regen (DONE)

**Important re-discovery**: the datamodule reads `history` from each
sample (`it["history"]`), not from `args.history` passed to the trainer.
So bumping `HISTORY=7→8` in `fair_defaults.py` alone did NOT shift the
forecast boundary at training time — the data itself stores the boundary.
Therefore all 4 regimes had to be regenerated with `--history 8.0` baked
into each sample.

### Generator changes
- Added `--history` CLI flag (default `HISTORY=7.0` for back-compat;
  passed explicitly as `--history 8.0` for all new generation).
- Added `--out_subdir` CLI flag so `nogap_*` variants don't overwrite
  `sparse_*`. Defaults to `--regime` for back-compat.
- `build_sample()` now takes `history` as an argument and stores it in
  the row; `norm_stats.json` records the value used.

### Old data preserved
The original `data/` (history=7.0) was moved to `data_history7_backup/`
before regeneration. The frozen `mvcompare_v1` log tree was trained on
that data; if anyone needs to re-run an old seed they should point
`--data_root data_history7_backup`.

### New data layout (history=8.0 throughout)
```
data/
├── sparse_independent/    K=1 variate gapped per sample, ~111 obs/var
├── sparse_dependent/      K=1 variate gapped per sample, ~111 obs/var
├── nogap_independent/     all dense, 120 obs/var
└── nogap_dependent/       all dense, 120 obs/var
```

Generation commands recorded for reproducibility:
```bash
python generate_longgap_multisin.py --regime sparse_independent \
    --history 8.0 --gap_variates_per_sample 1 \
    --n_train 1000 --n_val 200 --n_test 200
python generate_longgap_multisin.py --regime sparse_dependent \
    --history 8.0 --gap_variates_per_sample 1 \
    --n_train 1000 --n_val 200 --n_test 200
python generate_longgap_multisin.py --regime sparse_independent \
    --history 8.0 --gap_variates_per_sample 0 --n_gaps 0 \
    --out_subdir nogap_independent \
    --n_train 1000 --n_val 200 --n_test 200
python generate_longgap_multisin.py --regime sparse_dependent \
    --history 8.0 --gap_variates_per_sample 0 --n_gaps 0 \
    --out_subdir nogap_dependent \
    --n_train 1000 --n_val 200 --n_test 200
```

### Verification
- `_smoke_plot.py` extended to 4 regimes; image at `_smoke_plot.png` shows
  all 3 variates per regime with the `history=8.0` boundary marked.
- Datamodule sanity test (per-variate format, batch=4 on test split):
  - `sparse_*`: 332-339 valid obs / sample (gapped variate has ~80 obs)
  - `nogap_*`: 360 valid obs / sample (full 120 × 3)
  - All 4 regimes: pred_mask correctly counts obs with `timestamp >= 8.0`
    (~71-77 per sample across 3 variates).

## Phase 3 — New baseline adapters

### Phase 3a — mTAN (PyTorch) — DONE
- Vendored `_upstream/models.py` from `github.com/reml-lab/mTAN` (MIT). Only
  `multiTimeAttention`, `enc_mtan_rnn`, `dec_mtan_rnn` are used; remaining
  classes vendored verbatim for upstream fidelity.
- Adapter `mtan_forecaster.py`: builds per-sample sorted union of
  (timestamp, variate, value) tuples; encoder takes one-hot variate channels
  (value + mask = 2V dims) + sorted union times; decoder queries at union
  times and outputs all V variates per token; deterministic latent (no IWAE
  / KL); MSE on (forecast time, forecast variate) cells.
- Sized to **7.73M params** (-0.9% vs 7.80M target) via
  `rec_hidden=576, gen_hidden=576, latent_dim=64, num_ref_points=64,
  embed_time=128, num_heads=1, learn_emb=True`.
- Device caveat: vendored encoder/decoder hardcode `self.device` at init;
  adapter monkey-patches it on each forward so Lightning's device placement
  is authoritative.
- **Smoke test passed** (CPU, 20/5/5 dataset, batch=4, 2 epochs):
  loss decreases (train 0.336 → val 0.313 → test 0.302), checkpoint
  save/restore works, `test_metrics.csv` + `per_sample.jsonl` written in
  same schema as Mamba/RoMAE (mTAN uses `n_pred_v{d}` like RoMAE; aggregator
  handles both `n_obs_v{d}` and `n_pred_v{d}` via `make_plots.py:35`).

### Phase 3b — ContiFormer (PyTorch) — CODE COMPLETE, GPU SMOKE PENDING
- Vendored minimal `_upstream/physiopro/{network/contiformer.py,
  module/{linear,ode,interpolate,positional_encoding}.py}` from
  `github.com/microsoft/physiopro` (MIT). Patched `contiformer.py` to drop
  `from .tsrnn import NETWORKS` and the `@NETWORKS.register_module(...)`
  decorator (the registry framework requires `microsoft/utilsd`, irrelevant
  to direct instantiation).
- Pip deps added to env: `torchdiffeq`, `torchcde` (with transitive
  `torchsde`, `trampoline`).
- Adapter `contiformer_forecaster.py`: identical batch transform to mTAN
  (per-sample sorted union, 2V one-hot channels, hide forecast positions);
  ContiFormer encoder produces [B, T, d_model]; linear head -> V variates
  per query time; same MSE-on-cells loss.
- Sized to **7.65M params** (-1.9% vs 7.80M target) via
  `d_model=320, d_inner=1024, n_layers=6, n_head=4, d_k=80`.
- **Forward verified** (CPU, B=1, T=10 union tokens, ~9s — ODE-attention is
  intrinsically slow). **Backward not verified on CPU** (timed out at 90s
  even on B=2/T=30 due to `odeint_adjoint` cost). Pipeline is syntactically
  complete and uses the same API as the upstream `spiral.py` example;
  deferred to GPU smoke during Phase 4.
- **Memory caveat**: ContiFormer's MultiHeadAttention projects to
  [B, L, L, hidden, n_head, d_k] BEFORE attention — O(L^2) memory. For our
  T~360 at B=128, this can reach 10s of GB. Documented in
  `contiformer_forecaster.py` docstring; if H100 OOMs we'll drop to B=64
  or subsample the union token set per sample.

### Phase 3c — S5 (JAX, separate env) — PENDING

## Phase 4 — Final runs (PENDING)

140 runs across 5 models × 4 regimes × 5 seeds.

## Phase 5 — Evaluation extension (PENDING)

Extend `aggregate_results.py` and plot scripts to handle 5 models × 4
regimes. Add gap-vs-no-gap delta plot.

---

## Known caveats (carried forward from the plan)

- **history=8.0 + existing gap range**: gap generator uses
  `gap_start ∈ [2.5, 6.0]`, `gap_len ∈ [1.5, 3.0]`, so gaps can end as
  late as `t ≈ 9.0`. With history=8, ~half the long-gap samples will have
  gaps that partially overlap the forecast window `[8, 10]`. This is
  acceptable but should be noted in the paper's data section.
- **No escape valve**: if any baseline collapses to mean-predictor under
  the shared recipe, it is reported as-is.
- **Param budget**: ±10% of 7.8M per model. mTAN's paper defaults produce
  smaller models; scaling up `rec-hidden` / `num-ref-points` to reach
  budget is not a published config and should be validated on a small
  run before the full array launches.
