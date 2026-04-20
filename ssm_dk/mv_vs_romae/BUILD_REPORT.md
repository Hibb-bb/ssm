# Build Report: Mamba-MV vs RoMAE on Irregular Multivariate Forecasting

Audience: Dongho and collaborators — a record of what was built, what was
reused, what was modified, and why.

---

## 1. Starting point

Everything we reused came from **one canonical folder**:

```
/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm/mamba_experiments/
├── configs/
├── dataset_generation/
├── environment/
│   └── build_mamba_env.sh              # how pythonenvs/mamba was built
├── evaluation/
├── forecaster/
│   ├── __init__.py
│   ├── mamba_block.py                  # MambaBlock, MambaIrregularBlock
│   ├── mamba_forecaster.py             # univariate PL module
│   ├── plot_mamba_comparison.py
│   ├── sinusoidal_datamodule.py        # univariate DataModule
│   ├── test_cuda_kernel.py
│   └── train.py                        # univariate CLI entry point
├── scripts/
│   ├── run_mamba_cuda_fixed.sh         # SLURM template we copied
│   ├── run_mamba_fixed_comparison.sh
│   ├── run_mamba_sinusoidal.sh
│   └── test_mamba_cuda.sh
└── tsc/
```

Nothing in `ssm_model/mamba_forecaster/` — that older copy was removed by the
owner during development. We re-pointed all imports to the canonical
`ssm_model/ssm/mamba_experiments/forecaster/` subpackage.

Everything NEW lives in the writable directory we own:

```
ssm_dk/mv_vs_romae/
├── shared_config/fair_defaults.py        # controls one set of fair knobs
├── shared_data/multivariate_datamodule.py  # NEW — sparse-flat -> per-variate / flat-tokens
├── mamba_mv/                             # NEW — paper's full multivariate arch
│   ├── irregular_ssm.py
│   ├── shared_grid.py
│   ├── variable_axis_attention.py
│   ├── temporal_mamba.py
│   ├── query_readout.py
│   ├── multivariate_forecaster.py        # PL module
│   └── train_mv.py                       # CLI entry point
├── romae_forecaster/                     # NEW — transformer baseline
│   ├── _romae_repo/                      # cloned from github.com/Chromeilion/RoMAE (pip -e)
│   ├── romae_forecaster.py               # PL wrapper: MAE -> forecasting
│   └── train_romae.py
├── scripts/
│   ├── run_mamba_mv.sbatch
│   ├── run_romae.sbatch
│   └── run_aggregate.sbatch
└── eval/
    └── aggregate_results.py              # per-cell summary + paired Wilcoxon
```

---

## 2. What existed vs what we added

### 2.1 Univariate Mamba forecaster (existing)

`forecaster/mamba_forecaster.py` in `mamba_experiments/` is a single-variate
forecaster. Forward signature:

```python
def forward(self, values, delta_t=None):
    # values:  [B, L]
    # delta_t: [B, L]
    # returns: preds [B, L]
```

Three variants controlled by `dt_mode`:
- `"learned"` — vanilla Mamba (`MambaBlock`)
- `"replace"` — delta = true time gap (`MambaIrregularBlock`)
- `"additive"` — delta = softplus(learned + true dt)

Loss: MSE on positions with `timestamp >= history (7.0)`, inputs in that
region zeroed before forward to prevent leakage.

Training: AdamW, `lr=5e-4`, warmup 100 + cosine decay, bf16, `max_epochs=200`,
`patience=20`, `batch_size=128`. Metrics: MSE, MAE, R², Pearson.

### 2.2 What Mamba-MV changes

The paper defines a **multivariate** irregular-step SSM with five pieces:
1. Per-variate irregular-step encoder.
2. Alignment of per-variate hidden states onto a shared uniform grid.
3. Variable-axis attention across variates at each grid slot.
4. Temporal Mamba along the shared grid.
5. Query-time readout to the (variate, time) points we want to predict.

We reuse the existing `MambaBlock` / `MambaIrregularBlock` byte-for-byte for
pieces 1 and 4 (so the CUDA selective-scan kernel keeps working). Every other
piece was newly written, matching the paper's equations. Design notes:

- **Shared grid `s_1 < ... < s_K`** (`K=128` by default) uniform over
  `[0, t_max=10]`. The code can be swept over K later, we chose 128 because
  1 grid slot ≈ 0.078 time units, slightly finer than the mean observation
  spacing (~0.083).
- **Alignment (`shared_grid.py`)**. For each (sample, variate, slot) we gather
  the latent state `h_{pi_d(k)}^{(d)}` at the latest observed index before
  `s_k`, then apply a learnable exponential decay
  `z = h * exp(-gamma_d * (s_k - t_pi))` as a fast stand-in for
  `bar_A_{rho} h` from the paper. If no observation is available, we swap in
  a per-variate learned null state. Staleness + variate-id + availability
  are concatenated and projected.
- **Variable-axis attention (`variable_axis_attention.py`)**. At each grid
  slot, variates attend to each other. Additive embeddings for variate id,
  availability, and sinusoidal staleness are re-injected (per the paper's
  `tilde H`). A large negative bias on unavailable keys prevents them from
  providing evidence.
- **Temporal Mamba (`temporal_mamba.py`)**. Regular-grid Mamba (`dt_mode=
  learned` because the grid is uniform). Applied per variate via
  `[B,K,V,D] -> (B*V, K, D)`.
- **Query readout (`query_readout.py`)**. For each target `(v, t)`, gather
  `H^{(L)}[kappa, v, :]`, decay by `exp(-gamma_v * omega)`, concat sinusoidal
  `phi(omega)`, per-variate linear head.

PL module `MultivariateMambaForecaster` **copies the existing
optimizer/scheduler and test_step metric accounting verbatim** from the
univariate version so the only thing that differs between the two models is
the architecture — everything else is identical.

### 2.3 RoMAE adapter (new)

We vendored [github.com/Chromeilion/RoMAE](https://github.com/Chromeilion/RoMAE)
into `romae_forecaster/_romae_repo/` and installed it editable
(`pip install -e _romae_repo --no-deps`) into the same conda env as Mamba.
Python 3.12 in `pythonenvs/mamba` satisfies RoMAE's `requires-python>=3.11`,
so no second env is needed — which was a simplification over the plan.

RoMAE is an MAE (mask-then-reconstruct). We did **NOT** modify RoMAE's source;
instead we turned it into a forecaster entirely at the input-construction
layer:

1. **Tokenization**. Each observation `(value, timestamp, variate)` is one
   token. `tubelet_size=(1,1,1)`, `n_channels=1` → patchify is a no-op.
2. **Positions**. `positions=[B, 2, N]` = `[timestamp, variate_id]` (both
   float). RoPEND embeds both dims so the same attention scores are
   functions of real time gaps and variate identity.
3. **Forecast mask**. We set `mask[b, i] = (timestamp[b,i] >= history) AND
   real_token`. Encoder sees only non-masked tokens; decoder reconstructs
   the masked ones.
4. **Loss**. Built-in `MSELoss(reduction='none')` averaged on masked tokens —
   identical to Mamba's MSE on the prediction region. No code change inside
   RoMAE.

Param match at `enc d_model=288, nhead=6, depth=7` with the default
tiny-shallow decoder gives **7.80M trainable params**, matching Mamba-MV's
**7.79M** within 0.2%.

### 2.4 Shared infrastructure (new)

- **`shared_config/fair_defaults.py`**: a single `add_fair_args(parser)` call
  imported by both trainers. Any change to `lr`, `weight_decay`,
  `num_warmup_steps`, `max_epochs`, `patience`, `batch_size`, `precision`
  propagates to both models without a manual edit in two places.
- **`shared_data/multivariate_datamodule.py`**: reads the sparse-flat HF
  dataset (the format generated by `generate_longgap_multisin.py`) and
  splits per-variate using `n_obs_per_var`. Two export formats selected by
  `format=` kwarg:
    - `"per_variate"` → `[B, V, L_max]` tensors for Mamba-MV.
    - `"flat_tokens"` → one token per observation for RoMAE.
- **`eval/aggregate_results.py`**: walks `output/log/mvcompare_v1/...`, emits
  summary.csv (mean±std per cell), long.csv (for plotting), and paired
  Wilcoxon signed-rank tests on per-sample MSE.

---

## 3. Modifications to the existing SLURM template

We used
[`run_mamba_cuda_fixed.sh`](../../ssm_model/ssm/mamba_experiments/scripts/run_mamba_cuda_fixed.sh)
as the starting template. Identical headers kept:

```
#SBATCH --account=p32626
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
module purge; module load gcc/11.2.0
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
MAMBA_ENV=/projects/b1094/StarEmbed/pythonenvs/mamba
```

| Change | Old | New |
|---|---|---|
| Code dir | `ssm_model/ssm/mamba_experiments` | `ssm_dk` |
| Output root | `output/log/mamba_sinusoidal_cuda_fixed/` | `output/log/mvcompare_v1/` |
| Array size | 15 (3 variants × 5 seeds) | 30 for Mamba-MV, 10 for RoMAE |
| Array mapping | `seed = (id-1)//3`, `variant = (id-1)%3` | `regime, dt_mode, seed` packed as documented in each sbatch header |
| Entry point | `forecaster.train` | `mv_vs_romae.mamba_mv.train_mv` / `mv_vs_romae.romae_forecaster.train_romae` |
| Dataset path | `--data_root /projects/.../tpatchgnn_data_nobs160 --irregularity high_irreg` | `--regime sparse_{in,}dependent` (reads from `ssm_dk/data/` by default) |

The new RoMAE sbatch is structurally identical but runs the RoMAE CLI.
`run_aggregate.sbatch` is a short-partition CPU job that executes the
aggregator after the two array jobs finish (`--dependency=afterany`).

---

## 4. Fair-comparison controls

Every axis that could confound the comparison is pinned to a single source
of truth:

| Axis | Where it's set | Value |
|---|---|---|
| Dataset | `shared_config.fair_defaults.DATA_ROOT_DEFAULT` | `ssm_dk/data/{regime}` |
| Task | both `_compute_loss` implementations | MSE on `timestamp >= history AND real_token` |
| Optimizer | both `configure_optimizers` | AdamW, `lr=5e-4`, `wd=0.01`, bias/norm in no-decay group |
| LR schedule | both `configure_optimizers` | warmup 100 steps + cosine to 1600 |
| Epochs / patience | `fair_defaults.MAX_EPOCHS`, `PATIENCE` | 200 / 20 |
| Batch size | `fair_defaults.TRAIN_BATCH_SIZE`, `VAL_BATCH_SIZE` | 128 / 32 |
| Precision | `fair_defaults` → `Trainer(precision=...)` | `bf16-mixed` |
| Params (±5%) | model defaults | Mamba-MV 7.79M, RoMAE 7.80M (0.2% gap) |
| Seeds | sbatch arrays | `{1,2,3,4,5}` |
| Compute | sbatch header | H100, 32G mem, 4 CPUs, 2h walltime |
| Metrics | both `test_step` | MSE, MAE, R², Pearson, plus per-variate MSE |

Fairness is verified inside the aggregation script (`wall_fit_sec`, `n_params`
columns in summary.csv).

---

## 5. Pitfalls encountered and their fixes

These are worth calling out because they can easily regress:

1. **Path resolution off by one**. First draft used
   `Path(__file__).parents[4]` to reach `uni2ts_hongyu/` — that actually
   lands at `moirai/`. Fixed to `parents[3]`.
2. **Canonical ssm_model subpath moved**. The earlier
   `ssm_model/mamba_forecaster/` was removed. New imports target
   `ssm_model/ssm/mamba_experiments/forecaster/mamba_block`.
3. **RoMAE requires a uniform mask count per sample**. Its forward does
   `x[mask].reshape(b, -1, F)`, which fails whenever samples in a batch have
   different numbers of masked tokens. Our forecast mask is inherently
   variable-length. Fix: in `collate_flat_tokens` we pad each sample's
   `pred_mask` with dummies in the padding region so every sample has
   exactly `pred_max = max(pred_real_count)` True positions. RoMAE's own
   `loss[m_pad_mask] = 0` then zeroes those dummies. A companion
   `pred_real_count` tensor lets `test_step` recover the real targets.
4. **`pad_mask` convention is inverted**. RoMAE internally uses
   `per_sample_n = (~pad_mask).sum(dim=1)` and `loss[m_pad_mask] = 0` →
   `pad_mask=True` means PADDING. Our datamodule's convention is the
   opposite (True = real). The RoMAE wrapper flips the sign at the
   boundary.
5. **`dt_proj` has no grad in `dt_mode='replace'`**. Expected — the learned
   dt_proj is not used when delta is overridden with the true gap. Same
   behaviour as the existing univariate forecaster.

---

## 6. Testing performed before SLURM submission

1. `shared_data.MultivariateSinusoidalDataModule` produces the expected
   shapes for both formats; per-variate slicing round-trips on the
   sparse-flat HF data.
2. `MultivariateMambaForecaster.forward` + `loss.backward()` runs on CPU
   with 2-sample batches for `dt_mode ∈ {learned, replace, additive}`.
3. Parameter count check: at `d_model=d_hidden=384, n_perv=3, n_fusion=3`,
   `d_state=16` → 7.79M trainable.
4. `RoMAEForecaster.forward` + `loss.backward()` runs on CPU with 2-sample
   batches. All parameters receive gradients. Logits shape is
   `[B, pred_max, 1]`.
5. Per-variate MSE breakdowns emit correctly in both `test_step`s
   (inspected per-sample json).
6. `python -m mv_vs_romae.mamba_mv.train_mv --help` and
   `python -m mv_vs_romae.romae_forecaster.train_romae --help` both resolve
   and show the shared-fair args plus model-specific overrides.

Remaining: a first GPU run (one seed, short `max_epochs`) before launching
the full 40-job array — that step is cheap and worth doing before spending
a full cluster quota.

---

## 7. How to submit

```bash
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/mv_vs_romae/scripts
MAMBA_JID=$(sbatch --parsable run_mamba_mv.sbatch)
ROMAE_JID=$(sbatch --parsable run_romae.sbatch)
sbatch --dependency=afterany:${MAMBA_JID}:${ROMAE_JID} run_aggregate.sbatch
```

Outputs:
- Per-run: `output/log/mvcompare_v1/<model>/<regime>/<variant>/seed<S>/test_metrics.csv` + `per_sample.jsonl`
- Aggregate: `output/log/mvcompare_v1/summary.csv` + `long.csv` and the
  paired Wilcoxon table written to the aggregate job's stdout.

---

## 8. Future ablations that stay within this scaffolding

- **Grid K ∈ {64, 128, 256}** — pass `--grid_K` on Mamba-MV only.
- **Parameter-count Pareto** — pass `--d_model --n_fusion_blocks` (Mamba) or
  `--enc_d_model --enc_depth` (RoMAE). No other changes needed.
- **Gap-length sweep** — regenerate data with
  `--gap_len_range` modified and point `--data_root` at the new folder.
- **gap_variates_per_sample ∈ {1, 3}** — negative control when no
  cross-variate information is available.
