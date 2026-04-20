# Synthetic Long-Gap Multivariate Sinusoid — Build Notes

Working directory: `/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk/`

---

## 1. Reference Files Consulted

All references live under
`/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_model/ssm/`.

| File | What we used it for |
|------|--------------------|
| `mamba_experiments/dataset_generation/generate_multivariate_sinusodial_data.py` | Overall script skeleton: CLI args, `generate_timestamps` helper, seen-split min/max normalization, HF save pattern, `norm_stats.json` format. |
| `mamba_experiments/dataset_generation/convert_tpatchgnn_data.py` | Sparse multivariate flat storage format: `target` / `timestamp` / `past_feat_dynamic_real` / `n_obs_per_var` / `history`, plus the per-variate concat pattern in `records_to_hf_dataset` (`for d in range(n_vars)`). |
| `mamba_experiments/dataset_generation/generate_sinusoidal_data.py` | Confirmed the univariate storage conventions. Not reused directly. |
| `src/ncdssm/datasets/synthetic.py` | Reviewed for historical context (regular sampling + point masking). Not reused — too simple for our goal. |

## 2. What We Kept vs Changed vs Added

### Kept (copied / adapted verbatim)
- **`generate_timestamps(n_obs, t_max, frac_regular, rng)`** — lifted byte-for-byte from `generate_multivariate_sinusodial_data.py:51-76`. Controls irregularity by mixing regular and random gaps.
- **HF `Features` schema** — same five fields as `convert_tpatchgnn_data.py:212-221`, so downstream eval code (`eval_tpatchgnn.py`, MOIRAI pipelines) can read it without changes.
- **Seen-split (train+val) min/max normalization** — same rule as `generate_multivariate_sinusodial_data.py:269-277`.
- **Scales** — `T_MAX=10.0`, `HISTORY=7.0`, `NOISE_STD=0.05`, default `N_OBS_PER_VAR=120`, `N_VARS=3`. Matches the prior synthetic benchmark.

### Changed
- **Per-variate timestamps (not shared).** In the old code `stamp = ts_list * N_VARS` repeats one timestamp vector for all variates (`generate_multivariate_sinusodial_data.py:147`). We call `generate_timestamps` independently **once per variate** inside `build_sample`, producing async irregular sampling across variates — as the plan required.
- **Signal construction.** Replaced the hard-coded "two sinusoids + convex mix of the two" with two separate regimes, chosen via `--regime`:
  - `sparse_independent` — each variate has its own independent `(freq, amp, phase, noise)`.
  - `sparse_dependent` — shared latent `A0·sin(2π f0 (t − τ_d) + φ0)` with per-variate lag `τ_d`, mixing weight `a_d`, plus a small private sinusoid.

### Added
- **Long forbidden-interval mechanism.** `draw_forbidden_intervals` samples `n_gaps` non-overlapping intervals inside `[0, t_max]` (start ∈ `GAP_START_RANGE`, length ∈ `GAP_LEN_RANGE`). `apply_forbidden_mask` drops any timestamp falling inside.
- **Per-sample variate selection.** `--gap_variates_per_sample K` picks exactly `K` variates per sample (uniform without replacement) to receive the long gap; the remaining `n_vars − K` stay dense. Default = all. Set `K=1` for the "only one variate has the gap, others are reference channels" regime.
- **Metadata tracking.** `norm_stats.json` now records regime, gap ranges, K, every seed (`signal`, `ts`, `gap`, `noise`), plus per-split obs/var distribution and gap-length statistics.

---

## 3. Dataset Layout

Generated under `ssm_dk/data/{regime}/`:

```
ssm_dk/data/sparse_independent/
├── train/        # HF dataset dir (.arrow + dataset_info.json)
├── val/
├── test/
└── norm_stats.json

ssm_dk/data/sparse_dependent/
├── train/
├── val/
├── test/
└── norm_stats.json
```

Each row has the sparse-flat schema:

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | `sin_{i:05d}` |
| `target` | float32[] | All variates' observed values concatenated, then min-max normalized. |
| `timestamp` | float32[] | Same length as `target`; per-variate timestamps concatenated. |
| `past_feat_dynamic_real` | float32[] | Per-variate Δt (first entry per variate = 0). |
| `n_obs_per_var` | int32[] | Observation count per variate — use this to unpack into per-variate arrays. |
| `history` | float32 | Scalar, shared across samples. 7.0. |

To reconstruct per-variate arrays downstream:
```python
offsets = np.concatenate([[0], np.cumsum(row["n_obs_per_var"])])
for d, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
    ts_d   = row["timestamp"][lo:hi]
    vals_d = row["target"][lo:hi]
    dt_d   = row["past_feat_dynamic_real"][lo:hi]
```

---

## 4. Current Generation Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Train / Val / Test | 1000 / 200 / 200 | Matches old synthetic convention. |
| `n_vars` | 3 | Same as old synthetic. Real IMTS benchmarks (PhysioNet=41, Activity=12, USHCN=5) are larger. |
| `n_obs_per_var` | 120 | Pre-gap; post-gap means ≈ 111 (with K=1 averaged over variates). |
| `t_max` | 10.0 | Total window. |
| `history` | 7.0 | Observed prefix; forecast horizon is [7, 10]. |
| `frac_regular` | 0.0 | All per-variate Δt random (high irregularity). |
| `gap_start_range` | [2.5, 6.0] | Where a gap can begin. |
| `gap_len_range` | [1.5, 3.0] | 15–30% of the 10-unit window. |
| `n_gaps` | 1 | One forbidden interval per sample. |
| `gap_variates_per_sample` | 1 | One variate gets the gap; other two stay dense. |
| `noise_std` | 0.05 | Per-sample Gaussian noise. |
| Seeds | signal=42, ts=101, gap=202, noise=303 | Changeable via CLI. |

Two regimes generated with identical settings except the signal family.

---

## 5. Reproducing the Build

```bash
cd /projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk

# Dependent regime
python generate_longgap_multisin.py \
    --regime sparse_dependent \
    --gap_variates_per_sample 1

# Independent regime
python generate_longgap_multisin.py \
    --regime sparse_independent \
    --gap_variates_per_sample 1
```

Smoke test / visual check:
```bash
python _smoke_plot.py   # writes _smoke_plot.png with sample 0 from both regimes
```

---

## 6. What Was Deliberately *Not* Done

- **No eval-pipeline integration yet.** Dataset generation is isolated in `ssm_dk/` per the plan's "do not overwrite old sinusoid code" rule.
- **No training.** Plan only covers data; the draft-paper Mamba model does not yet exist.
- **`gap_start_range` and `gap_len_range` are module constants**, not CLI flags. Promoting them is a one-line change when we want a gap-difficulty sweep.
- **Which variate was gapped is not stored as a dataset column.** If downstream eval needs it (e.g. to measure error on the gapped variate specifically), we'll need to add it.

---

## 7. Open Decisions

1. **Scale.** 1000 train may be low if Mamba is trained from scratch; bump to 5–10k for robustness.
2. **`n_vars`.** Stick with 3 (comparable to old synthetic) or raise to 5–10 (more cross-variate headroom, still below real IMTS).
3. **Gap marker column.** Add `gapped_variate_index: int` to the HF dataset if eval wants to score the gap-recovery error specifically.
4. **Difficulty sweep.** Generate multiple datasets varying `gap_len_range` or `K` to plot error vs difficulty.
