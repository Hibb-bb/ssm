# Phase 5 Results — Real IMTS Datasets (Activity, USHCN)

**Status**: Patience=10 reruns submitted as jobs 6384307 (Mamba-MV), 6384308 (S5), 6384309 (RoMAE). Earlier patience=50 default-config runs aggregated in `output/log/imts_benchmark_v2_real/aggregate/summary_real.md`.

**Datasets**: T-PatchGNN preprocessed splits at `tpatchgnn_data/` (60/20/20, sparse-flat schema, identical to Phases 2-4). PhysioNet skipped for now per user decision (re-add when storyline is settled).

| Dataset | n_vars | history | t_max | Preprocessing |
|---|---|---|---|---|
| activity | 12 | 3000 ms | 4000 ms | normalized to [0,1] (T-PatchGNN convention) |
| ushcn | 5 | 24 mo | 48 mo | **raw** (T-PatchGNN convention; their Table 1 reports MSE×10⁻¹) |

---

## 1. Models and configs

All three models read `--auto_meta` and pull `n_vars / history / time_max` from `tpatchgnn_data/{dataset}/norm_stats.json` ([fair_defaults.py](../shared_config/fair_defaults.py) — `apply_auto_meta`).

| Model | Config (effective at sbatch run time) | Params (Activity) | Params (USHCN) |
|---|---|---|---|
| **S5** | paper-native: d_model=128, state=256, 6 layers, lr=1e-3, wd=0.05, bs=32+accum=4 | 1,395,969 (1.40M) | 1,395,969 (1.40M) |
| **RoMAE** | paper-native: enc_d=288×7, dec_d=180×2, lr=5e-4, default bs=128 | 7,803,073 (7.80M) | 7,803,073 (7.80M) |
| **Mamba-MV** | **CLI defaults** d_model=384, d_hidden=384, n_perv=3, n_fusion=3, d_state=16, expand=2, n_heads_varattn=4, dt_mode=replace, grid_K=128 + per-dataset HPO winner (lr, bs) | 7,814,604 (7.81M) | 7,794,997 (7.79M) |

**Note on Mamba-MV defaults pitfall**: the `MultivariateMambaForecaster.__init__` class default is `d_model=256`, but the CLI default in [`train_mv.py:63`](../mamba_mv/train_mv.py) is `d_model=384`. Sbatch scripts do not pass `--d_model`, so **the runtime config is d_model=384**. W&B configs confirm this for all colleague HPO and confirm runs. (See `BUILD_NOTES.md` for the corresponding code-hygiene fix.)

### Per-dataset Mamba-MV HPO winners (val/mse-best)

Verified 2026-04-25 by re-querying W&B `magicslabnorthwestern/TSKing` and ranking the 9 HPO sweep cells per dataset (lr ∈ {1e-4, 5e-4, 2e-3} × bs ∈ {64, 128, 256}, dt_mode=replace, seed=1, patience=10):

| Dataset | Winner (lr, bs) | val/mse | Note |
|---|---|---|---|
| Activity | **(2e-3, 128)** | 0.002832 | Colleague's confirm runs also use this. ✅ matches |
| USHCN | **(5e-4, 64)** | 0.682333 | Colleague's confirm runs use (1e-4, 256), val/mse=0.696294 — **2nd-best, NOT the val/mse winner.** Our p10 reruns use the principled val/mse winner. |

`grid_K = 128` for both datasets per `K = next_pow2(2 × p95(max_obs/var))` rule (Activity p95=37 → 128; USHCN p95=52 → 128).

---

## 2. T-PatchGNN paper Table 1 — comparison context

Headline numbers we are comparing against (from T-PatchGNN ICML 2024, Table 1, page 7). T-PatchGNN itself is bold (best); Warpformer is the underlined 2nd-best baseline on Activity and USHCN MSE.

| Model | Activity MSE×10⁻³ | Activity MAE×10⁻² | USHCN MSE×10⁻¹ | USHCN MAE×10⁻¹ |
|---|---|---|---|---|
| **T-PatchGNN (target)** | **2.66 ± 0.03** | **3.15 ± 0.02** | **5.00 ± 0.04** | **3.08 ± 0.04** |
| Warpformer (best baseline) | 2.79 ± 0.04 | 3.39 ± 0.03 | 5.25 ± 0.05 | 3.23 ± 0.05 |
| Graph WaveNet | 2.89 ± 0.03 | 3.40 ± 0.05 | 5.29 ± 0.04 | 3.16 ± 0.09 |
| GRU-D | 2.94 ± 0.05 | 3.53 ± 0.06 | 5.54 ± 0.38 | 3.40 ± 0.28 |
| mTAND | 3.22 ± 0.07 | 3.81 ± 0.07 | 5.33 ± 0.05 | 3.26 ± 0.10 |

### Parameter ratio (critical for storyline)

T-PatchGNN paper Section 5.1.2 standardizes **all 17 baselines + T-PatchGNN at hidden_dim=32** for Activity/USHCN (hidden_dim=64 for PhysioNet/MIMIC). Direct count from their repo at the paper's exact config (`history=3000, patch_size=300, hid_dim=32, te_dim=10, node_dim=10, nhead=1, tf_layer=1, nlayer=1`):

| Model | Activity params | USHCN params |
|---|---|---|
| T-PatchGNN (paper) | 165.6K | 167.5K |
| Our S5 | 1.40M (~8×) | 1.40M (~8×) |
| Our RoMAE | 7.80M (~47×) | 7.80M (~47×) |
| **Our Mamba-MV** | **7.81M (~47×)** | **7.79M (~47×)** |

**Implication**: We carry ~47× more parameters than T-PatchGNN on Activity/USHCN. This is comparable to TSFM-paper standards (Moirai-small ~14M, Chronos-T5-small ~46M), but reviewers will scrutinize the disparity. Storyline will need to either:
1. **Concede the size, justify by class** ("Mamba-MV is in the TSFM-scale model class; specialized IMTS baselines like T-PatchGNN run at ~150K"), OR
2. **Match the size** — re-run a shrunk Mamba-MV (`d_model=128, d_hidden=128, n_perv=2, n_fusion=2`, ~500K-1M params) and show the small version still beats T-PatchGNN. Decision pending.

---

## 3. Pre-p10 results (patience=50, mixed sources)

Snapshot from `output/log/imts_benchmark_v2_real/aggregate/summary_real.md` (S5/RoMAE = our 5-seed runs; Mamba-MV = colleague's W&B confirm runs):

| Model | Activity MSE×10⁻³ | Activity MAE×10⁻² | USHCN MSE×10⁻¹ | USHCN MAE×10⁻¹ |
|---|---|---|---|---|
| S5 | 16.27 ± 0.01 | 10.21 ± 0.00 | 5.35 ± 0.06 | 3.17 ± 0.12 |
| RoMAE | 2.62 ± 0.02 | 3.18 ± 0.02 | 5.03 ± 0.03 | 3.08 ± 0.14 |
| Mamba-MV (colleague's HPO at lr=1e-4, bs=256 on USHCN) | 2.86 ± 0.12 | 3.42 ± 0.16 | 5.29 ± 0.09 | 3.48 ± 0.17 |

Observations:
- **S5 collapses on Activity** (16.27 vs RoMAE 2.62) — exactly the channel-independence cost predicted in `audit/baseline_spec_s5.md`. Activity has 12 strongly-correlated accelerometer channels; S5 cannot fuse across them.
- **RoMAE essentially matches T-PatchGNN** on USHCN (5.03 vs 5.00) and is near it on Activity (2.62 vs 2.66). Encouraging.
- **Mamba-MV at the colleague's USHCN HPO pick (lr=1e-4, bs=256)** loses to RoMAE on USHCN (5.29 vs 5.03). This is likely because the colleague picked the 2nd-best HPO cell instead of the val/mse winner; the p10 reruns at the correct val/mse winner (lr=5e-4, bs=64) will tell us whether the principled choice changes the ordering.

---

## 4. p10 reruns — what's running now

All three jobs are 10-cell arrays (2 datasets × 5 seeds), patience=10 to match T-PatchGNN protocol exactly:

| Job | Script | Output dir |
|---|---|---|
| 6384307 | `scripts/run_mamba_mv_p10_real_small.sbatch` | `mamba_mv_p10/{ds}/replace_lr-{LR}_bs-{BS}/seed{N}/` |
| 6384308 | `scripts/run_s5_p10_real_small.sbatch` | `s5_p10/{ds}/default/seed{N}/` |
| 6384309 | `scripts/run_romae_p10_real_small.sbatch` | `romae_p10/{ds}/default/seed{N}/` |

Aggregator (post-completion): `eval/merge_real_results.py` reads per-seed `test_metrics.csv` from these dirs + W&B colleague runs, writes `aggregate/summary_real.csv` and `summary_real.md`.

Expected wall-clock: ~30-90 min per cell on H100 (Activity is largest per-sample due to L_max ≈ 130).

---

## 5. Decision points (post-completion)

After p10 results land, three branches to choose from (user decision; not pre-committed):

### Branch A — accept current results, write paper

If Mamba-MV at val/mse-winner HPO + p10 beats T-PatchGNN on at least one of (Activity, USHCN), the storyline is publishable:
- Concede the parameter disparity in Section 4.X ("Implementation details").
- Position Mamba-MV as a TSFM-class model competing with specialized IMTS baselines.
- Cite RoMAE and S5 as same-class baselines (~7.8M and ~1.4M).

### Branch B — shrink to match T-PatchGNN size

Add a 4th sbatch script `run_mamba_mv_p10_real_small_500K.sbatch` with `--d_model 128 --d_hidden 128 --n_perv_layer 2 --n_fusion_blocks 2 --n_heads_varattn 2` (estimated 500K-1M params). Run 5 seeds at the same HPO winners. **Strongest possible storyline** if it still beats T-PatchGNN.
- Cost: 10 more SLURM jobs, ~6 hours wall-clock.

### Branch C — re-explore dt_mode (learned, concat)

The colleague's HPO sweep was dt_mode=replace only. If `replace` underperforms after p10, run a small dt_mode ablation at the val/mse winner (lr, bs) for `learned` and `concat`. 10 jobs (2 ds × 5 seeds) per dt_mode = 20 jobs total.
- Justification from Phase 4-2: `learned` was the dt_mode winner on synthetic gap-on-random-variate data. May matter on USHCN (climate, slow dynamics).

---

## 6. Critical open questions for paper

1. **PhysioNet**: not run yet. If the paper claims "real IMTS benchmark," PhysioNet is expected. Decision: re-include after the Activity/USHCN storyline is settled, with `bs=32 + accum=4` for RoMAE OOM mitigation (per `run_romae_real.sbatch:72`).
2. **MIMIC**: not in our preprocessing pipeline; T-PatchGNN paper has it as the 4th dataset. Footnote in our paper as "MIMIC omitted; data not yet prepared."
3. **Fair-comparison reviewer ask**: very likely. Branch B (shrink Mamba-MV) is the cleanest preemption.
4. **APN paper (related work)**: APN uses different preprocessing (1,359 samples, 80/10/10, 300ms forecast); their numbers are NOT comparable to T-PatchGNN's Table 1 or to ours. Cite as related work, but anchor the comparison on T-PatchGNN.

---

## 7. Code paths (for re-runs)

| Purpose | File |
|---|---|
| Real-data trainers (auto_meta) | `mamba_mv/train_mv.py`, `s5_forecaster/train_s5.py`, `romae_forecaster/train_romae.py` |
| Phase-5 sbatch scripts (default p50) | `scripts/run_{mamba_mv,s5,romae}_real.sbatch` |
| Phase-5 sbatch scripts (T-PatchGNN-protocol p10) | `scripts/run_{mamba_mv,s5,romae}_p10_real_small.sbatch` |
| HPO sweep script | `scripts/run_mamba_mv_hpo_replace_real.sbatch` (run by colleague; not re-run here) |
| HPO winner picker | `eval/aggregate_hpo_real.py` |
| Aggregator (CSV + W&B + Markdown table) | `eval/merge_real_results.py` |
| Auto-meta helper | `shared_config/fair_defaults.py:apply_auto_meta` |
