# Mamba-RoPE Hybrid Experiment — 2026-04-29

A one-day exploration of a **hybrid architecture**: per-variate Mamba SSM front-end +
axial-RoPE attention back-end, motivated by a colleague's question about whether RoMAE's
RoPE attention could combine with our Mamba SSM. Trained and evaluated on USHCN and
PhysioNet, compared against Mamba-MV (3 dt_modes), RoMAE, S5, and the T-PatchGNN paper.

---

## TL;DR

| Dataset | Best baseline (5-seed) | Mamba-RoPE (5-seed) | Verdict |
|---|---|---|---|
| **USHCN** (V=5, dense, ~52 obs/var) | RoMAE 4.88 ± 0.04 | **4.95 ± 0.09** (2nd; wins MAE 3.03 vs paper 3.08) | ✅ Win |
| **PhysioNet** (V=41, sparse, ~9 obs/var) | Mamba-MV concat 4.58 ± 0.09 | **5.20 ± 0.35** (5th, ~13.5% worse, 7× larger std) | ❌ Loss |

(units: variable-averaged MSE×10⁻¹ for USHCN, MSE×10⁻³ for PhysioNet; see
[phase5_consolidated_v2.md](../../../../../output/log/imts_benchmark_v2_real/aggregate/phase5_consolidated_v2.md)
for the full leaderboard.)

**Decision: do not adopt as headline architecture.** The dataset-dependence is the
finding — flat-token attention helps on dense, low-V data; grid-based variate fusion
(Mamba-MV) handles sparse, high-V data better.

---

## 1. Motivation

A colleague suggested that RoMAE's continuous-position rotary attention (NDPRope, axial
over `(timestamp, variate_id)`) might combine well with our per-variate Mamba SSM.
Hypotheses:

1. **SSM front-end** captures per-variate temporal dynamics from irregularly-spaced
   observations (no need to align to a shared grid first).
2. **Axial-RoPE attention back-end** does cross-variate fusion with continuous-time
   awareness — better than Mamba-MV's discretised shared-grid + variate-axis attention.
3. The hybrid should be a **strict upgrade** over both pure Mamba-MV (which struggled
   on USHCN) and pure RoMAE (which lost to Mamba-MV on PhysioNet).

We tested hypothesis 3 — and falsified it.

---

## 2. Architecture

`imts_benchmark/mamba_rope/forecaster.py` (`MambaRoPEForecaster`).

**Pipeline**:

```
input: values [B,V,L], timestamps [B,V,L], deltat [B,V,L], valid_mask [B,V,L]

Stage 1: per-variate Mamba SSM
    PerVariateIrregularSSM(d_model=192, n_layer=3, dt_mode='learned')
    → h_pv [B, V, L, D]
    Reused from imts_benchmark.mamba_mv.irregular_ssm.

Stage 2: flatten to V*L flat tokens, build axial positions
    h_flat = h_pv.reshape(B, V*L, D)
    positions = stack([timestamps_flat, variate_id_flat])  # [B, 2, V*L]
    attn_mask: True for real obs, -inf for padding

Stage 3: axial-RoPE transformer encoder
    Encoder(d_model=192, nhead=6, depth=4)  +  NDPRope(head_dim=32, p=0.75, n_dims=2)
    → h_enc [B, V*L, D]
    Reused from romae library (positional_embeddings.NDPRope, model.Encoder).

Stage 4: scalar readout per token
    preds = Linear(d_model, 1).squeeze(-1) → [B, V, L]
    Loss: MSE on pred_mask positions (timestamps >= history).
```

**Param breakdown** (2.5M total — ~3× smaller than Mamba-MV (7.8M) and RoMAE (7.8M)):

| Module | Params | Share |
|---|---|---|
| `perv_ssm` (Mamba × 3 layers) | 756K | 30% |
| `encoder` (axial-RoPE × 4 layers) | 1.77M | 70% |
| `pos_emb` (NDPRope, no params) | 0 | — |
| `readout` + `norm` | <1K | <0.1% |

**Why `dt_mode='learned'`**: the RoPE attention layer already encodes timestamps
continuously, so duplicating Δt into the SSM (via `replace`/`concat`) felt redundant.
We only swept `learned` in HPO — see "What we didn't try" below.

---

## 3. Bug fix worth recording — NDPRope cache invalidation

First HPO submission (`6740053_[5]` + `6741130_[1-9]`) all failed identically after ~90s
during PyTorch Lightning's validation sanity-check. Root cause:

`NDPRope.forward()` populates `self.cache` with sin/cos of the first batch's positions
and **never invalidates it**. Batch 2 has a different `seq_len` (because `V*L` varies
per batch in dynamic padding) → cached `cos` shape (250) mismatches new `xq` shape (235)
inside `apply_ndprope`'s `first_half * cos` line.

Romae's own forecaster avoids this by calling `pos_embedding.reset_cache()` at the start
of every forward ([romae/model.py:311-314](../romae_forecaster/_romae_repo/romae/model.py)).
Fix: one line at [forecaster.py:160](../mamba_rope/forecaster.py) — `self.pos_emb.reset_cache()`
before the encoder call.

**Lesson**: when reusing third-party stateful modules, always grep their main forecaster
for `reset_*` calls and mirror them.

---

## 4. Experiments and results

All runs: `patience=10`, val/MSE early-stopping, AdamW + cosine LR with warmup, 5 seeds
for the confirm step.

### 4.1 USHCN — HPO + 5-seed confirm

- **HPO grid**: `lr ∈ {1e-4, 5e-4, 2e-3} × bs ∈ {64, 128, 256}` = 9 cells, seed=1
- **HPO winner** (val/MSE-best): `lr=5e-4, bs=64`, val/MSE = 0.6677
- **5-seed confirm** at winner:

| Metric | Mean ± std | Comparable best |
|---|---|---|
| `test_mse_tpg` | **0.4947 ± 0.0087** | RoMAE 0.4878 ± 0.0043 |
| `test_mae_tpg` | **0.3027 ± 0.0021** | RoMAE 0.3086 ± 0.0061 ← Mamba-RoPE **wins** |
| `test_r2` | 0.394 ± 0.018 | RoMAE 0.410 ± 0.010 |

Per the T-PatchGNN ×10⁻¹ scale, this reads **MSE 4.95±0.09, MAE 3.03±0.02**.

- **vs Mamba-MV** (5.11–5.18 across all 3 dt_modes): clear win, ~3-5% better.
- **vs RoMAE**: statistical tie on MSE (gap < 1σ); wins MAE outright.
- **vs T-PatchGNN paper** (5.00 / 3.08): wins both.

### 4.2 PhysioNet — HPO + 5-seed confirm

- **HPO grid**: `lr ∈ {1e-4, 5e-4, 2e-3} × eff_bs ∈ {64, 128, 256}` via `bs=8 × accum ∈ {8,16,32}`
  for memory safety on V=41 flat-token attention. 9 cells, seed=1.
- **HPO winner** (val/MSE-best): `lr=2e-3, eff_bs=64 (8×8)`, val/MSE = 0.003254
- **5-seed confirm** at winner:

| Metric | Mean ± std | Best baseline |
|---|---|---|
| `test_mse_tpg` | **0.005197 ± 0.000346** | Mamba-MV concat 0.004577 ± 0.000091 |
| `test_mae_tpg` | **0.038912 ± 0.001609** | Mamba-MV learned 0.0360 ± 0.00018 |
| `test_r2` | -1753 ± 450 | Mamba-MV concat -103 ± 25 |

Per ×10⁻³ scale, this reads **MSE 5.20±0.35, MAE 3.89±0.16**.

- **vs Mamba-MV (concat)**: ~13.5% worse, gap = 3.7σ — statistically significant loss.
- **vs T-PatchGNN paper** (4.98): worse by ~4%.
- **vs RoMAE** (6.88): better by ~25%, but RoMAE is the worst attention baseline on
  PhysioNet anyway.
- **Std is 7× larger** than Mamba-MV's — architecture is also more seed-sensitive
  on this dataset.

---

## 5. Why the dataset-dependence — proposed mechanism

| Factor | USHCN | PhysioNet |
|---|---|---|
| `V` (variates) | 5 | 41 |
| Avg observations per variate per sample | ~52 | ~9 |
| Padding ratio in `V × L_max` flat tokens | low | very high |
| Total flat tokens / sample | ~250-700 | ~3000-5000+ |

In Mamba-RoPE, the **encoder operates on `V × L_max` flat tokens with most positions
being padding** when the dataset is sparse-high-V. Even with the additive attention
mask, padding tokens degrade the signal-to-noise ratio at the attention layer because:

1. Real observations from variate `d` need to find their semantic neighbours among
   thousands of padding-shadow tokens.
2. The continuous `(t, d)` axial RoPE has many spurious neighbours within each rotation
   bucket on padding side — RoPE is built on the assumption that nearby positions are
   semantically related, which doesn't hold when half your "positions" are padding.

Mamba-MV instead uses a **fixed-K shared grid** (K=128 or 256) that explicitly aligns
all variates at the same time points. The variate-axis attention then operates on
`V × K` tokens that are all real — no padding shadow — so cross-variate fusion is
well-conditioned regardless of how sparse the original observations were.

So the architectural lesson: **flat-token attention without a learned grid alignment
breaks down when the padding ratio gets high**. The hybrid only helps when the data
is dense enough that flat-token attention has a high real:padding ratio.

---

## 6. What we didn't try (and why we stopped)

- **Other dt_modes on PhysioNet** (`replace`, `concat`): Mamba-MV's dt_mode spread on
  PhysioNet was only ~2% (concat 4.58 vs replace 4.68). Even at 4× larger sensitivity,
  the gap to close (13%) stays bigger than dt_mode could plausibly bridge.
- **Activity dataset** (V=12, ~37 obs/var, intermediate density): would have rounded
  out the dataset-dependence story but doesn't change the core conclusion.
- **Axial RoPE *inside* the SSM block** (colleague's alternative suggestion): would
  require non-trivial Mamba-block surgery; not justified given the negative PhysioNet
  result here.

---

## 7. Code locations

- Forecaster: [`imts_benchmark/mamba_rope/forecaster.py`](../mamba_rope/forecaster.py)
- Trainer: [`imts_benchmark/mamba_rope/train_mamba_rope.py`](../mamba_rope/train_mamba_rope.py)
- HPO scripts: [`scripts/run_mamba_rope_hpo_ushcn.sbatch`](../scripts/run_mamba_rope_hpo_ushcn.sbatch),
  [`scripts/run_mamba_rope_hpo_physionet.sbatch`](../scripts/run_mamba_rope_hpo_physionet.sbatch)
- 5-seed confirm scripts: [`scripts/run_mamba_rope_p10_ushcn.sbatch`](../scripts/run_mamba_rope_p10_ushcn.sbatch),
  [`scripts/run_mamba_rope_p10_physionet.sbatch`](../scripts/run_mamba_rope_p10_physionet.sbatch)
- Aggregator entry (locks the HPO winners as the canonical configs):
  [`eval/aggregate_phase5_v2.py:57-58`](../eval/aggregate_phase5_v2.py)

W&B project `magicslabnorthwestern/TSKing`, runs prefixed `hpo_mrope_*` and `p10_mrope_*`.

---

## 8. Updated table

The Mamba-RoPE row was added to [`results_table_phase5.tex`](results_table_phase5.tex):

```
Mamba-RoPE (learned) & 5.20$_{\pm 0.35}$ & 3.89$_{\pm 0.16}$ & --- & --- & --- & ---
                     & \underline{4.95$_{\pm 0.09}$} & \textbf{3.03$_{\pm 0.02}$} \\
```

PhysioNet: no bold/underline (rank 5). USHCN: underline on MSE (rank 2), **bold on MAE
(rank 1, beats T-PatchGNN paper)**.

---

## 9. Decision

- **Do not adopt** Mamba-RoPE as the headline architecture.
- **Mamba-MV remains the primary model** for the paper — competitive on all 3 datasets,
  best on PhysioNet, no architectural pathology.
- **Mamba-RoPE is worth a paragraph in the paper** as a "what didn't work" or "ablation"
  data point, since the dataset-dependence is itself a clean architectural finding.
- USHCN MAE = 3.03 ± 0.02 is now the best-in-paper number we've produced for that
  column (beats T-PatchGNN paper 3.08), even though the model isn't a strict upgrade.
