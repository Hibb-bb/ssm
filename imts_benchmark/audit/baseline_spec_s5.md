# S5 Baseline Spec — `s5-pytorch` Port

**Date**: 2026-04-22
**Status**: Port installed and Python-level import verified. Behavioral smoke pending.
**Verdict**: **YELLOW** — pragmatic choice of a community port over the JAX reference. Two concrete risks to track.

## Upstream summary (reference)

- **Paper**: Smith, Warrington, Linderman. "Simplified State Space Layers for Sequence Modeling". ICLR 2023. arXiv:2208.04933.
- **Reference code**: https://github.com/lindermanlab/S5 (JAX/Flax).
- **Key ideas**:
  - HiPPO-N initialization of diagonal SSM state matrices
  - Parallel associative-scan computation (vs sequential RNN)
  - Continuous-time parameterization via ZOH discretization with a per-step `Δt`

## Our source: `i404788/s5-pytorch` v0.2.1

- **GitHub**: https://github.com/i404788/s5-pytorch
- **PyPI**: `pip install s5-pytorch==0.2.1` (installed into `pythonenvs/mamba`)
- **License**: MPL-2.0 (repo) / MIT (PyPI metadata, inconsistent)
- **Community uptake**: 84 stars, low-activity maintenance
- **Key API (verified by import)**:
  ```python
  from s5 import S5
  m = S5(width=384, state_width=96, dt_min=0.001, dt_max=0.1)
  y = m(signal, step_scale=delta_t)   # signal: (B, T, width), step_scale: (B, T) or scalar
  ```
- **HiPPO-N init**: Yes — `make_DPLR_HiPPO(block_size)` called in `S5.__init__`, matches paper §3.3.
- **Parallel scan**: Yes — tree-reduction `associative_scan` in `s5/jax_compat.py` (not a sequential loop).
- **Per-step Δt**: Native. `step_scale` accepts `Tensor[B, T]`, feeds directly into ZOH discretization.
- **`S5Block`** (prebuilt GLU-FFN + LN + residual): does NOT thread `step_scale` through its `forward`. We therefore use raw `S5` and build our own block (`S5TemporalBlock` in `s5_forecaster.py`).

## Our wrapper's adaptation

- **File**: `s5_forecaster/s5_forecaster.py`
- **Architecture** (n_vars=3, d_model=384, n_layers=6, state_dim=96):
  1. Mask future values at `timestamps >= batch["history"]` (same pattern as Mamba-MV)
  2. Embed each observation: `concat(value, time_embed(t))` → project to d_model
  3. Flatten `(B, V)` into batch dim → run 6 `S5TemporalBlock`s (Pre-LN + S5 + FFN + residual)
  4. Head: linear `d_model → 1` at every position
  5. Loss: MSE on `pred_mask` positions (same as all other baselines)
- **Test metrics**: per-sample records include `ss_res`, `abs_err_sum`, `count`; `on_test_epoch_end` does global target-weighted aggregation for MSE/MAE.
- **Optimizer**: AdamW with `no_decay = {bias, norm}` split; `LambdaLR` warmup-cosine matching the shared recipe.
- **Param count**: 7,792,257 (−0.10% of 7.8M target) — verified via `audit/count_params.py`.

## Per-dimension verdicts

| Dimension | Status | Notes |
|---|---|---|
| Task adaptation (forecasting) | GREEN | Standard per-variate SSM + forecast-position-masked MSE. |
| Forward signature | GREEN | S5 API verified; shapes match. |
| Per-step Δt handling | GREEN | `step_scale=delta_t` tensor flows through; matches paper's continuous-time formulation. |
| HiPPO-N initialization | GREEN | Confirmed in port source (`make_DPLR_HiPPO`). |
| Parallel scan correctness | **YELLOW** | Port has recent commit "Fix bad state carrying in parallel formulation" (2026-04-26). v0.2.1 = post-fix, but no public LRA-reproduction evidence. Smoke test will check that training loss decreases. |
| License / paper-cite-ability | YELLOW | Port is not the reference implementation. Cite in paper as "PyTorch port by Kwaijtaal, `s5-pytorch` v0.2.1" with the proviso that LRA numbers have not been publicly reproduced. |
| Block design (custom vs `S5Block`) | GREEN | We use raw `S5` + our own Pre-LN + FFN, because `S5Block` drops `step_scale`. Clean workaround documented. |
| Param count | GREEN | 7.79M, within 0.10% of 7.8M budget. |

## Two concrete risks to track

### 1. Parallel-scan correctness (YELLOW)

The port author flagged a correctness issue on 2026-04-25 ("forward_rnn and forward are now compatible but not correct") that was fixed on 2026-04-26 ("Fix bad state carrying in parallel formulation"). We're installing v0.2.1, which is the post-fix release — so in principle we're OK. But:

- There's no public LRA-benchmark replication of this port vs. the JAX reference.
- The raw `S5` class does NOT have a `forward_rnn` method (only `S5Block` does), so we can't run the forward/forward_rnn equivalence test that the audit agent originally recommended on the raw class.

**Smoke test**: run training for 5 epochs on `multisin_med_irreg`. If train loss drops monotonically and test MSE is in the same ballpark as Mamba-MV (~0.05–0.15 at the 5-epoch mark), we have behavioral evidence that the port is working as intended. If train loss diverges or stays flat, the port has a correctness issue and we revert to building the JAX adapter from `_upstream/S5_pendulum/`.

### 2. License ambiguity (YELLOW)

MPL-2.0 on the GitHub repo, MIT declared in PyPI metadata. Since we `pip install` and do not vendor code, the distinction mainly affects whether we can paste snippets in a paper. For ICLR submission: safe to paste only the API-call lines (Fair Use / interface-only), cite the port by version, and let the full implementation live in the pip environment.

## Reference block (copy into RESULTS.md)

> "We benchmark against S5 (Smith et al., ICLR 2023) using the community PyTorch port `s5-pytorch` v0.2.1 by Kwaijtaal. Our wrapper uses the raw `S5` module (not the port's pre-built `S5Block`, which does not thread per-step Δt through its forward) and builds a Pre-LN + FFN block around it. Per-variate SSM stacks (6 layers, d_model=384, state_dim=96, 7.79M params) process each variate independently with a batched flatten over (batch × variate). Forecast positions are masked before the SSM stack (values zeroed where timestamps ≥ history) and MSE loss is computed only over those positions. Because no public reproduction of this port against the JAX reference's LRA benchmarks exists, we treat it as a qualitative S5 baseline and encourage the reader to interpret comparisons accordingly."
