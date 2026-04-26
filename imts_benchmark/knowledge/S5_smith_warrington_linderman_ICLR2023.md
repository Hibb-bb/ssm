# S5 — Simplified State Space Layers for Sequence Modeling
Smith, Warrington, Linderman. ICLR 2023. arXiv:2208.04933.

## Mechanism summary
S5 replaces S4's bank of H independent SISO SSMs with a single MIMO SSM of state size P. The MIMO SSM is diagonalized (complex eigenvalues Λ ∈ ℂᴾ) and applied via parallel scan rather than FFT convolution. This unlocks time-varying SSMs — each step can use its own Δ_k — efficiently.

## Sections we cite in this project

### §3.2 — Diagonalized dynamics, initialization
- State matrix is diagonal complex Λ, parameterized as `Ā = exp(Λ·Δ)`, `B̄ = Λ⁻¹(exp(Λ·Δ) − I)·B̃`.
- **HiPPO-N initialization**: diagonalization of the normal component of HiPPO-LegS. This is the long-range memory prior that the wrapper inherits and that Mamba-MV's S6 selective-init does NOT have.
- Conjugate symmetry reduces state/runtime by 2×.

### §3.3 — Time-varying SSMs for irregular sampling
> "Parallel scans and the continuous-time parameterization also allow for efficient handling of irregularly sampled time series and other time-varying SSMs, by simply supplying a different Ā_k matrix at each step."

This is the mechanism we use in [s5_forecaster.py:150](../../s5_forecaster/s5_forecaster.py#L150) via `step_scale=step`. The wrapper's per-step Δt correctly feeds into Λ_bar and B_bar per s5_model.py line 258 (see [Audit A](../audit/) report).

### §4.2 + Corollary 1 — Relation to S4 via multi-input HiPPO
HiPPO-N is an infinite-state-dim approximation of HiPPO-LegS for multi-input systems. Justifies initializing S5 with HiPPO-N even in MIMO setting.

### §6.3 — Pendulum regression (closest paper analogue to our task)
- Input: 50 noisy 24×24 images irregularly sampled from a 100-step pendulum simulation.
- Target: sin(θ), cos(θ) at observed timestamps. Regression, not forecasting, but with irregular timestamps and irregular temporal spacing.
- S5 gives 86× speedup vs CRU and lower MSE (0.00341 vs 0.00463).
- **Why this matters for us**: this is the only irregular-TS task in the S5 paper. Our wrapper ports this exact mechanism to async multivariate forecasting, which the paper does not test.

### §B.1.3 — Timescale initialization
Elements of `log Δ ∈ ℝᴾ` sampled uniformly from `[log 0.001, log 0.1)`. Wrapper inherits this via the `s5-pytorch` package defaults.

### Appendix A Listing 1 — JAX reference implementation
Canonical discretization:
```
Lambda_bar = exp(Lambda * Delta)
B_bar = (1/Lambda * (Lambda_bar - I))[..., None] * B_tilde
```
Translated to PyTorch in `/projects/b1094/StarEmbed/pythonenvs/mamba/lib/python3.12/site-packages/s5/s5_model.py:111-113`. Verified correct in Audit A.

## What the S5 paper does NOT benchmark
- **Forecasting**: all tasks are classification or regression at observed timestamps. There is no future-horizon forecast benchmark on irregular time series.
- **Async multivariate**: the pendulum task is univariate (or 24×24-image-valued per step, but shared timestamps). No async-variate evaluation.

This is the gap our benchmark fills, and also why wrapper-correctness audits were necessary.
