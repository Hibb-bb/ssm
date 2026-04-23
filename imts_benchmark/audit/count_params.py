"""Param-count audit for the Phase-2 IMTS benchmark.

Instantiates every model with the same defaults its own train_*.py uses,
prints total trainable parameters, and checks the fair-comparison budget
(7.8M target, ±10% tolerance).

Run:
    python -m imts_benchmark.audit.count_params
or
    /projects/b1094/StarEmbed/pythonenvs/mamba/bin/python \
        imts_benchmark/audit/count_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SSM_DK = _THIS.parents[2]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_config import fair_defaults as fd


TARGET_PARAMS = 7_800_000
TOLERANCE = 0.10


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mamba_mv():
    from imts_benchmark.mamba_mv.multivariate_forecaster import (
        MultivariateMambaForecaster,
    )
    return MultivariateMambaForecaster(
        d_model=384, d_hidden=384, n_vars=fd.N_VARS,
        n_perv_layer=3, n_fusion_blocks=3, n_heads_varattn=4,
        d_state=16, d_conv=4, expand=2, dt_mode="replace",
        grid_K=128, t_max=fd.T_MAX, n_freq=8,
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
        history=fd.HISTORY,
    )


def build_mtan():
    from imts_benchmark.mtan_forecaster.mtan_forecaster import MTANForecaster
    return MTANForecaster(
        n_vars=fd.N_VARS,
        rec_hidden=576, gen_hidden=576, latent_dim=64,
        embed_time=128, num_ref_points=64, num_heads=1, learn_emb=True,
        t_max=fd.T_MAX,
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
        history=fd.HISTORY,
    )


def build_contiformer():
    from imts_benchmark.contiformer_forecaster.contiformer_forecaster import (
        ContiFormerForecaster,
    )
    return ContiFormerForecaster(
        n_vars=fd.N_VARS,
        d_model=320, d_inner=1024, n_layers=6, n_head=4, d_k=80,
        dropout=0.1, atol_ode=1e-1, rtol_ode=1e-1,
        method_ode="rk4", actfn_ode="softplus",
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
        history=fd.HISTORY,
    )


def build_contiformer_reduced():
    """Reduced ContiFormer config to fit 80GB H100 memory. Target ~2M params.

    The 7.8M default doesn't fit because per-layer ODE-attention allocates
    `[B, L, L, hidden, n_head, d_k]` ≈ 5 GB at our L=360, n_head=4, d_k=80.
    Dropping d_model 320->160 (and d_k 80->40) brings that tensor to ~2.6 GB
    per layer, which should fit. Breaks param-budget fairness (2M vs 7.8M)
    but matches ContiFormer paper's own tested scale (hidden_dim 32-64).
    """
    from imts_benchmark.contiformer_forecaster.contiformer_forecaster import (
        ContiFormerForecaster,
    )
    return ContiFormerForecaster(
        n_vars=fd.N_VARS,
        d_model=160, d_inner=640, n_layers=4, n_head=4, d_k=40,
        dropout=0.1, atol_ode=0.5, rtol_ode=0.5,
        method_ode="rk4", actfn_ode="softplus",
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
        history=fd.HISTORY,
    )


def build_s5():
    from imts_benchmark.s5_forecaster.s5_forecaster import S5Forecaster
    return S5Forecaster(
        n_vars=fd.N_VARS,
        d_model=384, state_dim=96, n_layers=6, ff_mult=4.0,
        time_emb_dim=64, dropout=0.0,
        t_max=fd.T_MAX, history=fd.HISTORY,
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
    )


def build_romae():
    from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster
    return RoMAEForecaster(
        enc_d_model=288, enc_nhead=6, enc_depth=7,
        dec_d_model=180, dec_nhead=3, dec_depth=2,
        max_len=1500, p_rope_val=0.75, n_vars=fd.N_VARS,
        lr=fd.LR, weight_decay=fd.WEIGHT_DECAY,
        num_warmup_steps=fd.NUM_WARMUP_STEPS,
        num_training_steps=fd.NUM_TRAINING_STEPS,
        history=fd.HISTORY,
    )


BUILDERS = {
    "mamba_mv": build_mamba_mv,
    "mtan": build_mtan,
    "contiformer": build_contiformer,
    "contiformer_reduced": build_contiformer_reduced,
    "romae": build_romae,
    "s5": build_s5,
}

# Models whose param count is INTENTIONALLY outside the shared budget
# (printed for information only, doesn't count against pass/fail).
INFORMATIONAL = {"contiformer_reduced"}


def main():
    lo = int(TARGET_PARAMS * (1 - TOLERANCE))
    hi = int(TARGET_PARAMS * (1 + TOLERANCE))
    print(f"Fair-comparison target: {TARGET_PARAMS:,} params (±{int(TOLERANCE*100)}%)")
    print(f"Acceptable range: [{lo:,}, {hi:,}]")
    print()
    print(f"{'Model':<14} {'Params':>14}   {'Δ vs target':>12}   Status")
    print("-" * 60)

    any_fail = False
    for name, build in BUILDERS.items():
        try:
            model = build()
        except Exception as e:
            print(f"{name:<20} {'ERROR':>14}   {'-':>12}   FAIL ({type(e).__name__}: {e})")
            if name not in INFORMATIONAL:
                any_fail = True
            continue
        n = count_trainable(model)
        delta_pct = (n - TARGET_PARAMS) / TARGET_PARAMS * 100
        ok = lo <= n <= hi
        if name in INFORMATIONAL:
            status = "INFO"
        else:
            status = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True
        print(f"{name:<20} {n:>14,}   {delta_pct:>+11.2f}%   {status}")

    print()
    if any_fail:
        print("At least one model is outside the fair-comparison param budget.")
        sys.exit(1)
    print("All models within budget.")


if __name__ == "__main__":
    main()
