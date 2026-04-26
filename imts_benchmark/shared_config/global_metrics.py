"""Shared per-variable global R² / Pearson aggregator for forecaster test_step.

Why this exists
---------------
Earlier metric code computed R² and Pearson **per multivariate test sample**
and then averaged across samples.  For datasets with very sparse per-sample
target sets (e.g. USHCN, where one item often has only 1–5 observed targets
across 5 variables), this fails:

    ss_tot_sample = Σ (y - mean(y))²

is computed inside that single sample.  When the sample has only one observed
target, ``mean(y) == y`` so ``ss_tot_sample == 0``, the ``+ 1e-10`` denominator
guard makes ``R² → -1e10``, and a few such samples dominate the across-sample
mean.  Pearson likewise becomes NaN whenever a single sample has constant ``y``
or constant ``ŷ``.

This module instead computes **one R² per variable, pooled across the entire
test set, then averages over variables** – the standard regression-style R²
that GRU-ODE-Bayes / Latent-ODE / sklearn use.  It only needs cheap streaming
sums per (sample, variable):

    n_v        = Σ count of observed (sample, variable) targets
    sum_y_v    = Σ y           over those targets
    sum_y2_v   = Σ y²          over those targets
    sum_yh_v   = Σ ŷ           over those targets
    sum_yh2_v  = Σ ŷ²          over those targets
    sum_yyh_v  = Σ y · ŷ       over those targets
    ss_res_v   = Σ (y - ŷ)²    over those targets

From these:

    mean_y_v   = sum_y_v / n_v
    ss_tot_v   = sum_y2_v - n_v * mean_y_v²
    R²_v       = 1 - ss_res_v / ss_tot_v
    pearson_v  = (sum_yyh_v - n_v * mean_y_v * mean_yh_v) /
                  sqrt((sum_y2_v  - n_v * mean_y_v²)  *
                       (sum_yh2_v - n_v * mean_yh_v²))

Final scalar = mean over variables that have ``n_v >= 2`` and ``ss_tot_v > eps``.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def per_sample_variable_sums(y_v, yhat_v) -> dict:
    """Return the 6 streaming sums needed per (sample, variable).

    ``y_v`` and ``yhat_v`` are 1-D numpy arrays of equal length holding the
    observed targets and predictions for one variable in one sample (the
    variable's masked positions). All sums are returned as plain ``float`` so
    they JSON-serialise cleanly into ``per_sample.jsonl``.
    """
    import numpy as np
    n = int(y_v.size)
    return {
        "count": n,
        "sum_y": float(y_v.sum()),
        "sum_y2": float((y_v * y_v).sum()),
        "sum_yh": float(yhat_v.sum()),
        "sum_yh2": float((yhat_v * yhat_v).sum()),
        "sum_yyh": float((y_v * yhat_v).sum()),
        "ss_res": float(((y_v - yhat_v) ** 2).sum()),
        "abs_err_sum": float((y_v - yhat_v).__abs__().sum()),
    }


def aggregate_global_metrics(
    per_sample_outputs: Iterable[Mapping],
    n_vars: int,
    eps: float = 1e-12,
) -> dict:
    """Aggregate per-sample, per-variable streaming sums into global metrics.

    ``per_sample_outputs`` is the list each forecaster builds in test_step.
    For every variable ``d`` it must contain the keys produced by
    :func:`per_sample_variable_sums` prefixed with ``"_v{d}"`` – e.g.
    ``count_v0``, ``sum_y_v0``, ``ss_res_v0`` etc.

    Returns a dict with:

    * ``mse``            – global target-weighted MSE (same as before)
    * ``mae``            – global target-weighted MAE (same as before)
    * ``r2``             – mean over variables of per-variable global R²
    * ``pearson``        – mean over variables of per-variable global Pearson
    * ``r2_per_var``     – list[float] (length ``n_vars``); ``nan`` if dropped
    * ``pearson_per_var``– list[float] (length ``n_vars``); ``nan`` if dropped
    * ``n_per_var``      – list[int]; pooled obs count for each variable
    """
    out_n   = [0]   * n_vars
    out_sy  = [0.0] * n_vars
    out_sy2 = [0.0] * n_vars
    out_syh = [0.0] * n_vars
    out_syh2= [0.0] * n_vars
    out_syyh= [0.0] * n_vars
    out_ssr = [0.0] * n_vars
    out_aes = [0.0] * n_vars

    for o in per_sample_outputs:
        for d in range(n_vars):
            n_d = int(o.get(f"count_v{d}", 0))
            if n_d == 0:
                continue
            out_n[d]   += n_d
            out_sy[d]  += o[f"sum_y_v{d}"]
            out_sy2[d] += o[f"sum_y2_v{d}"]
            out_syh[d] += o[f"sum_yh_v{d}"]
            out_syh2[d]+= o[f"sum_yh2_v{d}"]
            out_syyh[d]+= o[f"sum_yyh_v{d}"]
            out_ssr[d] += o[f"ss_res_v{d}"]
            out_aes[d] += o.get(f"abs_err_sum_v{d}", 0.0)

    r2_per_var: list[float] = []
    pe_per_var: list[float] = []
    for d in range(n_vars):
        n_d = out_n[d]
        if n_d < 2:
            r2_per_var.append(float("nan"))
            pe_per_var.append(float("nan"))
            continue
        mean_y  = out_sy[d]  / n_d
        mean_yh = out_syh[d] / n_d
        ss_tot_y  = out_sy2[d]  - n_d * mean_y  * mean_y
        ss_tot_yh = out_syh2[d] - n_d * mean_yh * mean_yh
        if ss_tot_y < eps:
            r2_per_var.append(float("nan"))
            pe_per_var.append(float("nan"))
            continue
        r2_per_var.append(1.0 - out_ssr[d] / ss_tot_y)
        denom = ss_tot_y * max(ss_tot_yh, 0.0)
        if denom <= eps:
            pe_per_var.append(float("nan"))
        else:
            cov = out_syyh[d] - n_d * mean_y * mean_yh
            pe_per_var.append(cov / math.sqrt(denom))

    def _mean_skip_nan(xs):
        vals = [v for v in xs if not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    total_count = max(sum(out_n), 1)

    # Variable-averaged MSE/MAE — T-PatchGNN's compute_error(reduce='mean')
    # convention. Computes per-variable mean SE/AE pooled across the entire
    # test set, then averages over variables that have at least one obs.
    # This is what their published Table 1 numbers use; we report it
    # alongside our target-weighted aggregation for direct comparability.
    n_avai_var = sum(1 for n_d in out_n if n_d > 0)
    if n_avai_var > 0:
        mse_per_var = [out_ssr[d] / out_n[d] if out_n[d] > 0 else 0.0
                       for d in range(n_vars)]
        mae_per_var = [out_aes[d] / out_n[d] if out_n[d] > 0 else 0.0
                       for d in range(n_vars)]
        mse_tpg = sum(mse_per_var) / n_avai_var
        mae_tpg = sum(mae_per_var) / n_avai_var
    else:
        mse_per_var = [float("nan")] * n_vars
        mae_per_var = [float("nan")] * n_vars
        mse_tpg = float("nan")
        mae_tpg = float("nan")

    return {
        "mse":     sum(out_ssr) / total_count,        # target-weighted (ours)
        "mae":     sum(out_aes) / total_count,        # target-weighted (ours)
        "mse_tpg": mse_tpg,                           # variable-averaged (T-PatchGNN)
        "mae_tpg": mae_tpg,                           # variable-averaged (T-PatchGNN)
        "r2":      _mean_skip_nan(r2_per_var),
        "pearson": _mean_skip_nan(pe_per_var),
        "r2_per_var":      r2_per_var,
        "pearson_per_var": pe_per_var,
        "mse_per_var":     mse_per_var,
        "mae_per_var":     mae_per_var,
        "n_per_var":       out_n,
    }
