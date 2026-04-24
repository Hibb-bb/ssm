"""Render ONE sample's predictions for a slide deck.

Takes the 100th-percentile (highest dominant frequency) sample from a given
regime and plots 3 variates × 3 models as a single compact figure. Intended
for 1-slide-per-phase use.

Default target: high_irreg regime, 100th-percentile-freq sample.

Usage:
  python plot_slide_single_sample.py \
      --pred_root <path>/<phase>_results/predictions \
      --out_file <path>/<phase>_results/<phase>_slide_single.png \
      --phase_label "Phase 3 — async dense" \
      [--regime multisin_high_irreg] [--freq_percentile 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lombscargle

MODELS = ("mamba_mv", "s5", "romae")
MODEL_LABELS = {"mamba_mv": "Mamba-MV", "s5": "S5", "romae": "RoMAE"}
MODEL_COLORS = {"mamba_mv": "#d62728", "s5": "#ff7f0e", "romae": "#1f77b4"}


def load_predictions(pred_file: Path):
    with open(pred_file) as f:
        return [json.loads(l) for l in f]


def dominant_freq(entry, freq_grid):
    v0 = entry["variates"][0]
    ts = np.asarray(v0["ts_ctx"], dtype=float)
    vals = np.asarray(v0["val_ctx"], dtype=float)
    if len(ts) < 4:
        return 0.0
    vals = vals - np.mean(vals)
    angf = 2 * np.pi * freq_grid
    try:
        pgram = lombscargle(ts, vals, angf, normalize=True)
    except Exception:
        return 0.0
    return float(freq_grid[np.argmax(pgram)])


def sample_metrics(entry):
    y_t, y_p = [], []
    for v in entry["variates"]:
        y_t.extend(v["val_true_pred"]); y_p.extend(v["val_pred"])
    y_t = np.asarray(y_t); y_p = np.asarray(y_p)
    if len(y_t) == 0:
        return dict(mse=np.nan, r2=np.nan)
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    return dict(mse=float(np.mean((y_t - y_p) ** 2)),
                r2=float(1 - ss_res / (ss_tot + 1e-12)))


def _pick_representative(preds_by_model, freq_grid, constraint_winner: str | None = None):
    """Pick a sample whose Mamba MSE is closest to the Mamba test-set median,
    optionally constrained to samples where `constraint_winner` has the lowest
    MSE among all models (so the slide visually reinforces the aggregate
    headline). If no sample satisfies the constraint, falls back to the
    distance-to-median sample without constraint.
    """
    mamba_entries = preds_by_model["mamba_mv"]
    other_idx = {m: {e["item_id"]: e for e in es}
                 for m, es in preds_by_model.items() if m != "mamba_mv"}
    rows = []
    for i, e in enumerate(mamba_entries):
        iid = e["item_id"]
        mse_by_model = {"mamba_mv": sample_metrics(e)["mse"]}
        ok = True
        for m, idx in other_idx.items():
            other = idx.get(iid)
            if other is None:
                ok = False; break
            mse_by_model[m] = sample_metrics(other)["mse"]
        if not ok:
            continue
        rows.append((i, mse_by_model))
    if not rows:
        return None, None

    target_mamba = float(np.median([r[1]["mamba_mv"] for r in rows]))

    # Primary pass: apply winner constraint if given.
    if constraint_winner is not None:
        constrained = [
            (abs(r[1]["mamba_mv"] - target_mamba), r[0], r[1])
            for r in rows
            if min(r[1], key=lambda m: r[1][m]) == constraint_winner
        ]
        if constrained:
            constrained.sort()
            best_i = constrained[0][1]
            return best_i, dominant_freq(mamba_entries[best_i], freq_grid)

    # Fallback: no constraint, just closest-to-median Mamba MSE.
    unconstrained = [(abs(r[1]["mamba_mv"] - target_mamba), r[0], r[1]) for r in rows]
    unconstrained.sort()
    best_i = unconstrained[0][1]
    return best_i, dominant_freq(mamba_entries[best_i], freq_grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--phase_label", required=True)
    ap.add_argument("--regime", default="multisin_high_irreg")
    ap.add_argument("--picker", choices=("freq_percentile", "representative"),
                    default="representative",
                    help="'freq_percentile' picks by --freq_percentile; "
                         "'representative' picks a sample whose per-model "
                         "ranking matches the aggregate and whose Mamba R² "
                         "is near the test-set median (default).")
    ap.add_argument("--freq_percentile", type=int, default=75,
                    help="Used only when --picker=freq_percentile. 0=lowest.")
    ap.add_argument("--constraint_winner", default=None,
                    choices=(None, "mamba_mv", "s5", "romae"),
                    help="With --picker=representative: restrict to samples "
                         "where this model has the lowest MSE.")
    ap.add_argument("--history", type=float, default=8.0)
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    preds_by_model = {}
    for m in MODELS:
        globs = list((pred_root / m / args.regime).glob("*/predictions.jsonl"))
        if not globs:
            print(f"  [skip] {m} / {args.regime}: no predictions.jsonl"); continue
        preds_by_model[m] = load_predictions(globs[0])
    if "mamba_mv" not in preds_by_model:
        raise SystemExit(f"No Mamba predictions under {pred_root}/mamba_mv/{args.regime}")

    mamba_entries = preds_by_model["mamba_mv"]
    freq_grid = np.linspace(0.1, 2.5, 400)
    freqs = np.asarray([dominant_freq(e, freq_grid) for e in mamba_entries])

    if args.picker == "representative":
        pick_i, f_dom = _pick_representative(preds_by_model, freq_grid,
                                             constraint_winner=args.constraint_winner)
        if pick_i is None:
            raise SystemExit("no common item_ids across models — cannot pick representative sample")
    else:
        order = np.argsort(freqs)
        idx = int(np.clip(round((args.freq_percentile / 100.0) * (len(order) - 1)),
                          0, len(order) - 1))
        pick_i = int(order[idx])
        f_dom = float(freqs[pick_i])

    mamba_pick = mamba_entries[pick_i]
    item_id = mamba_pick["item_id"]

    other_idx = {m: {e["item_id"]: e for e in es}
                 for m, es in preds_by_model.items() if m != "mamba_mv"}

    n_vars = 3
    model_keys = [m for m in MODELS if m in preds_by_model]
    n_models = len(model_keys)
    fig, axes = plt.subplots(n_vars, n_models, figsize=(3.8 * n_models, 2.0 * n_vars),
                             sharex=True)
    axes = np.atleast_2d(axes)

    for v in range(n_vars):
        for c, model in enumerate(model_keys):
            ax = axes[v, c]
            entry = mamba_pick if model == "mamba_mv" else other_idx[model].get(item_id)
            if entry is None:
                ax.set_visible(False); continue
            var = entry["variates"][v]
            ts_ctx = np.asarray(var["ts_ctx"]); val_ctx = np.asarray(var["val_ctx"])
            ts_pred = np.asarray(var["ts_pred"])
            val_true = np.asarray(var["val_true_pred"])
            val_pred = np.asarray(var["val_pred"])
            ord_ctx = np.argsort(ts_ctx)
            ax.plot(ts_ctx[ord_ctx], val_ctx[ord_ctx], color="0.5", marker="o",
                    markersize=2.8, linewidth=0.8,
                    label="Context" if (v == 0 and c == 0) else None)
            ax.scatter(ts_pred, val_true, color="k", s=10, zorder=3,
                       label="True" if (v == 0 and c == 0) else None)
            ord_p = np.argsort(ts_pred)
            ax.plot(ts_pred[ord_p], val_pred[ord_p], color="red", marker="s",
                    markersize=3.5, linewidth=0.9, zorder=4,
                    label="Predicted" if (v == 0 and c == 0) else None)
            ax.fill_between(ts_pred[ord_p], val_true[ord_p], val_pred[ord_p],
                            alpha=0.15, color="steelblue", linewidth=0)
            ax.axvline(args.history, color="0.6", linestyle=":", linewidth=0.8)
            if v == 0:
                m_ = sample_metrics(entry)
                ax.set_title(f"{MODEL_LABELS[model]}   R²={m_['r2']:.3f}   MSE={m_['mse']:.4f}",
                             fontsize=10, color=MODEL_COLORS[model])
            if c == 0:
                ax.set_ylabel(f"v{v}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=9)
            if v == n_vars - 1:
                ax.set_xlabel("time", fontsize=9)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"{args.phase_label} — {args.regime}  |  dom. freq = {f_dom:.2f}",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (item_id={item_id}, f={f_dom:.2f})")


if __name__ == "__main__":
    main()
