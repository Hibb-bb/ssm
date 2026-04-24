"""Phase-general prediction plotter (3-model: Mamba-MV, S5, RoMAE; mTAN dropped).

For each regime, picks 4 samples at 25/50/75/100 dominant-frequency percentiles
(Lomb-Scargle on v0 context) and plots 3 variates × 3 models per sample. Slide
decks then crop the top strip (~high-freq sample) or the bottom strip
(~highest percentile) as needed.

Usage:
  python plot_phase_predictions_general.py \
      --pred_root <path>/<phase>_results/predictions \
      --out_dir <path>/<phase>_results \
      --phase_label "Phase 3 — async dense"
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
REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")


def load_predictions(pred_file: Path) -> list[dict]:
    with open(pred_file) as f:
        return [json.loads(l) for l in f]


def dominant_freq(entry: dict, freq_grid: np.ndarray) -> float:
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


def compute_sample_metrics(entry: dict) -> dict:
    y_t, y_p = [], []
    for v in entry["variates"]:
        y_t.extend(v["val_true_pred"])
        y_p.extend(v["val_pred"])
    y_t = np.asarray(y_t); y_p = np.asarray(y_p)
    if len(y_t) == 0:
        return dict(mse=np.nan, r2=np.nan)
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return dict(mse=float(np.mean((y_t - y_p) ** 2)), r2=float(r2))


def pick_samples_by_freq_percentile(entries, freq_grid, percentiles=(25, 50, 75, 100)):
    freqs = np.asarray([dominant_freq(e, freq_grid) for e in entries])
    order = np.argsort(freqs)
    picked = []
    for p in percentiles:
        idx_in_sorted = int(np.clip(round((p / 100.0) * (len(order) - 1)), 0, len(order) - 1))
        i = int(order[idx_in_sorted])
        picked.append((entries[i], float(freqs[i])))
    return picked


def index_by_item_id(entries):
    return {e["item_id"]: e for e in entries}


def plot_regime(pred_root: Path, regime: str, out_path: Path, history: float,
                phase_label: str) -> None:
    preds_by_model: dict[str, list[dict]] = {}
    for m in MODELS:
        glob = list((pred_root / m / regime).glob("*/predictions.jsonl"))
        if not glob:
            print(f"  [skip] {m} / {regime}: no predictions.jsonl"); continue
        preds_by_model[m] = load_predictions(glob[0])
    if "mamba_mv" not in preds_by_model:
        print(f"  [skip] {regime}: no Mamba predictions")
        return
    mamba_entries = preds_by_model["mamba_mv"]
    freq_grid = np.linspace(0.1, 2.5, 400)
    selected = pick_samples_by_freq_percentile(mamba_entries, freq_grid)

    other_idx = {m: index_by_item_id(entries) for m, entries in preds_by_model.items() if m != "mamba_mv"}
    model_keys = [m for m in MODELS if m in preds_by_model]

    n_samples = len(selected)
    n_vars = 3
    n_models = len(model_keys)
    fig, axes = plt.subplots(n_samples * n_vars, n_models,
                             figsize=(4.2 * n_models, 2.0 * n_samples * n_vars),
                             sharex=True)
    axes = np.atleast_2d(axes)

    for s_idx, (mamba_entry, f_dom) in enumerate(selected):
        item_id = mamba_entry["item_id"]
        for v in range(n_vars):
            for c, model in enumerate(model_keys):
                ax = axes[s_idx * n_vars + v, c]
                entry = mamba_entry if model == "mamba_mv" else other_idx[model].get(item_id)
                if entry is None:
                    ax.set_visible(False); continue
                var = entry["variates"][v]
                ts_ctx = np.asarray(var["ts_ctx"]); val_ctx = np.asarray(var["val_ctx"])
                ts_pred = np.asarray(var["ts_pred"])
                val_true = np.asarray(var["val_true_pred"])
                val_pred = np.asarray(var["val_pred"])
                ord_ctx = np.argsort(ts_ctx)
                ax.plot(ts_ctx[ord_ctx], val_ctx[ord_ctx], color="0.5", marker="o",
                        markersize=2.5, linewidth=0.8,
                        label="Context" if (s_idx == 0 and v == 0 and c == 0) else None)
                ax.scatter(ts_pred, val_true, color="k", s=8, zorder=3,
                           label="True" if (s_idx == 0 and v == 0 and c == 0) else None)
                ord_p = np.argsort(ts_pred)
                ax.plot(ts_pred[ord_p], val_pred[ord_p], color="red", marker="s",
                        markersize=3, linewidth=0.8, zorder=4,
                        label="Predicted" if (s_idx == 0 and v == 0 and c == 0) else None)
                ax.fill_between(ts_pred[ord_p], val_true[ord_p], val_pred[ord_p],
                                alpha=0.15, color="steelblue", linewidth=0,
                                label="|Error|" if (s_idx == 0 and v == 0 and c == 0) else None)
                ax.axvline(history, color="0.6", linestyle=":", linewidth=0.7)
                if v == 0:
                    metrics = compute_sample_metrics(entry)
                    ttl = (f"{MODEL_LABELS[model]}  variate 0  |  R²={metrics['r2']:.3f}, "
                           f"MSE={metrics['mse']:.4f}  (f={f_dom:.2f})")
                    ax.set_title(ttl, fontsize=9, color=MODEL_COLORS[model])
                else:
                    ax.set_title(f"{MODEL_LABELS[model]}  variate {v}", fontsize=8,
                                 color=MODEL_COLORS[model])
                if c == 0:
                    ax.set_ylabel(f"v{v}", fontsize=9)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
                if s_idx * n_vars + v == axes.shape[0] - 1:
                    ax.set_xlabel("time")

    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        f"{phase_label} — {regime} — test samples sorted by dominant frequency "
        f"(25/50/75/100 percentile, low→high).  Each sample: 3 variates stacked.",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--phase_label", type=str, required=True,
                    help="Human-readable phase label (used in suptitle).")
    ap.add_argument("--prefix", type=str, default="phase3",
                    help="Output filename prefix, e.g. phase3 / phase4_2.")
    ap.add_argument("--history", type=float, default=8.0)
    args = ap.parse_args()
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        out = out_dir / f"{args.prefix}_predictions_{regime}.png"
        plot_regime(pred_root, regime, out, history=args.history, phase_label=args.phase_label)


if __name__ == "__main__":
    main()
