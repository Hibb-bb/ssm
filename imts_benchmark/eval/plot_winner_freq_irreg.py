"""Per-phase "who wins where" chart across the (regime × dominant-frequency) grid.

For each test sample we compute the dominant frequency of v0's context via
Lomb-Scargle, bin into {low, mid, high}, and for each (regime, freq_bin) cell
we compute the mean per-sample MSE of each model (using seed=1 predictions
from predictions.jsonl). The cell is colored by the winning model and labeled
with its mean MSE.

Output: a 3-row × 4-column grid PNG per phase. Rows = freq bins (low → high),
columns = regimes (Regular → High). Cell bg color = winner model. Cell text =
"winner\nMSE" (mean across samples in that cell).

Inputs: predictions.jsonl files exported by export_test_predictions.py, which
exist under docs/<phase>/predictions/<model>/<regime>/<variant>/predictions.jsonl.
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
# Light tints for cell backgrounds (so black text reads).
MODEL_BG = {"mamba_mv": "#f7c8c8", "s5": "#ffe0c2", "romae": "#cfd9ec"}

REGIMES = ("multisin_regular", "multisin_low_irreg", "multisin_med_irreg", "multisin_high_irreg")
REGIME_LABELS = {"multisin_regular": "Regular\n(100%)", "multisin_low_irreg": "Low\n(80%)",
                 "multisin_med_irreg": "Med\n(30%)", "multisin_high_irreg": "High\n(0%)"}
FREQ_BINS = [(0.0, 0.8, "Low freq\nf < 0.8"), (0.8, 1.6, "Mid freq\n0.8 – 1.6"),
             (1.6, 3.0, "High freq\nf ≥ 1.6")]


def dominant_freq(entry, freq_grid):
    v0 = entry["variates"][0]
    ts = np.asarray(v0["ts_ctx"], dtype=float)
    vals = np.asarray(v0["val_ctx"], dtype=float)
    if len(ts) < 4:
        return 0.0
    vals = vals - np.mean(vals)
    try:
        pgram = lombscargle(ts, vals, 2 * np.pi * freq_grid, normalize=True)
    except Exception:
        return 0.0
    return float(freq_grid[int(np.argmax(pgram))])


def sample_mse(entry):
    ss = 0.0; n = 0
    for v in entry["variates"]:
        pred = np.asarray(v["val_pred"], dtype=float)
        true = np.asarray(v["val_true_pred"], dtype=float)
        if len(pred) == 0:
            continue
        ss += float(np.sum((pred - true) ** 2))
        n += int(len(pred))
    return ss / n if n > 0 else float("nan")


def load_predictions_for_model(pred_root: Path, model: str, regime: str):
    globs = list((pred_root / model / regime).glob("*/predictions.jsonl"))
    if not globs:
        return None
    with open(globs[0]) as f:
        return [json.loads(l) for l in f]


def build_cells(pred_root: Path):
    """Return dict[(regime, freq_bin_idx, model)] -> (mean_mse, n_samples)."""
    cells: dict[tuple, tuple] = {}
    freq_grid = np.linspace(0.1, 2.5, 400)
    # Accumulate per (regime, freq_bin, model) -> list of per-sample MSEs.
    bucket: dict[tuple, list[float]] = {}
    for regime in REGIMES:
        # Use Mamba's predictions to compute dom freq per item (ts_ctx depends only on data).
        mamba = load_predictions_for_model(pred_root, "mamba_mv", regime)
        if not mamba:
            continue
        freq_by_iid = {e["item_id"]: dominant_freq(e, freq_grid) for e in mamba}
        for model in MODELS:
            entries = load_predictions_for_model(pred_root, model, regime)
            if entries is None:
                continue
            for e in entries:
                iid = e["item_id"]
                f = freq_by_iid.get(iid)
                if f is None:
                    continue
                fi = None
                for i, (lo, hi, _) in enumerate(FREQ_BINS):
                    if lo <= f < hi:
                        fi = i; break
                if fi is None:
                    continue
                key = (regime, fi, model)
                bucket.setdefault(key, []).append(sample_mse(e))
    for key, vals in bucket.items():
        cells[key] = (float(np.mean(vals)), len(vals))
    return cells


def plot_winner_grid(cells, out_file: Path, phase_label: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax.set_ylim(-0.5, len(FREQ_BINS) - 0.5)
    ax.invert_yaxis()
    # Reverse freq order so "high freq" is on top.
    for xi, regime in enumerate(REGIMES):
        for yi, (lo, hi, fl_label) in enumerate(FREQ_BINS):
            cell_yi = (len(FREQ_BINS) - 1) - yi  # place high freq at top
            models_here = {m: cells.get((regime, yi, m)) for m in MODELS}
            valid = {m: v for m, v in models_here.items() if v is not None}
            if not valid:
                ax.add_patch(plt.Rectangle((xi - 0.48, cell_yi - 0.48), 0.96, 0.96,
                                           facecolor="#f0f0f0", edgecolor="white", linewidth=3))
                ax.text(xi, cell_yi, "no data", ha="center", va="center", fontsize=10, color="#888")
                continue
            winner = min(valid, key=lambda m: valid[m][0])
            ax.add_patch(plt.Rectangle((xi - 0.48, cell_yi - 0.48), 0.96, 0.96,
                                       facecolor=MODEL_BG[winner], edgecolor="white", linewidth=3))
            mse_w, n = valid[winner]
            # Also print runner-up gap for context.
            runners = sorted(valid.items(), key=lambda kv: kv[1][0])
            gap = runners[1][1][0] - runners[0][1][0] if len(runners) > 1 else 0.0
            lines = [
                f"{MODEL_LABELS[winner]}",
                f"MSE {mse_w:.4f}",
                f"Δ vs next {gap:+.4f}",
                f"n={n}",
            ]
            ax.text(xi, cell_yi - 0.05, "\n".join(lines),
                    ha="center", va="center", fontsize=10,
                    color=MODEL_COLORS[winner], fontweight="bold")
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([REGIME_LABELS[r] for r in REGIMES], fontsize=11)
    ax.set_yticks(range(len(FREQ_BINS)))
    ax.set_yticklabels([fl for _, _, fl in reversed(FREQ_BINS)], fontsize=11)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{phase_label} — winner by (irregularity × dominant frequency)\n"
                 f"Red = Mamba-MV   ·   Orange = S5   ·   Blue = RoMAE", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", required=True, type=Path)
    ap.add_argument("--out_file", required=True, type=Path)
    ap.add_argument("--phase_label", required=True)
    args = ap.parse_args()
    cells = build_cells(args.pred_root)
    plot_winner_grid(cells, args.out_file, args.phase_label)


if __name__ == "__main__":
    main()
