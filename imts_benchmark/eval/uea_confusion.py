"""Per-(dataset, dt_mode) confusion matrices + per-class precision/recall/F1
from `test_predictions.json` files written by `mamba_pretrain.train_cls`.

Walks `<root>/<dataset>/<dt_mode>/seed<S>/<run_name>/test_predictions.json`
and aggregates predictions across seeds (each test sample is predicted once
per seed; we sum the 3 confusion matrices into a single per-(ds,dt) matrix).

Why: the headline accuracies hide which classes the model gets wrong. On
LSST the classes are highly imbalanced; on Heartbeat we may collapse to the
majority class. Aggregating across seeds gives a sharper signal than any
single seed's matrix.

Usage:
    python -m imts_benchmark.eval.uea_confusion \
        --root output/log/uea_cls_pretrain_70d_repo/final \
        --out_dir output/log/uea_cls_pretrain_70d_repo/confusion
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_cells(root: Path) -> dict:
    """Return {(ds, dt): [pred_dicts...]} for all cells with predictions."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ds = ds_dir.name
        for dt_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            dt = dt_dir.name
            for seed_dir in sorted(p for p in dt_dir.iterdir() if p.is_dir()):
                if not seed_dir.name.startswith("seed"):
                    continue
                # cell layout: <seed_dir>/<run_name>/test_predictions.json
                for run_dir in seed_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    pp = run_dir / "test_predictions.json"
                    if not pp.exists():
                        continue
                    with open(pp) as f:
                        cells[(ds, dt)].append(json.load(f))
    return cells


def _confusion(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Build an n_classes×n_classes matrix; rows=true, cols=pred."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-class precision/recall/F1 from a confusion matrix.
    rows=true, cols=pred. Returns numpy arrays of length n_classes."""
    n = cm.shape[0]
    prec = np.zeros(n)
    rec = np.zeros(n)
    f1 = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec[c] = tp / (tp + fp) if (tp + fp) else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) else 0.0
        f1[c] = 2 * prec[c] * rec[c] / (prec[c] + rec[c]) if (prec[c] + rec[c]) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def plot_cm(ds: str, dt: str, cm: np.ndarray, idx_to_label: dict[int, str],
            metrics: dict[str, np.ndarray], n_seeds: int, out_path: Path) -> None:
    n = cm.shape[0]
    # Normalize per row for color, but annotate with raw counts.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / np.maximum(row_sums, 1), 0.0)

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n + 4), max(5, 0.4 * n + 3)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [idx_to_label.get(c, str(c))[:14] for c in range(n)]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    # Append per-class recall to y tick labels so the bad classes pop visually.
    ax.set_yticklabels(
        [f"{labels[c]}  (R={metrics['recall'][c]:.2f}, F1={metrics['f1'][c]:.2f})"
         for c in range(n)], fontsize=8,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=7, color=color)
    macro_f1 = float(metrics["f1"].mean())
    acc = float(np.diag(cm).sum() / max(cm.sum(), 1))
    ax.set_title(f"{ds}  /  dt_mode={dt}  ({n_seeds} seeds combined)\n"
                 f"acc={acc:.3f}  macro_F1={macro_f1:.3f}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str, required=True,
                   help="Final-eval tree, expects test_predictions.json per cell.")
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cells(root)
    if not cells:
        print(f"No test_predictions.json found under {root}")
        print("(Cells run with the older trainer that didn't dump predictions "
              "won't show up here. Re-run them with the updated train_cls.py "
              "to get per-sample preds.)")
        return

    print(f"Found predictions for {len(cells)} (ds, dt) groups:")
    summary_rows: list[dict] = []
    for (ds, dt), preds_list in sorted(cells.items()):
        n_classes = preds_list[0]["n_classes"]
        idx_to_label = {int(v): k for k, v in preds_list[0]["label_to_idx"].items()}
        # Sum per-seed confusion matrices into one (ds, dt) matrix.
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for pd_ in preds_list:
            cm += _confusion(np.asarray(pd_["labels"]),
                             np.asarray(pd_["preds"]), n_classes)
        metrics = _per_class_metrics(cm)
        macro_f1 = float(metrics["f1"].mean())
        acc = float(np.diag(cm).sum() / max(cm.sum(), 1))

        out_path = out_dir / f"{ds}__{dt}.png"
        plot_cm(ds, dt, cm, idx_to_label, metrics, len(preds_list), out_path)
        # Worst-performing classes by F1.
        worst = np.argsort(metrics["f1"])[:3]
        worst_str = ", ".join(
            f"{idx_to_label[int(c)]}(F1={metrics['f1'][c]:.2f})" for c in worst
        )
        print(f"  {ds:24s} {dt:8s}  {len(preds_list)} seeds  "
              f"acc={acc:.3f}  macro_F1={macro_f1:.3f}  "
              f"worst={worst_str}")
        summary_rows.append(dict(
            dataset=ds, dt_mode=dt, n_seeds=len(preds_list),
            acc=acc, macro_f1=macro_f1,
            worst_class=idx_to_label[int(worst[0])],
            worst_f1=float(metrics["f1"][worst[0]]),
        ))

    # Write a flat CSV-ish summary.
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nWrote {len(cells)} confusion matrices and summary to {out_dir}")


if __name__ == "__main__":
    main()
