"""Aggregate UEA classification HPO results and pick winners (v2).

Reads all summary.json files under the HPO output root, groups by
(dataset, dt_mode), picks the best per-cell config (LR + batch_size for v2),
and writes ``winner_configs.json`` plus a markdown summary.

v2 changes vs v1:
  - **Per-dataset selection metric.** Imbalanced datasets are picked by
    val/macro_f1 (HB, LSST, EP); balanced datasets stay on val/acc
    (BM, CT). Reason: on imbalanced data an LR that destabilizes the
    model into majority-class collapse can win the picker because
    constant-predictor val_acc equals the majority-class fraction.
  - **Collapse detector.** Any cell with
    ``val_acc - val_macro_f1 > COLLAPSE_GAP`` (default 0.2) is flagged
    and excluded from selection. Logged to stderr.
  - **Records batch_size in winner.** v2 sweeps batch_size as a real
    HPO axis; the winner needs it so the final-eval array can replay
    the correct config.

Usage:
    python -m imts_benchmark.eval.aggregate_uea_hpo \
        --hpo_root /projects/b1094/.../uea_cls/hpo_v2 \
        --out      /projects/b1094/.../uea_cls/winner_configs_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Per-dataset picker metric. Anything not listed here defaults to val_acc.
PICKER_METRIC = {
    "BasicMotions":          "best_val_acc",
    "CharacterTrajectories": "best_val_acc",
    "Epilepsy":              "val_macro_f1",   # 4-class with skew
    "Heartbeat":             "val_macro_f1",   # binary ~60/40
    "LSST":                  "val_macro_f1",   # 14-class heavy skew
}

COLLAPSE_GAP = 0.2  # val_acc - val_macro_f1 above this => suspected majority-class collapse.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_root", type=str, required=True,
                   help="HPO output root containing summary.json files (any depth).")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSON path for winner_configs.")
    p.add_argument("--collapse_gap", type=float, default=COLLAPSE_GAP,
                   help="val_acc - val_macro_f1 threshold above which a cell is flagged "
                        "as majority-class collapse and excluded from selection.")
    return p.parse_args()


def main():
    args = parse_args()
    hpo_root = Path(args.hpo_root)
    if not hpo_root.exists():
        raise FileNotFoundError(f"hpo_root does not exist: {hpo_root}")

    summaries = []
    for summary_path in hpo_root.rglob("summary.json"):
        with open(summary_path) as f:
            s = json.load(f)
        summaries.append(s)

    if not summaries:
        raise RuntimeError(f"No summary.json files found under {hpo_root}")

    print(f"[aggregate] read {len(summaries)} summaries")

    # Backward compat: v1 summaries don't have val_macro_f1; treat as missing.
    for s in summaries:
        s.setdefault("val_macro_f1", None)
        s.setdefault("batch_size", None)
        s.setdefault("head_dropout", None)

    # ---- Collapse detector ----
    n_collapsed = 0
    for s in summaries:
        if s["val_macro_f1"] is None:
            continue
        gap = s["best_val_acc"] - s["val_macro_f1"]
        if gap > args.collapse_gap:
            print(
                f"[COLLAPSE] {s['dataset']}/{s['dt_mode']} lr={s['lr']} "
                f"bs={s['batch_size']}: val_acc={s['best_val_acc']:.4f} "
                f"val_f1={s['val_macro_f1']:.4f} gap={gap:.3f} -> EXCLUDED",
                file=sys.stderr,
            )
            s["_collapsed"] = True
            n_collapsed += 1
        else:
            s["_collapsed"] = False
    if n_collapsed:
        print(f"[aggregate] {n_collapsed} cells flagged as collapsed and excluded")

    # ---- Group by (dataset, dt_mode); pick winner per group ----
    by_group: dict[tuple[str, str], list[dict]] = {}
    for s in summaries:
        key = (s["dataset"], s["dt_mode"])
        by_group.setdefault(key, []).append(s)

    winners = {}
    for (dataset, dt_mode), runs in sorted(by_group.items()):
        # Drop collapsed; if all collapsed for this group, fall back to all runs
        # so the pipeline keeps moving (final eval will still surface the issue).
        eligible = [r for r in runs if not r.get("_collapsed", False)]
        if not eligible:
            print(
                f"[warn] {dataset}/{dt_mode}: ALL cells collapsed; falling back "
                f"to full set so pipeline continues",
                file=sys.stderr,
            )
            eligible = runs

        metric_key = PICKER_METRIC.get(dataset, "best_val_acc")
        # If picker is val_macro_f1 but it's missing (v1-style summaries), fall back.
        if metric_key == "val_macro_f1" and any(r["val_macro_f1"] is None for r in eligible):
            print(
                f"[warn] {dataset}: requested picker val_macro_f1 but some cells "
                f"are missing it; falling back to best_val_acc",
                file=sys.stderr,
            )
            metric_key = "best_val_acc"

        eligible.sort(key=lambda r: r[metric_key], reverse=True)
        winner = eligible[0]
        winners.setdefault(dataset, {})[dt_mode] = {
            "lr": winner["lr"],
            "batch_size": winner["batch_size"],
            "head_dropout": winner["head_dropout"],
            "val_acc": winner["best_val_acc"],
            "val_macro_f1": winner["val_macro_f1"],
            "test_acc": winner["test_acc"],
            "macro_f1": winner["macro_f1"],
            "picker_metric": metric_key,
            "n_runs_seen": len(runs),
            "n_runs_eligible": len(eligible),
        }
        print(
            f"[winner] {dataset}/{dt_mode} (picker={metric_key}): "
            f"lr={winner['lr']:.0e} bs={winner['batch_size']} "
            f"drop={winner['head_dropout']} | "
            f"val_acc={winner['best_val_acc']:.4f} val_f1={winner['val_macro_f1']} "
            f"test_acc={winner['test_acc']:.4f} test_f1={winner['macro_f1']:.4f}; "
            f"eligible={len(eligible)}/{len(runs)}"
        )

    # ---- Write winner configs ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"[aggregate] wrote {out_path}")

    # ---- Markdown table ----
    md_path = out_path.with_suffix(".md")
    lines = ["# UEA HPO winners (v2: per-dataset picker, collapse-filtered)", ""]
    lines.append("| Dataset | dt_mode | picker | LR | BS | drop | val_acc | val_f1 | test_acc | test_f1 |")
    lines.append("|---------|---------|--------|----|----|------|---------|--------|----------|---------|")
    for dataset in sorted(winners.keys()):
        for dt_mode in ("replace", "learned"):
            if dt_mode not in winners[dataset]:
                continue
            w = winners[dataset][dt_mode]
            val_f1 = f"{w['val_macro_f1']:.4f}" if w['val_macro_f1'] is not None else "n/a"
            lines.append(
                f"| {dataset} | {dt_mode} | {w['picker_metric']} | "
                f"{w['lr']:.0e} | {w['batch_size']} | {w['head_dropout']} | "
                f"{w['val_acc']:.4f} | {val_f1} | "
                f"{w['test_acc']:.4f} | {w['macro_f1']:.4f} |"
            )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
