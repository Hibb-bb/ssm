"""Aggregate UEA classification HPO results and pick winners.

Reads all summary.json files under <hpo_root>/<dataset>/<dt_mode>/<lr_dir>/hpo/,
groups by (dataset, dt_mode), picks the LR with the highest val_acc per group,
and writes a single winner_configs.json plus a per-(dataset, dt_mode) markdown
summary table.

Usage:
    python -m imts_benchmark.eval.aggregate_uea_hpo \
        --hpo_root /projects/b1094/.../uea_cls/hpo \
        --out      /projects/b1094/.../uea_cls/winner_configs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_root", type=str, required=True,
                   help="HPO output root: <hpo_root>/<dataset>/<dt_mode>/<lr_dir>/hpo/summary.json")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSON path for winner_configs.")
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

    # Group by (dataset, dt_mode); pick highest val_acc.
    by_group: dict[tuple[str, str], list[dict]] = {}
    for s in summaries:
        key = (s["dataset"], s["dt_mode"])
        by_group.setdefault(key, []).append(s)

    winners = {}
    for (dataset, dt_mode), runs in sorted(by_group.items()):
        runs.sort(key=lambda r: r["best_val_acc"], reverse=True)
        winner = runs[0]
        winners.setdefault(dataset, {})[dt_mode] = {
            "lr": winner["lr"],
            "val_acc": winner["best_val_acc"],
            "test_acc": winner["test_acc"],
            "macro_f1": winner["macro_f1"],
            "n_runs_seen": len(runs),
        }
        print(f"[winner] {dataset}/{dt_mode}: lr={winner['lr']:.0e}, "
              f"val_acc={winner['best_val_acc']:.4f}, "
              f"(test_acc={winner['test_acc']:.4f}, "
              f"macro_f1={winner['macro_f1']:.4f}); "
              f"runs={len(runs)}")

    # Write winner configs.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"[aggregate] wrote {out_path}")

    # Markdown table for human eyeballing.
    md_path = out_path.with_suffix(".md")
    lines = ["# UEA HPO winners (val_acc-picked)", ""]
    lines.append("| Dataset | dt_mode | LR | val_acc | test_acc | macro_f1 |")
    lines.append("|---------|---------|----|---------|----------|----------|")
    for dataset in sorted(winners.keys()):
        for dt_mode in ("replace", "learned"):
            if dt_mode not in winners[dataset]:
                continue
            w = winners[dataset][dt_mode]
            lines.append(
                f"| {dataset} | {dt_mode} | {w['lr']:.0e} | "
                f"{w['val_acc']:.4f} | {w['test_acc']:.4f} | {w['macro_f1']:.4f} |"
            )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {md_path}")


if __name__ == "__main__":
    main()
