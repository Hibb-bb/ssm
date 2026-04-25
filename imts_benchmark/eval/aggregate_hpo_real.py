"""Aggregate the per-dataset Mamba-MV HPO sweep on real T-PatchGNN datasets.

Companion to aggregate_hpo_pick_winner.py — this variant scans ONE dt_mode
(default 'replace') across N datasets and writes a single winners_by_dataset.json
that the winner-confirm SLURM array consumes.

Reuses best_val_mse() from aggregate_hpo_pick_winner.py.

Usage:
  python aggregate_hpo_real.py \
      --hpo_log_dir /path/to/imts_benchmark_v2_real/hpo_mamba_mv/replace \
      --datasets physionet activity ushcn \
      --dt_mode replace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from imts_benchmark.eval.aggregate_hpo_pick_winner import best_val_mse, lr_to_float


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_log_dir", type=str, required=True,
                   help="Top-level HPO output dir; expected layout: "
                        "<hpo_log_dir>/<dataset>/dt-<dt>_lr-<lr>_bs-<bs>/seed<seed>/")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--dt_mode", type=str, default="replace")
    p.add_argument("--lrs", nargs="+", default=["1e-4", "5e-4", "2e-3"])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    hpo_dir = Path(args.hpo_log_dir)
    assert hpo_dir.exists(), f"HPO dir not found: {hpo_dir}"

    rows = []
    missing = []
    for ds in args.datasets:
        for lr in args.lrs:
            for bs in args.batch_sizes:
                tag = f"dt-{args.dt_mode}_lr-{lr}_bs-{bs}"
                cell_dir = hpo_dir / ds / tag / f"seed{args.seed}"
                val_mse = best_val_mse(cell_dir)
                if val_mse == float("inf"):
                    missing.append(str(cell_dir))
                rows.append({
                    "dataset": ds,
                    "dt_mode": args.dt_mode,
                    "lr": lr,
                    "train_batch_size": bs,
                    "seed": args.seed,
                    "min_val_mse": val_mse,
                    "cell_dir": str(cell_dir),
                })

    df = pd.DataFrame(rows)
    df["lr_float"] = df["lr"].map(lr_to_float)
    df = df.sort_values(
        by=["dataset", "min_val_mse", "lr_float", "train_batch_size"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    ranked_csv = hpo_dir / "hpo_ranked.csv"
    df.drop(columns=["lr_float"]).to_csv(ranked_csv, index=False)
    print(f"[aggregate_real] wrote {ranked_csv}")

    winners_by_dataset: dict = {}
    for ds in args.datasets:
        sub = df[df["dataset"] == ds]
        if len(sub) == 0 or sub["min_val_mse"].min() == float("inf"):
            print(f"[aggregate_real] WARNING: no usable cells for dataset={ds}",
                  file=sys.stderr)
            continue
        best = sub.iloc[0]
        winners_by_dataset[ds] = {
            "dt_mode": args.dt_mode,
            "lr": best["lr"],
            "train_batch_size": int(best["train_batch_size"]),
            "min_val_mse": float(best["min_val_mse"]),
            "source_cell_dir": best["cell_dir"],
        }

    winners_json = hpo_dir / "winners_by_dataset.json"
    with open(winners_json, "w") as f:
        json.dump(winners_by_dataset, f, indent=2)
    print(f"[aggregate_real] wrote {winners_json}")
    for ds, cfg in winners_by_dataset.items():
        print(f"  {ds}: lr={cfg['lr']} bs={cfg['train_batch_size']} "
              f"val_mse={cfg['min_val_mse']:.6f}")

    if missing:
        print(f"\n[aggregate_real] WARNING: {len(missing)} cells missing/unreadable:",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
