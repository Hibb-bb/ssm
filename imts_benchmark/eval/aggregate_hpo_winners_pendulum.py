"""Pick the val/mse-best (lr, batch_size) per dt_mode from the 27-cell
Pendulum mamba_pretrain HPO sweep and emit a winners JSON consumed by
the confirm sbatch.

Sweep layout (matches run_pendulum_hpo_pretrain_delta.sbatch):
  <hpo_log_dir>/<dt_mode>/lr-<lr>_b-<bs>/seed1/checkpoints/best.ckpt

Pendulum has no gradient accumulation (physical batch == effective batch),
so the schema is simpler than PhysioNet's eff_bs/phys_bs/accum split.

Usage:
  python -m imts_benchmark.eval.aggregate_hpo_winners_pendulum \\
      --hpo_log_dir /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from imts_benchmark.eval.aggregate_hpo_pick_winner import (
    best_val_mse,
    lr_to_float,
)


DT_MODES = ("replace", "learned", "concat")
LRS = ("5e-4", "2e-3", "1e-2")
BATCH_SIZES = (16, 32, 64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_log_dir", required=True,
                   help="Top dir; <hpo_log_dir>/<dt_mode>/lr-<lr>_b-<bs>/seed1/")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    hpo_dir = Path(args.hpo_log_dir)
    if not hpo_dir.exists():
        print(f"[hpo_pendulum] ERROR: HPO dir not found: {hpo_dir}",
              file=sys.stderr)
        return 1

    rows = []
    missing = []
    for dt in DT_MODES:
        for lr in LRS:
            for bs in BATCH_SIZES:
                cell_dir = hpo_dir / dt / f"lr-{lr}_b-{bs}" / f"seed{args.seed}"
                val_mse = best_val_mse(cell_dir)
                if val_mse == float("inf"):
                    missing.append(str(cell_dir))
                rows.append({
                    "dt_mode": dt,
                    "lr": lr,
                    "batch_size": bs,
                    "min_val_mse": val_mse,
                    "cell_dir": str(cell_dir),
                })

    df = pd.DataFrame(rows)
    df["lr_float"] = df["lr"].map(lr_to_float)
    df = df.sort_values(
        by=["dt_mode", "min_val_mse", "lr_float", "batch_size"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    ranked_csv = hpo_dir / "hpo_ranked.csv"
    df.drop(columns=["lr_float"]).to_csv(ranked_csv, index=False)
    print(f"[hpo_pendulum] wrote {ranked_csv}")

    winners: dict = {}
    for dt in DT_MODES:
        sub = df[(df["dt_mode"] == dt) & (df["min_val_mse"] != float("inf"))]
        if len(sub) == 0:
            print(f"[hpo_pendulum] WARNING: no usable cells for dt_mode={dt}",
                  file=sys.stderr)
            continue
        best = sub.iloc[0]
        winners[dt] = {
            "dt_mode": dt,
            "lr": best["lr"],
            "batch_size": int(best["batch_size"]),
            "min_val_mse": float(best["min_val_mse"]),
            "source_cell_dir": best["cell_dir"],
        }

    winners_json = hpo_dir / "winners_by_dt_mode.json"
    with open(winners_json, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"[hpo_pendulum] wrote {winners_json}")
    for dt, cfg in winners.items():
        print(f"  {dt}: lr={cfg['lr']} batch_size={cfg['batch_size']} "
              f"val_mse={cfg['min_val_mse']:.6f}")

    if missing:
        print(f"\n[hpo_pendulum] WARNING: {len(missing)} cells missing/unreadable:",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more", file=sys.stderr)

    if len(winners) < len(DT_MODES):
        print(f"[hpo_pendulum] ERROR: only {len(winners)}/{len(DT_MODES)} "
              f"dt_modes have winners", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
