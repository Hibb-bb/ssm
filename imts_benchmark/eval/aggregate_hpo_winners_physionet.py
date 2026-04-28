"""Pick the val/mse-best (lr, eff_bs) per dt_mode from the 27-cell PhysioNet
Mamba-MV HPO sweep and emit a winners JSON consumed by the confirm sbatch.

Sweep layout (matches run_mamba_mv_hpo_physionet_delta.sbatch):
  <hpo_log_dir>/<dt_mode>/lr-<lr>_ebs-<ebs>/seed1/checkpoints/best.ckpt

eff_bs -> (phys_bs, accum) translation is hard-coded to match the sweep.

Usage:
  python -m imts_benchmark.eval.aggregate_hpo_winners_physionet \\
      --hpo_log_dir /projects/bfrf/seojininus/ssm/output/log/imts_benchmark_v2_real/hpo_mamba_mv_phy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from imts_benchmark.eval.aggregate_hpo_pick_winner import best_val_mse, lr_to_float


DT_MODES = ("replace", "learned", "concat")
LRS = ("1e-4", "5e-4", "2e-3")
EFF_BS = (64, 128, 256)
EBS_TO_PHYS = {64: (16, 4), 128: (32, 4), 256: (32, 8)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_log_dir", required=True,
                   help="Top dir; <hpo_log_dir>/<dt_mode>/lr-<lr>_ebs-<ebs>/seed1/")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    hpo_dir = Path(args.hpo_log_dir)
    if not hpo_dir.exists():
        print(f"[hpo_phy] ERROR: HPO dir not found: {hpo_dir}", file=sys.stderr)
        return 1

    rows = []
    missing = []
    for dt in DT_MODES:
        for lr in LRS:
            for ebs in EFF_BS:
                cell_dir = hpo_dir / dt / f"lr-{lr}_ebs-{ebs}" / f"seed{args.seed}"
                val_mse = best_val_mse(cell_dir)
                if val_mse == float("inf"):
                    missing.append(str(cell_dir))
                phys_bs, accum = EBS_TO_PHYS[ebs]
                rows.append({
                    "dt_mode": dt,
                    "lr": lr,
                    "eff_bs": ebs,
                    "phys_bs": phys_bs,
                    "accum": accum,
                    "min_val_mse": val_mse,
                    "cell_dir": str(cell_dir),
                })

    df = pd.DataFrame(rows)
    df["lr_float"] = df["lr"].map(lr_to_float)
    df = df.sort_values(
        by=["dt_mode", "min_val_mse", "lr_float", "eff_bs"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    ranked_csv = hpo_dir / "hpo_ranked.csv"
    df.drop(columns=["lr_float"]).to_csv(ranked_csv, index=False)
    print(f"[hpo_phy] wrote {ranked_csv}")

    winners: dict = {}
    for dt in DT_MODES:
        sub = df[df["dt_mode"] == dt]
        if len(sub) == 0 or sub["min_val_mse"].min() == float("inf"):
            print(f"[hpo_phy] WARNING: no usable cells for dt_mode={dt}",
                  file=sys.stderr)
            continue
        best = sub.iloc[0]
        winners[dt] = {
            "dt_mode": dt,
            "lr": best["lr"],
            "eff_bs": int(best["eff_bs"]),
            "phys_bs": int(best["phys_bs"]),
            "accum": int(best["accum"]),
            "min_val_mse": float(best["min_val_mse"]),
            "source_cell_dir": best["cell_dir"],
        }

    winners_json = hpo_dir / "winners_by_dt_mode.json"
    with open(winners_json, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"[hpo_phy] wrote {winners_json}")
    for dt, cfg in winners.items():
        print(f"  {dt}: lr={cfg['lr']} phys_bs={cfg['phys_bs']} accum={cfg['accum']} "
              f"eff_bs={cfg['eff_bs']} val_mse={cfg['min_val_mse']:.6f}")

    if missing:
        print(f"\n[hpo_phy] WARNING: {len(missing)} cells missing/unreadable:",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more", file=sys.stderr)

    if len(winners) < len(DT_MODES):
        print(f"[hpo_phy] ERROR: only {len(winners)}/{len(DT_MODES)} dt_modes have winners",
              file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
