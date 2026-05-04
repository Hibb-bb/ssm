"""V2 Pendulum HPO aggregator: pick val/mse-best (weight_decay, dropout) per
dt_mode. Diagnosis from v1 runs: train/val gap of 0.3-0.6 NLL with no
regularization (wd=0, dropout=0) → memorization, val/mse floors at ~0.010.
v2 sweeps regularization while holding (lr, B) at v1 winners.

Sweep layout (matches run_pendulum_hpo_v2_reg_pretrain_delta.sbatch):
  <hpo_log_dir>/<dt_mode>/wd-<wd>_drop-<drop>/seed1/checkpoints/best.ckpt

(lr, B) is held per-dt_mode at the v1 confirm winner (read from v1's
winners_by_dt_mode.json) so the only varying axes here are (wd, dropout).

Usage:
  python -m imts_benchmark.eval.aggregate_hpo_winners_pendulum_v2_reg \\
      --hpo_log_dir /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain_v2_reg \\
      --v1_winners_json /projects/bfrf/seojininus/ssm/output/log/pendulum/hpo_mamba_pretrain/winners_by_dt_mode.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from imts_benchmark.eval.aggregate_hpo_pick_winner import best_val_mse


DT_MODES = ("replace", "learned", "concat")
WDS = ("0.0", "0.05", "0.2")
DROPOUTS = ("0.0", "0.1")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hpo_log_dir", required=True)
    p.add_argument("--v1_winners_json", required=True,
                   help="Path to v1's winners_by_dt_mode.json — used to carry "
                        "(lr, batch_size) into v2 confirm.")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    hpo_dir = Path(args.hpo_log_dir)
    if not hpo_dir.exists():
        print(f"[v2_reg] ERROR: {hpo_dir} not found", file=sys.stderr)
        return 1

    v1_winners = json.load(open(args.v1_winners_json))

    rows = []
    missing = []
    for dt in DT_MODES:
        for wd in WDS:
            for drop in DROPOUTS:
                cell_dir = hpo_dir / dt / f"wd-{wd}_drop-{drop}" / f"seed{args.seed}"
                val_mse = best_val_mse(cell_dir)
                if val_mse == float("inf"):
                    missing.append(str(cell_dir))
                rows.append({
                    "dt_mode": dt, "weight_decay": wd, "dropout": drop,
                    "min_val_mse": val_mse,
                    "cell_dir": str(cell_dir),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["dt_mode", "min_val_mse"], ascending=[True, True]).reset_index(drop=True)

    ranked_csv = hpo_dir / "hpo_ranked.csv"
    df.to_csv(ranked_csv, index=False)
    print(f"[v2_reg] wrote {ranked_csv}")

    winners: dict = {}
    for dt in DT_MODES:
        sub = df[(df["dt_mode"] == dt) & (df["min_val_mse"] != float("inf"))]
        if len(sub) == 0:
            print(f"[v2_reg] WARNING: no usable cells for dt_mode={dt}", file=sys.stderr)
            continue
        best = sub.iloc[0]
        v1 = v1_winners.get(dt, {})
        winners[dt] = {
            "dt_mode": dt,
            # carried from v1 winner — held fixed in v2
            "lr": v1.get("lr"),
            "batch_size": int(v1.get("batch_size")) if v1.get("batch_size") else None,
            # swept axes — picked by v2
            "weight_decay": float(best["weight_decay"]),
            "dropout": float(best["dropout"]),
            "min_val_mse": float(best["min_val_mse"]),
            "source_cell_dir": best["cell_dir"],
        }

    winners_json = hpo_dir / "winners_by_dt_mode.json"
    with open(winners_json, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"[v2_reg] wrote {winners_json}")
    for dt, cfg in winners.items():
        print(f"  {dt}: lr={cfg['lr']} B={cfg['batch_size']} wd={cfg['weight_decay']} "
              f"drop={cfg['dropout']} val_mse={cfg['min_val_mse']:.6f}")

    if missing:
        print(f"\n[v2_reg] WARNING: {len(missing)} cells missing/unreadable:", file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)

    if len(winners) < len(DT_MODES):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
