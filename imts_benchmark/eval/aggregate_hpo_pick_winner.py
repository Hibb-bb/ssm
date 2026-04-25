"""Aggregate the 27-cell Mamba-MV HPO sweep for one phase and pick winners.

For each cell (dt_mode, lr, train_batch_size), reads the Lightning
metrics.csv and extracts the minimum val/mse across training.
Selection on val/mse (not test_mse) to avoid test-set leakage in HPO.

Tie-break within a dt_mode: smallest lr first, then smallest batch size.

Writes:
  <phase_log_dir>/hpo_ranked.csv         — all 27 cells, ranked
  <phase_log_dir>/winner_config.json     — overall-best cell (any dt_mode)
  <phase_log_dir>/winners_by_mode.json   — best cell per dt_mode
      { "learned": {...}, "replace": {...}, "concat": {...} }
      Each value has keys: dt_mode, lr, train_batch_size, min_val_mse,
      source_cell_dir.

The per-dt_mode winners are what the 5-seed confirmation array consumes —
this gives us a full ablation table (learned vs replace vs concat) rather
than just one overall winner.

Usage:
  python aggregate_hpo_pick_winner.py \
      --phase_log_dir /path/to/hpo_mamba_mv/phase3 \
      --regime multisin_high_irreg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch


def best_val_mse(cell_dir: Path) -> float:
    """Return the best (min) val/mse for one HPO cell.

    Reads from the best.ckpt's ModelCheckpoint callback state — that's where
    Lightning persists `best_model_score`, which equals min val/mse since the
    callback is configured with monitor='val/mse', mode='min'.

    Falls back to Lightning's CSVLogger metrics.csv if present (older runs).
    Returns float('inf') on any failure so the caller can treat the cell as
    unusable.
    """
    # Primary: best_model_score from the checkpoint callback state.
    ckpt_path = cell_dir / "checkpoints" / "best.ckpt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cbs = ckpt.get("callbacks", {})
            for k, v in cbs.items():
                if "ModelCheckpoint" in k and isinstance(v, dict):
                    score = v.get("best_model_score")
                    if score is not None:
                        return float(score)
        except Exception as e:
            print(f"[aggregate] WARNING: could not read {ckpt_path}: {e}",
                  file=sys.stderr)

    # Fallback: old-style metrics.csv if a CSVLogger was used.
    metrics_csv = cell_dir / "lightning_logs" / "version_0" / "metrics.csv"
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if "val/mse" in df.columns:
                col = df["val/mse"].dropna()
                if len(col) > 0:
                    return float(col.min())
        except Exception:
            pass

    return float("inf")


def lr_to_float(lr_str: str) -> float:
    """Convert e.g. '1e-4' to 1e-4 for tie-breaking comparisons."""
    return float(lr_str)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase_log_dir", type=str, required=True,
                   help="e.g. output/log/imts_benchmark_v2/hpo_mamba_mv/phase3")
    p.add_argument("--regime", type=str, default="multisin_high_irreg")
    p.add_argument("--dt_modes", nargs="+", default=["learned", "replace", "concat"])
    p.add_argument("--lrs", nargs="+", default=["1e-4", "5e-4", "2e-3"])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--seed", type=int, default=1, help="HPO uses 1 seed per cell")
    args = p.parse_args()

    phase_dir = Path(args.phase_log_dir)
    assert phase_dir.exists(), f"phase dir not found: {phase_dir}"

    rows = []
    missing = []
    for dt in args.dt_modes:
        for lr in args.lrs:
            for bs in args.batch_sizes:
                tag = f"dt-{dt}_lr-{lr}_bs-{bs}"
                cell_dir = phase_dir / args.regime / tag / f"seed{args.seed}"
                val_mse = best_val_mse(cell_dir)
                if val_mse == float("inf"):
                    missing.append(str(cell_dir))
                rows.append({
                    "dt_mode": dt,
                    "lr": lr,
                    "train_batch_size": bs,
                    "seed": args.seed,
                    "min_val_mse": val_mse,
                    "cell_dir": str(cell_dir),
                })

    df = pd.DataFrame(rows)

    # Rank: strict min val/mse, tiebreak on smallest lr, then smallest batch size.
    df["lr_float"] = df["lr"].map(lr_to_float)
    df = df.sort_values(
        by=["min_val_mse", "lr_float", "train_batch_size"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1
    df = df.drop(columns=["lr_float"])

    ranked_csv = phase_dir / "hpo_ranked.csv"
    df.to_csv(ranked_csv, index=False)
    print(f"[aggregate] wrote {ranked_csv}")

    if missing:
        print(f"[aggregate] WARNING: {len(missing)}/{len(rows)} cells had no usable metrics.csv:",
              file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... +{len(missing) - 5} more", file=sys.stderr)

    # Refuse to pick a winner if every cell failed.
    n_usable = int((df["min_val_mse"] != float("inf")).sum())
    if n_usable == 0:
        print("[aggregate] ERROR: no usable cells, aborting before winner pick.",
              file=sys.stderr)
        return 1
    if n_usable < len(rows) // 2:
        print(f"[aggregate] ERROR: only {n_usable}/{len(rows)} cells usable (<50%); "
              f"aborting before winner pick.", file=sys.stderr)
        return 1

    winner = df.iloc[0].to_dict()
    winner_config = {
        "dt_mode": winner["dt_mode"],
        "lr": winner["lr"],
        "train_batch_size": int(winner["train_batch_size"]),
        "min_val_mse": float(winner["min_val_mse"]),
        "source_cell_dir": winner["cell_dir"],
    }

    winner_json = phase_dir / "winner_config.json"
    with open(winner_json, "w") as f:
        json.dump(winner_config, f, indent=2)
    print(f"[aggregate] overall winner: {winner_config}")
    print(f"[aggregate] wrote {winner_json}")

    # Per-dt_mode winners for the ablation table. Every dt_mode must have at
    # least one usable cell — otherwise the confirmation array would try to
    # run a missing config and we'd rather fail here than later.
    winners_by_mode: dict = {}
    for dt in args.dt_modes:
        sub = df[(df["dt_mode"] == dt) & (df["min_val_mse"] != float("inf"))]
        if len(sub) == 0:
            print(f"[aggregate] ERROR: no usable cells for dt_mode={dt}; "
                  f"confirmation would be incomplete, aborting.", file=sys.stderr)
            return 1
        best = sub.iloc[0].to_dict()  # df is already sorted
        winners_by_mode[dt] = {
            "dt_mode": best["dt_mode"],
            "lr": best["lr"],
            "train_batch_size": int(best["train_batch_size"]),
            "min_val_mse": float(best["min_val_mse"]),
            "source_cell_dir": best["cell_dir"],
        }

    by_mode_json = phase_dir / "winners_by_mode.json"
    with open(by_mode_json, "w") as f:
        json.dump(winners_by_mode, f, indent=2)
    print(f"[aggregate] per-dt_mode winners:")
    for dt, cfg in winners_by_mode.items():
        print(f"  {dt}: lr={cfg['lr']}, bs={cfg['train_batch_size']}, "
              f"val/mse={cfg['min_val_mse']:.5f}")
    print(f"[aggregate] wrote {by_mode_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
