"""Compare the HPO-tuned Mamba-MV winner against the existing S5 / RoMAE /
pre-HPO Mamba-MV benchmark rows on multisin_high_irreg.

For each phase (3, 4-2):
  - Reads the 5 seed-runs' test_metrics.csv under
    <LOG_BASE>/hpo_mamba_mv/{phase}_winner/multisin_high_irreg/<tag>/seed*/
  - Aggregates to a single row (mse_mean, mse_std, etc.) matching the
    schema in docs/phase*_results/phase*_summary_wide.csv.
  - Appends the row to that CSV (does NOT overwrite — treats the tuned row
    as a new 'variant').
  - Writes a markdown table comparing the tuned row against the existing
    S5/RoMAE rows for the same regime, to
    docs/HPO_COMPARISON_phase{3,4_2}_high_irreg.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/uni2ts_hongyu/ssm_dk")
LOG_BASE = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2/hpo_mamba_mv")
DOCS = REPO / "imts_benchmark" / "docs"


def aggregate_seeds(winner_dir: Path, regime: str, tag: str, seeds=(1, 2, 3, 4, 5)) -> dict:
    """Collate seed-wise test_metrics.csv into one aggregated row."""
    vals = {k: [] for k in ("mse", "mae", "r2", "pearson", "wall_fit_sec")}
    n_params = None
    for s in seeds:
        csv = winner_dir / regime / tag / f"seed{s}" / "test_metrics.csv"
        if not csv.exists():
            print(f"[compare] WARNING: missing {csv}", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        row = df.iloc[0]
        vals["mse"].append(float(row["test_mse"]))
        vals["mae"].append(float(row["test_mae"]))
        vals["r2"].append(float(row["test_r2"]))
        vals["pearson"].append(float(row["test_pearson"]))
        vals["wall_fit_sec"].append(float(row["wall_fit_sec"]))
        n_params = int(row["n_params"])

    def m(k): return float(np.mean(vals[k])) if vals[k] else float("nan")
    def s(k): return float(np.std(vals[k])) if len(vals[k]) > 1 else 0.0

    return {
        "n_seeds": len(vals["mse"]),
        "mse_mean": m("mse"), "mse_std": s("mse"),
        "mae_mean": m("mae"), "mae_std": s("mae"),
        "r2_mean":  m("r2"),  "r2_std":  s("r2"),
        "pearson_mean": m("pearson"), "pearson_std": s("pearson"),
        "wall_fit_sec_mean": m("wall_fit_sec"),
        "n_params": n_params if n_params is not None else 0,
    }


def compare_phase(phase_key: str, data_dir_name: str, summary_csv_path: Path) -> None:
    """Produce per-dt_mode comparison for one phase (learned vs replace vs concat)."""
    winners_json = LOG_BASE / phase_key / "winners_by_mode.json"
    if not winners_json.exists():
        print(f"[compare] ERROR: missing {winners_json}", file=sys.stderr)
        return
    winners = json.loads(winners_json.read_text())

    winner_dir = LOG_BASE / f"{phase_key}_winner"
    regime = "multisin_high_irreg"

    new_rows = []
    cfg_summary = {}
    for dt_mode, cfg in winners.items():
        lr, bs = cfg["lr"], cfg["train_batch_size"]
        tag = f"dt-{dt_mode}_lr-{lr}_bs-{bs}"
        agg = aggregate_seeds(winner_dir, regime, tag)
        if agg["n_seeds"] == 0:
            print(f"[compare] WARNING: dt_mode={dt_mode} has no seed runs at {tag}; "
                  f"skipping.", file=sys.stderr)
            continue
        new_rows.append({
            "model": "mamba_mv",
            "regime": regime,
            "variant": f"hpo_tuned_{dt_mode}",
            **agg,
        })
        cfg_summary[dt_mode] = f"lr={lr}, bs={bs}"

    if not new_rows:
        print(f"[compare] ERROR: no dt_mode had usable seed runs for {phase_key}.",
              file=sys.stderr)
        return

    # Drop any prior tuned rows for these dt_modes (idempotent re-run).
    tuned_variants = {r["variant"] for r in new_rows}
    if summary_csv_path.exists():
        existing = pd.read_csv(summary_csv_path)
        existing = existing[
            ~((existing["model"] == "mamba_mv") & (existing["variant"].isin(tuned_variants)))
        ]
        out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        out = pd.DataFrame(new_rows)
    out.to_csv(summary_csv_path, index=False)
    print(f"[compare] appended {len(new_rows)} tuned rows to {summary_csv_path}")

    # Build the comparison markdown — only high_irreg rows, sorted by MSE.
    high = out[out["regime"] == regime].copy()
    high = high.sort_values(by="mse_mean")
    md_path = DOCS / f"HPO_COMPARISON_{phase_key}_high_irreg.md"
    with open(md_path, "w") as f:
        f.write(f"# HPO winners vs baseline — {phase_key} × high_irreg\n\n")
        f.write(f"Per-dt_mode winners (picked by min val/mse on HPO sweep, "
                f"confirmed over 5 seeds):\n\n")
        for dt_mode, s in cfg_summary.items():
            f.write(f"- `{dt_mode}`: {s}\n")
        f.write("\n")
        f.write("| Rank | Model | Variant | n_seeds | MSE (mean ± std) | R² | Pearson | Params |\n")
        f.write("|------|-------|---------|---------|------------------|-----|---------|--------|\n")
        for i, r in enumerate(high.itertuples(index=False), 1):
            mse = f"{r.mse_mean:.5f} ± {r.mse_std:.5f}"
            f.write(f"| {i} | {r.model} | {r.variant} | {int(r.n_seeds)} | {mse} "
                    f"| {r.r2_mean:.3f} | {r.pearson_mean:.3f} | {int(r.n_params):,} |\n")
    print(f"[compare] wrote {md_path}")


def main() -> int:
    compare_phase(
        "phase3",
        "data_correct_async",
        DOCS / "phase3_results" / "phase3_summary_wide.csv",
    )
    compare_phase(
        "phase4_2",
        "data_correct_gap_random",
        DOCS / "phase4_2_results" / "phase4_2_summary_wide.csv",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
