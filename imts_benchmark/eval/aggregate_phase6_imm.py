"""Aggregate Mamba-MV TIME-IMM (Phase 6) confirm results into a per-dataset
table comparable to TIME-IMM paper Tables 3-11 ('Without Textual Data' rows).

Walks: <root>/mamba_mv_p10_<regime>/<regime>/<variant>/seed<N>/test_metrics.csv
Emits: <root>/aggregate/phase6_imm_consolidated.csv
       <root>/aggregate/phase6_imm_consolidated.md

Primary metric: per-variable averaged MSE (test_mse_tpg) — this matches
IMM-TSF's lib/evaluation.py exactly: `error_var_avg.sum() / n_avai_var`,
ignoring variates with no observations. So our numbers live in the same
space as the paper's Tables 3-11.

Paper anchors (Tables 3-11 'Without Textual Data' row, t-PatchGNN — strongest
unimodal IMTS baseline) are encoded in PAPER_ANCHORS for side-by-side
comparison.

Usage:
  python -m imts_benchmark.eval.aggregate_phase6_imm \
      [--root output/log/imts_benchmark_v2_imm]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


# t-PatchGNN unimodal MSE/MAE from TIME-IMM paper Tables 3-11
# (last 'tPatchGNN'/'Uni' row in each table).
PAPER_ANCHORS_TPATCHGNN_UNI = {
    "imm_gdelt":        {"mse": 1.0267, "mae": 0.6893},  # Table 3
    "imm_repohealth":   {"mse": 0.6196, "mae": 0.5000},  # Table 4
    "imm_mimic":        {"mse": 0.7250, "mae": 0.5479},  # Table 5
    "imm_fnspid":       {"mse": 0.1247, "mae": 0.2195},  # Table 6
    "imm_clustertrace": {"mse": 0.9671, "mae": 0.7119},  # Table 7
    "imm_studentlife": {"mse": 0.8661, "mae": 0.6672},  # Table 8
    "imm_ilinet":       {"mse": 1.6163, "mae": 0.9255},  # Table 9
    "imm_cesnet":       {"mse": 0.9796, "mae": 0.7713},  # Table 10
    "imm_epa_air":      {"mse": 0.6258, "mae": 0.6022},  # Table 11
}

# Best unimodal baseline per dataset, from the same TIME-IMM tables. Useful as
# a "strongest non-Mamba unimodal" anchor for context.
PAPER_ANCHORS_BEST_UNI = {
    "imm_gdelt":        {"mse": 1.0251, "mae": 0.6842, "model": "TimesNet/TimeMixer"},
    "imm_repohealth":   {"mse": 0.5376, "mae": 0.4081, "model": "DLinear"},
    "imm_mimic":        {"mse": 0.7250, "mae": 0.5457, "model": "tPatchGNN/TimeLLM"},
    "imm_fnspid":       {"mse": 0.1161, "mae": 0.1965, "model": "TimeMixer"},
    "imm_clustertrace": {"mse": 0.8418, "mae": 0.6765, "model": "DLinear"},
    "imm_studentlife": {"mse": 0.8661, "mae": 0.6672, "model": "tPatchGNN"},
    "imm_ilinet":       {"mse": 0.9718, "mae": 0.6456, "model": "CRU/TimeMixer"},
    "imm_cesnet":       {"mse": 0.9462, "mae": 0.7481, "model": "CRU/TimeMixer"},
    "imm_epa_air":      {"mse": 0.5361, "mae": 0.5279, "model": "DLinear"},
}


def stats(vals):
    arr = np.array([
        v for v in vals
        if v is not None
        and not (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v)))
    ])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    if len(arr) == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)


def read_csv_row(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def collect(root: Path):
    """Walk <root>/mamba_mv_p10_<regime>/<regime>/<variant>/seed*/test_metrics.csv.

    The {variant}/ name encodes dt_mode and HPO winner config:
        e.g. concat_lr-1e-4_bs-32_accum-8/
    """
    rows = []
    pattern = root / "mamba_mv_p10_imm_*" / "imm_*" / "*" / "seed*" / "test_metrics.csv"
    csv_paths = sorted(glob.glob(str(pattern)))
    if not csv_paths:
        print(f"[phase6_imm] no test_metrics.csv under {pattern}")

    grouped = defaultdict(list)
    for p in csv_paths:
        parts = Path(p).parts
        # ../mamba_mv_p10_imm_<regime>/imm_<regime>/<variant>/seed<N>/test_metrics.csv
        regime = parts[-4]                          # imm_<name>
        variant = parts[-3]                         # <dt_mode>_lr-...
        seed = parts[-2]                            # seed<N>
        dt_mode = variant.split("_", 1)[0]
        grouped[(regime, dt_mode, variant)].append((seed, p))

    for (regime, dt_mode, variant), seed_paths in grouped.items():
        seeds_data = {"mse": [], "mae": [], "mse_tpg": [], "mae_tpg": [], "r2": []}
        for seed, p in sorted(seed_paths):
            r = read_csv_row(p)
            for k in seeds_data:
                v = r.get(f"test_{k}")
                seeds_data[k].append(
                    float(v) if v not in (None, "", "nan") else None
                )
        row = {
            "regime": regime,
            "dt_mode": dt_mode,
            "variant": variant,
            "n_seeds": len([x for x in seeds_data["mse"] if x is not None]),
        }
        for m, vs in seeds_data.items():
            mean, std, n = stats(vs)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_n"] = n
        rows.append(row)
    return rows


def write_csv(rows, path: Path):
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt_mean_std(mean, std, n):
    if n == 0 or math.isnan(mean):
        return "—"
    if n == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def write_md(rows, path: Path):
    """Emit a Markdown table grouped by regime, with paper anchors."""
    by_regime = defaultdict(list)
    for r in rows:
        by_regime[r["regime"]].append(r)

    lines = [
        "# Mamba-MV on TIME-IMM (Phase 6) — vs paper Tables 3-11 (Uni rows)",
        "",
        "Primary metric: **test_mse_tpg** (per-variable averaged MSE).",
        "Matches IMM-TSF `lib/evaluation.py` exactly so our numbers are directly",
        "comparable to paper Tables 3-11 'Without Textual Data' rows.",
        "",
    ]

    for regime in sorted(by_regime):
        anchor_tpg = PAPER_ANCHORS_TPATCHGNN_UNI.get(regime, {})
        anchor_best = PAPER_ANCHORS_BEST_UNI.get(regime, {})
        lines.append(f"## {regime}")
        lines.append("")
        if anchor_tpg:
            lines.append(
                f"Paper anchor — t-PatchGNN Uni: MSE={anchor_tpg['mse']:.4f} "
                f"MAE={anchor_tpg['mae']:.4f}"
            )
        if anchor_best:
            lines.append(
                f"Paper anchor — best Uni baseline ({anchor_best['model']}): "
                f"MSE={anchor_best['mse']:.4f} MAE={anchor_best['mae']:.4f}"
            )
        lines.append("")
        lines.append("| dt_mode | variant | n | test_mse_tpg | test_mae_tpg | test_mse | test_mae |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(by_regime[regime], key=lambda x: x["dt_mode"]):
            lines.append(
                f"| {r['dt_mode']} | `{r['variant']}` | {r['n_seeds']} "
                f"| {fmt_mean_std(r['mse_tpg_mean'], r['mse_tpg_std'], r['mse_tpg_n'])} "
                f"| {fmt_mean_std(r['mae_tpg_mean'], r['mae_tpg_std'], r['mae_tpg_n'])} "
                f"| {fmt_mean_std(r['mse_mean'], r['mse_std'], r['mse_n'])} "
                f"| {fmt_mean_std(r['mae_mean'], r['mae_std'], r['mae_n'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="output/log/imts_benchmark_v2_imm",
        help="Root containing mamba_mv_p10_imm_*/ confirm output trees",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write the consolidated CSV/MD (default: <root>/aggregate)",
    )
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    rows = collect(root)
    csv_path = out / "phase6_imm_consolidated.csv"
    md_path = out / "phase6_imm_consolidated.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"[phase6_imm] wrote {csv_path}")
    print(f"[phase6_imm] wrote {md_path}")
    print(f"[phase6_imm] {len(rows)} (regime, dt_mode, variant) groups aggregated")


if __name__ == "__main__":
    main()
