"""Re-aggregate Phase-5 real-data results, reading BOTH metric formulas
(target-weighted and T-PatchGNN-style variable-averaged) from the new
test_metrics.csv columns.

Inputs:
  output/log/imts_benchmark_v2_real/{s5_p10, romae_p10, mamba_mv_p10}/...
                                       /<variant>/seed*/test_metrics.csv

Outputs:
  output/log/imts_benchmark_v2_real/aggregate/phase5_consolidated_v2.csv
  output/log/imts_benchmark_v2_real/aggregate/phase5_consolidated_v2.md
"""
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real")
OUT = ROOT / "aggregate"
OUT.mkdir(parents=True, exist_ok=True)

# T-PatchGNN paper Table 1 anchors (numbers in raw scale, before scaling)
PAPER = {
    "physionet": {"mse": 4.98e-3, "mae": 3.72e-2, "ms": 1e-3, "as": 1e-2,
                  "mse_disp": "MSE×10⁻³", "mae_disp": "MAE×10⁻²"},
    "activity":  {"mse": 2.66e-3, "mae": 3.15e-2, "ms": 1e-3, "as": 1e-2,
                  "mse_disp": "MSE×10⁻³", "mae_disp": "MAE×10⁻²"},
    "ushcn":     {"mse": 5.00e-1, "mae": 3.08e-1, "ms": 1e-1, "as": 1e-1,
                  "mse_disp": "MSE×10⁻¹", "mae_disp": "MAE×10⁻¹"},
}

DATASETS = tuple(PAPER.keys())

# Where each (model, variant) lives in the output tree.
# variant_dir = the directory name under model_p10/{ds}/<variant_dir>/
SOURCES = [
    ("S5",       "default", "s5_p10",       lambda ds: ["default"]),
    ("RoMAE",    "default", "romae_p10",    lambda ds: ["default"]),
    ("Mamba-MV", "replace", "mamba_mv_p10",
                 lambda ds: {"activity":  ["replace_lr-2e-3_bs-128"],
                             "ushcn":     ["replace_lr-5e-4_bs-64"],
                             "physionet": ["replace_lr-2e-3_bs-8x16"]}.get(ds, [])),
    ("Mamba-MV", "learned", "mamba_mv_p10",
                 lambda ds: {"activity":  ["learned_lr-2e-3_bs-128"],
                             "ushcn":     ["learned_lr-1e-4_bs-256"],
                             "physionet": ["learned_lr-2e-3_bs-8x8"]}.get(ds, [])),
    ("Mamba-MV", "concat",  "mamba_mv_p10",
                 lambda ds: {"activity":  ["concat_lr-5e-4_bs-128"],
                             "ushcn":     ["concat_lr-1e-4_bs-256"],
                             "physionet": ["concat_lr-2e-3_bs-16_accum-4"]}.get(ds, [])),
    ("Mamba-RoPE", "learned", "mamba_rope_p10",
                 lambda ds: {"ushcn":     ["learned_lr-5e-4_bs-64"],
                             "physionet": ["learned_lr-2e-3_bs-8_accum-8"]}.get(ds, [])),
]


def stats(vals):
    arr = np.array([v for v in vals if v is not None
                    and not (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v)))])
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    if len(arr) == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)


def read_csv_row(path: str) -> dict:
    with open(path) as f:
        return next(csv.DictReader(f))


def collect():
    rows = []
    for model, variant, top_dir, variant_dirs_fn in SOURCES:
        for ds in DATASETS:
            for variant_dir in variant_dirs_fn(ds):
                pattern = ROOT / top_dir / ds / variant_dir / "seed*" / "test_metrics.csv"
                seeds_data = {"mse": [], "mae": [], "mse_tpg": [], "mae_tpg": [],
                              "r2": [], "pearson": []}
                paths = sorted(glob.glob(str(pattern)))
                for p in paths:
                    r = read_csv_row(p)
                    for k in seeds_data:
                        v = r.get(f"test_{k}")
                        seeds_data[k].append(float(v) if v not in (None, "", "nan") else None)
                if not paths:
                    continue
                rows.append({
                    "model": model, "variant": variant, "dataset": ds,
                    "n_seeds": len([x for x in seeds_data["mse"] if x is not None]),
                    **{f"{m}_{stat}": s for m in seeds_data
                                       for stat, s in zip(("mean", "std", "n"),
                                                           stats(seeds_data[m]))},
                })
    return rows


def write_csv(rows, path):
    if not rows:
        print("No rows.")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def write_markdown(rows, path):
    """Markdown table comparing both metric formulas, per dataset."""
    lines = [
        "# Phase-5 consolidated results (both metric formulas)\n",
        "## How this file was built\n",
        "**Auto-generated** by `imts_benchmark/eval/aggregate_phase5_v2.py`. To regenerate after new",
        "per-seed runs land:",
        "```bash",
        "${MAMBA_ENV}/bin/python -m imts_benchmark.eval.aggregate_phase5_v2",
        "```",
        "Manual edits to this file will be overwritten on next run — edit the docstring/`write_markdown`",
        "in the aggregator instead.\n",
        "**Inputs.** Per-seed `test_metrics.csv` files written by each trainer at",
        "`mamba_mv.train_mv` / `mamba_rope.train_mamba_rope` / `s5_forecaster.train_s5` /",
        "`romae_forecaster.train_romae`. The aggregator walks `output/log/imts_benchmark_v2_real/",
        "<top_dir>/<dataset>/<variant_dir>/seed*/test_metrics.csv` for the (model, variant, dataset)",
        "tuples listed in the `SOURCES` dict at the top of the script. Add a row there to include",
        "a new model/variant in this report; missing per-seed dirs are silently skipped.\n",
        "**Methodology.** All bottom-block runs use `patience=10` early-stopping on `val/MSE` (the",
        "checkpoint with lowest val/MSE is loaded for test). 5 seeds (1-5) per cell. Hyperparameters",
        "are HPO winners from `hpo_mamba_*` sweeps (val/MSE-best of a 9- to 27-cell `lr × eff_bs ×",
        "dt_mode` grid at seed=1, then locked for the 5-seed confirm). The `SOURCES` dict in the",
        "aggregator names the winning `lr-X_bs-Y[_accum-Z]` directory per (model, variant, dataset).\n",
        "**Metrics.**",
        "- `MSE` / `MAE` (\"ours, target-weighted\"): aggregated as `Σ_n SS / Σ_n count` across all",
        "  test samples — every prediction contributes equally regardless of sample size.",
        "- `MSE_tpg` / `MAE_tpg` (**T-PatchGNN convention**): per-sample `(1/V_avail) Σ_d (1/N_d)",
        "  Σ_n (y_dn − ŷ_dn)²` — each variate weighted equally per sample, then averaged across",
        "  samples. **This is the headline metric** because it matches the published T-PatchGNN",
        "  paper (Zhang et al., ICML 2024, Table 1) and lets us cite directly against their numbers.",
        "- `R²` is per-variable global R² (`shared_config/global_metrics.py`), not the older",
        "  per-sample R² that collapsed on USHCN's near-constant variates. Strongly negative R²",
        "  on PhysioNet/USHCN reflects pathological variance ratios on a few rare-event variates,",
        "  not model failure — see `docs/r2_pathology.md` (if present) for details.\n",
        "**Scaling.** PhysioNet/Activity report `MSE×10⁻³` and `MAE×10⁻²` to match the paper's",
        "table-1 scale; USHCN reports `MSE×10⁻¹` and `MAE×10⁻¹` because USHCN values are not",
        "normalised to [0,1] (T-PatchGNN convention; see `convert_tpatchgnn_data.py:298`).\n",
        "**Per-dataset HPO winners (locked in `SOURCES`):**\n",
        "| Model | Variant | PhysioNet | Activity | USHCN |",
        "|---|---|---|---|---|",
        "| S5 | default | fair-defaults | fair-defaults | fair-defaults |",
        "| RoMAE | default | fair-defaults | fair-defaults | fair-defaults |",
        "| Mamba-MV | replace | lr=2e-3, eff_bs=128 (8×16) | lr=2e-3, bs=128 | lr=5e-4, bs=64 |",
        "| Mamba-MV | learned | lr=2e-3, eff_bs=64 (8×8)  | lr=2e-3, bs=128 | lr=1e-4, bs=256 |",
        "| Mamba-MV | concat  | lr=2e-3, eff_bs=64 (16×4) | lr=5e-4, bs=128 | lr=1e-4, bs=256 |",
        "| Mamba-RoPE | learned | lr=2e-3, eff_bs=64 (8×8) | (not run) | lr=5e-4, bs=64 |",
        "\nAll Mamba-MV runs use `grid_K=256` for PhysioNet, `grid_K=128` for Activity/USHCN.\n",
    ]
    for ds in DATASETS:
        cfg = PAPER[ds]
        lines.append(f"\n## {ds.upper()} ({cfg['mse_disp']} / {cfg['mae_disp']})\n")
        lines.append(f"| Model | Variant | n | MSE (ours, target-wgt) | **MSE (TPG, variable-avg)** | MAE (ours) | **MAE (TPG)** | R² |")
        lines.append(f"|---|---|---|---|---|---|---|---|")
        rs = [r for r in rows if r["dataset"] == ds]
        rs.sort(key=lambda r: r["mse_tpg_mean"] if not math.isnan(r["mse_tpg_mean"]) else 1e9)
        for r in rs:
            mse  = f"{r['mse_mean']/cfg['ms']:.2f}±{r['mse_std']/cfg['ms']:.2f}"
            mtpg = f"**{r['mse_tpg_mean']/cfg['ms']:.2f}±{r['mse_tpg_std']/cfg['ms']:.2f}**"
            mae  = f"{r['mae_mean']/cfg['as']:.2f}±{r['mae_std']/cfg['as']:.2f}"
            atpg = f"**{r['mae_tpg_mean']/cfg['as']:.2f}±{r['mae_tpg_std']/cfg['as']:.2f}**"
            r2   = f"{r['r2_mean']:+.3f}" if not math.isnan(r['r2_mean']) else "—"
            lines.append(f"| {r['model']} | {r['variant']} | {r['n_seeds']} | {mse} | {mtpg} | {mae} | {atpg} | {r2} |")
        # T-PatchGNN paper reference
        lines.append(f"| T-PatchGNN (paper) | — | 5 | — | **{cfg['mse']/cfg['ms']:.2f}** | — | **{cfg['mae']/cfg['as']:.2f}** | — |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    rows = collect()
    print(f"\nCollected {len(rows)} (model, variant, dataset) groups.\n")
    write_csv(rows, OUT / "phase5_consolidated_v2.csv")
    write_markdown(rows, OUT / "phase5_consolidated_v2.md")

    # quick on-screen summary
    print("\n--- Quick summary (variable-averaged MSE, the headline) ---")
    for ds in DATASETS:
        cfg = PAPER[ds]
        rs_ds = [r for r in rows if r["dataset"] == ds]
        rs_ds.sort(key=lambda r: r["mse_tpg_mean"])
        print(f"\n{ds.upper()} ({cfg['mse_disp']}):")
        for r in rs_ds:
            tpg = r["mse_tpg_mean"] / cfg["ms"]
            ours = r["mse_mean"]    / cfg["ms"]
            gap = (r["mse_tpg_mean"] - r["mse_mean"]) / r["mse_mean"] * 100 if r["mse_mean"] else float("nan")
            vs_paper = (r["mse_tpg_mean"] - cfg["mse"]) / cfg["mse"] * 100
            print(f"  {r['model']:<10} {r['variant']:<10} n={r['n_seeds']}  "
                  f"TPG={tpg:.2f}  ours={ours:.2f}  gap(TPG−ours)={gap:+.1f}%  vs T-PatchGNN={vs_paper:+.1f}%")


if __name__ == "__main__":
    main()
