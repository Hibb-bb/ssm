"""Aggregate test_metrics.csv + per_sample.jsonl across all (model, regime, variant, seed).

Expected directory layout:
  <results_root>/<model>/<regime>/<variant>/seed<S>/test_metrics.csv
  <results_root>/<model>/<regime>/<variant>/seed<S>/per_sample.jsonl  (optional)

Produces:
  <output_csv>  - wide summary (mean +/- std over seeds per cell). Includes
                  target-weighted per-variate MSE/MAE derived from
                  per_sample.jsonl when the per-variate fields are present.
                  For Phase 4 data, also includes gap-region metrics for v3.
  <long_csv>    - long-format raw rows for downstream plotting / stats

Significance:
  For each (regime, metric) we also run paired Wilcoxon signed-rank tests
  matching Mamba-MV (best dt_mode by val/mse) vs each baseline on per-sample
  metrics from per_sample.jsonl, reported alongside the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


METRIC_KEYS = ("mse", "mae", "r2", "pearson")
# Per-variate target-weighted aggregation fields saved in per_sample.jsonl by
# the 4 wrappers (added 2026-04-22). For Phase 2 runs that predate these
# fields, we fall back to sample-averaged mse_v{d} / mae_v{d}.
PER_VAR_TW_FIELDS = ("ss_res_v", "abs_err_sum_v", "count_v")
# Phase 4-only gap-region fields (in/out-of-gap restricted to variate 2 = v3).
GAP_FIELDS = ("ss_res_v2_in_gap", "abs_err_sum_v2_in_gap", "count_v2_in_gap",
              "ss_res_v2_out_gap", "abs_err_sum_v2_out_gap", "count_v2_out_gap")


def walk_results(root: Path):
    """Yield dicts (model, regime, variant, seed, csv_row, per_sample_list)."""
    for csv_path in root.rglob("test_metrics.csv"):
        parts = csv_path.relative_to(root).parts
        if len(parts) < 5:
            continue
        model, regime, variant, seed_dir, _ = parts[-5:]
        if not seed_dir.startswith("seed"):
            continue
        seed = int(seed_dir[len("seed"):])

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        row = rows[0]

        per_sample_path = csv_path.parent / "per_sample.jsonl"
        per_sample = []
        if per_sample_path.exists():
            with open(per_sample_path) as f:
                for line in f:
                    per_sample.append(json.loads(line))

        yield dict(
            model=model,
            regime=regime,
            variant=variant,
            seed=seed,
            row=row,
            per_sample=per_sample,
            path=str(csv_path),
        )


def _per_variate_target_weighted(per_sample_across_seeds: list[dict], n_vars: int = 3):
    """Compute target-weighted per-variate MSE/MAE by summing ss_res_v{d}
    across all samples (across seeds) and dividing by total count.

    If the new target-weighted fields are absent (Phase 2 runs that predate
    them), fall back to sample-averaging `mse_v{d}` across samples.

    Returns dict with keys `mse_v{d}_tw`, `mae_v{d}_tw`, `mse_v{d}_samp`,
    `mae_v{d}_samp`, `count_v{d}_total`.
    """
    out: dict[str, float] = {}
    for d in range(n_vars):
        # Target-weighted path (ss_res_v{d}, abs_err_sum_v{d}, count_v{d}).
        total_ss = 0.0
        total_ae = 0.0
        total_n = 0
        # Sample-averaged fallback path (mse_v{d}, mae_v{d}).
        mse_list: list[float] = []
        mae_list: list[float] = []
        for s in per_sample_across_seeds:
            if f"ss_res_v{d}" in s and f"count_v{d}" in s:
                total_ss += float(s[f"ss_res_v{d}"])
                total_ae += float(s.get(f"abs_err_sum_v{d}", 0.0))
                total_n += int(s[f"count_v{d}"])
            if f"mse_v{d}" in s:
                mse_list.append(float(s[f"mse_v{d}"]))
            if f"mae_v{d}" in s:
                mae_list.append(float(s[f"mae_v{d}"]))

        if total_n > 0:
            out[f"mse_v{d}_tw"] = round(total_ss / total_n, 6)
            out[f"mae_v{d}_tw"] = round(total_ae / total_n, 6)
            out[f"count_v{d}_total"] = total_n
        if mse_list:
            out[f"mse_v{d}_samp"] = round(statistics.mean(mse_list), 6)
        if mae_list:
            out[f"mae_v{d}_samp"] = round(statistics.mean(mae_list), 6)
    return out


def _phase4_gap_region(per_sample_across_seeds: list[dict]):
    """Compute target-weighted MSE/MAE restricted to in-gap vs out-gap, per
    gapped-variate stratum (Phase 4 only).

    Phase 4-1 (fixed_v3): all samples have d=2, output keys mse_v2_in_gap_tw etc.
    Phase 4-2 (random_uniform): samples stratified over d ∈ {0,1,2}, output
    keys mse_v{d}_in_gap_tw for each d present. Absent fields -> empty dict.
    """
    out: dict[str, float] = {}
    # Detect which gapped variate indices actually appear.
    for d in range(3):
        for label in ("in_gap", "out_gap"):
            total_ss = 0.0
            total_ae = 0.0
            total_n = 0
            for s in per_sample_across_seeds:
                k_ss = f"ss_res_v{d}_{label}"
                k_ae = f"abs_err_sum_v{d}_{label}"
                k_n = f"count_v{d}_{label}"
                if k_ss in s and k_n in s:
                    total_ss += float(s[k_ss])
                    total_ae += float(s.get(k_ae, 0.0))
                    total_n += int(s[k_n])
            if total_n > 0:
                out[f"mse_v{d}_{label}_tw"] = round(total_ss / total_n, 6)
                out[f"mae_v{d}_{label}_tw"] = round(total_ae / total_n, 6)
                out[f"count_v{d}_{label}_total"] = total_n
    return out


def summarize(entries):
    """(model, regime, variant) -> per-metric mean/std over seeds + per-variate
    target-weighted aggregates over pooled per_sample records across seeds."""
    by_key = defaultdict(list)
    for e in entries:
        key = (e["model"], e["regime"], e["variant"])
        by_key[key].append(e)
    summary_rows = []
    for key, runs in sorted(by_key.items()):
        row = {"model": key[0], "regime": key[1], "variant": key[2], "n_seeds": len(runs)}
        # Global metrics (from test_metrics.csv): mean ± std across seeds.
        for metric in METRIC_KEYS:
            vals = []
            for r in runs:
                v = r["row"].get(f"test_{metric}")
                if v is not None and v != "":
                    vals.append(float(v))
            if vals:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 6)
                row[f"{metric}_std"] = round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0
        # Per-variate target-weighted aggregates (pool per-sample records
        # across seeds -> single target-weighted number per variate).
        pooled = [s for r in runs for s in r["per_sample"]]
        row.update(_per_variate_target_weighted(pooled))
        # Phase 4 gap-region aggregates (empty for Phase 2/3).
        row.update(_phase4_gap_region(pooled))
        # Mean wall-clock and params (fairness checks).
        wall = [float(r["row"].get("wall_fit_sec", 0.0)) for r in runs if r["row"].get("wall_fit_sec")]
        params = [float(r["row"].get("n_params", 0.0)) for r in runs if r["row"].get("n_params")]
        if wall:
            row["wall_fit_sec_mean"] = round(statistics.mean(wall), 1)
        if params:
            row["n_params"] = int(statistics.mean(params))
        summary_rows.append(row)
    return summary_rows


def long_rows(entries):
    rows = []
    for e in entries:
        for metric in METRIC_KEYS:
            v = e["row"].get(f"test_{metric}")
            if v is None or v == "":
                continue
            rows.append(
                dict(
                    model=e["model"],
                    regime=e["regime"],
                    variant=e["variant"],
                    seed=e["seed"],
                    metric=metric,
                    value=float(v),
                    n_params=int(float(e["row"].get("n_params", 0))),
                    wall_fit_sec=float(e["row"].get("wall_fit_sec", 0.0)),
                )
            )
    return rows


def paired_significance(entries, metric: str = "mse",
                        challenger: str = "mamba_mv",
                        opponent: str = "romae"):
    """Paired per-sample Wilcoxon signed-rank: challenger (best variant) vs
    opponent on per-sample `metric` from per_sample.jsonl.

    Best variant of challenger is the one with lowest mean `metric` across
    seeds on the same regime. Pairing is by (regime, item_id) after averaging
    across seeds within each model.
    """
    if not HAS_SCIPY:
        return []
    buckets = defaultdict(dict)  # (model, regime, variant) -> item_id -> list
    for e in entries:
        key = (e["model"], e["regime"], e["variant"])
        for s in e["per_sample"]:
            if metric in s and "item_id" in s:
                buckets[key].setdefault(s["item_id"], []).append(float(s[metric]))
    regimes = sorted({k[1] for k in buckets.keys()})
    reports = []
    for regime in regimes:
        ch_keys = [k for k in buckets if k[0] == challenger and k[1] == regime]
        op_keys = [k for k in buckets if k[0] == opponent and k[1] == regime]
        if not ch_keys or not op_keys:
            continue

        def cell_mean(key):
            vals = [v for lst in buckets[key].values() for v in lst]
            return sum(vals) / len(vals) if vals else float("inf")

        best_ch = min(ch_keys, key=cell_mean)
        op_key = op_keys[0]

        ids = sorted(set(buckets[best_ch]) & set(buckets[op_key]))
        if len(ids) < 10:
            reports.append(
                dict(regime=regime, metric=metric, challenger=challenger, opponent=opponent,
                     n_paired=len(ids), note="too few pairs")
            )
            continue
        ch_vals = [statistics.mean(buckets[best_ch][i]) for i in ids]
        op_vals = [statistics.mean(buckets[op_key][i]) for i in ids]
        diffs = [c - o for c, o in zip(ch_vals, op_vals)]
        try:
            stat = sp_stats.wilcoxon(ch_vals, op_vals, alternative="less")
            reports.append(
                dict(
                    regime=regime,
                    metric=metric,
                    challenger=challenger,
                    opponent=opponent,
                    n_paired=len(ids),
                    best_challenger_variant=best_ch[2],
                    challenger_mean=round(statistics.mean(ch_vals), 6),
                    opponent_mean=round(statistics.mean(op_vals), 6),
                    mean_diff=round(statistics.mean(diffs), 6),
                    wilcoxon_stat=round(float(stat.statistic), 4),
                    wilcoxon_pvalue=float(stat.pvalue),
                )
            )
        except Exception as exc:
            reports.append(
                dict(regime=regime, metric=metric, challenger=challenger, opponent=opponent,
                     n_paired=len(ids), error=str(exc))
            )
    return reports


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        print(f"  (no rows to write to {path})")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  Wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--long_csv", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    entries = list(walk_results(root))
    print(f"Found {len(entries)} completed runs under {root}")

    summary = summarize(entries)
    long_table = long_rows(entries)

    write_csv(Path(args.output_csv), summary)
    write_csv(Path(args.long_csv), long_table)

    # Mamba-MV vs each baseline (paired Wilcoxon, alt=less means Mamba wins).
    print("\nPaired significance (Mamba-MV vs each baseline), alternative=less (Mamba wins):")
    for opponent in ("romae", "mtan", "s5"):
        for metric in METRIC_KEYS:
            for rep in paired_significance(entries, metric=metric, opponent=opponent):
                print("  ", rep)

    print("\nSummary preview:")
    for row in summary:
        cols = [row.get("model"), row.get("regime"), row.get("variant"),
                f"n_seeds={row.get('n_seeds')}",
                f"mse={row.get('mse_mean')}±{row.get('mse_std')}"]
        if "mse_v0_tw" in row:
            cols.append(f"v0_tw={row.get('mse_v0_tw')}")
            cols.append(f"v1_tw={row.get('mse_v1_tw')}")
            cols.append(f"v2_tw={row.get('mse_v2_tw')}")
        if "mse_v2_in_gap_tw" in row:
            cols.append(f"v2_in_gap={row.get('mse_v2_in_gap_tw')}")
            cols.append(f"v2_out_gap={row.get('mse_v2_out_gap_tw')}")
        print("  ", " ".join(str(c) for c in cols))


if __name__ == "__main__":
    main()
