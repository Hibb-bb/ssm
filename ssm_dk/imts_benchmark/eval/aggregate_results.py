"""Aggregate test_metrics.csv files across all (model, regime, variant, seed).

Expected directory layout:
  <results_root>/<model>/<regime>/<variant>/seed<S>/test_metrics.csv
  <results_root>/<model>/<regime>/<variant>/seed<S>/per_sample.jsonl  (optional)

Produces:
  <output_csv>  - wide summary (mean +/- std over seeds per cell)
  <long_csv>    - long-format raw rows for downstream plotting / stats

Significance:
  For each (regime, metric) we also run paired Wilcoxon signed-rank tests
  matching Mamba-MV (best dt_mode by val/mse) vs RoMAE on per-sample metrics
  from per_sample.jsonl, reported alongside the summary.
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


def summarize(entries):
    """(model, regime, variant) -> per-metric mean/std over seeds."""
    by_key = defaultdict(list)
    for e in entries:
        key = (e["model"], e["regime"], e["variant"])
        by_key[key].append(e)
    summary_rows = []
    for key, runs in sorted(by_key.items()):
        row = {"model": key[0], "regime": key[1], "variant": key[2], "n_seeds": len(runs)}
        for metric in METRIC_KEYS:
            vals = []
            for r in runs:
                v = r["row"].get(f"test_{metric}")
                if v is not None and v != "":
                    vals.append(float(v))
            if vals:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 6)
                row[f"{metric}_std"] = round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0
        # mean wall-clock and params (fairness checks)
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


def paired_significance(entries, metric: str = "mse"):
    """Paired per-sample Wilcoxon signed-rank: Mamba-MV (best dt_mode) vs RoMAE.

    Best dt_mode chosen as the mode with lowest mean `metric` across seeds on
    the same regime. Pairing is by (regime, seed, item_id). We average over
    seeds within each model before pairing per item_id.
    """
    if not HAS_SCIPY:
        return []
    # Gather per-(model, regime, variant, seed) item_id -> metric
    buckets = defaultdict(dict)  # (model, regime, variant) -> item_id -> list over seeds
    for e in entries:
        key = (e["model"], e["regime"], e["variant"])
        for s in e["per_sample"]:
            if metric in s and "item_id" in s:
                buckets[key].setdefault(s["item_id"], []).append(float(s[metric]))
    # Select best Mamba-MV variant per regime (lowest mean metric).
    regimes = sorted({k[1] for k in buckets.keys()})
    reports = []
    for regime in regimes:
        mamba_keys = [k for k in buckets if k[0] == "mamba_mv" and k[1] == regime]
        romae_keys = [k for k in buckets if k[0] == "romae" and k[1] == regime]
        if not mamba_keys or not romae_keys:
            continue

        def cell_mean(key):
            vals = [v for lst in buckets[key].values() for v in lst]
            return sum(vals) / len(vals) if vals else float("inf")

        best_mamba = min(mamba_keys, key=cell_mean)
        romae_key = romae_keys[0]  # only one "default" variant

        ids = sorted(set(buckets[best_mamba]) & set(buckets[romae_key]))
        if len(ids) < 10:
            reports.append(
                dict(regime=regime, metric=metric, n_paired=len(ids), note="too few pairs")
            )
            continue
        mamba_vals = [statistics.mean(buckets[best_mamba][i]) for i in ids]
        romae_vals = [statistics.mean(buckets[romae_key][i]) for i in ids]
        diffs = [m - r for m, r in zip(mamba_vals, romae_vals)]
        # H_A: Mamba < RoMAE  =>  diff < 0  =>  alternative='less'
        try:
            stat = sp_stats.wilcoxon(mamba_vals, romae_vals, alternative="less")
            reports.append(
                dict(
                    regime=regime,
                    metric=metric,
                    n_paired=len(ids),
                    best_mamba_variant=best_mamba[2],
                    mamba_mean=round(statistics.mean(mamba_vals), 6),
                    romae_mean=round(statistics.mean(romae_vals), 6),
                    mean_diff=round(statistics.mean(diffs), 6),
                    wilcoxon_stat=round(float(stat.statistic), 4),
                    wilcoxon_pvalue=float(stat.pvalue),
                )
            )
        except Exception as exc:  # degenerate input
            reports.append(
                dict(regime=regime, metric=metric, n_paired=len(ids), error=str(exc))
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

    print("\nPaired significance (Mamba-MV vs RoMAE), alternative=less (Mamba wins):")
    for metric in METRIC_KEYS:
        for rep in paired_significance(entries, metric=metric):
            print("  ", rep)

    print("\nSummary preview:")
    for row in summary:
        print(
            "  ",
            row.get("model"), row.get("regime"), row.get("variant"),
            "n_seeds=", row.get("n_seeds"),
            "mse=", row.get("mse_mean"), "+/-", row.get("mse_std"),
            "| n_params=", row.get("n_params"),
        )


if __name__ == "__main__":
    main()
