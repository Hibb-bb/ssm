"""Summarize a 7.D profiling run produced by ``profile_stages.sh``.

Reads three artifacts from ``--stage_dir``:

    nvidia_smi.csv             : GPU util + memory, polled every 2 s
    profiler_simple-fit.txt    : Lightning's SimpleProfiler per-action wall time
    csv/version_*/metrics.csv  : Lightning step-level metrics (gives steps/sec)

Prints a concise health check vs the README §7.D thresholds:

    Stage A: GPU util >= 95 %  ; aug hot-path < 5 ms/sample
    Stage B: GPU util >= 70 %  ; if not, raise num_workers or pre-cache LOTSA

Example::

    python imts_benchmark/pretrain/scripts/summarize_profile.py \\
        --stage_dir imts_benchmark/pretrain/runs/profile/stage_a
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# Maps the simple profiler's "action name" column to the bucket we
# care about for §7.D analysis.  The exact action names come from
# pytorch_lightning.profilers.SimpleProfiler.summary().
_ACTION_BUCKETS = {
    "data": [
        "[_TrainingEpochLoop].train_dataloader_next",
        "[_DataLoaderIter].next",
        "[Trainer].train_dataloader_next",
    ],
    "fwd": [
        "[LightningModule]MultivariateMambaForecaster.forward",
        "[LightningModule]MultivariateMambaForecaster.training_step",
        "[Strategy]SingleDeviceStrategy.training_step",
    ],
    "bwd": [
        "[Strategy]SingleDeviceStrategy.backward",
        "run_training_batch",
    ],
    "optim": [
        "[Strategy]SingleDeviceStrategy.optimizer_step",
        "[LightningModule]MultivariateMambaForecaster.optimizer_step",
    ],
}


def _read_nvidia_smi(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    util_col = next((c for c in df.columns if "utilization.gpu" in c), None)
    mem_col = next((c for c in df.columns if "memory.used" in c), None)
    if util_col is None or mem_col is None:
        return None
    util = df[util_col].astype(str).str.replace(" %", "").astype(float)
    mem = df[mem_col].astype(str).str.replace(" MiB", "").astype(float)
    # Drop the first 30 s of warmup samples (dataloader spinup, CUDA
    # context init); use steady-state numbers only.
    if len(util) > 15:
        util = util.iloc[15:]
        mem = mem.iloc[15:]
    return {
        "n_samples": int(len(util)),
        "util_mean": float(util.mean()),
        "util_p50": float(util.quantile(0.5)),
        "util_p10": float(util.quantile(0.1)),
        "util_p90": float(util.quantile(0.9)),
        "mem_max_mib": float(mem.max()),
    }


def _read_simple_profile(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text()
    # SimpleProfiler.summary() prints a tab-padded markdown-style table:
    #   |  Action  | Mean duration (s) | Num calls | Total time (s) | Percentage % |
    #   |  Total   |        -          |   39558   |     316.53     |     100 %    |
    #   |  ...     |       0.117       |    806    |      94.5      |    29.85     |
    # Note: column order is Mean / Num / Total / Pct  (NOT Mean / Total / Num).
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 5:
            continue
        name, mean_s, num_s, total_s, pct_s = parts
        if name in ("Action", ""):
            continue  # header row
        try:
            num_i = int(num_s)
            total_f = float(total_s)
            pct_f = float(pct_s.rstrip("%").strip())
            mean_f = float("nan") if mean_s == "-" else float(mean_s)
        except ValueError:
            continue
        rows.append({
            "name": name,
            "mean_s": mean_f,
            "num": num_i,
            "total_s": total_f,
            "pct": pct_f,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("total_s", ascending=False)
    # Bucket totals.
    bucket_totals = {}
    for bucket, names in _ACTION_BUCKETS.items():
        bucket_totals[bucket] = float(
            df.loc[df["name"].isin(names), "total_s"].sum()
        )
    return {
        "rows_top10": df.head(10).to_dict(orient="records"),
        "buckets": bucket_totals,
        "total_fit_s": float(df["total_s"].sum()),
    }


def _read_metrics(stage_dir: Path) -> dict | None:
    """Best-effort read of Lightning step-level metrics for steps/sec."""
    csv_root = stage_dir / "csv"
    if not csv_root.exists():
        return None
    candidates = sorted(csv_root.glob("version_*/metrics.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[-1])
    if "step" not in df.columns:
        return None
    n_steps = int(df["step"].max() + 1) if len(df) else 0
    return {"n_steps_logged": n_steps, "metrics_path": str(candidates[-1])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage_dir", required=True, type=Path,
                   help="e.g. imts_benchmark/pretrain/runs/profile/stage_a")
    p.add_argument(
        "--gpu_target",
        type=float,
        default=None,
        help="GPU util threshold to compare against; defaults to 95 for "
             "stage_a, 70 for stage_b based on directory name.",
    )
    args = p.parse_args()

    stage_name = args.stage_dir.name
    if args.gpu_target is None:
        args.gpu_target = 95.0 if stage_name.endswith("_a") else 70.0

    print(f"\n=== Profile summary for {stage_name} ({args.stage_dir}) ===\n")

    abl_path = args.stage_dir / "ablation.json"
    if abl_path.exists():
        ab = json.loads(abl_path.read_text())
        print(f"  config       : stage={ab.get('stage')} bs={ab.get('batch_size')} "
              f"nw={ab.get('num_workers')} prec={ab.get('precision')}")
        print(f"  git_sha      : {ab.get('git_sha')}")
        print()

    nv = _read_nvidia_smi(args.stage_dir / "nvidia_smi.csv")
    if nv is None:
        print("  nvidia-smi    : <missing>")
    else:
        print(
            f"  GPU util     : mean={nv['util_mean']:.1f}%  "
            f"p10={nv['util_p10']:.1f}%  p50={nv['util_p50']:.1f}%  "
            f"p90={nv['util_p90']:.1f}%  (n={nv['n_samples']})"
        )
        print(f"  GPU mem peak : {nv['mem_max_mib']:.0f} MiB")
        ok = nv["util_mean"] >= args.gpu_target
        print(f"  vs target    : >= {args.gpu_target:.0f}%  -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"               : (suggest: raise num_workers, "
                  f"check augmentation hot path, or pre-cache LOTSA)")
    print()

    sp_path = args.stage_dir / "fit-profiler_simple.txt"
    if not sp_path.exists():
        sp_path = args.stage_dir / "profiler_simple-fit.txt"
    sp = _read_simple_profile(sp_path)
    if sp is None:
        print("  simple_prof  : <missing>  "
              "(check that --profiler simple was passed)")
    else:
        b = sp["buckets"]
        total = sp["total_fit_s"]
        print(f"  fit total    : {total:.1f} s")
        print(f"  buckets (s)  : data={b.get('data', 0):.1f}  "
              f"fwd={b.get('fwd', 0):.1f}  "
              f"bwd={b.get('bwd', 0):.1f}  "
              f"optim={b.get('optim', 0):.1f}")
        if total > 0 and b.get("data"):
            data_pct = 100 * b["data"] / total
            print(f"  data %       : {data_pct:.1f}%  "
                  f"(>30% suggests DataLoader bottleneck)")
        print()
        print("  top-10 by total_s:")
        for r in sp["rows_top10"]:
            print(f"    {r['name'][:60]:60s} {r['total_s']:8.2f} s  "
                  f"({r['num']:5d} calls, mean {r['mean_s']*1000:8.2f} ms)")
    print()

    mets = _read_metrics(args.stage_dir)
    if mets is not None:
        print(f"  steps logged : {mets['n_steps_logged']}  "
              f"(metrics: {mets['metrics_path']})")
    print()


if __name__ == "__main__":
    main()
