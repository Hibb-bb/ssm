"""CPU-only diagnostic: time the LOTSA loading + degradation pipeline.

Stage B's GPU-side profile (see pretrain/runs/profile/stage_b/) showed
``train_dataloader_next`` averaging 3.3 s per batch at bs=32 nw=12
while GPU compute was only ~110 ms — i.e. the data path is the
bottleneck.  This script isolates the data path, with NO model and NO
GPU, so we can localize the slow stage to either:

  - HF Arrow random-access read    (via ``_HFArrowIndexer.getitem``);
  - shape massage to (V, T);                            (cheap, np);
  - ``LOTSAToIrregular`` degradation pipeline      (regime+jitter+...);
  - the per-batch task-sampler overhead in mixed_dataset / collate.

After the §7.D.4 indexer fix this should be entirely degrade-bound.
Outputs a table and a JSON sidecar so it's reproducible.

Usage::

    python imts_benchmark/pretrain/scripts/profile_lotsa_pipeline.py \
        --n_samples 200 --max_per_source 50

Wall-clock cost: ~1-2 minutes on the current LOTSA disk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import yaml

# Local imports.
import sys
_THIS_DIR = Path(__file__).resolve().parent
_PRETRAIN_DIR = _THIS_DIR.parent
_REPO_ROOT = _PRETRAIN_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from imts_benchmark.pretrain.sources import (
    LOTSAHFSource,
    discover_lotsa_sources,
)
from imts_benchmark.synthetic_data.degradation import LOTSAToIrregular


def _stats(name: str, samples: list[float]) -> dict:
    if not samples:
        return {"name": name, "n": 0}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "name": name,
        "n": int(len(arr)),
        "mean_ms": float(arr.mean() * 1000),
        "p50_ms": float(np.percentile(arr, 50) * 1000),
        "p90_ms": float(np.percentile(arr, 90) * 1000),
        "p99_ms": float(np.percentile(arr, 99) * 1000),
        "total_s": float(arr.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sources_yaml",
        type=Path,
        default=_PRETRAIN_DIR / "configs/sources.yaml",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Total samples to draw across LOTSA sources.",
    )
    p.add_argument(
        "--max_per_source",
        type=int,
        default=50,
        help="Cap per LOTSA sub-dataset so we cover diversity.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_PRETRAIN_DIR / "runs/profile/lotsa_pipeline.json",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.sources_yaml.read_text())
    lotsa_root = cfg["lotsa_root"]
    include = cfg.get("lotsa_include")
    exclude = cfg.get("lotsa_exclude") or []

    print(f"[discover] lotsa_root = {lotsa_root}")
    sources = discover_lotsa_sources(
        root=lotsa_root,
        include=include,
        exclude=exclude,
    )
    print(f"[discover] {len(sources)} LOTSA sources discovered")
    if not sources:
        raise SystemExit("No LOTSA sources discovered; aborting.")

    transform = LOTSAToIrregular(
        min_length=128,
        max_length=512,
        max_variates=20,
        source_tag_prefix="lotsa",
        apply_jitter=False,
    )

    # Buckets: keep per-stage timings separately.
    t_open = []         # one-off __post_init__ already happened in discover; but we time the random __getitem__
    t_random_get = []   # ``self._ds[i]`` access in LOTSAHFSource.__iter__
    t_to_2d = []        # _to_2d on returned target
    t_degrade = []      # LOTSAToIrregular.__call__
    t_yield = []        # full source.__iter__ yield (sum of above + dict construction)
    per_source_yield: dict[str, list[float]] = {}
    per_source_degrade: dict[str, list[float]] = {}

    rng = np.random.default_rng(42)
    n_per_src = max(1, args.max_per_source)

    # Strategy: walk EVERY source for a fixed sample budget per source so
    # the slowest offenders cannot hide behind the cheap-but-frequent ones.
    src_order = list(range(len(sources)))
    rng.shuffle(src_order)
    samples_taken = 0
    print(f"[profile] budget = {n_per_src} samples per source × {len(src_order)} sources")

    print("[profile] starting per-source loop ...")
    overall_start = time.perf_counter()
    for si in src_order:
        if samples_taken >= args.n_samples:
            break
        src: LOTSAHFSource = sources[si]
        if not isinstance(src, LOTSAHFSource):
            continue
        ds = src._ds
        L = src._len
        if L == 0:
            continue
        per_source_yield.setdefault(src.name, [])
        per_source_degrade.setdefault(src.name, [])

        for _ in range(n_per_src):
            if samples_taken >= args.n_samples:
                break
            idx = int(rng.integers(0, L))

            # Step 1: HF random access through the new zero-copy
            # indexer (see §7.D.3).  ``ex['target']`` is a numpy view
            # of shape (T,) univariate or (V, T) multivariate.
            t0 = time.perf_counter()
            try:
                ex = src._indexer.getitem(idx)
            except Exception:
                continue
            t1 = time.perf_counter()
            t_random_get.append(t1 - t0)

            target = ex.get("target")
            if target is None:
                continue

            # Step 2: shape massage to (V, T).  This is the analog of
            # the pre-fix _to_2d but trivial because the indexer
            # already returns ndarray.
            t2a = time.perf_counter()
            if target.ndim == 1:
                arr = target[None, :]
            else:
                arr = target
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            t2b = time.perf_counter()
            t_to_2d.append(t2b - t2a)
            if arr.size == 0 or arr.shape[1] < 2:
                continue

            # Step 3: degradation.
            entry = {
                "target": arr,
                "freq": str(ex.get("freq", "H")),
                "item_id": str(ex.get("item_id", idx)),
                "source_name": src.name,
            }
            t3a = time.perf_counter()
            out = transform(entry)
            t3b = time.perf_counter()
            t_degrade.append(t3b - t3a)
            per_source_degrade[src.name].append(t3b - t3a)

            yield_total = t3b - t0
            t_yield.append(yield_total)
            per_source_yield[src.name].append(yield_total)
            samples_taken += 1

            if samples_taken % 50 == 0:
                elapsed = time.perf_counter() - overall_start
                print(
                    f"  ... {samples_taken}/{args.n_samples} samples,"
                    f" {elapsed:.1f}s elapsed,"
                    f" mean yield {1000 * mean(t_yield):.1f} ms/sample"
                )

    overall_s = time.perf_counter() - overall_start
    if samples_taken == 0:
        raise SystemExit("Got 0 samples; check sources_yaml / lotsa_root.")

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_samples": samples_taken,
        "n_sources_used": len(per_source_yield),
        "wall_s": overall_s,
        "stage": {
            "hf_random_get": _stats("hf_random_get", t_random_get),
            "to_2d":         _stats("to_2d",         t_to_2d),
            "degrade":       _stats("degrade",       t_degrade),
            "yield_total":   _stats("yield_total",   t_yield),
        },
        "per_source_yield_ms": {
            k: {
                "n": len(v),
                "mean_ms": 1000 * mean(v) if v else 0.0,
                "p50_ms":  1000 * median(v) if v else 0.0,
            }
            for k, v in sorted(per_source_yield.items())
        },
        "per_source_degrade_ms": {
            k: {
                "n": len(v),
                "mean_ms": 1000 * mean(v) if v else 0.0,
                "p50_ms":  1000 * median(v) if v else 0.0,
            }
            for k, v in sorted(per_source_degrade.items())
        },
    }
    args.output.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 78)
    print(f"LOTSA pipeline profile  ({samples_taken} samples,"
          f" {summary['n_sources_used']} sources, wall {overall_s:.1f}s)")
    print("=" * 78)
    print()
    print(f"{'Stage':<20s} {'mean':>10s} {'p50':>10s} {'p90':>10s} {'p99':>10s} {'total':>10s}")
    for k, st in summary["stage"].items():
        if not st.get("n"):
            continue
        print(
            f"{k:<20s} {st['mean_ms']:>9.2f}  {st['p50_ms']:>9.2f}  "
            f"{st['p90_ms']:>9.2f}  {st['p99_ms']:>9.2f}  "
            f"{st['total_s']:>9.2f}s"
        )

    # Throughput projection: with 12 workers, what's our ceiling?
    if t_yield:
        per_sample_s = mean(t_yield)
        per_worker_qps = 1.0 / per_sample_s
        # bs=32 means 32 samples/batch; 12 workers → 12*per_worker_qps samples/sec
        bs = 32
        nw = 12
        total_qps = nw * per_worker_qps
        batch_qps = total_qps / bs
        target_step_s = 0.110  # GPU step time from Stage A profile
        target_batch_qps = 1.0 / target_step_s
        print()
        print(f"Throughput math (per-sample wall = {per_sample_s*1000:.1f} ms):")
        print(f"  per-worker  : {per_worker_qps:.1f} samples/sec")
        print(f"  bs={bs}, nw={nw}: {total_qps:.1f} samples/sec → {batch_qps:.2f} batches/sec")
        print(f"  GPU target  : {target_batch_qps:.2f} batches/sec ({target_step_s*1000:.0f} ms/step)")
        gap = target_batch_qps / batch_qps if batch_qps else 999
        print(f"  GPU starves by {gap:.1f}× → need {int(np.ceil(nw * gap))} workers,"
              f" or pre-cache to cut per-sample wall by {gap:.1f}×")

    print()
    # Top-5 slowest sources by mean yield wall.
    pairs = sorted(
        ((k, v["mean_ms"], v["n"]) for k, v in summary["per_source_yield_ms"].items()),
        key=lambda x: -x[1],
    )
    if pairs:
        print("Top-5 slowest LOTSA sources (by mean yield wall):")
        for k, ms, n in pairs[:5]:
            print(f"  {k:<55s}  {ms:7.2f} ms/sample  (n={n})")
        print()
        print("Top-5 fastest LOTSA sources:")
        for k, ms, n in pairs[-5:][::-1]:
            print(f"  {k:<55s}  {ms:7.2f} ms/sample  (n={n})")

    print()
    print(f"JSON sidecar : {args.output}")


if __name__ == "__main__":
    main()
