"""Pick HPO winner per (model, dataset) from IMM-TSF baseline architectural sweep.

Walks
  <root>/<MODEL>/<DS_NAME>/<TAG>/seed1/result.json
and writes
  <root>/<MODEL>/<DS_NAME>/winner.json
containing the architectural HPs of the cell with lowest val/MSE (or test/MSE
fallback — see below).

IMM-TSF's `trainable()` returns the test_res evaluated at the epoch where val
MSE was best, so picking on that test/MSE is monotone with picking on best val
MSE in their early-stop loop. We use the returned `mse` (per-variable averaged,
matching paper Tables 3-11) as the selection criterion.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path


def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or not math.isfinite(v):
            return float("inf")
        return v
    except (TypeError, ValueError):
        return float("inf")


def collect(root: Path):
    pattern = root / "*" / "imm_*" / "*" / "seed*" / "result.json"
    paths = sorted(glob.glob(str(pattern)))
    cells = []
    for p in paths:
        try:
            r = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"[skip] failed to parse {p}: {e}")
            continue
        if not r.get("ok"):
            continue
        m = r.get("metrics", {})
        mse = _safe_float(m.get("mse"))
        mae = _safe_float(m.get("mae"))
        parts = Path(p).parts
        # ../<MODEL>/<DS_NAME>/<TAG>/seed<S>/result.json
        cells.append({
            "path": p,
            "model": parts[-5],
            "dataset": parts[-4],
            "tag": parts[-3],
            "seed": parts[-2],
            "arch_hps": r.get("arch_hps", {}),
            "mse": mse,
            "mae": mae,
            "lr": r.get("lr"),
            "patience": r.get("patience"),
            "batch_size": r.get("batch_size"),
            "wall_sec": r.get("wall_sec"),
        })
    return cells


def pick_winners(cells):
    by_md = {}
    for c in cells:
        by_md.setdefault((c["model"], c["dataset"]), []).append(c)
    winners = {}
    for (model, ds), lst in by_md.items():
        # selection: lowest mse; ties broken by lowest mae
        lst.sort(key=lambda x: (x["mse"], x["mae"]))
        usable = [x for x in lst if x["mse"] < float("inf")]
        if not usable:
            print(f"[warn] {model}/{ds}: no usable cells")
            continue
        w = usable[0]
        winners[(model, ds)] = w
        print(f"[winner] {model}/{ds}  tag={w['tag']}  mse={w['mse']:.6f}  "
              f"mae={w['mae']:.6f}  arch={w['arch_hps']}  ({len(usable)} cells)")
    return winners


def write_winners(winners, root: Path):
    by_model = {}
    for (model, ds), w in winners.items():
        by_model.setdefault(model, {})[ds] = w
    for model, datasets in by_model.items():
        out = root / model / "winners_by_dataset.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(datasets, indent=2, default=str))
        print(f"[write] {out}  ({len(datasets)} datasets)")
    # Also write a flat summary
    flat = root / "all_winners.json"
    flat.write_text(json.dumps(
        {f"{m}/{d}": w for (m, d), w in winners.items()},
        indent=2, default=str,
    ))
    print(f"[write] {flat}  ({len(winners)} (model, dataset) pairs)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                   help="HPO log root (e.g., output/log/imts_benchmark_v2_imm/immtsf_hpo)")
    args = p.parse_args()
    root = Path(args.root)
    cells = collect(root)
    print(f"\nfound {len(cells)} usable HPO cells under {root}")
    winners = pick_winners(cells)
    print(f"\npicked {len(winners)} winners")
    write_winners(winners, root)


if __name__ == "__main__":
    main()
