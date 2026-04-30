"""Diagnose post-fix Moirai-style LOTSA sampling distribution.

Replays the pipeline up to source selection and stacking *without*
launching a full training run.  Reports:
    - per-dataset effective sampling probability after the
      (num_ts × yaml_weight) fix,
    - which datasets get StackedLOTSASource wrapping,
    - empirical V distribution from drawing N samples,
    - sanity check against Moirai's intent (no single dataset > a few %).

Usage::

    /home/ubuntu/envs/mamba/bin/python -m imts_benchmark.pretrain.scripts.diagnose_moirai_sampling \\
        --n_samples 4096 --output_dir /tmp/moirai_diag

Run from /home/ubuntu/hongyu/ssm.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from imts_benchmark.pretrain.datamodule import _load_yaml
from imts_benchmark.pretrain.mixed_dataset import (
    LogicalSource,
    load_lotsa_weight_map,
    make_logical_source,
)
from imts_benchmark.pretrain.sources import (
    MOIRAI_STACKING_DATASETS,
    STACK_ALL,
    StackedLOTSASource,
    discover_lotsa_sources,
)
from imts_benchmark.pretrain.task_sampler import (
    StageSpec,
    make_transform,
    stage_from_dict,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage_cfg",
        default="imts_benchmark/pretrain/configs/stage_a_single_phase.yaml",
    )
    p.add_argument(
        "--sources_cfg", default="imts_benchmark/pretrain/configs/sources.yaml"
    )
    p.add_argument("--n_samples", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="/tmp/moirai_diag")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stage_cfg = _load_yaml(args.stage_cfg)
    sources_cfg = _load_yaml(args.sources_cfg)
    stage: StageSpec = stage_from_dict(stage_cfg.get("stage", {}))

    lotsa_root = sources_cfg["lotsa_root"]
    weight_yaml = sources_cfg.get("lotsa_weighted_yaml")
    weight_map = load_lotsa_weight_map(weight_yaml) if weight_yaml else None
    include = sources_cfg.get("lotsa_include")
    exclude = sources_cfg.get("lotsa_exclude")

    physical = discover_lotsa_sources(
        lotsa_root,
        include=include,
        exclude=exclude,
        seed=args.seed,
        stacking_datasets=STACK_ALL,
        variate_count_dist=tuple(stage.variate_count_dist),
        max_variates=stage.max_variates,
    )

    # --------- Effective sampling probability per dataset ------------
    print("\n" + "=" * 78)
    print(" POST-FIX LOTSA EFFECTIVE SAMPLING PROBABILITY")
    print("=" * 78)

    rows = []
    for s in physical:
        nm = s.name.replace("lotsa:", "")
        n_ts = len(s)
        yaml_w = weight_map.get(nm, 1.0) if weight_map else 1.0
        product = n_ts * yaml_w
        is_stacked = isinstance(s, StackedLOTSASource)
        rows.append(
            {
                "name": nm,
                "num_ts": n_ts,
                "yaml_w": yaml_w,
                "product": product,
                "stacked": is_stacked,
            }
        )

    total = sum(r["product"] for r in rows)
    for r in rows:
        r["pct"] = 100.0 * r["product"] / total if total > 0 else 0.0

    rows.sort(key=lambda r: -r["pct"])
    print(
        f"{'dataset':<40s}{'num_ts':>10s}{'yaml_w':>10s}"
        f"{'product':>14s}{'pct':>8s}{'stack':>8s}"
    )
    print("-" * 90)
    for r in rows[:30]:
        print(
            f"{r['name'][:40]:<40s}"
            f"{r['num_ts']:>10d}"
            f"{r['yaml_w']:>10.4f}"
            f"{r['product']:>14.2f}"
            f"{r['pct']:>7.2f}%"
            f"{'   ✓' if r['stacked'] else '   .':>8s}"
        )
    print("...")
    n_stacked = sum(1 for r in rows if r["stacked"])
    pct_stacked = sum(r["pct"] for r in rows if r["stacked"])
    print(
        f"\nstacked datasets: {n_stacked} / {len(rows)} "
        f"({pct_stacked:.1f}% of LOTSA mass)"
    )
    sw_pct = sum(r["pct"] for r in rows if r["name"] in ("solar_power", "wind_power"))
    print(f"solar_power + wind_power combined: {sw_pct:.2f}%")

    # --------- Empirical: build LogicalSource and draw N samples ----
    print("\n" + "=" * 78)
    print(" EMPIRICAL V DISTRIBUTION (N=" + str(args.n_samples) + ")")
    print("=" * 78)

    logical = make_logical_source(
        name="lotsa_degraded",
        physical_sources=physical,
        transform=make_transform(stage, "lotsa_degraded", None),
        weight_map=weight_map,
    )
    rng = np.random.default_rng(args.seed)
    weights = logical.physical_weights / logical.physical_weights.sum()

    # Sample dataset indices according to weights
    chosen_ds = rng.choice(len(physical), size=args.n_samples, p=weights)
    ds_counter = Counter(chosen_ds)

    # Empirical V distribution at TWO points:
    #   - v_pre: shape[0] coming out of the source iterator (before
    #     LOTSAToIrregular).  Reflects stacking output for univariate-
    #     on-disk and native V for natively-MV.
    #   - v_post: actual V the model sees after LOTSAToIrregular runs.
    #     This should follow variate_count_dist — it's the source of
    #     truth for "model-seen V".
    v_pre_counter: Counter = Counter()
    v_post_counter: Counter = Counter()
    n_per_ds_to_draw = 50
    transform = make_transform(stage, "lotsa_degraded", None)
    for di, count in ds_counter.most_common(40):
        src = physical[di]
        try:
            it = iter(src)
        except Exception:
            continue
        n_drawn = 0
        weight = count // n_per_ds_to_draw + 1
        for ex in it:
            tgt = ex.get("target")
            if tgt is None:
                continue
            v_pre = int(tgt.shape[0]) if tgt.ndim == 2 else 1
            v_pre_counter[v_pre] += weight

            # Run the actual transform to get model-seen V
            try:
                out_ex = transform(ex)
            except Exception as e:
                v_post = 0
            else:
                # output uses ragged per-variate lists; V = len(values_per_var)
                if out_ex.get("_empty"):
                    v_post = 0
                else:
                    vpv = out_ex.get("values_per_var")
                    v_post = len(vpv) if vpv is not None else 0
            if v_post > 0:
                v_post_counter[v_post] += weight

            n_drawn += 1
            if n_drawn >= n_per_ds_to_draw:
                break

    def _print_dist(counter: Counter, label: str) -> dict[str, float]:
        total = sum(counter.values())
        print(f"\n{'-' * 60}\n{label}\n{'-' * 60}")
        print(f"{'V':<10s}{'count':>10s}{'pct':>10s}")
        for v in sorted(counter.keys()):
            c = counter[v]
            pct = 100.0 * c / total if total else 0.0
            print(f"{f'V = {v}':<10s}{c:>10d}{pct:>9.2f}%")
        bucket_pcts = {"V=1": 0.0, "V=2..8": 0.0, "V=9..16": 0.0, "V=17..20": 0.0}
        for v, c in counter.items():
            pct = 100.0 * c / total if total else 0.0
            if v == 1:
                bucket_pcts["V=1"] += pct
            elif 2 <= v <= 8:
                bucket_pcts["V=2..8"] += pct
            elif 9 <= v <= 16:
                bucket_pcts["V=9..16"] += pct
            else:
                bucket_pcts["V=17..20"] += pct
        print(f"\n  BUCKETED:")
        for k, v in bucket_pcts.items():
            print(f"    {k:<12s}{v:>6.2f}%")
        return bucket_pcts

    pre_b = _print_dist(v_pre_counter, "PRE-TRANSFORM (stacked-source output)")
    post_b = _print_dist(v_post_counter, "POST-TRANSFORM (model-seen V)")

    target = stage.variate_count_dist
    print(
        f"\n  target  [V=1: {target[0]*100:.0f}%, V=2..8: {target[1]*100:.0f}%, "
        f"V=9..16: {target[2]*100:.0f}%, V=17..20: {target[3]*100:.0f}%]"
    )

    out_json = {
        "per_dataset": rows,
        "stacked_count": n_stacked,
        "stacked_pct": pct_stacked,
        "solar_wind_pct": sw_pct,
        "v_pre_transform": dict(v_pre_counter),
        "v_post_transform": dict(v_post_counter),
        "bucket_pcts_pre": pre_b,
        "bucket_pcts_post": post_b,
        "target_dist": list(stage.variate_count_dist),
    }
    out_file = out / "moirai_sampling_diag.json"
    out_file.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
