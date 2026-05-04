"""Generate the CRU pendulum dataset and convert to HF Arrow for our datamodule.

Bypasses CRU's run_experiment.py wrapper (which requires geotorch even with
--exit_after_generation, because it calls load_model() before the exit check).
We instead import the data-side functions directly:

  - lib.pendulum_generation.generate_pendulums  (saves pend_regression.npz)
  - lib.data_utils.Pendulum_regression          (subsamples at sample_rate=0.5)

Both are pure-numpy/torch and don't pull geotorch.

Output schema (matches imts_benchmark.shared_data.pendulum_datamodule):
    item_id:                  string
    images:                   list[float], length T * 24 * 24, in [0, 1]
    timestamp:                list[float], length T (integer indices in 0..99,
                                            stored as float for Arrow uniformity)
    past_feat_dynamic_real:   list[float], length T (per-step delta_t, first=0)
    target:                   list[float], length T * 2 (sin, cos pairs)
    n_obs:                    int (T = 50 by default at sample_rate=0.5)

Splits: train (2000), val (1000), test (1000). Random state for subsampling
is fixed at 0, matching CRU/S5's default and giving byte-reproducible output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import datasets as hfds


CRU_REPO = "/u/seojininus/ssm/_s5_repo/Continuous-Recurrent-Units"
CACHE_DIR = "/u/seojininus/ssm/pendulum_data/cache/pendulum/"
OUT_ROOT = Path("/u/seojininus/ssm/pendulum_data/pendulum")
SAMPLE_RATE = 0.5      # matches RoMAE / S5 paper (L=50 from T=100)
RANDOM_STATE = 0       # CRU/S5 default
IMG_H = IMG_W = 24
IMG_DIM = IMG_H * IMG_W


def _ensure_imports():
    """Make CRU's lib importable without polluting our package namespace.

    CRU's pendulum_generation.py:393 uses PIL.Image.ANTIALIAS, which was
    removed in Pillow 10. Patch it before import so the 2022-era CRU code
    runs against our 2024+ Pillow.
    """
    import PIL.Image
    if not hasattr(PIL.Image, "ANTIALIAS"):
        # ANTIALIAS was always an alias for LANCZOS; keep behavior identical.
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    if CRU_REPO not in sys.path:
        sys.path.insert(0, CRU_REPO)
    from lib.pendulum_generation import generate_pendulums
    from lib.data_utils import Pendulum_regression
    return generate_pendulums, Pendulum_regression


def _build_split(ds, split_label: str) -> hfds.Dataset:
    rows = []
    for i in range(len(ds)):
        obs, targets, time_points, _ = ds[i]
        # obs:        [T, 1, 24, 24] float64 in [0,1] (already /255)
        # targets:    [T, 2]         float64 (sin, cos)
        # time_points: [T]           int     (indices in 0..99, sorted)
        obs_np = obs.numpy().astype(np.float32)
        tgt_np = targets.numpy().astype(np.float32)
        ts_np = time_points.numpy().astype(np.float32)
        T = obs_np.shape[0]

        deltat = np.zeros(T, dtype=np.float32)
        if T > 1:
            deltat[1:] = ts_np[1:] - ts_np[:-1]

        rows.append({
            "item_id": f"pendulum_{split_label}_{i:05d}",
            "images": obs_np.reshape(-1).tolist(),       # T * 1 * 24 * 24
            "timestamp": ts_np.tolist(),                 # T
            "past_feat_dynamic_real": deltat.tolist(),   # T
            "target": tgt_np.reshape(-1).tolist(),       # T * 2
            "n_obs": int(T),
        })
    return hfds.Dataset.from_list(rows)


def main():
    generate_pendulums, Pendulum_regression = _ensure_imports()

    os.makedirs(CACHE_DIR, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    npz_path = os.path.join(CACHE_DIR, "pend_regression.npz")
    if not os.path.exists(npz_path):
        print(f"[convert_pendulum] generating npz at {npz_path} (first time only)...")
        generate_pendulums(CACHE_DIR, task="regression")
    else:
        print(f"[convert_pendulum] reusing existing {npz_path}")

    # CRU uses 'valid'; we save under 'val' to match our datamodule convention.
    split_map = [("train", "train"), ("valid", "val"), ("test", "test")]
    for cru_mode, our_dir in split_map:
        ds = Pendulum_regression(
            file_path=CACHE_DIR,
            name="pend_regression.npz",
            mode=cru_mode,
            sample_rate=SAMPLE_RATE,
            random_state=RANDOM_STATE,
        )
        # Sanity: confirm the per-sample shape matches what we expect.
        obs0, tgt0, ts0, _ = ds[0]
        assert obs0.shape == (50, 1, IMG_H, IMG_W), \
            f"unexpected obs shape: {tuple(obs0.shape)}"
        assert tgt0.shape == (50, 2), f"unexpected target shape: {tuple(tgt0.shape)}"
        assert ts0.shape == (50,), f"unexpected ts shape: {tuple(ts0.shape)}"

        hf = _build_split(ds, our_dir)
        out_dir = OUT_ROOT / our_dir
        hf.save_to_disk(str(out_dir))
        print(f"[convert_pendulum] wrote {out_dir} ({len(hf)} samples)")

    # One-line summary so the smoke test can sanity-check the dataset.
    info = {
        "split_sizes": {
            "train": 2000, "val": 1000, "test": 1000,
        },
        "T_per_sample": int(SAMPLE_RATE * 100),
        "image_dim": IMG_DIM,
        "target_dim": 2,
        "t_max": 100.0,
        "sample_rate": SAMPLE_RATE,
        "random_state": RANDOM_STATE,
        "source": "CRU @ andrewwarrington/Continuous-Recurrent-Units (pendulum branch)",
    }
    import json
    info_path = OUT_ROOT / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[convert_pendulum] wrote {info_path}")
    print(f"[convert_pendulum] DONE — datamodule will read from {OUT_ROOT}/")


if __name__ == "__main__":
    main()
