"""Audit C — is the shared-grid `exp(-gamma * rho)` decay a real bottleneck?

For Mamba-MV `replace` checkpoints on Phase 3 × high_irreg (N=5 seeds):
  1. Extract trained `gamma = softplus(grid.gamma_raw)` per variate / channel.
  2. Compute `rho = s_k - t_{pi_d(k)}` on the test split.
  3. Report the distribution of the implied decay factor `exp(-gamma * rho)`.

A decay factor near 1 => approximation near-identity => §5 is minor.
A decay factor near 0 => state is heavily attenuated => §5 is severe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CKPT_ROOT = Path(
    "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/"
    "imts_benchmark_v2/phase3/mamba_mv/multisin_high_irreg/replace"
)
DATA_DIR = Path(
    "/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/moirai/"
    "uni2ts_hongyu/ssm_dk/data_correct_async/multisin_high_irreg"
)
T_MAX = 10.0
K_GRID = 128
N_VARS = 3


def load_gamma_raw(ckpt_path: Path) -> torch.Tensor:
    """Return grid.gamma_raw of shape [V, D_h]."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # Key will look like "model.grid.gamma_raw" (PL LightningModule prefix).
    for k, v in sd.items():
        if k.endswith("grid.gamma_raw"):
            return v
    raise KeyError(f"grid.gamma_raw not found in {ckpt_path}")


def _split_per_variate(arr, n_obs):
    out = []
    c = 0
    for n in n_obs:
        out.append(arr[c:c + n])
        c += n
    return out


def compute_rho_samples():
    """Load test split and compute the distribution of rho = s_k - t_pi per variate.
    Returns rho array of shape [n_samples_total, V, K]. Invalid slots => np.nan."""
    from datasets import load_from_disk

    ds = load_from_disk(str(DATA_DIR / "test"))
    grid = np.linspace(0.0, T_MAX, K_GRID)

    all_rho = []
    for row in ds:
        n_obs = np.asarray(row["n_obs_per_var"], dtype=np.int64)
        ts_flat = np.asarray(row["timestamp"], dtype=np.float32)
        ts_pv = _split_per_variate(ts_flat, n_obs)

        rho_sample = np.full((N_VARS, K_GRID), np.nan, dtype=np.float32)
        for d in range(N_VARS):
            ts = ts_pv[d]
            if len(ts) == 0:
                continue
            for k, s_k in enumerate(grid):
                le = ts[ts <= s_k]
                if len(le) == 0:
                    continue
                t_pi = le.max()
                rho_sample[d, k] = s_k - t_pi
        all_rho.append(rho_sample)
    return np.stack(all_rho, axis=0)  # [N, V, K]


def main():
    # === 1. Gamma stats across seeds
    gammas = []
    for seed in range(1, 6):
        ckpt = CKPT_ROOT / f"seed{seed}" / "checkpoints" / "best.ckpt"
        if not ckpt.exists():
            print(f"  [skip] {ckpt} not found")
            continue
        g_raw = load_gamma_raw(ckpt)               # [V, D_h]
        g = F.softplus(g_raw).numpy()              # [V, D_h]
        gammas.append(g)
        print(f"seed={seed}  gamma softplus stats per variate:")
        for d in range(g.shape[0]):
            print(f"   v{d}  min={g[d].min():.4f}  mean={g[d].mean():.4f}  "
                  f"median={np.median(g[d]):.4f}  max={g[d].max():.4f}")
    gammas = np.stack(gammas, axis=0)              # [n_seeds, V, D_h]
    gamma_mean_over_seeds = gammas.mean(axis=0)     # [V, D_h]
    print(f"\n#seeds = {gammas.shape[0]}, D_h = {gammas.shape[-1]}")

    # === 2. Rho distribution on test split
    print("\nComputing rho distribution on test split...")
    rho = compute_rho_samples()                     # [N, V, K]
    print(f"Rho shape: {rho.shape}")

    # Strip NaNs per-variate, summarize
    for d in range(N_VARS):
        rd = rho[:, d, :].flatten()
        rd = rd[~np.isnan(rd)]
        print(f"  v{d}: rho stats over {len(rd)} (sample, grid_slot) pairs:"
              f" min={rd.min():.4f}  mean={rd.mean():.4f}  median={np.median(rd):.4f}"
              f"  p90={np.percentile(rd, 90):.4f}  max={rd.max():.4f}")

    # === 3. Implied decay factor exp(-gamma * rho)
    # For each variate, compute the decay using channel-mean gamma and summarize
    # distribution over (sample, grid_slot, channel).
    print("\nImplied decay factor = exp(-gamma * rho):")
    for d in range(N_VARS):
        g_d = gamma_mean_over_seeds[d]              # [D_h]
        rho_d = rho[:, d, :]                        # [N, K]
        mask = ~np.isnan(rho_d)
        rho_flat = rho_d[mask]                      # [M_d]
        # Outer product: decay[m, c] = exp(-g_d[c] * rho_flat[m])
        # This can blow up memory; sample a subset if needed.
        if len(rho_flat) > 5000:
            idx = np.random.default_rng(0).choice(len(rho_flat), 5000, replace=False)
            rho_flat = rho_flat[idx]
        decay = np.exp(-g_d[None, :] * rho_flat[:, None])  # [M, D_h]
        d_mean = decay.mean()
        d_med = np.median(decay)
        d_p10 = np.percentile(decay, 10)
        d_p90 = np.percentile(decay, 90)
        print(f"  v{d}: mean_decay={d_mean:.4f}  median={d_med:.4f}"
              f"  p10={d_p10:.4f}  p90={d_p90:.4f}"
              f"  (gamma_channel_mean={g_d.mean():.4f}, rho_mean={rho_flat.mean():.4f})")

    # Aggregate: "fraction of (slot, channel) pairs where decay < 0.5"
    print("\nSeverity — fraction of (slot, channel) pairs where decay < 0.5:")
    for d in range(N_VARS):
        g_d = gamma_mean_over_seeds[d]
        rho_d = rho[:, d, :]
        rho_flat = rho_d[~np.isnan(rho_d)]
        if len(rho_flat) > 5000:
            idx = np.random.default_rng(0).choice(len(rho_flat), 5000, replace=False)
            rho_flat = rho_flat[idx]
        decay = np.exp(-g_d[None, :] * rho_flat[:, None])
        frac = (decay < 0.5).mean()
        print(f"  v{d}: {frac:.3%}  of (slot, channel) pairs have decay < 0.5")


if __name__ == "__main__":
    main()
