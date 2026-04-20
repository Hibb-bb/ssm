"""Smoke-test plot: one sample per regime, overlaying all variates with
async timestamps and the forbidden long-gap interval visible."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent / "data"
SAMPLE_IDX = 0

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for ax, regime in zip(axes, ("sparse_independent", "sparse_dependent")):
    ds = load_from_disk(str(ROOT / regime / "train"))
    row = ds[SAMPLE_IDX]
    n_obs = row["n_obs_per_var"]
    target = np.array(row["target"], dtype=np.float32)
    ts = np.array(row["timestamp"], dtype=np.float32)

    offsets = np.concatenate([[0], np.cumsum(n_obs)])
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for d, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        ax.plot(ts[lo:hi], target[lo:hi], "o-", ms=3, lw=0.6,
                color=colors[d % len(colors)], label=f"var{d} (n={hi - lo})")

    ax.set_title(f"{regime} — sample {SAMPLE_IDX}")
    ax.set_ylabel("normalized value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("time")
out = Path(__file__).resolve().parent / "_smoke_plot.png"
plt.tight_layout()
plt.savefig(out, dpi=120)
print(f"Saved {out}")
