"""Smoke-test plot: one sample per regime, overlaying all variates with
async timestamps. Shows the forecast boundary (history=) and any forbidden
long-gap interval. Regenerated for the 4-regime imts_benchmark setup."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent / "data"
SAMPLE_IDX = 0
REGIMES = (
    "sparse_independent",
    "sparse_dependent",
    "nogap_independent",
    "nogap_dependent",
)

fig, axes = plt.subplots(len(REGIMES), 1, figsize=(11, 2.6 * len(REGIMES)), sharex=True)

for ax, regime in zip(axes, REGIMES):
    ds = load_from_disk(str(ROOT / regime / "train"))
    row = ds[SAMPLE_IDX]
    n_obs = row["n_obs_per_var"]
    target = np.array(row["target"], dtype=np.float32)
    ts = np.array(row["timestamp"], dtype=np.float32)
    history = float(row["history"])

    offsets = np.concatenate([[0], np.cumsum(n_obs)])
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for d, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        ax.plot(ts[lo:hi], target[lo:hi], "o-", ms=3, lw=0.6,
                color=colors[d % len(colors)], label=f"var{d} (n={hi - lo})")

    ax.axvline(history, color="gray", lw=0.8, ls="--", label=f"history={history}")
    ax.set_title(f"{regime} — sample {SAMPLE_IDX}")
    ax.set_ylabel("normalized value")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("time")
out = Path(__file__).resolve().parent / "_smoke_plot.png"
plt.tight_layout()
plt.savefig(out, dpi=120)
print(f"Saved {out}")
