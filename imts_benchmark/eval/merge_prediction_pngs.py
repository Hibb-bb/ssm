"""Merge per-model prediction PNGs into one composite per (dataset, sample_idx).

For each test sample 0..N-1, stack the same sample's plot from every available
model vertically (S5 / RoMAE / Mamba-MV per dt_mode), so model behavior on the
same test instance can be compared at a glance.

Output: aggregate/predictions_test/composites/composite_{dataset}_sample_{i}.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path("/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/imts_benchmark_v2_real/aggregate/predictions_test")

# Discover available model rows per dataset.
# Layout written by run_plot_predictions_after_mamba.sbatch:
#   <variant_dir>/<model_name>/<regime>/sample_*.png
# Plus the older flat format we wrote earlier:
#   <model_name>/<regime>/sample_*.png   (for s5 / romae)
ROWS = [
    # (display label, [dataset -> Path to dir containing sample_*.png])
    ("S5",                {"activity": ROOT / "default" / "s5"    / "activity",
                           "ushcn":    ROOT / "default" / "s5"    / "ushcn"}),
    ("RoMAE",             {"activity": ROOT / "default" / "romae" / "activity",
                           "ushcn":    ROOT / "default" / "romae" / "ushcn"}),
    ("Mamba-MV (replace)",{"activity": ROOT / "replace_lr-2e-3_bs-128" / "mamba_mv" / "activity",
                           "ushcn":    ROOT / "replace_lr-5e-4_bs-64"  / "mamba_mv" / "ushcn"}),
    ("Mamba-MV (learned)",{"activity": ROOT / "learned_lr-2e-3_bs-128" / "mamba_mv" / "activity",
                           "ushcn":    ROOT / "learned_lr-1e-4_bs-256" / "mamba_mv" / "ushcn"}),
    ("Mamba-MV (concat)", {"activity": ROOT / "concat_lr-5e-4_bs-128"  / "mamba_mv" / "activity",
                           "ushcn":    ROOT / "concat_lr-1e-4_bs-256"  / "mamba_mv" / "ushcn"}),
]


def make_composite(dataset: str, sample_idx: int, out_dir: Path):
    rows_present = []
    for label, ds_to_dir in ROWS:
        png = ds_to_dir[dataset] / f"sample_{sample_idx:02d}.png"
        if png.exists():
            rows_present.append((label, png))
    if not rows_present:
        print(f"[skip] {dataset} sample {sample_idx}: no PNGs found")
        return

    # Stack vertically. Each subplot uses imshow on the original PNG and the
    # model label sits on the left as a y-axis label.
    n = len(rows_present)
    # Get aspect ratio from the first image so the grid is well-proportioned
    sample_img = mpimg.imread(str(rows_present[0][1]))
    h, w = sample_img.shape[0], sample_img.shape[1]
    aspect = h / w
    fig_w = 9
    fig_h = fig_w * aspect * n + 0.4 * n  # extra for spacing/labels
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    for ax, (label, png) in zip(axes, rows_present):
        img = mpimg.imread(str(png))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, fontsize=11, rotation=0, ha="right", va="center",
                      labelpad=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"{dataset.capitalize()} — test sample #{sample_idx}  (cross-model comparison)",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"composite_{dataset}_sample_{sample_idx:02d}.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    out_dir = ROOT / "composites"
    for dataset in ("activity", "ushcn"):
        for i in range(4):
            make_composite(dataset, i, out_dir)


if __name__ == "__main__":
    main()
