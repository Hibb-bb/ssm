"""Print per-dataset class counts + the inverse-frequency class weights the
training loss currently sees.

Why: our loss is weighted (`F.cross_entropy(weight=class_weights)`) while
RoMAE App. C.2 uses only label_smoothing. If our datasets are nearly balanced
the weights are ~1 and harmless; if they're imbalanced (LSST has 14 classes,
HB has 2 with imbalance) the weighting can change behaviour. Run this to see
the magnitudes before deciding whether `--use_class_weights 0` is worth a
seed.

Usage:
    python -m imts_benchmark.eval.uea_class_balance \
        --data_root /projects/bfrf/seojininus/ssm/data_uea
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from imts_benchmark.shared_data.uea_classification_datamodule import (
    UEAClassificationDataModule,
)


DATASETS = ["BasicMotions", "CharacterTrajectories", "Epilepsy", "Heartbeat", "LSST"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Match the train_cls.py default so the train subset "
                        "(and therefore counts) is identical.")
    args = p.parse_args()

    for ds in DATASETS:
        dm = UEAClassificationDataModule(
            data_root=args.data_root,
            dataset_name=ds,
            batch_size=8,
            num_workers=0,
            split_seed=args.split_seed,
        )
        dm.setup()

        # Recompute counts from the same train subset the datamodule used.
        labels = np.asarray([dm.train_ds.y[i] for i in range(len(dm.train_ds))])
        counts = np.bincount(labels, minlength=dm.n_classes)

        weights = dm.class_weights.numpy()
        idx_to_label = {v: k for k, v in dm.label_to_idx.items()}

        print(f"\n=== {ds}  (V={dm.n_vars}, |train|={len(dm.train_ds)}, "
              f"|val|={len(dm.val_ds)}, |test|={len(dm.test_ds)}, "
              f"C={dm.n_classes}) ===")
        print(f"  imbalance ratio (max/min count): "
              f"{counts.max() / max(counts.min(), 1):.2f}")
        print(f"  weight ratio    (max/min weight): "
              f"{weights.max() / weights.min():.2f}")
        print(f"  per-class:")
        for c in range(dm.n_classes):
            print(f"    [{c:2d}] {idx_to_label[c]:30s}  "
                  f"count={counts[c]:5d}  weight={weights[c]:.3f}")


if __name__ == "__main__":
    main()
