"""Pretraining infrastructure for the multivariate Mamba IMTS forecaster.

Subpackages / modules:
  - sources        : uniform iterable wrappers around LOTSA HF datasets
                     and the two synthetic Arrow files
  - task_sampler   : per-example task parameter sampler (V, W, ctx_frac,
                     target subset, irregularity regime)
  - mixed_dataset  : stage-weighted multi-source IterableDataset with
                     Moirai-style dataset capping
  - collate        : pad to max_dim, normalize time, build masks
  - datamodule     : Lightning DataModule glue
  - train_pretrain : entry-point training script
"""
