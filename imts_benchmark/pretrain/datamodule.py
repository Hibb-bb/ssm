"""Lightning DataModule for the multivariate Mamba pretraining pipeline.

Wires together (sources -> mixed_dataset -> collate) given a fully
resolved stage config. The stage config specifies:

  - ``stage`` block:    StageSpec fields (per-variate window, V dist,
                        regime dist, etc.).
  - ``mix`` block:      logical-source weights (e.g.
                        ``{lotsa_degraded: 0.30, chronos2_synth: 0.35,
                          ...}``).
  - ``sources`` block (passed in separately or merged in): which
                        physical sources to load for each logical
                        source, and which weight_map to use.

A separate ``sources.yaml`` registry holds source paths so we can
swap stages without re-stating data locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from .collate import make_collate
from .mixed_dataset import (
    LogicalSource,
    MixedTorchDataset,
    load_lotsa_weight_map,
    make_logical_source,
)
from .sources import (
    ChronosSynthSource,
    KernelSynthSource,
    LOTSAHFSource,
    discover_lotsa_sources,
)
from .task_sampler import (
    SourceTaskOverride,
    StageSpec,
    default_overrides,
    make_transform,
    overrides_from_dict,
    stage_from_dict,
)
from .val_imts import build_imts_val_loaders


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _expand_env(s: Any) -> Any:
    if isinstance(s, str):
        return os.path.expandvars(os.path.expanduser(s))
    return s


def build_logical_sources(
    sources_cfg: dict[str, Any],
    stage: StageSpec,
    overrides: dict[str, SourceTaskOverride],
    seed: int,
) -> list[LogicalSource]:
    """Build all LogicalSources from a sources.yaml dict.

    Format of ``sources_cfg``::

        lotsa_root: "/home/ubuntu/lotsa_data"
        lotsa_weighted_yaml: "/home/ubuntu/uni2ts/cli/conf/.../lotsa_v1_weighted.yaml"
        synthetic:
          chronos2_synth: "/path/to/chronos2-synth.arrow"
          kernelsynth:    "/path/to/kernelsynth-irregular.arrow"
        # optional: limit which LOTSA datasets are loaded
        lotsa_include: [PEMS04, PEMS08, ...]
        lotsa_exclude: []
    """
    out: list[LogicalSource] = []

    # ---- Synthetic logical sources ---------------------------------
    syn = sources_cfg.get("synthetic", {})
    chronos_path = _expand_env(syn.get("chronos2_synth"))
    kernel_path = _expand_env(syn.get("kernelsynth"))

    if chronos_path and Path(chronos_path).exists():
        out.append(make_logical_source(
            name="chronos2_synth",
            physical_sources=[ChronosSynthSource(path=chronos_path, seed=seed)],
            transform=make_transform(stage, "chronos2_synth", overrides.get("chronos2_synth")),
            weight_map=None,
        ))
    if kernel_path and Path(kernel_path).exists():
        out.append(make_logical_source(
            name="kernelsynth",
            physical_sources=[KernelSynthSource(path=kernel_path, seed=seed)],
            transform=make_transform(stage, "kernelsynth", overrides.get("kernelsynth")),
            weight_map=None,
        ))

    # ---- LOTSA logical sources -------------------------------------
    lotsa_root = _expand_env(sources_cfg.get("lotsa_root"))
    if lotsa_root and Path(lotsa_root).exists():
        weight_yaml = _expand_env(sources_cfg.get("lotsa_weighted_yaml"))
        weight_map = (
            load_lotsa_weight_map(weight_yaml)
            if weight_yaml and Path(weight_yaml).exists()
            else None
        )
        include = sources_cfg.get("lotsa_include")
        exclude = sources_cfg.get("lotsa_exclude")

        physical = discover_lotsa_sources(
            lotsa_root, include=include, exclude=exclude, seed=seed
        )
        if physical:
            # lotsa_regular: clean data only, no degradation
            out.append(make_logical_source(
                name="lotsa_regular",
                physical_sources=physical,
                transform=make_transform(stage, "lotsa_regular", overrides.get("lotsa_regular")),
                weight_map=weight_map,
            ))
            # lotsa_degraded: degraded context, clean future targets
            out.append(make_logical_source(
                name="lotsa_degraded",
                physical_sources=physical,
                transform=make_transform(stage, "lotsa_degraded", overrides.get("lotsa_degraded")),
                weight_map=weight_map,
            ))

    if not out:
        raise RuntimeError(
            "No logical sources were built. Check that sources.yaml points to "
            "existing arrow files and/or LOTSA root."
        )
    return out


@dataclass
class PretrainDataModuleArgs:
    stage_cfg_path: str
    sources_cfg_path: str
    batch_size: int = 32
    num_workers: int = 4
    max_dim: int = 20
    seed: int = 42
    val_steps: int = 0  # 0 disables synthetic val
    persistent_workers: bool = True
    # Downstream IMTS validation (see pretrain/README.md §3.4).
    # When ``val_imts_datasets`` is non-empty, validate against the
    # listed regimes from ``tpatchgnn_data/`` every val cycle.
    val_imts_data_root: Optional[str] = None
    val_imts_datasets: tuple[str, ...] = ()
    val_imts_split: str = "val"
    val_imts_subset: Optional[int] = 1024
    val_imts_batch_size: int = 64
    val_imts_num_workers: int = 2
    # Time-IMM / IMM-TSF downstream val (paper-reproducible).
    # See pretrain/README.md §3.4 and §7.B'.  These eight datasets share
    # the same sparse-flat schema as ``tpatchgnn_data`` but use the
    # converter's per-record GLOBAL z-score (norm_mode="precomputed_per_record")
    # so test MSE/MAE matches the IMM-TSF paper Table.
    val_imm_tsf_data_root: Optional[str] = None
    val_imm_tsf_datasets: tuple[str, ...] = ()
    val_imm_tsf_split: str = "test"
    val_imm_tsf_subset: Optional[int] = None  # None => full test split
    val_imm_tsf_batch_size: int = 32
    val_imm_tsf_num_workers: int = 2


class PretrainDataModule(pl.LightningDataModule):
    """Per-stage Lightning DataModule.

    The "training set" is an infinite IterableDataset. Lightning iterates
    until ``trainer.max_steps`` is reached; ``len()`` is intentionally
    not defined to flag the iterable nature.

    A small synthetic-only validation loop is exposed when
    ``val_steps > 0`` to track loss without depending on a real held-out
    LOTSA split (which we leave for downstream IMTS benchmarking).
    """

    def __init__(self, args: PretrainDataModuleArgs):
        super().__init__()
        self.args = args
        self.stage_cfg = _load_yaml(args.stage_cfg_path)
        self.sources_cfg = _load_yaml(args.sources_cfg_path)

        self.stage: StageSpec = stage_from_dict(self.stage_cfg.get("stage", {}))
        merged_overrides = default_overrides()
        for k, v in overrides_from_dict(self.stage_cfg.get("overrides", {})).items():
            merged_overrides[k] = v
        self.overrides = merged_overrides

        self.mix_weights: dict[str, float] = self.stage_cfg["mix"]
        self._train_ds: Optional[MixedTorchDataset] = None
        self._val_ds: Optional[MixedTorchDataset] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_ds is not None:
            return
        logical = build_logical_sources(
            self.sources_cfg,
            stage=self.stage,
            overrides=self.overrides,
            seed=self.args.seed,
        )
        self._train_ds = MixedTorchDataset(
            logical_sources=logical,
            logical_weights=self.mix_weights,
            seed=self.args.seed,
        )
        if self.args.val_steps > 0:
            # validation: small synthetic-only dataset to track val loss
            val_logical = [
                ls for ls in logical if ls.name in {"chronos2_synth", "kernelsynth"}
            ]
            if val_logical:
                vw = {ls.name: 1.0 for ls in val_logical}
                self._val_ds = MixedTorchDataset(
                    logical_sources=val_logical,
                    logical_weights=vw,
                    seed=self.args.seed + 7,
                )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _make_loader(self, ds: MixedTorchDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=make_collate(max_dim=self.args.max_dim),
            persistent_workers=self.args.persistent_workers and self.args.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return self._make_loader(self._train_ds)

    def val_dataloader(self):
        loaders = []
        if self._val_ds is not None:
            loaders.append(self._make_loader(self._val_ds))
        if self.args.val_imts_data_root and self.args.val_imts_datasets:
            loaders.extend(build_imts_val_loaders(
                data_root=self.args.val_imts_data_root,
                datasets_list=list(self.args.val_imts_datasets),
                split=self.args.val_imts_split,
                subset_size=self.args.val_imts_subset,
                batch_size=self.args.val_imts_batch_size,
                num_workers=self.args.val_imts_num_workers,
                max_dim=self.args.max_dim,
                seed=self.args.seed + 17,
            ))
        if self.args.val_imm_tsf_data_root and self.args.val_imm_tsf_datasets:
            loaders.extend(build_imts_val_loaders(
                data_root=self.args.val_imm_tsf_data_root,
                datasets_list=list(self.args.val_imm_tsf_datasets),
                split=self.args.val_imm_tsf_split,
                subset_size=self.args.val_imm_tsf_subset,
                batch_size=self.args.val_imm_tsf_batch_size,
                num_workers=self.args.val_imm_tsf_num_workers,
                max_dim=self.args.max_dim,
                seed=self.args.seed + 23,
                norm_mode="precomputed_per_record",
                name_prefix="imm_",
            ))
        if not loaders:
            return None
        return loaders if len(loaders) > 1 else loaders[0]
