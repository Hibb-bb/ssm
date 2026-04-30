"""Pretraining entry-point.

Example::

    python -m imts_benchmark.pretrain.train_pretrain \\
        --stage a \\
        --loss huber \\
        --batch_size 32 \\
        --max_steps 1000 \\
        --output_dir /home/ubuntu/hongyu/ssm/imts_benchmark/pretrain/runs/stage_a_smoke

Stage YAMLs live in ``imts_benchmark/pretrain/configs/stage_*.yaml``.
The source registry lives in
``imts_benchmark/pretrain/configs/sources.yaml`` and is reused across
all stages.

The script trains the existing
``MultivariateMambaForecaster`` (or ``MultivariateMambaSandwichForecaster``)
with three pretraining-specific overrides:

  - ``max_dim`` set from the stage config (default 20),
  - ``t_max=1.0`` because the data layer normalizes time per sample,
  - ``loss_type`` set from CLI (default Huber, per Time-MoE).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from ..mamba_mv.multivariate_forecaster import (
    MultivariateMambaForecaster,
    MultivariateMambaSandwichForecaster,
)
from ..shared_config.wandb_lightning import build_wandb_logger
from .datamodule import PretrainDataModule, PretrainDataModuleArgs


_STAGE_CFG_PATHS = {
    "a": Path(__file__).parent / "configs" / "stage_a.yaml",
    "b": Path(__file__).parent / "configs" / "stage_b.yaml",
    "c": Path(__file__).parent / "configs" / "stage_c.yaml",
}


def _build_model(args: argparse.Namespace) -> pl.LightningModule:
    cls = (
        MultivariateMambaSandwichForecaster
        if args.arch == "sandwich"
        else MultivariateMambaForecaster
    )
    kwargs = dict(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        max_dim=args.max_dim,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        n_heads_varattn=args.n_heads_varattn,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        grid_K=args.grid_K,
        # Data layer normalizes timestamps to [0, 1] per sample.
        t_max=1.0,
        n_freq=args.n_freq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_steps,
        # Pretraining-specific
        loss_type=args.loss,
        huber_delta=args.huber_delta,
        lr_schedule=args.lr_schedule,
    )
    if args.arch == "sandwich":
        kwargs.update(n_tail_grid_mamba=args.n_tail_grid_mamba)
    return cls(**kwargs)


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.stage_cfg is None:
        key = args.stage.lower().strip()
        if key.startswith("stage_"):
            key = key[len("stage_") :]
        if key not in _STAGE_CFG_PATHS:
            raise ValueError(f"unknown stage '{args.stage}'; pick one of a/b/c")
        args.stage_cfg = str(_STAGE_CFG_PATHS[key])
    if args.sources_cfg is None:
        args.sources_cfg = str(Path(__file__).parent / "configs" / "sources.yaml")
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "runs" / f"stage_{args.stage}")
    return args


def _git_sha(start: Path) -> str:
    """Best-effort git SHA of the repo containing ``start``; '<no-git>' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()
    except Exception:
        return "<no-git>"


def _dump_ablation_sidecar(args: argparse.Namespace) -> None:
    """Write a JSON sidecar with all knobs that uniquely identify this run.

    Drives the §3.6 systematic-adjustment strategy: every run is tagged
    by (axis, level) so we can later collate sweeps with a one-liner.
    """
    info = {
        "ablation_axis": args.ablation_axis,
        "ablation_level": args.ablation_level,
        "stage": args.stage,
        "stage_cfg": args.stage_cfg,
        "sources_cfg": args.sources_cfg,
        "loss": args.loss,
        "huber_delta": args.huber_delta,
        "arch": args.arch,
        "max_dim": args.max_dim,
        "d_model": args.d_model,
        "d_hidden": args.d_hidden,
        "n_perv_layer": args.n_perv_layer,
        "n_fusion_blocks": args.n_fusion_blocks,
        "grid_K": args.grid_K,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_warmup_steps": args.num_warmup_steps,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "max_steps": args.max_steps,
        "precision": args.precision,
        "seed": args.seed,
        "val_imts_datasets": list(args.val_imts_datasets or []),
        "val_imts_split": args.val_imts_split,
        "val_imm_tsf_datasets": list(args.val_imm_tsf_datasets or []),
        "val_imm_tsf_split": args.val_imm_tsf_split,
        "profiler": args.profiler,
        "num_workers": args.num_workers,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_mode": args.wandb_mode,
        "wandb_tags": list(args.wandb_tags or []),
        "git_sha": _git_sha(Path(__file__).resolve().parent),
        "init_from": args.init_from,
        "init_from_strict": args.init_from_strict,
        "ckpt_resume": args.ckpt,
        "argv": sys.argv,
    }
    out_path = Path(args.output_dir) / "ablation.json"
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    print(f"[ablation] wrote {out_path}")


class _SynthInDistValCallback(pl.Callback):
    """Run model on a fixed snapshot of synthetic windows every val cycle.

    Train loss alone is hard to read because each batch is a different random
    sample.  An in-distribution val set (= same windows every cycle, in the
    same data distribution as train) gives a much cleaner signal for "is the
    model still learning."

    Snapshot is built at trainer setup time from a fresh dataloader with
    `seed=stage_seed + 1000` so it doesn't overlap with training samples
    in the early epochs.  The snapshot is held in CPU memory.

    Logs (in asinh-z space, same as train/huber):
        val/synth_huber           — mean huber loss over the snapshot
        val/synth_mse_z           — z-MSE in raw z-space (sinh-inverted preds)
        val/synth_mae_z           — z-MAE in raw z-space
        val/synth_r2_overall      — pooled R^2 over all observations
        val/synth_chronos2_r2     — R^2 restricted to chronos2_synth windows
        val/synth_kernelsynth_r2  — R^2 restricted to kernelsynth windows
    """

    def __init__(
        self,
        n_windows: int = 512,
        batch_size: int = 32,
        seed_offset: int = 1000,
    ):
        super().__init__()
        self.n_windows = int(n_windows)
        self.batch_size = int(batch_size)
        self.seed_offset = int(seed_offset)
        self._batches: list[dict] = []        # CPU tensors
        self._sources: list[list[str]] = []   # per-batch source names
        self._snapshot_built = False

    def _build_snapshot(self, trainer: pl.Trainer) -> None:
        if self._snapshot_built:
            return
        # Construct a fresh dataloader from the same datamodule but with
        # a different seed.  Using train_dataloader directly because val
        # in this codebase is OOD; we want IN-distribution for this.
        dm = trainer.datamodule
        if dm is None:
            print("[synth-val] no datamodule attached; skipping snapshot")
            return

        from .datamodule import PretrainDataModule, PretrainDataModuleArgs
        if not isinstance(dm, PretrainDataModule):
            print(f"[synth-val] datamodule type {type(dm).__name__} unsupported; skipping")
            return

        snap_args = PretrainDataModuleArgs(
            stage_cfg_path=dm.args.stage_cfg_path,
            sources_cfg_path=dm.args.sources_cfg_path,
            batch_size=self.batch_size,
            num_workers=2,
            max_dim=dm.args.max_dim,
            seed=dm.args.seed + self.seed_offset,
            persistent_workers=False,
        )
        snap_dm = PretrainDataModule(snap_args)
        snap_dm.setup()
        snap_dl = snap_dm.train_dataloader()

        n_collected = 0
        for batch in snap_dl:
            # Move to CPU + clone so the snapshot is independent.
            cpu_batch = {
                k: (v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            self._batches.append(cpu_batch)
            self._sources.append(list(batch.get("source_name", [])))
            n_collected += int(batch["values"].shape[0])
            if n_collected >= self.n_windows:
                break
        self._snapshot_built = True
        print(f"[synth-val] snapshot built: {n_collected} windows in {len(self._batches)} batches "
              f"(seed={snap_args.seed})")

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            self._build_snapshot(trainer)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._batches:
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        sum_huber = 0.0
        sum_mse_z = 0.0
        sum_mae_z = 0.0
        n_pred = 0
        per_src_truth: dict[str, list[torch.Tensor]] = {}
        per_src_pred: dict[str, list[torch.Tensor]] = {}
        all_truth: list[torch.Tensor] = []
        all_pred: list[torch.Tensor] = []

        for batch_cpu, sources in zip(self._batches, self._sources):
            batch_g = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch_cpu.items()
            }
            preds_asinh = pl_module(batch_g)
            pm = batch_g["pred_mask"]
            n = int(pm.sum().item())
            if n == 0:
                continue

            # Train loss is in asinh-z, so synth_huber is comparable to train/huber.
            truth_asinh = batch_g["values"]
            diff_asinh = (preds_asinh - truth_asinh)[pm]
            huber = torch.nn.functional.smooth_l1_loss(
                preds_asinh[pm], truth_asinh[pm], beta=1.0, reduction="sum",
            )
            sum_huber += float(huber.item())

            # raw-z metrics: invert asinh.
            #
            # 2026-04-29: clamp BOTH preds_asinh AND truth_asinh to ±10
            # before sinh inversion (same fix as §7.L for IMM-TSF eval).
            # Some lotsa snapshot windows have truth_asinh ≈ 20 from
            # heavy-tailed real-world series (rare big spikes in
            # otherwise near-zero data — e.g., sensor faults, sparse
            # counts).  Without clamping, sinh(20) ≈ 3e8 dominates the
            # pooled R²/MSE calc → val/synth_lotsa_degraded_r2 stays
            # frozen at -0.0005 across all steps.  See README §7.O.
            preds_a_c = preds_asinh.clamp(-10.0, 10.0)
            truth_a_c = truth_asinh.clamp(-10.0, 10.0)
            preds_z = torch.sinh(preds_a_c)
            truth_z = torch.sinh(truth_a_c)
            diff_z = (preds_z - truth_z)[pm].float()
            sum_mse_z += float((diff_z ** 2).sum().item())
            sum_mae_z += float(diff_z.abs().sum().item())
            n_pred += n

            # Per-source for R^2 — flatten and assign each (b, v, l)
            # contribution to the source of its batch element b.
            B = preds_z.shape[0]
            for b in range(B):
                m = pm[b]
                if not bool(m.any()):
                    continue
                t_b = truth_z[b][m].float().detach().cpu()
                p_b = preds_z[b][m].float().detach().cpu()
                src = sources[b] if b < len(sources) else "?"
                per_src_truth.setdefault(src, []).append(t_b)
                per_src_pred.setdefault(src, []).append(p_b)
                all_truth.append(t_b)
                all_pred.append(p_b)

        if was_training:
            pl_module.train()

        if n_pred == 0:
            return

        pl_module.log("val/synth_huber", sum_huber / n_pred, sync_dist=False, add_dataloader_idx=False)
        pl_module.log("val/synth_mse_z", sum_mse_z / n_pred, sync_dist=False, add_dataloader_idx=False)
        pl_module.log("val/synth_mae_z", sum_mae_z / n_pred, sync_dist=False, add_dataloader_idx=False)

        # Pooled R^2 = 1 - SS_res / SS_tot
        if all_truth:
            t_all = torch.cat(all_truth)
            p_all = torch.cat(all_pred)
            ss_res = float(((t_all - p_all) ** 2).sum().item())
            ss_tot = float(((t_all - t_all.mean()) ** 2).sum().item())
            if ss_tot > 1e-12:
                pl_module.log("val/synth_r2_overall", 1.0 - ss_res / ss_tot,
                              sync_dist=False, add_dataloader_idx=False)

        for src in sorted(per_src_truth):
            t = torch.cat(per_src_truth[src])
            p = torch.cat(per_src_pred[src])
            ss_res = float(((t - p) ** 2).sum().item())
            ss_tot = float(((t - t.mean()) ** 2).sum().item())
            if ss_tot > 1e-12:
                # Strip "synth:" prefix if present so wandb names are clean.
                tag = src.split(":", 1)[-1] if ":" in src else src
                pl_module.log(f"val/synth_{tag}_r2", 1.0 - ss_res / ss_tot,
                              sync_dist=False, add_dataloader_idx=False)


class _AggregateValMetricsCallback(pl.Callback):
    """Compute aggregate val metrics after each validation epoch.

    Two metric "spaces" are aggregated:

    1. ``mse_z`` / ``mae_z`` over the **8 IMM-TSF datasets** only.
       This is paper-comparable space; the per-record per-feature
       z-score is the IMM-TSF normalization (parse_datasets.py:103-111
       + evaluation.py:27-30).  Activity / USHCN are intentionally
       excluded because per-(b,v) z collapses for near-constant series
       (USHCN precipitation, see README §Q4) and the published baselines
       for those datasets report **original-unit** MSE anyway.

    2. ``mse`` / ``mae`` over both IMM-TSF and the activity/ushcn
       loaders.  Original units; useful for sanity / dashboards but
       not directly comparable across datasets due to scale heterogeneity.

    Logs:
        ``val/mse_z_imm_avg``  : mean of ``val/mse_z_imm_{ds}`` across 8 IMM-TSF
        ``val/mae_z_imm_avg``  : same for MAE
        ``val/mse_imm_avg``    : mean of ``val/mse_imm_{ds}`` (original units)
        ``val/mae_imm_avg``    : same for MAE
        ``val/mse_imts_avg``   : mean of ``val/mse_{activity,ushcn}`` (orig)
        ``val/mae_imts_avg``   : same
        ``val/mse_avg_all``    : mean of all 10 original-unit MSEs

    ``val/mse_z_imm_avg`` is the primary axis-4 / ES decision number
    per pretrain/README.md §7.F.
    """

    IMM_DATASETS = (
        "EPA-Air", "ILINet", "GDELT", "FNSPID",
        "CESNET", "StudentLife", "RepoHealth", "ClusterTrace",
    )
    IMTS_DATASETS = ("activity", "ushcn")

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics

        def _mean_of(keys: list[str]) -> float | None:
            vals: list[float] = []
            for k in keys:
                if k in cm:
                    try:
                        v = float(cm[k])
                        # Filter non-finite (a single bad ckpt eval can
                        # produce inf on the IMM-TSF side; we don't want
                        # one inf to poison the average and trip ES).
                        if np.isfinite(v):
                            vals.append(v)
                    except Exception:
                        pass
            return None if not vals else sum(vals) / len(vals)

        # z-space metrics: only over IMM-TSF (paper-comparable)
        for root in ("mse_z", "mae_z"):
            imm_keys = [f"val/{root}_imm_{d}" for d in self.IMM_DATASETS]
            imm_avg = _mean_of(imm_keys)
            if imm_avg is not None:
                pl_module.log(f"val/{root}_imm_avg", imm_avg,
                              sync_dist=False, add_dataloader_idx=False)

        # Original-unit metrics: aggregate over IMM-TSF, IMTS, and all
        for root in ("mse", "mae"):
            imm_keys = [f"val/{root}_imm_{d}" for d in self.IMM_DATASETS]
            imts_keys = [f"val/{root}_{d}" for d in self.IMTS_DATASETS]

            imm_avg = _mean_of(imm_keys)
            imts_avg = _mean_of(imts_keys)
            all_avg = _mean_of(imm_keys + imts_keys)

            if imm_avg is not None:
                pl_module.log(f"val/{root}_imm_avg", imm_avg,
                              sync_dist=False, add_dataloader_idx=False)
            if imts_avg is not None:
                pl_module.log(f"val/{root}_imts_avg", imts_avg,
                              sync_dist=False, add_dataloader_idx=False)
            if all_avg is not None:
                pl_module.log(f"val/{root}_avg_all", all_avg,
                              sync_dist=False, add_dataloader_idx=False)


def _load_weights_from_ckpt(
    model: pl.LightningModule, ckpt_path: str, strict: bool
) -> dict:
    """Load only the model `state_dict` from a Lightning .ckpt.

    Returns a small dict of provenance info so the caller can persist
    it in `transition.json`.  This is the right primitive for stage
    A → B → C handoff: optimizer Adam moments / LR schedule / step
    counter / RNG / dataloader cursor are all tuned to the *previous*
    stage's data distribution and step budget, so re-using them
    typically destabilizes the first hundreds of steps in the new
    stage.  See pretrain/README.md §7.H for the rationale.
    """
    src = Path(ckpt_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"--init_from: ckpt not found at {src}")

    blob = torch.load(str(src), map_location="cpu", weights_only=False)
    if "state_dict" not in blob:
        raise KeyError(
            f"--init_from: {src} is not a Lightning checkpoint "
            f"(no 'state_dict' key); got top-level keys {list(blob)[:5]}"
        )
    sd = blob["state_dict"]
    result = model.load_state_dict(sd, strict=strict)
    missing, unexpected = (
        list(getattr(result, "missing_keys", [])),
        list(getattr(result, "unexpected_keys", [])),
    )
    print(f"[init_from] loaded weights from {src}")
    print(f"[init_from]   matched   : {len(sd) - len(unexpected)} tensors")
    if missing:
        print(f"[init_from]   missing   : {len(missing)} -> {missing[:5]}{' …' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[init_from]   unexpected: {len(unexpected)} -> {unexpected[:5]}{' …' if len(unexpected) > 5 else ''}")

    src_step = int(blob.get("global_step", -1))
    src_epoch = int(blob.get("epoch", -1))
    src_pl_ver = blob.get("pytorch-lightning_version", "?")
    return {
        "init_from_ckpt": str(src),
        "init_from_global_step": src_step,
        "init_from_epoch": src_epoch,
        "init_from_pl_version": str(src_pl_ver),
        "init_from_strict": bool(strict),
        "init_from_missing_keys": missing,
        "init_from_unexpected_keys": unexpected,
    }


def _dump_transition_metadata(
    args: argparse.Namespace, init_info: dict | None
) -> None:
    """Write transition.json describing where this run was initialized
    from (if any).  Mirrors `ablation.json` but is specifically for
    cross-stage continuity provenance.
    """
    if init_info is None:
        return
    src_run_dir = Path(init_info["init_from_ckpt"]).resolve().parent
    src_ablation = src_run_dir / "ablation.json"
    src_summary: dict = {}
    if src_ablation.exists():
        try:
            src_summary = json.loads(src_ablation.read_text())
        except Exception:
            pass
    info = {
        **init_info,
        "this_run": {
            "stage": args.stage,
            "stage_cfg": args.stage_cfg,
            "ablation_axis": args.ablation_axis,
            "ablation_level": args.ablation_level,
            "wandb_run_name": args.wandb_run_name,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "num_warmup_steps": args.num_warmup_steps,
        },
        "source_run": {
            "ablation": src_summary,
            "run_dir": str(src_run_dir),
        },
    }
    out_path = Path(args.output_dir) / "transition.json"
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2, sort_keys=True, default=str)
    print(f"[transition] wrote {out_path}")


def _print_pretrain_banner(args: argparse.Namespace) -> None:
    with open(args.stage_cfg) as f:
        sc = yaml.safe_load(f)
    print("=" * 72)
    print(f"  Mamba IMTS Pretraining — stage {args.stage}, loss {args.loss}")
    if args.ablation_axis or args.ablation_level:
        print(f"  ablation      : axis={args.ablation_axis} level={args.ablation_level}")
    print("=" * 72)
    print(f"  stage_cfg     : {args.stage_cfg}")
    print(f"  sources_cfg   : {args.sources_cfg}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  max_steps     : {args.max_steps}")
    print(f"  arch          : {args.arch}")
    print(f"  max_dim       : {args.max_dim}")
    print(f"  d_model       : {args.d_model}")
    print(f"  loss_type     : {args.loss} (delta={args.huber_delta})")
    print("  per-stage mix :")
    for name, w in sc.get("mix", {}).items():
        print(f"      {name:20s} {w}")
    if args.val_imts_datasets:
        print(f"  imts val      : {' '.join(args.val_imts_datasets)} "
              f"(split={args.val_imts_split}, subset={args.val_imts_subset})")
    if args.val_imm_tsf_datasets:
        print(f"  imm-tsf val   : {' '.join(args.val_imm_tsf_datasets)} "
              f"(split={args.val_imm_tsf_split})")
    print("=" * 72)


def main() -> None:
    p = argparse.ArgumentParser()
    # Stage / loss
    p.add_argument("--stage", default="a", choices=["a", "b", "c", "A", "B", "C"])
    p.add_argument("--loss", default="huber", choices=["huber", "mse"])
    p.add_argument("--huber_delta", type=float, default=1.0)
    # Override config paths
    p.add_argument("--stage_cfg", default=None,
                   help="override per-stage config path (default: configs/stage_<stage>.yaml)")
    p.add_argument("--sources_cfg", default=None,
                   help="override sources registry path (default: configs/sources.yaml)")
    # Data
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_steps", type=int, default=0,
                   help="if > 0, run a small synthetic-only val loop")
    # Downstream IMTS val (see pretrain/README.md §3.4).
    p.add_argument("--val_imts_data_root", default=None,
                   help="root of tpatchgnn_data; when set together with "
                        "--val_imts_datasets, runs downstream IMTS val")
    p.add_argument("--val_imts_datasets", nargs="*", default=[],
                   help="space-separated list of regimes (e.g. activity ushcn)")
    p.add_argument("--val_imts_split", default="val", choices=["val", "test"])
    p.add_argument("--val_imts_subset", type=int, default=1024,
                   help="subsample each split to this many windows; -1 to use full split")
    p.add_argument("--val_imts_batch_size", type=int, default=64)
    p.add_argument("--val_imts_num_workers", type=int, default=2)
    p.add_argument("--val_check_steps", type=int, default=2000,
                   help="run val every N training steps (default: every 2k)")
    # Time-IMM / IMM-TSF downstream val (see pretrain/README.md §3.4 + §7.B').
    # Wired here on top of the activity/ushcn val so we get paper-comparable
    # MSE/MAE on 8 extra real-world IMTS test splits.
    p.add_argument("--val_imm_tsf_data_root", default=None,
                   help="root of imm_tsf_sparse/ converted via "
                        "pretrain/scripts/convert_imm_tsf_to_sparse.py")
    p.add_argument("--val_imm_tsf_datasets", nargs="*", default=[],
                   help="space-separated dataset names to evaluate, e.g. "
                        "EPA-Air ILINet GDELT (default: all detected on disk)")
    p.add_argument("--val_imm_tsf_split", default="test",
                   choices=["train", "val", "test"],
                   help="default 'test' (matches IMM-TSF paper Table reporting)")
    p.add_argument("--val_imm_tsf_subset", type=int, default=-1,
                   help="subsample each split to this many windows; -1 = full split")
    p.add_argument("--val_imm_tsf_batch_size", type=int, default=32)
    p.add_argument("--val_imm_tsf_num_workers", type=int, default=2)
    # Model
    p.add_argument("--arch", default="vanilla", choices=["vanilla", "sandwich"])
    # Match the production architecture used by `imts_benchmark.mamba_mv.train_mv`
    # (the script we use for downstream fine-tuning on activity / ushcn /
    # physionet). Pretraining at the same d_model means the resulting
    # checkpoint can be loaded into the fine-tune setup with no
    # architectural surgery (param-shape compatible). See
    # imts_benchmark/mamba_mv/train_mv.py:63-64 and pretrain/README.md §7.D.
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--d_hidden", type=int, default=384)
    p.add_argument("--n_perv_layer", type=int, default=3)
    p.add_argument("--n_fusion_blocks", type=int, default=3)
    p.add_argument("--n_tail_grid_mamba", type=int, default=2,
                   help="only used when --arch sandwich")
    p.add_argument("--n_heads_varattn", type=int, default=4)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--dt_mode", default="replace")
    p.add_argument("--grid_K", type=int, default=128)
    p.add_argument("--n_freq", type=int, default=8)
    # Optim
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_warmup_steps", type=int, default=200)
    # LR schedule choice. ``cosine`` (default) decays peak LR -> 0 over
    # the run; ``constant`` holds peak LR after warmup.  Use ``constant``
    # to test whether the late-training val plateau is an LR-decay
    # artifact vs a real model-capacity ceiling.
    p.add_argument("--lr_schedule", choices=["cosine", "constant", "multistep"], default="cosine")
    # Trainer
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--max_minutes", type=float, default=None,
                   help="(optional) wall-clock cap in minutes; useful for "
                        "fixed-length profiling runs. Stops cleanly so "
                        "profiler output is fully flushed.")
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--save_every_n_steps", type=int, default=500)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--ckpt", default=None,
                   help="optional: full Lightning resume from a previous .ckpt "
                        "(model + optimizer + scheduler + step counter + RNG). "
                        "Use ONLY for crash recovery within the same stage.")
    p.add_argument("--init_from", default=None,
                   help="optional: load model weights only from a previous "
                        ".ckpt (no optimizer / scheduler / step counter). "
                        "Use this for stage transitions, e.g. stage A → B → C: "
                        "the new run starts at step 0 with a fresh LR schedule "
                        "but inherits the pretrained representation. See "
                        "pretrain/README.md §7.H for the rationale.")
    p.add_argument("--init_from_strict", action="store_true",
                   help="when --init_from is set, require an exact key match "
                        "between the checkpoint and the current model "
                        "(default: allow missing/unexpected keys with a warning, "
                        "useful when stage B adds new heads / changes max_dim)")
    # Ablation tagging (see pretrain/README.md §3.6). Free-form strings;
    # convention is something like --ablation_axis loss --ablation_level huber.
    p.add_argument("--ablation_axis", default=None,
                   help="(optional) name of the ablation axis this run belongs to")
    p.add_argument("--ablation_level", default=None,
                   help="(optional) level within the ablation axis")
    # Early stopping (see pretrain/README.md §7.K). Disabled by default;
    # turn on with --early_stop_metric to pick a val key to monitor.
    # Only fires after the *first* val epoch where the metric appears.
    p.add_argument("--early_stop_metric", default=None,
                   help="val/* metric key to monitor; e.g. 'val/mse_z_imm_avg'. "
                        "Empty disables early stopping.")
    p.add_argument("--early_stop_patience", type=int, default=6,
                   help="number of val epochs without improvement before stopping (default: 6)")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="minimum improvement to count as 'better' (default: 0.0)")
    p.add_argument("--early_stop_mode", choices=["min", "max"], default="min",
                   help="'min' for losses, 'max' for accuracy-like metrics (default: min)")
    # Synthetic in-distribution val (see _SynthInDistValCallback).
    # Not enabled by default — only useful when training on synth-bearing
    # mixes (chronos2 / kernelsynth).
    p.add_argument("--synth_val_n_windows", type=int, default=0,
                   help="if >0, snapshot N windows from a fresh dataloader at "
                        "trainer setup and evaluate the model on them every val cycle. "
                        "Logs val/synth_huber, val/synth_mse_z, val/synth_r2_overall, "
                        "val/synth_<source>_r2.")
    p.add_argument("--synth_val_batch_size", type=int, default=32)
    p.add_argument("--synth_val_seed_offset", type=int, default=1000)
    # W&B (see pretrain/README.md §7.G). Gated by --use_wandb so profiling
    # and smoke tests stay local-only (CSVLogger is always on).  Mirrors
    # the same flag names as imts_benchmark/mamba_mv/train_mv.py so the
    # downstream fine-tune pipeline and pretraining share dashboards.
    p.add_argument("--use_wandb", action="store_true",
                   help="enable WandbLogger alongside CSVLogger; project "
                        "and run name come from --wandb_project / --wandb_run_name "
                        "(or the WANDB_PROJECT / WANDB_NAME env vars)")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name (default: env WANDB_PROJECT or 'mamba-imts-pretrain')")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity / team name (default: env WANDB_ENTITY)")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run display name (default: auto from stage + ablation tag)")
    p.add_argument("--wandb_mode", default=None,
                   choices=[None, "online", "offline", "disabled"],
                   help="W&B mode override; default uses env WANDB_MODE if set, "
                        "else 'online'.  Use 'offline' on machines without "
                        "wandb credentials so the run still streams to ./wandb/.")
    p.add_argument("--wandb_tags", nargs="*", default=[],
                   help="space-separated tags applied to the W&B run "
                        "(in addition to stage / ablation auto-tags)")
    # Profiling (see pretrain/README.md §7.D). When set, Lightning's
    # built-in profiler dumps a per-action time breakdown at end-of-fit:
    #   "simple"   - per-action wall time (data, fwd, bwd, optim, val)
    #   "advanced" - same plus per-callback hooks; a bit more overhead
    #   "pytorch"  - chrome-trace flame graph (heavyweight, large output)
    p.add_argument("--profiler", default=None,
                   choices=[None, "simple", "advanced", "pytorch"],
                   help="(optional) Lightning profiler; None disables profiling")

    args = p.parse_args()
    args.stage = args.stage.lower()
    args = _resolve_paths(args)

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)
    _print_pretrain_banner(args)
    _dump_ablation_sidecar(args)

    # Data
    dm = PretrainDataModule(
        PretrainDataModuleArgs(
            stage_cfg_path=args.stage_cfg,
            sources_cfg_path=args.sources_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_dim=args.max_dim,
            seed=args.seed,
            val_steps=args.val_steps,
            val_imts_data_root=args.val_imts_data_root,
            val_imts_datasets=tuple(args.val_imts_datasets),
            val_imts_split=args.val_imts_split,
            val_imts_subset=(args.val_imts_subset if args.val_imts_subset >= 0 else None),
            val_imts_batch_size=args.val_imts_batch_size,
            val_imts_num_workers=args.val_imts_num_workers,
            val_imm_tsf_data_root=args.val_imm_tsf_data_root,
            val_imm_tsf_datasets=tuple(args.val_imm_tsf_datasets),
            val_imm_tsf_split=args.val_imm_tsf_split,
            val_imm_tsf_subset=(
                args.val_imm_tsf_subset if args.val_imm_tsf_subset >= 0 else None
            ),
            val_imm_tsf_batch_size=args.val_imm_tsf_batch_size,
            val_imm_tsf_num_workers=args.val_imm_tsf_num_workers,
        )
    )

    # Model
    model = _build_model(args)

    # Optional weight-only init for stage transitions (A → B → C).
    # NOT a Lightning resume: optimizer / scheduler / step counter / RNG
    # / dataloader cursor all start fresh.  See README §7.H.
    init_info: dict | None = None
    if args.init_from is not None:
        if args.ckpt is not None:
            raise ValueError(
                "--init_from and --ckpt are mutually exclusive: "
                "--init_from = weights only (stage transition); "
                "--ckpt = full Lightning resume (crash recovery)."
            )
        init_info = _load_weights_from_ckpt(
            model, args.init_from, strict=args.init_from_strict
        )
        _dump_transition_metadata(args, init_info)

    # Trainer
    has_imts_val = bool(args.val_imts_data_root and args.val_imts_datasets)
    has_imm_tsf_val = bool(
        args.val_imm_tsf_data_root and args.val_imm_tsf_datasets
    )
    callbacks: list = [
        LearningRateMonitor(logging_interval="step"),
        # (1) Periodic step-keyed ckpts: cheap, every 5K steps + always
        # the latest "last.ckpt" so a crash recovers via --ckpt last.ckpt.
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="ckpt-{step:08d}",
            save_top_k=-1,
            every_n_train_steps=args.save_every_n_steps,
            save_last=True,
        ),
    ]
    # (2) Aggregate val metrics + "best" ckpt by val/mse_z_imm_avg.
    # Only meaningful when downstream val is on; with synthetic-only val
    # there's no z-scored space to pick a best by.
    if has_imm_tsf_val:
        callbacks.append(_AggregateValMetricsCallback())
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="best-step{step:08d}-mse{val/mse_z_imm_avg:.4f}",
                monitor="val/mse_z_imm_avg",
                mode="min",
                save_top_k=3,
                save_last=False,
                auto_insert_metric_name=False,
            )
        )
    elif has_imts_val:
        callbacks.append(_AggregateValMetricsCallback())
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename="best-step{step:08d}-mse{val/mse_z_imts_avg:.4f}",
                monitor="val/mse_z_imts_avg",
                mode="min",
                save_top_k=3,
                save_last=False,
                auto_insert_metric_name=False,
            )
        )

    # (3) Synthetic in-distribution val (opt-in via --synth_val_n_windows).
    # Snapshot is built once at fit-start; per-cycle inference adds ~5s.
    if args.synth_val_n_windows > 0:
        callbacks.append(
            _SynthInDistValCallback(
                n_windows=args.synth_val_n_windows,
                batch_size=args.synth_val_batch_size,
                seed_offset=args.synth_val_seed_offset,
            )
        )

    # (4) Early stopping (opt-in via --early_stop_metric).  Lightning will
    # only consult this callback at validation_epoch_end, so cadence is
    # tied to --val_check_steps.  Patience is in *val epochs*, not steps.
    #
    # ``check_finite=False`` because the asinh→sinh inversion in
    # _log_imts_val can produce a single inf z-MSE if the model emits
    # an out-of-range asinh-z prediction (see README §7.K).  Aggregate
    # callback already filters non-finite values from the IMM-TSF mean
    # before logging, so val/mse_z_imm_avg is always finite as long as
    # ≥1 of 8 datasets evaluated cleanly.  But we want ES to gracefully
    # skip a transient blowup rather than terminate the whole run.
    if args.early_stop_metric:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stop_metric,
                mode=args.early_stop_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                strict=True,
                verbose=True,
                check_finite=False,
            )
        )
    csv_logger = CSVLogger(save_dir=args.output_dir, name="csv")
    loggers: list = [csv_logger]
    if args.use_wandb:
        # Default to a stable project name + a run name that encodes the
        # ablation tag so dashboards stay legible.  These are pure
        # defaults; the CLI flags still take precedence.
        if args.wandb_project is None:
            args.wandb_project = os.environ.get(
                "WANDB_PROJECT", "mamba-imts-pretrain"
            )
        if args.wandb_run_name is None:
            run_bits = [f"stage_{args.stage}"]
            if args.ablation_axis:
                run_bits.append(f"axis-{args.ablation_axis}")
            if args.ablation_level:
                run_bits.append(f"lvl-{args.ablation_level}")
            run_bits.append(f"seed{args.seed}")
            args.wandb_run_name = "_".join(run_bits)
        # Auto-tags so the W&B project view groups runs by stage / axis.
        auto_tags = [f"stage_{args.stage}", f"loss_{args.loss}"]
        if args.ablation_axis:
            auto_tags.append(f"axis_{args.ablation_axis}")
        if args.ablation_level:
            auto_tags.append(f"level_{args.ablation_level}")
        args.wandb_tags = list(args.wandb_tags) + [
            t for t in auto_tags if t not in args.wandb_tags
        ]
        # Build the W&B logger now (before fit) so Lightning's setup
        # streams metrics from step 1.  We snapshot some hparams that
        # only become available after the model is built.
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(args.stage_cfg) as f:
            _stage_cfg = yaml.safe_load(f)
        wandb_logger = build_wandb_logger(
            args,
            extra_config={
                "git_sha": _git_sha(Path(__file__).resolve().parent),
                "stage_cfg_resolved": args.stage_cfg,
                "sources_cfg_resolved": args.sources_cfg,
                "n_params": n_params,
                "n_params_trainable": n_trainable,
                "stage_mix": _stage_cfg.get("mix", {}),
            },
        )
        if wandb_logger is not False:
            loggers.append(wandb_logger)
            print(f"[wandb] enabled: project={args.wandb_project!r} "
                  f"run={args.wandb_run_name!r} mode={args.wandb_mode or 'online'}")
            print(f"[wandb] tags: {args.wandb_tags}")
            print(f"[wandb] n_params: {n_params:,} ({n_trainable:,} trainable)")
    else:
        wandb_logger = False

    logger = loggers if len(loggers) > 1 else loggers[0]

    if has_imts_val or has_imm_tsf_val:
        # Downstream val is the primary source of "is the model getting
        # better at the actual downstream task?" (§3.6). Run it on its
        # own cadence; the synthetic val_steps gate is independent.
        val_interval = max(1, args.val_check_steps)
        # When we have downstream val loaders, do NOT clamp val batches
        # with limit_val_batches=0 — we want full subsets / test splits.
        limit_val = 1.0
    else:
        val_interval = max(1, args.save_every_n_steps)
        limit_val = args.val_steps if args.val_steps > 0 else 0

    profiler = None
    if args.profiler is not None:
        # Build the profiler so its output lands next to the run logs
        # rather than the cwd, which is the Lightning default.
        if args.profiler == "simple":
            from pytorch_lightning.profilers import SimpleProfiler
            profiler = SimpleProfiler(
                dirpath=args.output_dir, filename="profiler_simple"
            )
        elif args.profiler == "advanced":
            from pytorch_lightning.profilers import AdvancedProfiler
            profiler = AdvancedProfiler(
                dirpath=args.output_dir, filename="profiler_advanced"
            )
        elif args.profiler == "pytorch":
            from pytorch_lightning.profilers import PyTorchProfiler
            profiler = PyTorchProfiler(
                dirpath=args.output_dir, filename="profiler_pytorch"
            )

    trainer_kwargs = dict(
        max_steps=args.max_steps,
        precision=args.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        enable_progress_bar=True,
        # IterableDataset has no len; Lightning needs val_check_interval
        # in steps and check_val_every_n_epoch=None.
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val,
        num_sanity_val_steps=0,
    )
    if args.max_minutes is not None:
        trainer_kwargs["max_time"] = {"minutes": args.max_minutes}

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
