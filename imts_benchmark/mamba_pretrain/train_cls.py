"""Train Mamba-MV (pretrain-arch) classifier on UEA datasets with Kidger 30% sync drop.

Entry point for both smoke runs and HPO array tasks. Per-dataset defaults
(batch_size, label_smoothing, grad_clip, grid_K, head_dropout) are
inherited from RoMAE Table 12; LR and batch_size are typically the swept
axes for v2 HPO. Per-dataset defaults still apply when the corresponding
CLI flag is left at its sentinel ``-1``.

Example:
    python -m imts_benchmark.mamba_mv.train_cls \
        --dataset BasicMotions --data_root <data_root> \
        --dt_mode replace --lr 1e-3 --batch_size 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.mamba_pretrain.multivariate_classifier import MultivariateMambaClassifier
from imts_benchmark.shared_data.uea_classification_datamodule import (
    UEAClassificationDataModule,
)


# Per-dataset defaults inherited from RoMAE Table 12, with label_smoothing
# converted from RoMAE's confidence-c convention (App. A.1: "reducing each
# correct class label from 1 to a confidence value c") to PyTorch's smoothing-
# amount convention (label_smoothing=0 means no smoothing). Mapping: p = 1 - c.
# RoMAE c values: BM=1.0, CT=0.9, EP=0.8, HB=1.0, LSST=0.9
# PyTorch p:      BM=0.0, CT=0.1, EP=0.2, HB=0.0, LSST=0.1
ROMAE_DATASET_DEFAULTS = {
    "BasicMotions":          {"batch_size": 8,  "label_smoothing": 0.0, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "CharacterTrajectories": {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "Epilepsy":              {"batch_size": 16, "label_smoothing": 0.2, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "Heartbeat":             {"batch_size": 16, "label_smoothing": 0.0, "grad_clip": 2.0,  "grid_K": 128, "head_dropout": 0.0},
    "LSST":                  {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "grid_K": 64,  "head_dropout": 0.2},
}


def parse_args():
    p = argparse.ArgumentParser(description="Mamba-MV UEA classification")

    # Data.
    p.add_argument("--dataset", type=str, required=True,
                   choices=list(ROMAE_DATASET_DEFAULTS.keys()))
    p.add_argument("--data_root", type=str, required=True,
                   help="Path containing <Dataset>/<Dataset>_TRAIN.ts and _TEST.ts")
    p.add_argument("--drop_rate", type=float, default=0.3)
    p.add_argument("--drop_seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for the train/val 80/20 stratified split. Fixed "
                        "across our HPO and final eval so val identity is stable.")
    p.add_argument("--num_workers", type=int, default=2)

    # Architecture.
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--n_perv_layer", type=int, default=3)
    p.add_argument("--n_fusion_blocks", type=int, default=3)
    p.add_argument("--n_heads_varattn", type=int, default=4)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--n_freq", type=int, default=8)
    p.add_argument("--dt_mode", type=str, default="replace",
                   choices=["learned", "replace", "concat"])
    p.add_argument("--head_use_avail_mask", type=int, default=1)
    # grid_K override (else inherit from ROMAE_DATASET_DEFAULTS).
    p.add_argument("--grid_K", type=int, default=-1)
    # head_dropout override (sentinel -1.0 means "use per-dataset default").
    p.add_argument("--head_dropout", type=float, default=-1.0)

    # Optimization.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--warmup_pct", type=float, default=0.1,
                   help="LR warmup as fraction of total training steps.")
    # Per-dataset overrides (else use ROMAE_DATASET_DEFAULTS).
    p.add_argument("--batch_size", type=int, default=-1)
    p.add_argument("--label_smoothing", type=float, default=-1.0)
    p.add_argument("--grad_clip", type=float, default=-1.0)

    # Run management.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./uea_cls_outputs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--precision", type=str, default="32-true",
                   help="Lightning precision. Default 32-true matches the "
                        "forecasting trainer; the Mamba selective_scan CUDA "
                        "kernel asserts delta.dtype == u.dtype, which fails "
                        "under bf16-mixed unless deltat is explicitly cast.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: small max_epochs and small patience.")

    return p.parse_args()


def main():
    args = parse_args()

    # Apply RoMAE per-dataset defaults where CLI args are unset (-1).
    defaults = ROMAE_DATASET_DEFAULTS[args.dataset]
    if args.batch_size < 0:
        args.batch_size = defaults["batch_size"]
    if args.label_smoothing < 0:
        args.label_smoothing = defaults["label_smoothing"]
    if args.grad_clip < 0:
        args.grad_clip = defaults["grad_clip"]
    if args.grid_K < 0:
        args.grid_K = defaults["grid_K"]
    if args.head_dropout < 0:
        args.head_dropout = defaults["head_dropout"]

    if args.smoke:
        args.max_epochs = min(args.max_epochs, 30)
        args.patience = min(args.patience, 10)

    pl.seed_everything(args.seed, workers=True)

    run_name = args.run_name or (
        f"{args.dataset}_{args.dt_mode}_lr{args.lr}_bs{args.batch_size}"
        f"_drop{args.head_dropout}_seed{args.seed}"
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    dm = UEAClassificationDataModule(
        data_root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        drop_rate=args.drop_rate,
        drop_seed=args.drop_seed,
        normalize=True,
        t_max=1.0,
    )
    dm.setup()

    # Estimated total training steps for the LR schedule (cap at max_epochs).
    steps_per_epoch = max(1, math.ceil(len(dm.train_ds) / args.batch_size))
    num_training_steps = max(1, steps_per_epoch * args.max_epochs)
    num_warmup_steps = max(1, int(num_training_steps * args.warmup_pct))

    # ---- Model ----
    model = MultivariateMambaClassifier(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_vars=dm.n_vars,
        n_classes=dm.n_classes,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        n_heads_varattn=args.n_heads_varattn,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        grid_K=args.grid_K,
        t_max=1.0,
        n_freq=args.n_freq,
        head_use_avail_mask=bool(args.head_use_avail_mask),
        head_dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        class_weights=dm.class_weights,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run={run_name}] n_vars={dm.n_vars}, n_classes={dm.n_classes}, "
          f"|train|={len(dm.train_ds)}, |val|={len(dm.val_ds)}, |test|={len(dm.test_ds)}, "
          f"n_params={n_params:,}")

    # ---- Trainer ----
    callbacks = [
        EarlyStopping(monitor="val/acc", mode="max", patience=args.patience),
    ]
    # LearningRateMonitor requires a logger; skipped for now since smoke and
    # HPO runs use logger=False. Re-enable when wandb logger is wired in.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        deterministic=False,  # Mamba's SSM kernel isn't fully deterministic; cosine schedule + seed cover most.
        log_every_n_steps=10,
        logger=False,         # smoke runs don't need logging; HPO/final wire wandb separately.
        enable_checkpointing=False,  # We never load the saved .ckpt; metrics come from
                                     # callback_metrics + immediate trainer.test(). Default
                                     # ckpt path is CWD-relative, and 30 array tasks cd to
                                     # the same CODE_DIR -> race on shared checkpoints/ dir.
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    fit_time = time.time() - t0

    # Final-epoch val metrics, used by the v2 aggregator. We capture both
    # acc and macro_f1; the picker may use either (collapse detection compares
    # them).
    best_val_acc = float(trainer.callback_metrics.get("val/acc", torch.tensor(0.0)).item())
    val_macro_f1 = float(trainer.callback_metrics.get("val/macro_f1", torch.tensor(0.0)).item())
    print(f"[done] val/acc={best_val_acc:.4f}, val/macro_f1={val_macro_f1:.4f}, "
          f"fit_time={fit_time:.1f}s")

    # Test eval.
    test_metrics = trainer.test(model, datamodule=dm)[0]
    test_acc = float(test_metrics.get("test/acc_final", test_metrics.get("test/acc", 0.0)))
    macro_f1 = float(test_metrics.get("test/macro_f1", 0.0))
    print(f"[test] acc={test_acc:.4f}, macro_f1={macro_f1:.4f}")

    # Persist a flat summary for downstream aggregation.
    summary = dict(
        dataset=args.dataset,
        dt_mode=args.dt_mode,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        grid_K=args.grid_K,
        head_dropout=args.head_dropout,
        n_vars=dm.n_vars,
        n_classes=dm.n_classes,
        n_params=n_params,
        best_val_acc=best_val_acc,
        val_macro_f1=val_macro_f1,
        test_acc=test_acc,
        macro_f1=macro_f1,
        fit_time=fit_time,
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
