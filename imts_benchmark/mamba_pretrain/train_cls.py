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
from pytorch_lightning.loggers import CSVLogger

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.mamba_pretrain.multivariate_classifier import MultivariateMambaClassifier
from imts_benchmark.shared_config.wandb_lightning import (
    build_wandb_logger,
    log_wandb_after_fit,
    log_wandb_run_summary,
)
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
    # The 5 RoMAE Table 12 datasets — exact RoMAE App. C.2 values.
    "BasicMotions":             {"batch_size": 8,  "label_smoothing": 0.0, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "CharacterTrajectories":    {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "Epilepsy":                 {"batch_size": 16, "label_smoothing": 0.2, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "Heartbeat":                {"batch_size": 16, "label_smoothing": 0.0, "grad_clip": 2.0,  "grid_K": 128, "head_dropout": 0.0},
    "LSST":                     {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "grid_K": 64,  "head_dropout": 0.2},
    # ContiFormer Table 8 datasets without RoMAE Table 12 values. Defaults
    # picked from ContiFormer Appendix C.2.1 (lr=1e-2 SGD, bs=64, lr=1e-3
    # AdamW per their actual repo) plus light per-dataset adjustments for
    # small training-set sizes (smaller bs) and strong overfitting risk
    # (more head_dropout). label_smoothing=0.1 is the common-default for
    # multi-class; grid_K=128 for short sequences, smaller for very long.
    "ArticularyWordRecognition": {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0, "grid_K": 128, "head_dropout": 0.0},
    "ERing":                    {"batch_size": 8,  "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "FingerMovements":          {"batch_size": 16, "label_smoothing": 0.0, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.0},
    "HandMovementDirection":    {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "Handwriting":              {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "JapaneseVowels":           {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.0},
    "Libras":                   {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.2},
    "NATOPS":                   {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.0},
    "PenDigits":                {"batch_size": 64, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 32,  "head_dropout": 0.0},
    "RacketSports":             {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.0},
    "SelfRegulationSCP1":       {"batch_size": 16, "label_smoothing": 0.0, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "SpokenArabicDigits":       {"batch_size": 64, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "UWaveGestureLibrary":      {"batch_size": 32, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    # High-V datasets: V-axis attention is O(V^2). Keep physical bs=1 and
    # use --accumulate_grad_batches 16 in the sbatch to recover effective
    # bs=16. grid_K halved to 64 to roughly keep the [B, K, V, V] tensor
    # comparable to a normal cell. bf16-mixed precision required.
    "DuckDuckGeese":            {"batch_size": 1,  "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.2},
    "PEMS-SF":                  {"batch_size": 1,  "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 64,  "head_dropout": 0.0},
}


# Architecture presets. Width-only axis, depth fixed at 3+3 (per the
# colleague's BIG-vs-SMALL framing on the IMM/PhysioNet runs):
#     SMALL  d=64   ≈ 259K params   (head-of-table sweet spot from IMM HPO)
#     BASE   d=256  ≈ 3.5M params   (the historical UEA default we've been running)
#     BIG    d=384  ≈ 7.8M params   (colleague's BIG)
# Apply via --model_size; depth/heads/state/conv/expand remain on their
# individual CLI defaults so this preset only moves width.
MODEL_SIZE_PRESETS = {
    "small": {"d_model": 64,  "d_hidden": 64,  "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
    "base":  {"d_model": 256, "d_hidden": 256, "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
    "big":   {"d_model": 384, "d_hidden": 384, "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
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
    p.add_argument("--model_size", type=str, default="base",
                   choices=list(MODEL_SIZE_PRESETS.keys()),
                   help="Apply a preset for d_model/d_hidden/n_perv_layer/"
                        "n_fusion_blocks/n_heads_varattn. Individual --d_model "
                        "etc. flags below are ignored unless model_size=base "
                        "(in which case they are honoured for backward compat).")
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
    # Lets us run high-V datasets (DDG V=1345, PEMS V=963) at physical
    # batch_size=1 while keeping effective batch_size=16 — the V-axis
    # attention is O(V^2) in memory so B=1 is the only feasible per-step
    # config. No-op (=1) for normal datasets.
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    # Loss weighting. Default ON to match prior runs; flip OFF to match RoMAE,
    # which uses only label_smoothing and no inverse-frequency class weights.
    p.add_argument("--use_class_weights", type=int, default=1,
                   choices=[0, 1],
                   help="If 0, override dm.class_weights with all-ones so the "
                        "cross-entropy is unweighted (matches RoMAE App. C.2).")

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

    # Weights & Biases (online cloud logging). We always keep CSVLogger as a
    # local backup; wandb adds live dashboards under the configured project.
    p.add_argument("--use_wandb", action="store_true",
                   help="Stream per-step + per-epoch metrics to wandb.")
    p.add_argument("--wandb_project", type=str, default="TSKing")
    p.add_argument("--wandb_entity", type=str, default="magicslabnorthwestern",
                   help="wandb entity (team / user). The ML lab account.")
    p.add_argument("--wandb_run_name", type=str, default="",
                   help="If empty, a descriptive default is generated from "
                        "dataset/dt_mode/size/seed.")

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

    # Apply architecture preset (overrides individual --d_model etc. unless
    # the user picked the "base" preset, in which case the CLI values stand
    # so existing scripts don't shift behaviour silently).
    if args.model_size != "base":
        preset = MODEL_SIZE_PRESETS[args.model_size]
        for k, v in preset.items():
            setattr(args, k, v)

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

    # Class weights toggle: optionally drop the inverse-frequency weighting so
    # the loss matches RoMAE App. C.2 (label_smoothing only).
    cw = dm.class_weights if args.use_class_weights else None

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
        class_weights=cw,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run={run_name}] n_vars={dm.n_vars}, n_classes={dm.n_classes}, "
          f"|train|={len(dm.train_ds)}, |val|={len(dm.val_ds)}, |test|={len(dm.test_ds)}, "
          f"n_params={n_params:,}")

    # ---- Trainer ----
    callbacks = [
        EarlyStopping(monitor="val/acc", mode="max", patience=args.patience),
    ]
    # CSVLogger writes per-epoch metrics.csv to <output_dir>/<run_name>/csv/
    # version_0/metrics.csv — used downstream for loss-curve plots / convergence
    # diagnosis. Lightweight (just a CSV file). Replaces the prior logger=False.
    csv_logger = CSVLogger(
        save_dir=str(out_dir),
        name="csv",
        flush_logs_every_n_steps=50,
    )

    # Optional wandb dashboard. Defaults to a descriptive run name so sweeps
    # are easy to filter in the UI.
    if args.use_wandb and not args.wandb_run_name:
        args.wandb_run_name = (
            f"uea_cls_{args.dataset}_{args.dt_mode}_size{args.model_size}"
            f"_seed{args.seed}_drop{args.drop_rate}"
        )
    wandb_logger = build_wandb_logger(args)
    loggers = [csv_logger]
    if wandb_logger is not False:
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        deterministic=False,  # Mamba's SSM kernel isn't fully deterministic; cosine schedule + seed cover most.
        log_every_n_steps=10,
        logger=loggers,
        enable_checkpointing=False,  # We never load the saved .ckpt; metrics come from
                                     # callback_metrics + immediate trainer.test(). Default
                                     # ckpt path is CWD-relative, and 30 array tasks cd to
                                     # the same CODE_DIR -> race on shared checkpoints/ dir.
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    fit_time = time.time() - t0
    log_wandb_after_fit(wandb_logger, fit_time, best_val_mse=None)

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
        model_size=args.model_size,
        d_model=args.d_model,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        use_class_weights=bool(args.use_class_weights),
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
    log_wandb_run_summary(wandb_logger, summary)

    # Per-sample test predictions for downstream confusion-matrix /
    # per-class diagnostics. Read by eval/uea_confusion.py. We pull from the
    # `_test_agg` dict stashed by the model in on_test_epoch_end. Shape:
    # one JSON object with parallel labels/preds/logits arrays.
    agg = getattr(model, "_test_agg", None)
    if agg and "labels" in agg:
        preds_path = out_dir / "test_predictions.json"
        with open(preds_path, "w") as f:
            json.dump(
                dict(
                    dataset=args.dataset,
                    dt_mode=args.dt_mode,
                    seed=args.seed,
                    n_classes=dm.n_classes,
                    label_to_idx=dm.label_to_idx,
                    class_weights=dm.class_weights.tolist(),
                    labels=agg["labels"],
                    preds=agg["preds"],
                    logits=agg["logits"],
                ),
                f,
            )


if __name__ == "__main__":
    main()
