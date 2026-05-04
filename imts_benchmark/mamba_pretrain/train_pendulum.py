"""Train ``mamba_pretrain`` (slot-anonymous variant) on the Pendulum
image-regression task.

This is a sibling of ``imts_benchmark.mamba_pretrain.train_mv`` for the
Pendulum task (Becker 2019; Schirmer 2022; Smith 2023; Zivanovic 2025).
Per-timestep regression on irregular 24x24 pendulum images, predicting
(sin theta_t, cos theta_t) at every observed timestep.

Key differences from train_mv:
  - No ``add_fair_args``: Pendulum is per-token regression with its own
    optimizer recipe (matching the published baseline protocol). The
    forecasting flags ``--regime``, ``--history``, ``--t_max``,
    ``--auto_meta`` don't apply.
  - Datamodule emits image batches (V=1, single image stream).
  - Loss is per-timestep MSE on (sin, cos), no history split.
  - Optional Gaussian-NLL head (matches S5/CRU/RKN protocol).

Output layout matches mamba_pretrain (``test_metrics.csv`` +
``per_sample.jsonl`` under ``--output_dir``) so the Pendulum HPO
aggregator can glob the same way.

Usage:
    python -m imts_benchmark.mamba_pretrain.train_pendulum \\
        --dt_mode replace \\
        --lr 5e-3 \\
        --train_batch_size 32 \\
        --seed 1 \\
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_config.wandb_lightning import (
    build_wandb_logger,
    log_wandb_after_fit,
    log_wandb_run_summary,
)
from imts_benchmark.shared_data.pendulum_datamodule import PendulumDataModule
from imts_benchmark.mamba_pretrain.pendulum_forecaster import (
    PendulumMambaForecaster,
)


def main():
    parser = argparse.ArgumentParser(
        description="Mamba-pretrain Pendulum image-regression"
    )

    # ---- data ----
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root containing pendulum/{train,val,test}/")
    parser.add_argument("--regime", type=str, default="pendulum",
                        help="Subdirectory under data_root.")
    parser.add_argument("--num_workers", type=int, default=2)

    # ---- training ----
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Kept for API parity with train_mv; default 1 "
                             "since pendulum batches are small.")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="100 matches S5 Table 11 / CRU default. "
                             "RoMAE used 50 (Transformer family, different).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="32-true")

    # ---- optim ----
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="0.0 matches S5/CRU pendulum protocol "
                             "(SSM family). RoMAE used 0.01.")
    parser.add_argument("--warmup_pct", type=float, default=0.10,
                        help="Warmup as fraction of total optimizer steps. "
                             "Computed at runtime so it scales with B.")
    parser.add_argument("--warmup_min_steps", type=int, default=50)
    parser.add_argument(
        "--lr_schedule", type=str, default="cosine",
        choices=("cosine", "constant"),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout p applied after image_embed and at head input. "
             "v2 sweep adds this to address v1 overfitting (see RESULTS_pendulum.md §v2).",
    )
    parser.add_argument(
        "--front_end", type=str, default="linear",
        choices=("linear", "schirmer_cnn"),
        help="linear: Linear(576, d_hidden) patch-embed (v1 default; matches RoMAE). "
             "schirmer_cnn: Schirmer 2022 CNN encoder used by S5/CRU/RKN family "
             "(v3; matches S5 App. G.3.8 verbatim). See RESULTS_pendulum.md §v3-plan.",
    )

    # ---- model ----
    parser.add_argument(
        "--dt_mode", type=str, default="replace",
        choices=("learned", "replace", "additive", "concat"),
    )
    parser.add_argument(
        "--head_type", type=str, default="gaussian",
        choices=("linear", "gaussian"),
        help="gaussian: separate mu/log-var heads + Gaussian NLL "
             "(matches S5/CRU/RKN — our SSM family). "
             "linear: deterministic Linear->2 + MSE (matches RoMAE; ablation).",
    )
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--n_perv_layer", type=int, default=3)
    parser.add_argument("--n_fusion_blocks", type=int, default=3)
    parser.add_argument("--n_heads_varattn", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--grid_K", type=int, default=256)
    parser.add_argument("--n_freq", type=int, default=8)
    parser.add_argument(
        "--t_max", type=float, default=100.0,
        help="Max time horizon for the grid; CRU/S5 use T=100 on pendulum.",
    )

    # ---- wandb ----
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    monitor_key = "val/mse"  # selection metric for both head types

    model = PendulumMambaForecaster(
        head_type=args.head_type,
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        n_heads_varattn=args.n_heads_varattn,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        grid_K=args.grid_K,
        t_max=args.t_max,
        n_freq=args.n_freq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_pct=args.warmup_pct,
        warmup_min_steps=args.warmup_min_steps,
        lr_schedule=args.lr_schedule,
        dropout=args.dropout,
        front_end=args.front_end,
    )
    # Stash seed so test_metrics.csv can record it.
    model._seed = args.seed
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pendulum] Total trainable params: {n_params:,}")

    dm = PendulumDataModule(
        data_root=args.data_root,
        regime=args.regime,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best",
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            save_last=False,
        ),
        EarlyStopping(monitor=monitor_key, patience=args.patience, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = build_wandb_logger(args, extra_config={"n_params": n_params})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=wandb_logger if wandb_logger else False,
    )

    start = time.time()
    trainer.fit(model, dm)
    wall_fit = time.time() - start

    best_ckpt = callbacks[0].best_model_path
    best_val = None
    if best_ckpt and callbacks[0].best_model_score is not None:
        best_val = float(callbacks[0].best_model_score)
    log_wandb_after_fit(wandb_logger, wall_fit, best_val)

    if best_ckpt:
        print(f"[pendulum] Best checkpoint: {best_ckpt}", flush=True)
        if best_val is not None:
            print(f"[pendulum] Best {monitor_key}: {best_val:.6f}", flush=True)
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("[pendulum] No checkpoint saved, testing with last model",
              flush=True)
        trainer.test(model, dm)

    log_wandb_run_summary(wandb_logger, {
        "n_params": n_params,
        "best_val_mse": best_val,
        "wall_fit_seconds": wall_fit,
    })


if __name__ == "__main__":
    main()
