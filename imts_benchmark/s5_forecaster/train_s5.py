"""Train the S5 forecaster on the multisin IMTS benchmark.

Mirrors the pattern of train_mv.py / train_romae.py / train_mtan.py. Shared
optimizer/scheduler/precision args come from shared_config.fair_defaults.

Usage:
    python -m imts_benchmark.s5_forecaster.train_s5 \\
        --regime multisin_med_irreg --seed 1 --output_dir /path/to/out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.shared_config.fair_defaults import (
    add_fair_args,
    apply_auto_meta,
    derive_phase_tags,
)
from imts_benchmark.shared_config.wandb_lightning import (
    build_wandb_logger,
    log_wandb_after_fit,
    log_wandb_run_summary,
)
from imts_benchmark.shared_data.multivariate_datamodule import (
    MultivariateSinusoidalDataModule,
)
from imts_benchmark.s5_forecaster.s5_forecaster import S5Forecaster


def main():
    parser = argparse.ArgumentParser(
        description="S5 forecasting (Kwaijtaal s5-pytorch port of Smith et al. 2023)"
    )
    add_fair_args(parser)

    # Paper-native config (Smith et al. ICLR 2023, S5_pendulum/run_train.py).
    # Previous 384/96 was chosen to match a 7.8M param budget but (combined
    # with weight decay applied to SSM-spectral params) caused 0/5 seeds to
    # escape mean-prediction. See docs/RESULTS_phase2.md "Per-model recipe
    # deviations". Native lands at ~1.4M params.
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--state_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--ff_mult", type=float, default=4.0)
    parser.add_argument("--time_emb_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    args = apply_auto_meta(args)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[args] train_batch_size={args.train_batch_size}  "
          f"val_batch_size={args.val_batch_size}  "
          f"accumulate_grad_batches={args.accumulate_grad_batches}  "
          f"precision={args.precision}  "
          f"d_model={args.d_model} n_layers={args.n_layers} state_dim={args.state_dim}",
          flush=True)

    model = S5Forecaster(
        n_vars=args.n_vars,
        d_model=args.d_model,
        state_dim=args.state_dim,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
        t_max=args.t_max,
        history=args.history,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {n_params:,}", flush=True)

    dm = MultivariateSinusoidalDataModule(
        data_root=args.data_root,
        regime=args.regime,
        format="per_variate",
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="best",
            monitor="val/mse",
            mode="min",
            save_top_k=1,
            save_last=False,
        ),
        EarlyStopping(monitor="val/mse", patience=args.patience, mode="min"),
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = build_wandb_logger(args, extra_config={"n_params": n_params})
    if wandb_logger is not False:
        wandb_logger.log_hyperparams(derive_phase_tags(args))

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

    # See train_mv.py for context: long fits on PhysioNet trigger a
    # SIGABRT during trainer.test() below Python. Defensive cleanup
    # before test phase consistently across S5/RoMAE/Mamba-MV.
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}", flush=True)
        print(f"Best val/mse: {float(callbacks[0].best_model_score):.6f}", flush=True)
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("No checkpoint saved, testing with last model", flush=True)
        trainer.test(model, dm)

    metrics_row = {
        "model": "s5",
        "variant": "default",
        "regime": args.regime,
        "seed": args.seed,
        "n_params": n_params,
        "wall_fit_sec": round(wall_fit, 2),
    }
    if hasattr(model, "_test_agg"):
        for k, v in model._test_agg.items():
            metrics_row[f"test_{k}"] = round(v, 6)

    out_csv = os.path.join(args.output_dir, "test_metrics.csv")
    # Write header + single row.
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        w.writeheader()
        w.writerow(metrics_row)
    print(f"Test metrics saved to {out_csv}", flush=True)

    # Per-sample JSONL.
    per_sample_path = os.path.join(args.output_dir, "per_sample.jsonl")
    with open(per_sample_path, "w") as f:
        for o in model._test_outputs:
            f.write(json.dumps(o) + "\n")
    print(f"Per-sample results: {per_sample_path}", flush=True)

    for k, v in metrics_row.items():
        print(f"  {k}: {v}", flush=True)

    log_wandb_run_summary(wandb_logger, metrics_row)


if __name__ == "__main__":
    main()
