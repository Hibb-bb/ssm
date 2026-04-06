"""
CLI training script for Mamba sinusoidal forecasting.

Usage:
    python -m mamba_forecaster.train \
        --dt_mode learned \
        --data_root /path/to/tpatchgnn_data_nobs160 \
        --irregularity high_irreg \
        --seed 1 \
        --output_dir /path/to/output
"""

import argparse
import csv
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from .mamba_forecaster import MambaForecaster
from .sinusoidal_datamodule import SinusoidalDataModule


def main():
    parser = argparse.ArgumentParser(description="Mamba sinusoidal forecasting")
    parser.add_argument(
        "--dt_mode",
        type=str,
        default="learned",
        choices=["learned", "replace", "additive"],
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--irregularity", type=str, default="high_irreg")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--num_training_steps", type=int, default=1600)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--history", type=float, default=7.0)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    model = MambaForecaster(
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        history=args.history,
    )

    dm = SinusoidalDataModule(
        data_root=args.data_root,
        irregularity=args.irregularity,
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
        EarlyStopping(
            monitor="val/mse",
            patience=args.patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=args.output_dir,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)

    best_ckpt = callbacks[0].best_model_path
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}")
        print(f"Best val/mse: {callbacks[0].best_model_score:.6f}")
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("No checkpoint saved, testing with last model")
        trainer.test(model, dm)

    if hasattr(model, "_test_agg"):
        metrics = model._test_agg
        csv_path = os.path.join(args.output_dir, "test_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Test metrics saved to {csv_path}")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
