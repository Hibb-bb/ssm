"""Train the RoMAE forecaster on the sparse-flat synthetic data.

Identical optimizer / scheduler / early-stopping / batch-size / precision as
mamba_mv/train_mv.py via shared_config.fair_defaults. Only the model-specific
knobs (encoder/decoder dims) differ.

Usage:
    python -m imts_benchmark.romae_forecaster.train_romae \
        --regime sparse_dependent --seed 1 --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
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

from imts_benchmark.shared_config.fair_defaults import add_fair_args
from imts_benchmark.shared_config.wandb_lightning import (
    build_wandb_logger,
    log_wandb_after_fit,
    log_wandb_run_summary,
)
from imts_benchmark.shared_data.multivariate_datamodule import (
    MultivariateSinusoidalDataModule,
)
from imts_benchmark.romae_forecaster.romae_forecaster import RoMAEForecaster


def main():
    parser = argparse.ArgumentParser(description="RoMAE forecasting (transformer baseline)")
    add_fair_args(parser)

    parser.add_argument("--enc_d_model", type=int, default=288)
    parser.add_argument("--enc_nhead", type=int, default=6)
    parser.add_argument("--enc_depth", type=int, default=7)
    parser.add_argument("--dec_d_model", type=int, default=180)
    parser.add_argument("--dec_nhead", type=int, default=3)
    parser.add_argument("--dec_depth", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=1500)
    parser.add_argument("--p_rope_val", type=float, default=0.75)

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = RoMAEForecaster(
        enc_d_model=args.enc_d_model,
        enc_nhead=args.enc_nhead,
        enc_depth=args.enc_depth,
        dec_d_model=args.dec_d_model,
        dec_nhead=args.dec_nhead,
        dec_depth=args.dec_depth,
        max_len=args.max_len,
        p_rope_val=args.p_rope_val,
        n_vars=args.n_vars,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        history=args.history,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {n_params:,}")

    dm = MultivariateSinusoidalDataModule(
        data_root=args.data_root,
        regime=args.regime,
        format="flat_tokens",
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
        print(f"Best checkpoint: {best_ckpt}")
        print(f"Best val/mse: {float(callbacks[0].best_model_score):.6f}")
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("No checkpoint saved, testing with last model")
        trainer.test(model, dm)

    metrics_row = {
        "model": "romae", "variant": "default", "regime": args.regime, "seed": args.seed,
        "n_params": n_params, "wall_fit_sec": round(wall_fit, 2),
    }
    if hasattr(model, "_test_agg"):
        metrics_row.update({f"test_{k}": round(float(v), 6) for k, v in model._test_agg.items()})

    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)
    print(f"Test metrics saved to {csv_path}")

    if hasattr(model, "_test_outputs") and model._test_outputs:
        samples_path = os.path.join(args.output_dir, "per_sample.jsonl")
        with open(samples_path, "w") as f:
            for o in model._test_outputs:
                f.write(json.dumps(o) + "\n")
        print(f"Per-sample results: {samples_path}")

    for k, v in metrics_row.items():
        print(f"  {k}: {v}")

    log_wandb_run_summary(wandb_logger, metrics_row)


if __name__ == "__main__":
    main()
