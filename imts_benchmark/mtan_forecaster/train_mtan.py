"""Train mTAN as a forecaster on the sparse-flat synthetic data.

Identical optimizer / scheduler / early-stopping / batch / precision as the
other imts_benchmark trainers via shared_config.fair_defaults. Architectural
defaults (rec_hidden=576, gen_hidden=576, latent_dim=64) chosen to land
within +/-10% of the 7.8M trainable param budget shared with Mamba-MV/RoMAE.

Usage:
    python -m imts_benchmark.mtan_forecaster.train_mtan \
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
from imts_benchmark.mtan_forecaster.mtan_forecaster import MTANForecaster


def main():
    parser = argparse.ArgumentParser(description="mTAN forecasting (Multi-Time Attention Networks)")
    add_fair_args(parser)

    # Paper-native config (Shukla & Marlin, ICLR 2021). Previous 576/576/64
    # was chosen to match a 7.8M param budget but caused 0/5 seeds to escape
    # mean-prediction. See docs/RESULTS_phase2.md "Per-model recipe deviations".
    parser.add_argument("--rec_hidden", type=int, default=32)
    parser.add_argument("--gen_hidden", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=20)
    parser.add_argument("--embed_time", type=int, default=128)
    parser.add_argument("--num_ref_points", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--learn_emb", action="store_true", default=True)

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = MTANForecaster(
        n_vars=args.n_vars,
        rec_hidden=args.rec_hidden,
        gen_hidden=args.gen_hidden,
        latent_dim=args.latent_dim,
        embed_time=args.embed_time,
        num_ref_points=args.num_ref_points,
        num_heads=args.num_heads,
        learn_emb=args.learn_emb,
        t_max=args.t_max,
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
        "model": "mtan", "variant": "default", "regime": args.regime, "seed": args.seed,
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
