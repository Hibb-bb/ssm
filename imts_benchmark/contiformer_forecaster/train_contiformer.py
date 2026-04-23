"""Train ContiFormer as a forecaster on the sparse-flat synthetic data.

Sized to ~7.65M params via d_model=320, d_inner=1024, n_layers=6, n_head=4,
d_k=80 (within +/-10% of the 7.8M shared budget). Uses identical optimizer /
scheduler / batch / precision as Mamba-MV / RoMAE / mTAN trainers via
shared_config.fair_defaults.

Usage:
    python -m imts_benchmark.contiformer_forecaster.train_contiformer \
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
from imts_benchmark.contiformer_forecaster.contiformer_forecaster import (
    ContiFormerForecaster,
)


def main():
    parser = argparse.ArgumentParser(description="ContiFormer forecasting (continuous-time transformer)")
    add_fair_args(parser)

    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--d_inner", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--d_k", type=int, default=80)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--atol_ode", type=float, default=1e-1)
    parser.add_argument("--rtol_ode", type=float, default=1e-1)
    parser.add_argument("--method_ode", type=str, default="rk4")
    parser.add_argument("--actfn_ode", type=str, default="softplus")  # matches upstream physiopro default

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Log resolved batch/memory-sensitive args so OOM diagnostics can be
    # read directly from the .out file.
    print(f"[args] train_batch_size={args.train_batch_size}  "
          f"val_batch_size={args.val_batch_size}  "
          f"accumulate_grad_batches={args.accumulate_grad_batches}  "
          f"precision={args.precision}", flush=True)

    model = ContiFormerForecaster(
        n_vars=args.n_vars,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        dropout=args.dropout,
        atol_ode=args.atol_ode,
        rtol_ode=args.rtol_ode,
        method_ode=args.method_ode,
        actfn_ode=args.actfn_ode,
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
        "model": "contiformer", "variant": "default", "regime": args.regime, "seed": args.seed,
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
