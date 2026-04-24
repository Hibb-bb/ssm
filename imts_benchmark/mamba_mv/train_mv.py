"""Train the multivariate Mamba forecaster on the sparse-flat synthetic data.

Usage:
    python -m imts_benchmark.mamba_mv.train_mv \
        --dt_mode replace \
        --regime sparse_dependent \
        --seed 1 \
        --output_dir /path/to/output
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
_SSM_DK = _THIS_DIR.parents[1]  # .../ssm_dk
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
from imts_benchmark.mamba_mv.multivariate_forecaster import (
    MultivariateMambaForecaster,
    MultivariateMambaSandwichForecaster,
)


def main():
    parser = argparse.ArgumentParser(description="Multivariate Mamba forecasting (paper arch)")
    add_fair_args(parser)

    # Model-specific knobs.
    parser.add_argument(
        "--dt_mode",
        type=str,
        default="replace",
        choices=["learned", "replace", "additive", "concat"],
    )
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--d_hidden", type=int, default=384)
    parser.add_argument(
        "--mamba_arch",
        type=str,
        default="standard",
        choices=("standard", "sandwich"),
        help="standard: n_perv_layer × irregular SSM, n_fusion_blocks × (attn+grid mamba). "
        "sandwich: same kwargs but adds n_tail_grid_mamba extra TemporalMambaOnGrid "
        "after fusion; canonical sandwich uses --n_perv_layer 2 --n_fusion_blocks 2 "
        "(defaults below stay 3,3 for standard).",
    )
    parser.add_argument("--n_tail_grid_mamba", type=int, default=2)
    parser.add_argument("--n_perv_layer", type=int, default=3)
    parser.add_argument("--n_fusion_blocks", type=int, default=3)
    parser.add_argument("--n_heads_varattn", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--grid_K", type=int, default=128)
    parser.add_argument("--n_freq", type=int, default=8)

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model_kw = dict(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_vars=args.n_vars,
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
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
        history=args.history,
    )
    if args.mamba_arch == "sandwich":
        model = MultivariateMambaSandwichForecaster(
            n_tail_grid_mamba=args.n_tail_grid_mamba,
            **model_kw,
        )
    else:
        model = MultivariateMambaForecaster(**model_kw)
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
    # Regime is not in LightningModule.save_hyperparameters(); log it once as W&B
    # config (same path as model hparams), not as a time-series metric.
    if wandb_logger is not False:
        wandb_logger.log_hyperparams({"regime": args.regime})

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

    metrics_row = {"model": "mamba_mv", "variant": args.dt_mode, "regime": args.regime, "seed": args.seed}
    metrics_row["n_params"] = n_params
    metrics_row["wall_fit_sec"] = round(wall_fit, 2)

    if hasattr(model, "_test_agg"):
        metrics_row.update({f"test_{k}": round(float(v), 6) for k, v in model._test_agg.items()})

    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)
    print(f"Test metrics saved to {csv_path}")

    # Also dump per-sample outputs for downstream significance tests.
    if hasattr(model, "_test_outputs") and model._test_outputs:
        samples_path = os.path.join(args.output_dir, "per_sample.jsonl")
        with open(samples_path, "w") as f:
            for o in model._test_outputs:
                f.write(json.dumps(o) + "\n")
        print(f"Per-sample results: {samples_path}")

    for k, v in metrics_row.items():
        print(f"  {k}: {v}")

    # Keep regime in wandb.config only (see log_hyperparams above), not run/regime scalars.
    wandb_summary = {k: v for k, v in metrics_row.items() if k != "regime"}
    log_wandb_run_summary(wandb_logger, wandb_summary)


if __name__ == "__main__":
    main()
