"""Train the Mamba-RoPE hybrid forecaster.

Per-variate Mamba SSM front-end + axial-RoPE attention back-end.

Usage:
    python -m imts_benchmark.mamba_rope.train_mamba_rope \
        --data_root /path/to/tpatchgnn_data \
        --regime ushcn --auto_meta \
        --seed 1 --output_dir /path/to/output --use_wandb
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
from imts_benchmark.mamba_rope.forecaster import MambaRoPEForecaster


def main():
    parser = argparse.ArgumentParser(description="Mamba-RoPE hybrid forecasting")
    add_fair_args(parser)

    # Model-specific knobs
    parser.add_argument("--dt_mode", type=str, default="learned",
                        choices=["learned", "replace", "concat"])
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_perv_layer", type=int, default=3)
    parser.add_argument("--n_attn_layer", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--p_rope", type=float, default=0.75)
    parser.add_argument("--max_len_per_var", type=int, default=256)

    args = parser.parse_args()
    args = apply_auto_meta(args)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = MambaRoPEForecaster(
        d_model=args.d_model,
        n_perv_layer=args.n_perv_layer,
        n_attn_layer=args.n_attn_layer,
        nhead=args.nhead,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        p_rope=args.p_rope,
        n_vars=args.n_vars,
        max_len_per_var=args.max_len_per_var,
        history=args.history,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps,
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

    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}")
        print(f"Best val/mse: {float(callbacks[0].best_model_score):.6f}")
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("No checkpoint saved, testing with last model")
        trainer.test(model, dm)

    metrics_row = {"model": "mamba_rope", "variant": args.dt_mode,
                   "regime": args.regime, "seed": args.seed}
    metrics_row["n_params"] = n_params
    metrics_row["wall_fit_sec"] = round(wall_fit, 2)

    if hasattr(model, "_test_agg"):
        metrics_row.update({f"test_{k}": round(float(v), 6)
                            for k, v in model._test_agg.items()})

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

    wandb_summary = {k: v for k, v in metrics_row.items() if k != "regime"}
    log_wandb_run_summary(wandb_logger, wandb_summary)


if __name__ == "__main__":
    main()
