"""Train the slot-anonymous Mamba-MV variant (``mamba_pretrain``) on the
sparse-flat synthetic / IMTS data, using the same fair-defaults recipe as
``imts_benchmark.mamba_mv.train_mv``.

The architecture comes from ``mamba_pretrain.multivariate_forecaster`` —
it differs from ``mamba_mv`` only in the controlled axes we want to test:

  - no per-variate slot embedding (``shared_grid``),
  - shared decoder head across variates (``query_readout``),
  - Moirai-style same/different attention bias instead of variate IDs
    (``variable_axis_attention``),
  - bf16-mixed dtype safety in the SSM kernel (``mamba_block``).

Training data path, datamodule, optimizer/scheduler defaults, val metric
key (``val/mse``), checkpoint/early-stop monitor, and the test-side
emission (``test_metrics.csv`` + ``per_sample.jsonl``) are all unchanged
so HPO / aggregator / confirm pipelines compare like-for-like.

Optional pass-through knobs new to this trainer (defaults preserve
`mamba_mv` behavior for the comparison run):

  --loss_type {mse,huber}   default mse        (huber: Time-MoE §3.2.2)
  --huber_delta             default 1.0
  --lr_schedule {cosine,constant,multistep}    default cosine
  --max_dim                 default None       (FM-style padded slot count)

Usage:
    python -m imts_benchmark.mamba_pretrain.train_mv \\
        --dt_mode replace \\
        --regime physionet \\
        --auto_meta \\
        --seed 1 \\
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
from imts_benchmark.mamba_pretrain.multivariate_forecaster import (
    MultivariateMambaForecaster,
    MultivariateMambaSandwichForecaster,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multivariate Mamba forecasting (slot-anonymous / pretrain-arch variant)"
    )
    add_fair_args(parser)

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
        "after fusion; canonical sandwich uses --n_perv_layer 2 --n_fusion_blocks 2.",
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

    # Pretrain-arch passthroughs. Defaults preserve mamba_mv behavior so
    # the head-to-head HPO compares architecture only.
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=("mse", "huber"),
        help="mse keeps val/mse as the selection metric (matches mamba_mv); "
        "huber uses Smooth-L1 loss (Time-MoE §3.2.2) and logs val/huber.",
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=("cosine", "constant", "multistep"),
    )
    parser.add_argument(
        "--max_dim",
        type=int,
        default=None,
        help="Optional FM-style padded variate-slot count. Leave unset for "
        "downstream IMTS training (n_vars from --auto_meta is used).",
    )

    args = parser.parse_args()
    args = apply_auto_meta(args)
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # The pretrain-arch model selects val/mse vs val/huber based on
    # loss_type. ModelCheckpoint/EarlyStopping below monitor the matching
    # key so HPO uses a consistent metric.
    monitor_key = "val/huber" if args.loss_type == "huber" else "val/mse"

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
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        lr_schedule=args.lr_schedule,
    )
    if args.max_dim is not None:
        model_kw["max_dim"] = args.max_dim
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
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            save_last=False,
        ),
        EarlyStopping(monitor=monitor_key, patience=args.patience, mode="min"),
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

    # Mirror mamba_mv: drop CUDA caches before test() to avoid SIGABRT
    # observed on long PhysioNet fits on Delta GH (jobs 2197568, 2197588).
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}", flush=True)
        if callbacks[0].best_model_score is not None:
            print(f"Best {monitor_key}: {float(callbacks[0].best_model_score):.6f}",
                  flush=True)
        trainer.test(model, dm, ckpt_path=best_ckpt)
    else:
        print("No checkpoint saved, testing with last model", flush=True)
        trainer.test(model, dm)

    metrics_row = {
        "model": "mamba_pretrain",
        "variant": args.dt_mode,
        "regime": args.regime,
        "seed": args.seed,
    }
    metrics_row["n_params"] = n_params
    metrics_row["wall_fit_sec"] = round(wall_fit, 2)

    if hasattr(model, "_test_agg"):
        metrics_row.update(
            {f"test_{k}": round(float(v), 6) for k, v in model._test_agg.items()}
        )

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
