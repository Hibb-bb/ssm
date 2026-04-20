"""Centralised fair-comparison defaults shared by Mamba-MV and RoMAE trainers.

Any knob listed here MUST be identical across both models. Model-specific
architectural knobs (d_model, depth, etc.) are set by each model's own
train script so that parameter counts can be matched within +/-5%.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DATA_ROOT_DEFAULT = (
    "/projects/b1094/StarEmbed/skai_universal_forecaster/src/train/"
    "moirai/uni2ts_hongyu/ssm_dk/data"
)
LOG_ROOT_DEFAULT = (
    "/projects/b1094/StarEmbed/skai_universal_forecaster/output/log/mvcompare_v1"
)

HISTORY = 7.0
T_MAX = 10.0
N_VARS = 3

LR = 5e-4
WEIGHT_DECAY = 0.01
NUM_WARMUP_STEPS = 100
NUM_TRAINING_STEPS = 1600
MAX_EPOCHS = 200
PATIENCE = 20
GRADIENT_CLIP_VAL = 1.0
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 32
NUM_WORKERS = 2

SEEDS = (1, 2, 3, 4, 5)
REGIMES = ("sparse_independent", "sparse_dependent")


def add_fair_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the fair-comparison knobs. Both trainers call this."""
    parser.add_argument("--data_root", type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--regime", type=str, required=True, choices=REGIMES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num_warmup_steps", type=int, default=NUM_WARMUP_STEPS)
    parser.add_argument("--num_training_steps", type=int, default=NUM_TRAINING_STEPS)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--gradient_clip_val", type=float, default=GRADIENT_CLIP_VAL)

    parser.add_argument("--train_batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--val_batch_size", type=int, default=VAL_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    parser.add_argument("--history", type=float, default=HISTORY)
    parser.add_argument("--t_max", type=float, default=T_MAX)
    parser.add_argument("--n_vars", type=int, default=N_VARS)

    parser.add_argument("--precision", type=str, default="bf16-mixed")
    return parser


def regime_data_path(data_root: str, regime: str) -> Path:
    return Path(data_root) / regime
