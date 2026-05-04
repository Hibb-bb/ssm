"""Train Mamba-MV (pretrain-arch) classifier on UEA datasets with Kidger 30% sync drop.

Entry point for both smoke runs and HPO array tasks. Per-dataset defaults
(batch_size, label_smoothing, grad_clip, grid_K, head_dropout) are
inherited from RoMAE Table 12; LR and batch_size are typically the swept
axes for v2 HPO. Per-dataset defaults still apply when the corresponding
CLI flag is left at its sentinel ``-1``.

Example:
    python -m imts_benchmark.mamba_mv.train_cls \
        --dataset BasicMotions --data_root <data_root> \
        --dt_mode replace --lr 1e-3 --batch_size 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from matplotlib.pylab import sample
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

_THIS_DIR = Path(__file__).resolve().parent
_SSM_DK = _THIS_DIR.parents[1]
if str(_SSM_DK) not in sys.path:
    sys.path.insert(0, str(_SSM_DK))

from imts_benchmark.mamba_pretrain.multivariate_classifier import MultivariateMambaClassifier
from imts_benchmark.shared_config.wandb_lightning import (
    build_wandb_logger,
    log_wandb_after_fit,
    log_wandb_run_summary,
)
from imts_benchmark.shared_data.uea_classification_datamodule import (
    UEAClassificationDataModule,
    collate_uea_per_variate,
)

from oracle.custom_datasets.ELAsTiCC import ELAsTiCC_LC_Dataset, flag_value, n_static_features, n_channels, img_height, img_width

LSST_passband_to_wavelengths = {
    'u': (320 + 400) / (2 * 1000),
    'g': (400 + 552) / (2 * 1000),
    'r': (552 + 691) / (2 * 1000),
    'i': (691 + 818) / (2 * 1000),
    'z': (818 + 922) / (2 * 1000),
    'Y': (950 + 1080) / (2 * 1000),
}

classes = ['SNII', 'SNIb/c', 'CART', 'EB', 'Delta Scuti', 'AGN', 'PISN', 'Cepheid', 'TDE', 'SNI91bg', 'SLSN', 'SNIax', 'SNIa', 'KN', 'Dwarf Novae', 'uLens', 'RR Lyrae', 'M-dwarf Flare', 'ILOT' ]


def reformat_ts_matrix(ts_sample, max_length):

    feature__order = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'BAND', 'PHOTFLAG']
    
    final_matrix = np.zeros((ts_sample.shape[0], 6), dtype=np.float32)
    final_mask = np.zeros((ts_sample.shape[0], 6), dtype=np.bool)
    for i, f in enumerate(['u', 'g', 'r', 'i', 'z', 'Y']):
        
        filter_mask = ts_sample[:,feature__order.index('BAND')] == LSST_passband_to_wavelengths[f]

        final_matrix[:,i] = ts_sample[:,1] * filter_mask #* detec_mask
        final_mask[:,i] = filter_mask

    times = ts_sample[:,feature__order.index('MJD')]
    delta_times = np.diff(times, prepend=times[0])

    # pad to max_length
    if ts_sample.shape[0] < max_length:
            pad_length = max_length - ts_sample.shape[0]
            final_matrix = np.pad(
                final_matrix, ((pad_length, 0), (0, 0)),
                mode='constant', constant_values=0
            )
            final_mask = np.pad(
                final_mask, ((pad_length, 0), (0, 0)),
                mode='constant', constant_values=False
            )
            times = np.pad(times, (pad_length, 0), mode='constant', constant_values=0)
            delta_times = np.pad(
                delta_times, (pad_length, 0),
                mode='constant', constant_values=0
            )

    return final_matrix.T, final_mask.T, times, delta_times


def get_weights(labels):
    counts = np.zeros((len(classes)))
    for l in labels:
        counts[classes.index(l)] += 1
    weights = 1 / counts
    return torch.from_numpy(weights).float()


def custom_collate_ELAsTiCC(batch):
    """
    Custom collation function for processing a batch of ELAsTiCC dataset samples.

    Parameters:
        batch (list): A list of dictionaries, each representing a sample.

    Returns:
        dict: A dictionary containing the collated batch with the following keys:
            - 'ts': A padded tensor of time series data with shape (batch_size, max_length, ...), where padding is applied using the predefined flag_value.
            - 'static': A tensor of static features with shape (batch_size, n_static_features).
            - 'length': A tensor containing the lengths of each time series in the batch.
            - 'label': A numpy array of labels for the batch (array-like).
            - 'raw_label': A numpy array of raw ELAsTiCC class labels (array-like).
            - 'id': A numpy array of SNIDs corresponding to each sample.
            - 'lc_plot' (if present in the input samples): A tensor of light curve plots with shape (batch_size, n_channels, img_height, img_width).
    """

    batch_size = len(batch)

    max_length = max(sample['ts'].shape[0] for sample in batch)

    ELASTICC_class_array = []
    snid_array = np.zeros((batch_size))

    ts_tensor = torch.zeros((batch_size, 6, max_length), dtype=torch.float32, device='cpu')
    mask_tensor = torch.zeros((batch_size, 6, max_length), dtype=torch.bool, device='cpu')
    times_tensor = torch.zeros((batch_size, 6, max_length), dtype=torch.float32, device='cpu')
    delta_times_tensor = torch.zeros((batch_size, 6, max_length), dtype=torch.float32, device='cpu')
    labels_tensor = torch.zeros((batch_size), dtype=torch.long, device='cpu')

    for i, sample in enumerate(batch):

        obs, final_mask, times, delta_ts = reformat_ts_matrix(sample['ts'], max_length)

        ts_tensor[i,:,:] = torch.from_numpy(obs)
        mask_tensor[i,:,:] = torch.from_numpy(final_mask)

        times_t = torch.as_tensor(times, dtype=torch.float32)
        delta_t = torch.as_tensor(delta_ts, dtype=torch.float32)

        for j in range(6):
            delta_times_tensor[i, j, :] = delta_t
            times_tensor[i, j, :] = times_t

        labels_tensor[i] = classes.index(sample['label'])

        # book keeping
        ELASTICC_class_array.append(sample['ELASTICC_class'])
        snid_array[i] = sample['SNID']

    ELASTICC_class_array = np.array(ELASTICC_class_array)


    d = {
        'values': ts_tensor,
        'timestamps': times_tensor, 
        'deltat': delta_times_tensor,
        'valid_mask': mask_tensor,
        'label': labels_tensor,
        'raw_label': ELASTICC_class_array,
        'id': snid_array,
    }

    return d

class ELAsTiCCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=16,
        max_n_per_class=200,
        num_workers=2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_n_per_class = max_n_per_class
        self.n_vars = 6
        self.n_classes = 19
        self.val_truncation_days = np.array([2000])
        self.num_workers = num_workers

    def setup(self, stage=None):

        self.train_ds = ELAsTiCC_LC_Dataset(
            parquet_file_path="/Users/vedshah/Documents/Research/NU-Miller/Projects/Hierarchical-VT/data/ELAsTiCC/train.parquet",
            max_n_per_class=self.max_n_per_class,
        )

        self.val_ds = ELAsTiCC_LC_Dataset(
            parquet_file_path="/Users/vedshah/Documents/Research/NU-Miller/Projects/Hierarchical-VT/data/ELAsTiCC/val.parquet",
            max_n_per_class=self.max_n_per_class,
        )

        self.test_ds = ELAsTiCC_LC_Dataset(
            parquet_file_path="/Users/vedshah/Documents/Research/NU-Miller/Projects/Hierarchical-VT/data/ELAsTiCC/test.parquet",
            max_n_per_class=self.max_n_per_class,
        )

        self.class_weights = get_weights(self.train_ds.get_all_labels())


    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_ELAsTiCC,
                          persistent_workers=self.num_workers > 0,)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_ELAsTiCC,
                          persistent_workers=self.num_workers > 0,)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=custom_collate_ELAsTiCC,
                          persistent_workers=self.num_workers > 0,)

# Per-dataset defaults inherited from RoMAE Table 12, with label_smoothing
# converted from RoMAE's confidence-c convention (App. A.1: "reducing each
# correct class label from 1 to a confidence value c") to PyTorch's smoothing-
# amount convention (label_smoothing=0 means no smoothing). Mapping: p = 1 - c.
# RoMAE c values: BM=1.0, CT=0.9, EP=0.8, HB=1.0, LSST=0.9
# PyTorch p:      BM=0.0, CT=0.1, EP=0.2, HB=0.0, LSST=0.1
ROMAE_DATASET_DEFAULTS = {
    "BasicMotions":          {"batch_size": 8,  "label_smoothing": 0.0, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "CharacterTrajectories": {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.0},
    "Epilepsy":              {"batch_size": 16, "label_smoothing": 0.2, "grad_clip": 1.0,  "grid_K": 128, "head_dropout": 0.2},
    "Heartbeat":             {"batch_size": 16, "label_smoothing": 0.0, "grad_clip": 2.0,  "grid_K": 128, "head_dropout": 0.0},
    "ELAsTiCC":              {"batch_size": 16, "label_smoothing": 0.1, "grad_clip": 10.0, "grid_K": 64,  "head_dropout": 0.2},
}


# Architecture presets. Width-only axis, depth fixed at 3+3 (per the
# colleague's BIG-vs-SMALL framing on the IMM/PhysioNet runs):
#     SMALL  d=64   ≈ 259K params   (head-of-table sweet spot from IMM HPO)
#     BASE   d=256  ≈ 3.5M params   (the historical UEA default we've been running)
#     BIG    d=384  ≈ 7.8M params   (colleague's BIG)
# Apply via --model_size; depth/heads/state/conv/expand remain on their
# individual CLI defaults so this preset only moves width.
MODEL_SIZE_PRESETS = {
    "small": {"d_model": 64,  "d_hidden": 64,  "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
    "base":  {"d_model": 256, "d_hidden": 256, "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
    "big":   {"d_model": 384, "d_hidden": 384, "n_perv_layer": 3, "n_fusion_blocks": 3, "n_heads_varattn": 4},
}


def parse_args():
    p = argparse.ArgumentParser(description="Mamba-MV UEA classification")

    # Data.
    p.add_argument("--dataset", type=str, required=True,
                   choices=list(ROMAE_DATASET_DEFAULTS.keys()))
    p.add_argument("--drop_rate", type=float, default=0.3)
    p.add_argument("--drop_seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42,
                   help="Seed for the train/val 80/20 stratified split. Fixed "
                        "across our HPO and final eval so val identity is stable.")
    p.add_argument("--num_workers", type=int, default=2)

    # Architecture.
    p.add_argument("--model_size", type=str, default="base",
                   choices=list(MODEL_SIZE_PRESETS.keys()),
                   help="Apply a preset for d_model/d_hidden/n_perv_layer/"
                        "n_fusion_blocks/n_heads_varattn. Individual --d_model "
                        "etc. flags below are ignored unless model_size=base "
                        "(in which case they are honoured for backward compat).")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_hidden", type=int, default=256)
    p.add_argument("--n_perv_layer", type=int, default=3)
    p.add_argument("--n_fusion_blocks", type=int, default=3)
    p.add_argument("--n_heads_varattn", type=int, default=4)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--n_freq", type=int, default=8)
    p.add_argument("--dt_mode", type=str, default="replace",
                   choices=["learned", "replace", "concat"])
    p.add_argument("--head_use_avail_mask", type=int, default=1)
    # grid_K override (else inherit from ROMAE_DATASET_DEFAULTS).
    p.add_argument("--grid_K", type=int, default=-1)
    # head_dropout override (sentinel -1.0 means "use per-dataset default").
    p.add_argument("--head_dropout", type=float, default=-1.0)

    # Optimization.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_epochs", type=int, default=800)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--warmup_pct", type=float, default=0.1,
                   help="LR warmup as fraction of total training steps.")
    # Per-dataset overrides (else use ROMAE_DATASET_DEFAULTS).
    p.add_argument("--batch_size", type=int, default=-1)
    p.add_argument("--label_smoothing", type=float, default=-1.0)
    p.add_argument("--grad_clip", type=float, default=-1.0)
    # Loss weighting. Default ON to match prior runs; flip OFF to match RoMAE,
    # which uses only label_smoothing and no inverse-frequency class weights.
    p.add_argument("--use_class_weights", type=int, default=1,
                   choices=[0, 1],
                   help="If 0, override dm.class_weights with all-ones so the "
                        "cross-entropy is unweighted (matches RoMAE App. C.2).")

    # Run management.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./uea_cls_outputs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--precision", type=str, default="32-true",
                   help="Lightning precision. Default 32-true matches the "
                        "forecasting trainer; the Mamba selective_scan CUDA "
                        "kernel asserts delta.dtype == u.dtype, which fails "
                        "under bf16-mixed unless deltat is explicitly cast.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: small max_epochs and small patience.")

    # Weights & Biases (online cloud logging). We always keep CSVLogger as a
    # local backup; wandb adds live dashboards under the configured project.
    p.add_argument("--use_wandb", action="store_true",
                   help="Stream per-step + per-epoch metrics to wandb.")
    p.add_argument("--wandb_project", type=str, default="TSKing")
    p.add_argument("--wandb_entity", type=str, default="magicslabnorthwestern",
                   help="wandb entity (team / user). The ML lab account.")
    p.add_argument("--wandb_run_name", type=str, default="",
                   help="If empty, a descriptive default is generated from "
                        "dataset/dt_mode/size/seed.")

    return p.parse_args()


def main():
    args = parse_args()

    # Apply RoMAE per-dataset defaults where CLI args are unset (-1).
    defaults = ROMAE_DATASET_DEFAULTS[args.dataset]
    if args.batch_size < 0:
        args.batch_size = defaults["batch_size"]
    if args.label_smoothing < 0:
        args.label_smoothing = defaults["label_smoothing"]
    if args.grad_clip < 0:
        args.grad_clip = defaults["grad_clip"]
    if args.grid_K < 0:
        args.grid_K = defaults["grid_K"]
    if args.head_dropout < 0:
        args.head_dropout = defaults["head_dropout"]

    # Apply architecture preset (overrides individual --d_model etc. unless
    # the user picked the "base" preset, in which case the CLI values stand
    # so existing scripts don't shift behaviour silently).
    if args.model_size != "base":
        preset = MODEL_SIZE_PRESETS[args.model_size]
        for k, v in preset.items():
            setattr(args, k, v)

    if args.smoke:
        args.max_epochs = min(args.max_epochs, 30)
        args.patience = min(args.patience, 10)

    pl.seed_everything(args.seed, workers=True)

    run_name = args.run_name or (
        f"{args.dataset}_{args.dt_mode}_lr{args.lr}_bs{args.batch_size}"
        f"_drop{args.head_dropout}_seed{args.seed}"
    )
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    dm = ELAsTiCCDataModule(num_workers=args.num_workers)
    dm.setup()

    # Estimated total training steps for the LR schedule (cap at max_epochs).
    # steps_per_epoch = max(1, math.ceil(len(dm.train_ds) / args.batch_size))
    # num_training_steps = max(1, steps_per_epoch * args.max_epochs)
    # num_warmup_steps = max(1, int(num_training_steps * args.warmup_pct))

    # Class weights toggle: optionally drop the inverse-frequency weighting so
    # the loss matches RoMAE App. C.2 (label_smoothing only).
    cw = dm.class_weights if args.use_class_weights else None

    # ---- Model ----
    model = MultivariateMambaClassifier(
        d_model=args.d_model,
        d_hidden=args.d_hidden,
        n_vars=dm.n_vars,
        n_classes=dm.n_classes,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        n_heads_varattn=args.n_heads_varattn,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        dt_mode=args.dt_mode,
        grid_K=args.grid_K,
        t_max=1.0,
        n_freq=args.n_freq,
        head_use_avail_mask=bool(args.head_use_avail_mask),
        head_dropout=args.head_dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        class_weights=cw,
    )

    n_params = sum(p.numel() for p in model.parameters())
    # print(f"[run={run_name}] n_vars={dm.n_vars}, n_classes={dm.n_classes}, "
    #       f"|train|={len(dm.train_ds)}, |val|={len(dm.val_ds)}, |test|={len(dm.test_ds)}, "
    #       f"n_params={n_params:,}")

    # ---- Trainer ----
    callbacks = [
        EarlyStopping(monitor="val/acc", mode="max", patience=args.patience),
    ]
    # CSVLogger writes per-epoch metrics.csv to <output_dir>/<run_name>/csv/
    # version_0/metrics.csv — used downstream for loss-curve plots / convergence
    # diagnosis. Lightweight (just a CSV file). Replaces the prior logger=False.
    csv_logger = CSVLogger(
        save_dir=str(out_dir),
        name="csv",
        flush_logs_every_n_steps=50,
    )

    # Optional wandb dashboard. Defaults to a descriptive run name so sweeps
    # are easy to filter in the UI.
    if args.use_wandb and not args.wandb_run_name:
        args.wandb_run_name = (
            f"uea_cls_{args.dataset}_{args.dt_mode}_size{args.model_size}"
            f"_seed{args.seed}_drop{args.drop_rate}"
        )
    wandb_logger = build_wandb_logger(args)
    loggers = [csv_logger]
    if wandb_logger is not False:
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision if torch.cuda.is_available() else 32,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        deterministic=False,  # Mamba's SSM kernel isn't fully deterministic; cosine schedule + seed cover most.
        log_every_n_steps=10,
        logger=loggers,
        enable_checkpointing=False,  # We never load the saved .ckpt; metrics come from
                                     # callback_metrics + immediate trainer.test(). Default
                                     # ckpt path is CWD-relative, and 30 array tasks cd to
                                     # the same CODE_DIR -> race on shared checkpoints/ dir.
    )

    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    fit_time = time.time() - t0
    log_wandb_after_fit(wandb_logger, fit_time, best_val_mse=None)

    # Final-epoch val metrics, used by the v2 aggregator. We capture both
    # acc and macro_f1; the picker may use either (collapse detection compares
    # them).
    best_val_acc = float(trainer.callback_metrics.get("val/acc", torch.tensor(0.0)).item())
    val_macro_f1 = float(trainer.callback_metrics.get("val/macro_f1", torch.tensor(0.0)).item())
    print(f"[done] val/acc={best_val_acc:.4f}, val/macro_f1={val_macro_f1:.4f}, "
          f"fit_time={fit_time:.1f}s")

    # Test eval.
    test_metrics = trainer.test(model, datamodule=dm)[0]
    test_acc = float(test_metrics.get("test/acc_final", test_metrics.get("test/acc", 0.0)))
    macro_f1 = float(test_metrics.get("test/macro_f1", 0.0))
    print(f"[test] acc={test_acc:.4f}, macro_f1={macro_f1:.4f}")

    # Persist a flat summary for downstream aggregation.
    summary = dict(
        dataset=args.dataset,
        dt_mode=args.dt_mode,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        grid_K=args.grid_K,
        head_dropout=args.head_dropout,
        model_size=args.model_size,
        d_model=args.d_model,
        n_perv_layer=args.n_perv_layer,
        n_fusion_blocks=args.n_fusion_blocks,
        use_class_weights=bool(args.use_class_weights),
        n_vars=dm.n_vars,
        n_classes=dm.n_classes,
        n_params=n_params,
        best_val_acc=best_val_acc,
        val_macro_f1=val_macro_f1,
        test_acc=test_acc,
        macro_f1=macro_f1,
        fit_time=fit_time,
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log_wandb_run_summary(wandb_logger, summary)

    # Per-sample test predictions for downstream confusion-matrix /
    # per-class diagnostics. Read by eval/uea_confusion.py. We pull from the
    # `_test_agg` dict stashed by the model in on_test_epoch_end. Shape:
    # one JSON object with parallel labels/preds/logits arrays.
    agg = getattr(model, "_test_agg", None)
    if agg and "labels" in agg:
        preds_path = out_dir / "test_predictions.json"
        with open(preds_path, "w") as f:
            json.dump(
                dict(
                    dataset=args.dataset,
                    dt_mode=args.dt_mode,
                    seed=args.seed,
                    n_classes=dm.n_classes,
                    label_to_idx=dm.label_to_idx,
                    class_weights=dm.class_weights.tolist(),
                    labels=agg["labels"],
                    preds=agg["preds"],
                    logits=agg["logits"],
                ),
                f,
            )


if __name__ == "__main__":
    main()
