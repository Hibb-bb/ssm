# Mamba SSM Experiments for Irregular Time Series

Self-contained directory with all code, configs, and scripts for the Mamba
sinusoidal forecasting and time-series classification experiments.

## Directory Structure

```
experiments/
├── forecaster/              # Mamba sinusoidal forecasting (3 dt variants)
│   ├── mamba_block.py         Pure PyTorch Mamba block + MambaIrregularBlock
│   ├── mamba_forecaster.py    Lightning module (learned / replace / additive dt)
│   ├── sinusoidal_datamodule.py   Lightning DataModule for HF Arrow datasets
│   ├── train.py               CLI training entrypoint (argparse)
│   ├── plot_mamba_comparison.py   Comparison plots sorted by frequency
│   └── test_cuda_kernel.py    Validates mamba_ssm CUDA kernel vs PyTorch ref
│
├── tsc/                     # Mamba time-series classification
│   └── mamba_tsc.py           5 Mamba-1 variants for irregular TSC
│                              (LTI, Selective, GapDelta, GapSelective, GapFeature)
│
├── dataset_generation/      # Sinusoidal data generation pipeline
│   ├── generate_sinusoidal_raw.py  Generates raw .npz sinusoidal data
│   └── convert_to_moirai.py       Converts .npz → HuggingFace Arrow format
│
├── evaluation/              # Downstream evaluation
│   └── eval_tpatchgnn.py      T-PatchGNN benchmark eval (MSE/RMSE/MAE)
│
├── configs/                 # Hydra YAML configs for sinusoidal datasets
│   ├── data/                  12 configs (4 irreg levels × 3 nobs variants)
│   ├── val_data/              12 validation configs
│   └── test_data/             12 test configs
│
├── scripts/                 # SLURM job scripts
│   ├── run_mamba_sinusoidal.sh    Array job: 3 variants × 4 irreg × 5 seeds = 60
│   └── test_mamba_cuda.sh         Validate CUDA kernel installation
│
└── environment/             # Conda environment setup
    └── build_mamba_env.sh     Builds mamba env with mamba_ssm CUDA kernels
```

## Quick Start

### 1. Build the environment (GPU node required for CUDA compilation)

```bash
sbatch experiments/environment/build_mamba_env.sh
```

This creates a conda env at `/projects/b1094/StarEmbed/pythonenvs/mamba` with:
- PyTorch + CUDA 12.4
- `mamba_ssm` (compiled from source with CUDA selective scan kernels)
- `causal_conv1d`
- `pytorch-lightning`, `datasets`, `scipy`, `einops`

### 2. Generate sinusoidal data

**Step 1: Generate raw `.npz` data** (all 4 irregularity levels):
```bash
python experiments/dataset_generation/generate_sinusoidal_raw.py \
    --output_dir /path/to/raw \
    --all_levels \
    --n_obs 160 --seed 42
```

This produces 4 directories (`sinusoidal_{regular,low_irreg,med_irreg,high_irreg}/`)
each containing `train.npz`, `val.npz`, `test.npz`, and `meta.json`.

Irregularity is controlled by `frac_regular` (fraction of uniformly-spaced samples):
- `regular` (1.0), `low_irreg` (0.8), `med_irreg` (0.3), `high_irreg` (0.0)

**Step 2: Convert to HuggingFace Arrow** (for DataModule):
```bash
python experiments/dataset_generation/convert_to_moirai.py \
    --input_root /path/to/raw \
    --output_root /path/to/hf_data \
    --all_levels
```

### 3. Run Mamba sinusoidal forecasting

```bash
sbatch --array=1-60 experiments/scripts/run_mamba_sinusoidal.sh
```

This trains 60 models (3 variants × 4 irregularity levels × 5 seeds):

| Variant | `dt_mode` | Description |
|---------|-----------|-------------|
| `vanilla_mamba` | `learned` | Standard Mamba, delta fully learned |
| `mamba_true_dt` | `replace` | Delta = true inter-observation time gap |
| `mamba_hybrid_dt` | `additive` | Delta = softplus(learned + true dt) |

### 4. Validate CUDA kernels

```bash
sbatch experiments/scripts/test_mamba_cuda.sh
```

Runs correctness checks (CUDA vs PyTorch reference), backward pass, and
speed benchmarks for various sequence lengths.

## Mamba Variants Explained

### Forecaster (`forecaster/`)

The core idea: inject real inter-observation time gaps into the SSM
discretization step (delta), replacing or augmenting the fully-learned delta.

- **MambaBlock**: Standard Mamba-1 with fully learned delta (baseline)
- **MambaIrregularBlock**: Accepts external `delta_t` from data
  - `replace` mode: delta = broadcast(true_dt) — pure physics
  - `additive` mode: delta = softplus(learned + true_dt) — hybrid

### TSC Classifier (`tsc/`)

Five controlled variants testing different parameterization strategies:
- **LTI**: Delta, B, C all input-independent (linear SSM / S4-like)
- **Selective**: Delta, B, C all input-dependent (standard Mamba-1)
- **GapDelta**: Delta from inter-observation time gaps only
- **GapSelective**: Delta from input + time gaps (additive)
- **GapFeature**: Time gaps concatenated as input feature (control)

## Dependencies

The `mamba` conda environment (`pythonenvs/mamba/`) or `moirai_train`
environment depending on the experiment:

- **Mamba forecasting** (`run_mamba_sinusoidal.sh`): uses `moirai_train` env
  (pure PyTorch selective scan, no CUDA kernels needed)
- **CUDA kernel tests** (`test_mamba_cuda.sh`): uses `mamba` env
  (requires compiled `mamba_ssm` with CUDA kernels)

## Parent Directory

The parent `mamba/` directory contains the official
[mamba-ssm](https://github.com/state-spaces/mamba) library source with
CUDA kernels for the selective scan operation. It is used as a build
dependency for the `mamba` conda environment.
