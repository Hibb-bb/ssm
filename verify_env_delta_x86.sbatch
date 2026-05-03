#!/bin/bash
#SBATCH --account=bfrf-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --job-name=verify_mamba_x86
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/logs/verify_env_x86_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/logs/verify_env_x86_%j.err

# Tiny GPU job for the Delta (non-AI) cluster.
# Account/partition match Delta conventions:
#   - account=bfrf-dtai (no -gh suffix; that's the DeltaAI variant)
#   - partition=gpuA100x4 (most common GPU partition; alternates: gpuA40x4, gpuA100x8)
#
# Adjust ENV_PATH if the build was made with a different env path.

set -euo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/logs

ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba_x86}"

# Delta exposes conda via the miniforge3-python lmod module.
module load miniforge3-python 2>/dev/null || true
CONDA_BIN="$(command -v conda)"
CONDA_REAL="$(readlink -f "${CONDA_BIN}")"
source "$(dirname "$(dirname "${CONDA_REAL}")")/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "host: $(hostname)"
echo "date: $(date)"
nvidia-smi

python - <<'PY'
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
assert torch.cuda.is_available(), 'no CUDA visible inside the job'
print('device:', torch.cuda.get_device_name(0), 'cap:', torch.cuda.get_device_capability(0))

import mamba_ssm, causal_conv1d
print('mamba_ssm:', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)

import selective_scan_cuda  # noqa: F401
print('selective_scan_cuda: imported OK')

import romae
print('romae: imported OK')

from mamba_ssm import Mamba
x = torch.randn(2, 64, 16, device='cuda')
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to('cuda')
y = m(x)
assert y.shape == x.shape, (y.shape, x.shape)
print(f'Mamba forward pass on {torch.cuda.get_device_name(0)}: {x.shape} -> {y.shape} OK')

import sys
sys.path.insert(0, '/projects/bfrf/seojininus/ssm')
from imts_benchmark.mamba_mv.multivariate_forecaster import (
    MultivariateMambaForecaster, MultivariateMambaSandwichForecaster,
)
print('repo imports: OK')

print()
print('ALL VERIFICATION PASSED')
PY
