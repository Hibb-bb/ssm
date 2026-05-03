#!/bin/bash
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --job-name=verify_mamba_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/bfrf/seojininus/ssm/logs/verify_env_%j.out
#SBATCH --error=/projects/bfrf/seojininus/ssm/logs/verify_env_%j.err

# Tiny GPU job: import mamba_ssm/causal_conv1d, run a forward pass on GH200.
# Assumes the env was already built on login via build_env_delta.sh.

set -euo pipefail

mkdir -p /projects/bfrf/seojininus/ssm/logs

ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba}"

source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh
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

import romae  # distribution name 'RoMAE', import name 'romae'
print('romae: imported OK')

from mamba_ssm import Mamba
x = torch.randn(2, 64, 16, device='cuda')
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to('cuda')
y = m(x)
assert y.shape == x.shape, (y.shape, x.shape)
print(f'Mamba forward pass on {torch.cuda.get_device_name(0)}: {x.shape} -> {y.shape} OK')

# Also exercise the repo entrypoint imports (catches missing transitive deps).
import sys, os
sys.path.insert(0, '/projects/bfrf/seojininus/ssm')
from imts_benchmark.mamba_mv.multivariate_forecaster import (
    MultivariateMambaForecaster,
    MultivariateMambaSandwichForecaster,
)
print('repo imports: OK')

print()
print('ALL VERIFICATION PASSED')
PY
