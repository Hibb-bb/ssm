#!/bin/bash
#SBATCH --account=b1094
#SBATCH --job-name=build_mamba
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/pythonenvs/build_mamba_env_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/pythonenvs/build_mamba_env_%j.err
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

module purge all
module load gcc/11.2.0

source ~/.bashrc
eval "$(conda shell.bash hook)"

set -eo pipefail

export CONDA_PKGS_DIRS=/projects/b1094/StarEmbed/pythonenvs/.conda_pkgs
export PYTHONNOUSERSITE=1
export MAMBA_FORCE_BUILD=TRUE
mkdir -p "$CONDA_PKGS_DIRS"

ENV_PATH=/projects/b1094/StarEmbed/pythonenvs/mamba

echo "============================================="
echo "Step 1: Create conda environment"
echo "============================================="
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists, removing..."
    conda env remove -p "$ENV_PATH" -y
fi

conda create -p "$ENV_PATH" python=3.12 -y
conda activate "$ENV_PATH"

echo "Python: $(python --version)"
echo "which python: $(which python)"

echo "============================================="
echo "Step 2: Install PyTorch + CUDA toolkit via conda"
echo "============================================="
conda install -p "$ENV_PATH" pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

echo "============================================="
echo "Step 3: Verify nvcc is available"
echo "============================================="
CONDA_NVCC=$(find "$ENV_PATH" -name "nvcc" -type f 2>/dev/null | head -1)
if [ -z "$CONDA_NVCC" ]; then
    echo "nvcc not found in conda env, installing cuda-toolkit..."
    conda install -p "$ENV_PATH" nvidia::cuda-toolkit -y
    CONDA_NVCC=$(find "$ENV_PATH" -name "nvcc" -type f 2>/dev/null | head -1)
fi
echo "nvcc found at: $CONDA_NVCC"
export CUDA_HOME=$(dirname $(dirname "$CONDA_NVCC"))
echo "CUDA_HOME set to: $CUDA_HOME"
$CONDA_NVCC --version

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
nvidia-smi

echo "============================================="
echo "Step 4: Install causal-conv1d (build from source)"
echo "============================================="
pip install causal-conv1d --no-build-isolation --no-cache-dir 2>&1

echo "============================================="
echo "Step 5: Install mamba-ssm (build from source)"
echo "============================================="
pip install mamba-ssm --no-build-isolation --no-cache-dir 2>&1

echo "============================================="
echo "Step 6: Install remaining dependencies"
echo "============================================="
pip install pytorch-lightning datasets scipy einops ninja packaging transformers

echo "============================================="
echo "Step 7: Verify installation"
echo "============================================="
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')

import mamba_ssm
print('mamba_ssm:', mamba_ssm.__version__)

import causal_conv1d
print('causal_conv1d:', causal_conv1d.__version__)

import selective_scan_cuda
print('selective_scan_cuda: OK')

from mamba_ssm import Mamba
x = torch.randn(2, 64, 16).to('cuda')
model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to('cuda')
y = model(x)
print(f'Mamba forward pass: input {x.shape} -> output {y.shape}')
assert y.shape == x.shape, 'Shape mismatch!'
print('All checks passed!')
"

echo "============================================="
echo "Environment ready at: $ENV_PATH"
echo "============================================="
echo "Done!"
