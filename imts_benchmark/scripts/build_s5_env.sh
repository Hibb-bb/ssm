#!/bin/bash
#SBATCH --account=b1094
#SBATCH --job-name=build_s5_jax
#SBATCH --nodes=1
#SBATCH --output=/projects/b1094/StarEmbed/pythonenvs/build_s5_jax_env_%j.out
#SBATCH --error=/projects/b1094/StarEmbed/pythonenvs/build_s5_jax_env_%j.err
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Build a modern JAX environment for the S5 forecaster baseline.
#
# Notes on JAX version choice vs upstream S5:
#   - lindermanlab/S5 `pendulum` branch pins jax==0.3.5 (mid-2022). Those
#     wheels predate bundled CUDA12 support and don't install cleanly on
#     current drivers.
#   - We install modern jax[cuda12] (0.4.x+) with matched flax/optax. The
#     S5 layer code (ssm.py + associated initializers) is small and
#     straightforward JAX; we'll port it to the current API when vendoring.
#
# Deps requested by the plan:
#   jax[cuda12], flax, optax, datasets, numpy, pyyaml
# Plus: scipy (S5 init uses scipy.linalg), einops, chex (flax utility),
#       tqdm (progress), wandb-optional.

module purge all
module load gcc/11.2.0

source ~/.bashrc
eval "$(conda shell.bash hook)"

set -eo pipefail

# Use a user-writable pkgs cache (shared dir is owned by another user).
export CONDA_PKGS_DIRS=${HOME}/.conda_pkgs_s5
export PYTHONNOUSERSITE=1
mkdir -p "$CONDA_PKGS_DIRS"

ENV_PATH=/projects/b1094/StarEmbed/pythonenvs/s5-jax

echo "============================================="
echo "Step 1: Create conda environment (python 3.11)"
echo "============================================="
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists, removing..."
    conda env remove -p "$ENV_PATH" -y
fi

conda create -p "$ENV_PATH" python=3.11 pip -y
conda activate "$ENV_PATH"

# Force the env's python/pip ahead of any user-site or system bins.
export PATH="$ENV_PATH/bin:$PATH"
# Belt + suspenders: disable loading packages from user site ($HOME/.local).
unset PYTHONPATH

echo "Python: $(python --version)"
echo "which python: $(which python)"
echo "python -m pip --version: $(python -m pip --version)"

echo "============================================="
echo "Step 2: Install JAX with bundled CUDA 12 wheels"
echo "============================================="
# jax[cuda12] pulls prebuilt CUDA 12 + cuDNN as pip packages. No conda
# cuda-toolkit needed. Always invoke via `python -m pip` to avoid any stale
# user-site pip in PATH.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]"

python -c "
import jax
print('jax:', jax.__version__)
print('jaxlib:', jax.lib.__version__ if hasattr(jax, 'lib') else '?')
print('devices:', jax.devices())
print('default_backend:', jax.default_backend())
"

echo "============================================="
echo "Step 3: Install Flax + Optax + numerical deps"
echo "============================================="
python -m pip install flax optax chex scipy einops numpy

echo "============================================="
echo "Step 4: Install HuggingFace datasets + utilities"
echo "============================================="
# datasets for reading the same arrow files the PyTorch pipeline uses.
python -m pip install datasets pyyaml tqdm pandas

echo "============================================="
echo "Step 5: Verify GPU + full stack"
echo "============================================="
nvidia-smi
python -c "
import jax
import jax.numpy as jnp
import flax
import optax
import chex
import datasets
print('jax:', jax.__version__)
print('flax:', flax.__version__)
print('optax:', optax.__version__)
print('chex:', chex.__version__)
print('datasets:', datasets.__version__)
print('devices:', jax.devices())
assert any(d.platform == 'gpu' for d in jax.devices()), 'No GPU device visible to JAX!'
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1024, 1024))
y = jnp.matmul(x, x.T).block_until_ready()
print('GPU matmul OK, result mean:', float(y.mean()))
print('All checks passed!')
"

echo "============================================="
echo "Environment ready at: $ENV_PATH"
echo "============================================="
echo "Done!"
