#!/bin/bash
# Build the Quest-matched mamba env on Delta GH (aarch64 + sm_90 / GH200).
#
# Default: runs on login node (login has nvcc + libcuda stub, enough to
# compile causal_conv1d/mamba_ssm). Skips the final GPU forward pass since
# login has no GPU. Run verify_env_delta.sbatch afterward to confirm CUDA.
#
# To run inline GPU verification (only inside a SLURM GPU allocation):
#     RUN_GPU_VERIFY=1 bash build_env_delta.sh
#
# Re-runnable: removes the env path before recreating.

set -euo pipefail

# ---------- paths ----------
ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba}"
ROMAE_DIR="${ROMAE_DIR:-/projects/bfrf/seojininus/envs/RoMAE}"
ROMAE_SHA="480cfaf80cf1f0630774998cb5c595ee684007e2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${REPO_DIR}/requirements_delta.txt"

# Project-local conda pkg cache so we don't fill $HOME or the shared cache.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/projects/bfrf/seojininus/envs/.conda_pkgs}"
mkdir -p "${CONDA_PKGS_DIRS}" "$(dirname "${ENV_PATH}")"

# Force-build mamba_ssm/causal_conv1d sources (no x86_64 wheel sneak-ins) and
# disable PYTHONUSERBASE picking up stale pip installs.
export PYTHONNOUSERSITE=1
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_SKIP_CUDA_BUILD=FALSE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

# Hopper sm_90; matches Quest H100 (also sm_90). One arch -> faster build.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "============================================="
echo "build_env_delta.sh"
echo "  ENV_PATH:         ${ENV_PATH}"
echo "  CONDA_PKGS_DIRS:  ${CONDA_PKGS_DIRS}"
echo "  REQS:             ${REQS}"
echo "  TORCH_CUDA_ARCH:  ${TORCH_CUDA_ARCH_LIST}"
echo "  host:             $(hostname)"
echo "  date:             $(date)"
echo "============================================="

# ---------- conda hook ----------
# Use the system miniforge install. We don't need to activate anything from it
# beyond `conda` itself.
source /sw/user/python/miniforge3-pytorch-2.5.0/etc/profile.d/conda.sh

echo ""
echo "============================================="
echo "Step 1: Create conda env (Python 3.12 to match Quest 3.12.13)"
echo "============================================="
# Default: keep the env if it already exists (idempotent re-runs after a
# Step-N failure). Set RECREATE_ENV=1 to force a clean rebuild.
if [[ -d "${ENV_PATH}" ]]; then
    if [[ "${RECREATE_ENV:-0}" == "1" ]]; then
        echo "RECREATE_ENV=1 -> wiping ${ENV_PATH}..."
        conda env remove -p "${ENV_PATH}" -y
        conda create -p "${ENV_PATH}" python=3.12 pip -y
    else
        echo "Env already exists at ${ENV_PATH}, reusing. (RECREATE_ENV=1 to wipe.)"
    fi
else
    conda create -p "${ENV_PATH}" python=3.12 pip -y
fi
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"

python --version
which python
which pip

echo ""
echo "============================================="
echo "Step 2: System CUDA visible to nvcc (build only; runtime libs come from torch wheel)"
echo "============================================="
# Delta provides a recent CUDA via NVHPC SDK. Quest used cu124; here we'll
# compile with whatever 12.x nvcc the system exposes (forward-compat at the
# driver/runtime level for sm_90).
NVCC_BIN="$(command -v nvcc || true)"
if [[ -z "${NVCC_BIN}" ]]; then
    echo "ERROR: nvcc not found in PATH" >&2
    exit 1
fi
# NVHPC SDK ships nvcc under .../compilers/bin while CUDA toolkit lives under
# .../cuda/. The .../cuda/ dir itself has bin/include/lib64 symlinks to the
# active version — use it directly so version-dir layout doesn't matter.
NVCC_REAL="$(readlink -f "${NVCC_BIN}")"
if [[ "${NVCC_REAL}" == */hpc_sdk/*/compilers/bin/nvcc ]]; then
    HPC_ROOT="$(dirname "$(dirname "$(dirname "${NVCC_REAL}")")")"   # .../25.5
    if [[ -x "${HPC_ROOT}/cuda/bin/nvcc" ]]; then
        CUDA_HOME="${HPC_ROOT}/cuda"
    fi
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(dirname "$(dirname "${NVCC_BIN}")")"
fi
# Sanity: CUDA_HOME must contain bin/nvcc, not just nvvm/bin.
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "ERROR: CUDA_HOME='${CUDA_HOME}' has no bin/nvcc" >&2
    exit 1
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "nvcc:      ${NVCC_BIN}  (real: ${NVCC_REAL})"
echo "CUDA_HOME: ${CUDA_HOME}"
"${CUDA_HOME}/bin/nvcc" --version
nvidia-smi || echo "(nvidia-smi failed — expected on login node, fine.)"

echo ""
echo "============================================="
echo "Step 3: Install torch==2.5.1 + cu124 (aarch64 wheel from PyTorch index)"
echo "============================================="
python -m pip install --upgrade pip
python -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.5.1"

python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print('compute capability:', cap)
"

echo ""
echo "============================================="
echo "Step 4: Install build prereqs (setuptools/wheel/ninja/packaging) BEFORE C-ext builds"
echo "============================================="
python -m pip install \
    "setuptools==82.0.1" \
    "wheel==0.46.3" \
    "ninja==1.13.0" \
    "packaging"

echo ""
echo "============================================="
echo "Step 5: Build causal-conv1d 1.6.1 from source (no-build-isolation, against torch 2.5.1)"
echo "============================================="
python -m pip install --no-build-isolation --no-cache-dir "causal-conv1d==1.6.1"

echo ""
echo "============================================="
echo "Step 6: Build mamba-ssm 2.3.1 from source (no-build-isolation, against torch 2.5.1)"
echo "============================================="
python -m pip install --no-build-isolation --no-cache-dir "mamba-ssm==2.3.1"

echo ""
echo "============================================="
echo "Step 7: Install rest of Quest pin set"
echo "============================================="
python -m pip install -r "${REQS}"

echo ""
echo "============================================="
echo "Step 8: Install RoMAE (editable, pinned to commit ${ROMAE_SHA})"
echo "============================================="
if [[ -d "${ROMAE_DIR}/.git" ]]; then
    echo "RoMAE checkout exists at ${ROMAE_DIR}; updating to pin..."
    (cd "${ROMAE_DIR}" && git fetch --all && git checkout "${ROMAE_SHA}")
else
    git clone https://github.com/Chromeilion/RoMAE.git "${ROMAE_DIR}"
    (cd "${ROMAE_DIR}" && git checkout "${ROMAE_SHA}")
fi
python -m pip install --no-build-isolation -e "${ROMAE_DIR}"

echo ""
echo "============================================="
echo "Step 9: CPU-side verification (imports + versions)"
echo "============================================="
python - <<'PY'
import torch
print('torch:', torch.__version__, '| cuda(version):', torch.version.cuda)
import mamba_ssm, causal_conv1d
print('mamba_ssm:', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)
import pytorch_lightning, transformers, datasets, wandb, scipy, numpy, pandas
print('lightning:', pytorch_lightning.__version__)
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('wandb:', wandb.__version__)
print('scipy:', scipy.__version__)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
import torchcde, torchdiffeq, torchsde, s5
print('torchcde:', torchcde.__version__)
print('torchdiffeq:', torchdiffeq.__version__)
print('torchsde:', torchsde.__version__)
import romae  # distribution name 'RoMAE', import name 'romae'
print('romae:', getattr(romae, '__version__', '(no __version__)'))
print('CPU-side checks: OK')
PY

if [[ "${RUN_GPU_VERIFY:-0}" == "1" ]]; then
    echo ""
    echo "============================================="
    echo "Step 10: GPU verification (Mamba forward pass)"
    echo "============================================="
    python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not available — set RUN_GPU_VERIFY=1 only inside a GPU allocation.'
print('device:', torch.cuda.get_device_name(0), 'cap:', torch.cuda.get_device_capability(0))
import selective_scan_cuda  # noqa: F401
print('selective_scan_cuda: imported OK')
from mamba_ssm import Mamba
x = torch.randn(2, 64, 16, device='cuda')
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).to('cuda')
y = m(x)
assert y.shape == x.shape, (y.shape, x.shape)
print(f'Mamba forward pass on {torch.cuda.get_device_name(0)}: {x.shape} -> {y.shape} OK')
print('ALL GPU CHECKS PASSED')
PY
else
    echo ""
    echo "(GPU verify skipped — run sbatch verify_env_delta.sbatch to test CUDA kernel.)"
fi

echo ""
echo "============================================="
echo "Env ready: ${ENV_PATH}"
echo "Activate with: conda activate ${ENV_PATH}"
echo "============================================="
