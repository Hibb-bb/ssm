#!/bin/bash
# Build the Quest-matched mamba env on the *Delta* (non-AI) cluster:
# x86_64 + A100 (sm_80) or A40 (sm_86).
#
# This is the Delta sibling of build_env_delta.sh (which targets DeltaAI /
# Grace Hopper / sm_90). The Python/pip layer is identical; what changes:
#   - TORCH_CUDA_ARCH_LIST: "8.0" (A100) instead of "9.0" (GH200)
#   - Conda base location is detected from `module load python` if needed
#   - Env path uses a `_x86` suffix so it can't collide with the DeltaAI env
#     if /projects/bfrf is shared across both clusters
#
# Delta-side assumptions (verify in delta_x86_preflight.sh before running):
#   - /projects/bfrf/seojininus is mounted with the same path
#   - Delta has a CUDA 12.x toolkit (module load cuda/12.4 or similar)
#   - GPU partition is one of: gpuA100x4, gpuA40x4, gpuA100x8
#
# Default: runs on a Delta login node (requires nvcc + libcuda stub).
# Re-runnable: keeps env unless RECREATE_ENV=1.
#
# Inline GPU verify (only inside a GPU SLURM allocation):
#     RUN_GPU_VERIFY=1 bash build_env_delta_x86.sh

set -euo pipefail

# ---------- paths ----------
ENV_PATH="${ENV_PATH:-/projects/bfrf/seojininus/envs/mamba_x86}"
ROMAE_DIR="${ROMAE_DIR:-/projects/bfrf/seojininus/envs/RoMAE_x86}"
ROMAE_SHA="480cfaf80cf1f0630774998cb5c595ee684007e2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="${REPO_DIR}/requirements_delta.txt"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/projects/bfrf/seojininus/envs/.conda_pkgs_x86}"
mkdir -p "${CONDA_PKGS_DIRS}" "$(dirname "${ENV_PATH}")"

export PYTHONNOUSERSITE=1
export MAMBA_FORCE_BUILD=TRUE
export MAMBA_SKIP_CUDA_BUILD=FALSE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

# A100 = sm_80, A40 = sm_86. Build for both so either partition works.
# If you know you'll only use A100, set TORCH_CUDA_ARCH_LIST="8.0" to save build time.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "============================================="
echo "build_env_delta_x86.sh"
echo "  ENV_PATH:         ${ENV_PATH}"
echo "  CONDA_PKGS_DIRS:  ${CONDA_PKGS_DIRS}"
echo "  REQS:             ${REQS}"
echo "  TORCH_CUDA_ARCH:  ${TORCH_CUDA_ARCH_LIST}"
echo "  host:             $(hostname)"
echo "  date:             $(date)"
echo "============================================="

# ---------- conda hook ----------
# Delta uses lmod modules. The relevant one is `miniforge3-python` which
# puts conda on PATH and exposes its conda.sh hook.
if ! command -v conda &>/dev/null; then
    if command -v module &>/dev/null; then
        module load miniforge3-python 2>/dev/null \
            || module load miniforge3 2>/dev/null \
            || module load python 2>/dev/null \
            || true
    fi
fi
CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "ERROR: conda not found after 'module load miniforge3-python'." >&2
    echo "Try: module avail miniforge3 ; then load the right name and rerun." >&2
    exit 1
fi
# Resolve the conda root from the binary location so we can source conda.sh.
CONDA_REAL="$(readlink -f "${CONDA_BIN}")"
CONDA_PREFIX_GUESS="$(dirname "$(dirname "${CONDA_REAL}")")"
CONDA_SH="${CONDA_PREFIX_GUESS}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: derived conda.sh '${CONDA_SH}' does not exist." >&2
    exit 1
fi
echo "conda.sh: ${CONDA_SH}"
source "${CONDA_SH}"

echo ""
echo "============================================="
echo "Step 1: Create conda env (Python 3.12 to match Quest 3.12.13)"
echo "============================================="
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
# Delta typically uses lmod modules for CUDA. If nvcc isn't on PATH, try to
# load a 12.x cuda module.
if ! command -v nvcc &>/dev/null && command -v module &>/dev/null; then
    for m in cuda/12.4 cuda/12.6 cuda/12.5 cuda/12.3 cuda/12.2 cuda/12.1 cuda; do
        module load "$m" 2>/dev/null && break
    done
fi
NVCC_BIN="$(command -v nvcc || true)"
if [[ -z "${NVCC_BIN}" ]]; then
    echo "ERROR: nvcc not found. On Delta, try 'module load cuda/12.4' before re-running." >&2
    exit 1
fi
NVCC_REAL="$(readlink -f "${NVCC_BIN}")"
# Try to derive CUDA_HOME from nvcc location.
CUDA_HOME="$(dirname "$(dirname "${NVCC_REAL}")")"
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
echo "Step 3: Install torch==2.5.1 + cu124 (x86_64 wheel from PyTorch index)"
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
    print('compute capability:', torch.cuda.get_device_capability(0))
"

echo ""
echo "============================================="
echo "Step 4: Install build prereqs (setuptools/wheel/ninja/packaging)"
echo "============================================="
python -m pip install \
    "setuptools==82.0.1" \
    "wheel==0.46.3" \
    "ninja==1.13.0" \
    "packaging"

echo ""
echo "============================================="
echo "Step 5: Build causal-conv1d 1.6.1 from source"
echo "============================================="
python -m pip install --no-build-isolation --no-cache-dir "causal-conv1d==1.6.1"

echo ""
echo "============================================="
echo "Step 6: Build mamba-ssm 2.3.1 from source"
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
    (cd "${ROMAE_DIR}" && git fetch --all && git checkout "${ROMAE_SHA}")
else
    git clone https://github.com/Chromeilion/RoMAE.git "${ROMAE_DIR}"
    (cd "${ROMAE_DIR}" && git checkout "${ROMAE_SHA}")
fi
python -m pip install --no-build-isolation -e "${ROMAE_DIR}"

echo ""
echo "============================================="
echo "Step 9: CPU-side verification"
echo "============================================="
python - <<'PY'
import torch
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
import mamba_ssm, causal_conv1d
print('mamba_ssm:', mamba_ssm.__version__)
print('causal_conv1d:', causal_conv1d.__version__)
import pytorch_lightning, transformers, datasets, wandb, scipy, numpy, pandas
print('lightning:', pytorch_lightning.__version__,
      '| transformers:', transformers.__version__,
      '| datasets:', datasets.__version__,
      '| wandb:', wandb.__version__,
      '| scipy:', scipy.__version__,
      '| numpy:', numpy.__version__,
      '| pandas:', pandas.__version__)
import torchcde, torchdiffeq, torchsde, s5
print('torchcde:', torchcde.__version__,
      '| torchdiffeq:', torchdiffeq.__version__,
      '| torchsde:', torchsde.__version__)
import romae
print('romae: imported OK')
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
    echo "(GPU verify skipped — run sbatch verify_env_delta_x86.sbatch to test CUDA kernel.)"
fi

echo ""
echo "============================================="
echo "Env ready: ${ENV_PATH}"
echo "Activate with: conda activate ${ENV_PATH}"
echo "============================================="
