#!/bin/bash
# Pre-flight script for the Delta (non-AI / x86_64) cluster.
# Run this *on a Delta login node* the moment you get access. It sanity-checks
# everything build_env_delta_x86.sh assumes, so we know whether to run the
# build as-is or whether minor edits are needed.
#
# Usage on Delta:
#   ssh user@login.delta.ncsa.illinois.edu
#   cd /projects/bfrf/seojininus/ssm   # if /projects/bfrf is mounted; else clone
#   bash delta_x86_preflight.sh

set -uo pipefail

PASS=0
FAIL=0
ok()  { echo "  [OK]   $*"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
hdr() { echo ""; echo "== $* =="; }

hdr "host"
hostname
uname -m   # expect x86_64
echo "user: $USER"

hdr "1. SLURM accounts"
ACCT_OUT="$(accounts 2>&1 || true)"
echo "$ACCT_OUT"
if echo "$ACCT_OUT" | grep -qE "bfrf-dtai($|[^-])"; then
    ok "non-AI 'bfrf-dtai' allocation visible"
else
    bad "no 'bfrf-dtai' (non-AI) account found in 'accounts'"
fi

hdr "2. Cluster associations (sacctmgr)"
sacctmgr show association where user=$USER format=Cluster,Account -np 2>&1 | sort -u || true

hdr "3. GPU partitions visible"
sinfo -h -o "%P" 2>&1 | sort -u | head -20
for p in gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8; do
    if sinfo -h -p "$p" -o "%P" 2>/dev/null | grep -q "$p"; then
        ok "partition $p exists"
    else
        bad "partition $p not visible"
    fi
done

hdr "4. Shared filesystem (/projects/bfrf mounted?)"
if [[ -d /projects/bfrf/seojininus ]]; then
    ok "/projects/bfrf/seojininus exists"
    if [[ -d /projects/bfrf/seojininus/ssm ]]; then
        ok "ssm repo present at /projects/bfrf/seojininus/ssm"
    else
        bad "ssm repo not found — may need to clone it on Delta"
    fi
    if [[ -f /projects/bfrf/seojininus/ssm/requirements_delta.txt ]]; then
        ok "requirements_delta.txt present"
    else
        bad "requirements_delta.txt missing"
    fi
else
    bad "/projects/bfrf/seojininus not mounted on Delta — would need to scp/clone artifacts"
fi

hdr "5. Conda / miniforge available"
CONDA_FOUND=""
for c in /sw/external/python/miniforge3-pytorch-2.5.0 /sw/external/python/miniforge3 /sw/user/python/miniforge3-pytorch-2.5.0; do
    if [[ -f "${c}/etc/profile.d/conda.sh" ]]; then
        ok "miniforge at ${c}"
        CONDA_FOUND="${c}"
        break
    fi
done
if [[ -z "${CONDA_FOUND}" ]]; then
    bad "no miniforge in known paths — try 'module avail python' below"
fi
module avail python 2>&1 | head -30 || true

hdr "6. CUDA toolkit / nvcc"
if command -v nvcc &>/dev/null; then
    ok "nvcc on PATH: $(command -v nvcc)"
    nvcc --version | head -4
else
    bad "nvcc not on PATH; trying module load..."
    if command -v module &>/dev/null; then
        for m in cuda/12.4 cuda/12.6 cuda/12.5 cuda/12.3 cuda; do
            if module avail "$m" 2>&1 | grep -q "$m"; then
                echo "  -> available: $m"
            fi
        done
    fi
fi

hdr "7. WANDB credentials"
if [[ -f ~/.netrc ]] && grep -q api.wandb.ai ~/.netrc; then
    ok "~/.netrc has wandb credentials"
else
    bad "no wandb credentials in ~/.netrc — re-add on Delta if home isn't shared"
fi
if [[ -f ~/.wandb_env ]]; then
    ok "~/.wandb_env exists"
else
    bad "~/.wandb_env not found"
fi

hdr "summary"
echo "passed: $PASS / failed: $FAIL"
if [[ $FAIL -eq 0 ]]; then
    echo "All checks green. Run:  bash build_env_delta_x86.sh"
else
    echo "Some checks failed. Inspect above and adjust build_env_delta_x86.sh paths/modules."
fi
