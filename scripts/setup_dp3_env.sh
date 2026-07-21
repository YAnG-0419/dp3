#!/usr/bin/env bash
# Recreate the offline DP3 training env on a new Linux + NVIDIA machine.
#
# Prerequisites:
#   - conda / miniforge
#   - NVIDIA driver new enough for the chosen CUDA wheel
#     (this repo's reference machine uses torch cu128 + driver ~580)
#   - this repository already cloned
#
# Usage (from repo root):
#   conda env create -f environment_dp3.yml   # first time only
#   conda activate dp3
#   bash scripts/setup_dp3_env.sh
#
# Optional: override CUDA wheel channel, e.g. CUDA 12.1 GPUs:
#   TORCH_CUDA=cu121 bash scripts/setup_dp3_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup_dp3_env] python=$(python -V 2>&1)  TORCH_CUDA=${TORCH_CUDA}"

python -m pip install -U pip setuptools wheel

echo "[setup_dp3_env] installing torch/torchvision from ${TORCH_INDEX}"
python -m pip install torch torchvision --index-url "${TORCH_INDEX}"

echo "[setup_dp3_env] installing requirements_dp3.txt"
python -m pip install -r "${ROOT}/requirements_dp3.txt"

echo "[setup_dp3_env] editable install: diffusion_policy_3d"
python -m pip install -e "${ROOT}/3D-Diffusion-Policy"

echo "[setup_dp3_env] editable install: pytorch3d_simplified"
python -m pip install -e "${ROOT}/third_party/pytorch3d_simplified"

echo "[setup_dp3_env] verifying imports"
python - <<'PY'
import torch
import zarr
import hydra
import diffusers
import wandb
import pytorch3d
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("ok")
PY

echo "[setup_dp3_env] done. Activate with: conda activate dp3"
