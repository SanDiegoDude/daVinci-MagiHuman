#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# daVinci-MagiHuman — one-shot installer
#
# Sets up a Python 3.12 venv with the correct PyTorch build for your
# CUDA driver, then installs all remaining dependencies.
#
# Usage:
#   bash install.sh          # auto-detect CUDA version
#   bash install.sh cu128    # force a specific CUDA tag
# -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

# ---- Python check ------------------------------------------------
PYTHON=""
for candidate in python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ "$ver" == "3.12" || "$ver" == "3.13" ]]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.12+ is required but not found."
    echo "       Install it (e.g. pyenv install 3.12) and try again."
    exit 1
fi
echo "[1/5] Using $PYTHON ($($PYTHON --version 2>&1))"

# ---- Venv --------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[2/5] Creating venv at ${VENV_DIR}..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "[2/5] Venv already exists at ${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

# ---- Detect CUDA version ----------------------------------------
CUDA_TAG="${1:-}"

if [[ -z "$CUDA_TAG" ]]; then
    echo "[3/5] Detecting CUDA driver version..."
    if command -v nvidia-smi &>/dev/null; then
        DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)

        if [[ -n "$CUDA_VER" ]]; then
            MAJOR="${CUDA_VER%%.*}"
            MINOR="${CUDA_VER##*.}"
            CUDA_TAG="cu${MAJOR}${MINOR}"
            echo "       Driver: ${DRIVER_CUDA}  →  CUDA ${CUDA_VER}  →  ${CUDA_TAG}"
        fi
    fi

    if [[ -z "$CUDA_TAG" ]]; then
        echo "       Could not detect CUDA version. Defaulting to cu128."
        CUDA_TAG="cu128"
    fi
else
    echo "[3/5] Using user-specified CUDA tag: ${CUDA_TAG}"
fi

# Validate the tag is something PyTorch actually ships
VALID_TAGS="cu118 cu121 cu124 cu126 cu128"
if ! echo "$VALID_TAGS" | grep -qw "$CUDA_TAG"; then
    echo "WARNING: ${CUDA_TAG} may not have a PyTorch wheel. Falling back to cu128."
    CUDA_TAG="cu128"
fi

# ---- Install PyTorch ---------------------------------------------
echo "[4/5] Installing PyTorch (${CUDA_TAG})..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# Verify CUDA works
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install — check your driver!'
dev = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'       GPU: {dev}  ({mem:.0f} GB)')
"

# ---- Install remaining deps -------------------------------------
echo "[5/5] Installing remaining dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"
pip install -r "${SCRIPT_DIR}/requirements-nodeps.txt" --no-deps

echo ""
echo "============================================================"
echo " Install complete!"
echo " Activate:  source ${VENV_DIR}/bin/activate"
echo " Run:       python app.py"
echo "============================================================"
