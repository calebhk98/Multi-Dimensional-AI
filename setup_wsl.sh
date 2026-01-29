#!/bin/bash
#
# WSL2 Setup Script for Multi-Dimensional-AI
#
# Usage:
#   1. Open WSL2 terminal
#   2. Navigate to project: cd /mnt/g/AI/multiDimAI/Multi-Dimensional-AI
#   3. Run: bash setup_wsl.sh
#
# This script will:
#   - Install Python 3 and dependencies
#   - Create a virtual environment
#   - Install PyTorch with CUDA support
#   - Install project requirements
#   - Verify CUDA is working
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Multi-Dimensional-AI - WSL2 Setup Script"
echo "========================================================================"
echo ""

# Check if running in WSL
if ! grep -q microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be WSL2. Continuing anyway...${NC}"
fi

# Step 1: Update system
echo -e "${GREEN}[1/6] Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Step 2: Install Python and dependencies
echo -e "${GREEN}[2/6] Installing Python and build dependencies...${NC}"
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential

# Step 3: Create virtual environment
echo -e "${GREEN}[3/6] Creating virtual environment...${NC}"
VENV_DIR="venv_wsl"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing and recreating...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Step 4: Install PyTorch with CUDA
echo -e "${GREEN}[4/6] Installing PyTorch with CUDA support...${NC}"
echo "This may take a few minutes..."

# Detect CUDA version if possible
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "Detected CUDA version: $CUDA_VERSION"
else
    CUDA_VERSION="12.1"
    echo "Could not detect CUDA version, defaulting to $CUDA_VERSION"
fi

# Install PyTorch (cu121 works for CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 5: Install project requirements
echo -e "${GREEN}[5/6] Installing project requirements...${NC}"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}No requirements.txt found. Installing common dependencies...${NC}"
    pip install numpy tqdm pyyaml
fi

# Step 6: Verify CUDA
echo -e "${GREEN}[6/6] Verifying CUDA installation...${NC}"

CUDA_TEST=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(f'SUCCESS: CUDA is available')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'PyTorch Version: {torch.__version__}')
else:
    print('FAILED: CUDA not available')
    exit(1)
" 2>&1)

echo "$CUDA_TEST"

if echo "$CUDA_TEST" | grep -q "SUCCESS"; then
    echo ""
    echo -e "${GREEN}========================================================================"
    echo "  Setup Complete!"
    echo "========================================================================${NC}"
    echo ""
    echo "To start training:"
    echo ""
    echo "  1. Activate the environment:"
    echo "     ${YELLOW}source venv_wsl/bin/activate${NC}"
    echo ""
    echo "  2. Run staged training (recommended):"
    echo "     ${YELLOW}python scripts/train_staged.py \\"
    echo "         --config configs/training_fast.yaml \\"
    echo "         --model-config configs/model_100m.yaml \\"
    echo "         --data \"/mnt/g/Datasets/wiki_tokenized.bin\" \\"
    echo "         --phase text${NC}"
    echo ""
    echo "  Or run the fast text-only script:"
    echo "     ${YELLOW}python scripts/train_text_fast.py \\"
    echo "         --config configs/training_fast.yaml \\"
    echo "         --model-config configs/model_100m.yaml \\"
    echo "         --data \"/mnt/g/Datasets/wiki_tokenized.bin\"${NC}"
    echo ""
    echo "  Pro tip: Copy your data to WSL's native filesystem for faster I/O:"
    echo "     ${YELLOW}mkdir -p ~/data"
    echo "     cp /mnt/g/Datasets/wiki_tokenized.bin ~/data/${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}========================================================================"
    echo "  CUDA Setup Failed!"
    echo "========================================================================${NC}"
    echo ""
    echo "Troubleshooting steps:"
    echo ""
    echo "  1. Make sure you have an NVIDIA GPU and drivers installed on Windows"
    echo "     Run 'nvidia-smi' in PowerShell to verify"
    echo ""
    echo "  2. Ensure WSL2 is updated:"
    echo "     wsl --update"
    echo ""
    echo "  3. Restart WSL2:"
    echo "     wsl --shutdown"
    echo "     wsl"
    echo ""
    echo "  4. Try reinstalling PyTorch:"
    echo "     pip uninstall torch torchvision torchaudio"
    echo "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    exit 1
fi
