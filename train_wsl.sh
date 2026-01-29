#!/bin/bash
#
# Quick training launcher for WSL2
#
# Usage:
#   bash train_wsl.sh /path/to/data.bin [phase]
#
# Examples:
#   bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin
#   bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin text
#   bash train_wsl.sh ~/data/wiki_tokenized.bin audio --checkpoint checkpoints/staged/text/text_final.pt
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Usage: bash train_wsl.sh <data_path> [phase] [extra_args...]${NC}"
    echo ""
    echo "Arguments:"
    echo "  data_path    Path to tokenized .bin file"
    echo "  phase        Training phase: text, audio, vision, full (default: text)"
    echo "  extra_args   Additional arguments passed to training script"
    echo ""
    echo "Examples:"
    echo "  bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin"
    echo "  bash train_wsl.sh ~/data/wiki.bin text --max-steps 50000"
    echo "  bash train_wsl.sh ~/data/wiki.bin audio --checkpoint checkpoints/staged/text/text_final.pt"
    exit 1
fi

DATA_PATH="$1"
PHASE="${2:-text}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}Error: Data file not found: $DATA_PATH${NC}"
    exit 1
fi

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv_wsl" ]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source venv_wsl/bin/activate
    else
        echo -e "${RED}Error: Virtual environment not found. Run setup_wsl.sh first.${NC}"
        exit 1
    fi
fi

# Check CUDA
echo -e "${GREEN}Checking CUDA...${NC}"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo -e "${GREEN}========================================================================"
echo "  Starting Training"
echo "========================================================================${NC}"
echo ""
echo "  Data:    $DATA_PATH"
echo "  Phase:   $PHASE"
echo "  Extra:   $EXTRA_ARGS"
echo ""

# Get file size for info
FILE_SIZE=$(du -h "$DATA_PATH" | cut -f1)
echo "  Data size: $FILE_SIZE"
echo ""

# Run training
python scripts/train_staged.py \
    --config configs/training_fast.yaml \
    --model-config configs/model_100m.yaml \
    --data "$DATA_PATH" \
    --phase "$PHASE" \
    $EXTRA_ARGS
