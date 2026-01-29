# WSL2 Setup for Training

Using WSL2 (Windows Subsystem for Linux) provides significant performance benefits:
- **Multi-worker data loading** (4-8x faster data throughput)
- **Better PyTorch multiprocessing** support
- **Native Linux CUDA drivers** (since WSL2 2.0)

## Quick Start

### 1. Open WSL2 Terminal

```bash
# From Windows, open PowerShell or CMD and type:
wsl

# Or open "Ubuntu" from Start menu if you installed Ubuntu
```

### 2. Navigate to Your Project

Windows drives are mounted at `/mnt/c`, `/mnt/d`, `/mnt/g`, etc.

```bash
# Navigate to your project
cd /mnt/g/AI/multiDimAI/Multi-Dimensional-AI

# Verify you're in the right place
ls -la
```

### 3. Set Up Python Environment

```bash
# Install Python and venv if not already installed
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv_wsl

# Activate it
source venv_wsl/bin/activate

# Install PyTorch with CUDA (check https://pytorch.org for latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### 4. Verify CUDA Works

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

### 5. Run Training

```bash
# Staged training (recommended)
python scripts/train_staged.py \
    --config configs/training_fast.yaml \
    --model-config configs/model_100m.yaml \
    --data "/mnt/g/Datasets/wiki_tokenized.bin" \
    --phase text

# Or the fast text-only script
python scripts/train_text_fast.py \
    --config configs/training_fast.yaml \
    --model-config configs/model_100m.yaml \
    --data "/mnt/g/Datasets/wiki_tokenized.bin"
```

## Performance Comparison

| Setting | Windows CMD | WSL2 |
|---------|-------------|------|
| `num_workers` | 0 (forced) | 4+ |
| Data loading | Single-threaded | Multi-threaded |
| Expected speedup | Baseline | **2-4x faster** |

## Troubleshooting

### CUDA Not Found

If CUDA isn't detected in WSL2:

```bash
# Check NVIDIA driver version (run in Windows PowerShell)
nvidia-smi

# In WSL2, you need the WSL2-specific CUDA toolkit
# DON'T install the full CUDA toolkit, the driver is shared from Windows

# Just ensure PyTorch is installed with CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

WSL2 has memory limits by default. Create/edit `~/.wslconfig` in Windows:

```
# In Windows: C:\Users\<YourUsername>\.wslconfig
[wsl2]
memory=48GB  # Adjust based on your RAM
swap=8GB
```

Then restart WSL2:
```powershell
# In PowerShell
wsl --shutdown
wsl
```

### Slow File Access on /mnt/

Accessing Windows files through `/mnt/` has overhead. For best performance:

```bash
# Option 1: Copy data to WSL2 native filesystem (faster)
cp /mnt/g/Datasets/wiki_tokenized.bin ~/data/

# Then train from there
python scripts/train_staged.py --data ~/data/wiki_tokenized.bin ...

# Option 2: Use Windows path directly (slower but convenient)
python scripts/train_staged.py --data /mnt/g/Datasets/wiki_tokenized.bin ...
```

### Permission Issues

```bash
# If you get permission errors
chmod +x scripts/*.py

# For file access issues
sudo chown -R $USER:$USER /path/to/project
```

## One-Line Setup Script

Save this as `setup_wsl.sh` and run it:

```bash
#!/bin/bash
set -e

echo "Setting up WSL2 training environment..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install -y python3 python3-pip python3-venv

# Create venv
python3 -m venv venv_wsl
source venv_wsl/bin/activate

# Install PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt

# Test CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "Setup complete! Activate with: source venv_wsl/bin/activate"
```

## Expected Performance

With WSL2 and all optimizations:

| Metric | Windows CMD | WSL2 + Optimizations |
|--------|-------------|---------------------|
| Tokens/sec | ~665 | ~15,000-20,000 |
| Speedup | 1x | **~25-30x** |

The majority of the speedup comes from:
1. `num_workers=4` instead of 0 (removes data loading bottleneck)
2. TF32 tensor cores enabled
3. Proper FP16 mixed precision
