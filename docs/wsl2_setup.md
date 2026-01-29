# WSL2 Setup for Training

Using WSL2 (Windows Subsystem for Linux) provides significant performance benefits:
- **Multi-worker data loading** (4-8x faster data throughput)
- **Better PyTorch multiprocessing** support
- **Native Linux CUDA drivers** (since WSL2 2.0)

## Quick Start (Recommended)

We provide scripts to automate setup and training.

### Step 1: Open WSL2

```bash
# From Windows PowerShell or CMD:
wsl

# Or open "Ubuntu" from Start menu
```

### Step 2: Navigate to Project

Windows drives are mounted at `/mnt/c`, `/mnt/d`, `/mnt/g`, etc.

```bash
cd /mnt/g/AI/multiDimAI/Multi-Dimensional-AI
```

### Step 3: Run Setup Script

```bash
bash setup_wsl.sh
```

This script will:
- Update system packages
- Install Python 3 and build dependencies
- Create virtual environment (`venv_wsl`)
- Install PyTorch with CUDA support
- Install project requirements
- Verify CUDA is working

### Step 4: Start Training

```bash
# Simple usage - trains text phase by default
bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin

# Specify a phase
bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin text

# With extra arguments
bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin text --max-steps 50000

# Continue from checkpoint (e.g., adding audio after text training)
bash train_wsl.sh /mnt/g/Datasets/audio_data.bin audio --checkpoint checkpoints/staged/text/text_final.pt
```

## Training Launcher Usage

The `train_wsl.sh` script simplifies training:

```
bash train_wsl.sh <data_path> [phase] [extra_args...]

Arguments:
  data_path    Path to tokenized .bin file (required)
  phase        Training phase: text, audio, vision, full (default: text)
  extra_args   Additional arguments passed to training script
```

### Examples

```bash
# Phase 1: Train on text (Wikipedia, etc.)
bash train_wsl.sh /mnt/g/Datasets/wiki_tokenized.bin text

# Phase 2: Add audio capability
bash train_wsl.sh /mnt/g/Datasets/audio_data.bin audio \
    --checkpoint checkpoints/staged/text/text_final.pt

# Phase 3: Add vision
bash train_wsl.sh /mnt/g/Datasets/vision_data.bin vision \
    --checkpoint checkpoints/staged/audio/audio_final.pt

# Custom settings
bash train_wsl.sh ~/data/wiki.bin text --max-steps 100000 --batch-size 16
```

## Performance Tip: Copy Data to WSL Filesystem

Accessing Windows files through `/mnt/` has I/O overhead. For best performance, copy your data to WSL's native filesystem:

```bash
# Create data directory in WSL home
mkdir -p ~/data

# Copy your tokenized data (one-time)
cp /mnt/g/Datasets/wiki_tokenized.bin ~/data/

# Train from WSL filesystem (faster)
bash train_wsl.sh ~/data/wiki_tokenized.bin text
```

## Performance Comparison

| Setting | Windows CMD | WSL2 |
|---------|-------------|------|
| `num_workers` | 0 (forced) | 4+ |
| Data loading | Single-threaded | Multi-threaded |
| Expected tokens/sec | ~665 | ~15,000-20,000 |
| Speedup | 1x | **~25-30x** |

The speedup comes from:
1. `num_workers=4` instead of 0 (removes data loading bottleneck)
2. TF32 tensor cores enabled
3. FP16 mixed precision

## Manual Setup (Alternative)

If you prefer to set up manually instead of using `setup_wsl.sh`:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential

# 3. Create virtual environment
python3 -m venv venv_wsl
source venv_wsl/bin/activate

# 4. Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install requirements
pip install -r requirements.txt

# 6. Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## Troubleshooting

### CUDA Not Found

```bash
# Check NVIDIA driver in Windows PowerShell first
nvidia-smi

# In WSL2, DON'T install full CUDA toolkit (driver is shared from Windows)
# Just ensure PyTorch has CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory

WSL2 has memory limits by default. Create/edit `.wslconfig` in Windows:

```
# File: C:\Users\<YourUsername>\.wslconfig
[wsl2]
memory=48GB
swap=8GB
```

Then restart WSL2:
```powershell
wsl --shutdown
wsl
```

### Permission Issues

```bash
# Make scripts executable
chmod +x setup_wsl.sh train_wsl.sh scripts/*.py

# Fix ownership if needed
sudo chown -R $USER:$USER .
```

### "Module not found" Errors

```bash
# Make sure virtual environment is activated
source venv_wsl/bin/activate

# Verify you're using the right Python
which python3  # Should show: /path/to/venv_wsl/bin/python3
```

## Full Workflow Summary

```bash
# One-time setup
wsl
cd /mnt/g/AI/multiDimAI/Multi-Dimensional-AI
bash setup_wsl.sh

# Copy data for better performance (optional but recommended)
mkdir -p ~/data
cp /mnt/g/Datasets/wiki_tokenized.bin ~/data/

# Train!
bash train_wsl.sh ~/data/wiki_tokenized.bin text
```

## Staged Training Phases

| Phase | What's Trained | Use Case |
|-------|----------------|----------|
| `text` | Text encoders/decoders + transformer | Initial LLM training |
| `audio` | + Audio encoder/decoder | Add speech/transcription |
| `vision` | + Visual encoder | Add image understanding |
| `full` | + Body (proprioception/touch/animation) | Full VR embodiment |

Each phase builds on the previous, preserving learned weights.
