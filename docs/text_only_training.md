# Text-Only Training Guide

Step-by-step guide for the 1st round of training: text in, text out (like a standard LLM).

## Overview

The first training round focuses on **text-only mode** to validate the entire pipeline before scaling to multi-modal training. All input and output modalities remain connected in the model, but only text inputs/outputs are actively trained while others are masked or ignored.

### What This Means

- **Input**: `internal_voice_tokens` (text tokens representing thoughts/speech)
- **Output**: `internal_text` (predicted next tokens)
- **Model Architecture**: Full multi-modal model is used, but other encoders/decoders receive null/zero inputs
- **Loss**: Only `internal_text` loss is computed (weight = 1.0), all other losses have weight = 0.0

This is essentially standard language model training (like GPT) using the multi-modal architecture.

---

## Prerequisites

### 1. Environment Setup

The project uses a Python virtual environment located at `venv/`.

**IMPORTANT**: Each new terminal session requires venv activation:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Or use the python directly
.\venv\Scripts\python.exe <command>
```

**Verify Environment:**

```powershell
# Check Python version (should be 3.13.1)
.\venv\Scripts\python.exe --version

# Verify key packages are installed
.\venv\Scripts\python.exe -c "import torch; import transformers; print('Environment OK')"
```

### 2. Training Data

You need tokenized text data in one of these formats:

- `.bin` - NumPy binary format (uint16), recommended for large datasets
- `.pt` - PyTorch tensor file
- `.npy` - NumPy array file

**Creating Training Data:**

Use the provided tokenization script:

```powershell
# Tokenize a text file
.\venv\Scripts\python.exe scripts/tokenize_corpus.py `
	--input data/your_corpus.txt `
	--output data/corpus.bin
```

The script:

- Uses GPT-2 tokenizer (vocab_size = 50257)
- Saves as `uint16` binary for efficiency
- Reports compression ratio and file size

**Recommended Data Size:**

- **Minimum**: 10M tokens (~20MB file) for basic validation
- **Recommended**: 100M+ tokens for meaningful training
- **Production**: 1B+ tokens for quality results

### 3. Hardware Requirements

The configuration in `configs/text_only_config.yaml` is now set to the **"Tiny" (10M parameter)** model for rapid validation on **RTX 3090**:

| Config Parameter | Value | RTX 3090 Notes                     |
| ---------------- | ----- | ---------------------------------- |
| `batch_size`     | 32    | Very fast training, low VRAM usage |
| `hidden_dim`     | 512   | Tiny model (10M params)            |
| `num_layers`     | 6     | Quick convergence                  |
| `max_seq_length` | 2048  | Full context window                |

**For RTX 3090 (24GB VRAM):**

- This "Tiny" model will train extremely fast (~15 hours for 1M samples)
- Ideal for verifying the entire pipeline works before scaling up

---

## Configuration

The text-only configuration is in [`configs/text_only_config.yaml`](file:///c:/Users/caleb/Documents/School/AI%20Documents/AntiGrav/Multi%20Dimensional%20AI/configs/text_only_config.yaml).

### Key Settings (Tiny Model)

```yaml
training:
	max_steps: 31250        # ~1M samples
	batch_size: 32          # Efficient on 3090
	log_interval: 100
	save_interval: 5000

model:
	transformer:
		num_layers: 6       # Tiny model depth
		hidden_dim: 512     # Tiny model width
		num_attention_heads: 8
		ffn_dim: 2048
		max_position_embeddings: 2048

	encoders:
		internal_voice:
			vocab_size: 50257
			embedding_dim: 512
			max_seq_length: 2048
```

### Adjusting for RTX 3090

Current config is already optimized for a fast start on RTX 3090. No changes needed.

---

## Running Training

### Step 1: Dry Run (CRITICAL - Do This First)

Before actual training, run a 10-step dry run to validate the pipeline:

```powershell
.\venv\Scripts\python.exe scripts/train_text_only.py `
	--config configs/text_only_config.yaml `
	--data data/corpus.bin `
	--dry-run
```

**Expected Output:**

```
Loading config from configs/text_only_config.yaml...
Using device: cuda
Creating dataset from data/corpus.bin...
Loaded X,XXX,XXX tokens from data/corpus.bin
Creating X,XXX samples of length 512...
Creating DataLoader with batch_size=8...
Initializing MultiModalCreature model...
Model parameters: X,XXX,XXX total, X,XXX,XXX trainable

*** DRY RUN MODE: Training for only 10 steps ***

Initializing Trainer...
================================================================================
Starting training...
================================================================================

Step 1/10 | Loss: X.XXX | loss_internal_text: X.XXX | samples/sec: XX.X
...
Step 10/10 | Loss: X.XXX | loss_internal_text: X.XXX | samples/sec: XX.X

================================================================================
Training complete!
================================================================================
```

**What the Dry Run Validates:**

- Environment is correctly set up
- Data file loads successfully
- Model initializes without errors
- Forward/backward pass works
- No CUDA OOM errors
- Checkpointing works

> [!CAUTION]
> **Do NOT proceed to full training until the dry run completes successfully!**

### Step 2: Full Training

Once dry run passes, start full training:

```powershell
.\venv\Scripts\python.exe scripts/train_text_only.py `
	--config configs/text_only_config.yaml `
	--data data/corpus.bin
```

**On Desktop (RTX 3090):**

Training will run for the configured `max_steps`. Monitor the output for:

- Loss decreasing over time
- Stable samples/sec throughput
- No OOM errors

**Stopping/Resuming:**

- Press `Ctrl+C` to stop gracefully (saves checkpoint)
- Resume from checkpoint with `Trainer.load_checkpoint()` (see Validation section below)

### Step 3: Monitor Progress

Checkpoints are saved to `checkpoints/text_only/`:

- `model_step_1000.pt`
- `model_step_2000.pt`
- ...
- `model_final.pt` (when training completes)

**Key Metrics:**

| Metric             | Good                           | Warning              | Action                 |
| ------------------ | ------------------------------ | -------------------- | ---------------------- |
| Total Loss         | Decreasing                     | Plateaus immediately | Check LR, data quality |
| loss_internal_text | < 4.0 initially, < 2.0 trained | > 5.0 or NaN         | Check data, reduce LR  |
| samples/sec        | > 10                           | < 5                  | Profile bottlenecks    |

---

## Validation After Training

Use the inference script to test the trained model.

### Simple Text Generation Test

Create `scripts/test_text_generation.py`:

```python
"""
Quick text generation test for trained model.
"""
import torch
import yaml
from transformers import GPT2Tokenizer
from src.models.multimodal_transformer import MultiModalCreature

# Load checkpoint
checkpoint_path = "checkpoints/text_only/model_final.pt"
checkpoint = torch.load(checkpoint_path, map_location="cuda")

# Initialize model
config = checkpoint["config"]
model = MultiModalCreature(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")
model.eval()

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode prompt
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Generate
print(f"Prompt: {prompt}")
print("Generating...")

with torch.no_grad():
	for _ in range(50):  # Generate 50 tokens
		outputs = model(
			internal_voice_tokens=input_ids,
			return_hidden_states=True
		)

		# Get next token prediction
		next_token_logits = outputs["logits_internal_text"][0, -1, :]
		next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

		# Append to sequence
		input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

		# Decode and print
		generated_text = tokenizer.decode(input_ids[0])
		print(f"\r{generated_text}", end="")

print("\n\nDone!")
```

Run:

```powershell
.\venv\Scripts\python.exe scripts/test_text_generation.py
```

**Expected Behavior (Untrained Model):** Nonsensical or repetitive text
**Expected Behavior (Trained Model):** Coherent continuation of the prompt

---

## Troubleshooting

### Environment Issues

**Problem:** `python` or `pytest` not found

**Solution:** The venv is not activated. Use full path:

```powershell
# Instead of: python script.py
# Use:
.\venv\Scripts\python.exe script.py
```

**OR activate venv first:**

```powershell
.\venv\Scripts\Activate.ps1
python script.py  # Now works
```

### Data Issues

**Problem:** `FileNotFoundError: Token file not found`

**Solution:** Create tokenized data first:

```powershell
.\venv\Scripts\python.exe scripts/tokenize_corpus.py `
	--input your_text_file.txt `
	--output data/corpus.bin
```

### CUDA OOM

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size in config:

```yaml
training:
	batch_size: 4  # Reduce from 8
```

### Loss Not Decreasing

**Problem:** Loss stays high or increases

**Possible Causes:**

1. Learning rate too high → Reduce `lr` in config
2. Data quality issues → Check tokenized data
3. Model/data mismatch → Verify vocab_size matches tokenizer

---

## Next Steps After Text Training

Once text-only training is successful:

1. **Validate text generation quality** with inference script
2. **Analyze checkpoints** to understand loss progression
3. **Scale up** to bigger model (hidden_dim=1536) if desired
4. **Move to multi-modal training** (Phase 4: Pairwise Integration)

---

## Summary Checklist

Before starting 1st training round:

- [ ] Environment verified (`.\venv\Scripts\python.exe --version`)
- [ ] Dependencies installed (check with `pip list`)
- [ ] Training data prepared (tokenized .bin file exists)
- [ ] Configuration reviewed and adjusted for RTX 3090
- [ ] **Dry run completed successfully**
- [ ] Monitoring plan in place
- [ ] Text generation test script ready

Once checklist is complete → **Ready for training on desktop!**
