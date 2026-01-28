# Scripts Reference

Complete reference for all available scripts in the Multi-Dimensional AI project.

---

## Training Scripts

### train.py

**Purpose:** Main training script for full multi-modal model.

**Usage:**

```bash
python scripts/train.py --config configs/training_config.yaml
```

**Options:**

- `--config`: Path to training configuration file (required)
- `--resume`: Resume from checkpoint path (optional)

**Example:**

```bash
python scripts/train.py --config configs/training_config.yaml --resume checkpoints/model_step_50000.pt
```

**When to use:** Phase 5+ backpropagation training on full multi-modal data.

---

### train_single_modality.py

**Purpose:** Train individual encoder-decoder pairs in isolation.

**Usage:**

```bash
python scripts/train_single_modality.py --config configs/single_modality_config.yaml
```

**Options:**

- `--config`: Path to single-modality configuration file
- `--modality`: Specific modality to train (e.g., 'text', 'audio', 'vision')

**Example:**

```bash
python scripts/train_single_modality.py --config configs/single_modality_config.yaml --modality text
```

**When to use:** Phase 3 single-modality training to validate encoders/decoders work correctly.

---

### train_baseline.py

**Purpose:** Train a baseline model for comparison purposes.

**Usage:**

```bash
python scripts/train_baseline.py --config configs/model_config.yaml
```

**When to use:** Establish baseline performance metrics before full multi-modal training.

---

### train_text_only.py

**Purpose:** Train using only text modalities (internal/external text).

**Usage:**

```bash
python scripts/train_text_only.py --config configs/text_only_config.yaml
```

**Example:**

```bash
python scripts/train_text_only.py --config configs/text_only_config.yaml
```

**When to use:** Test pipeline with LLM-like training (text input → text output) before adding other modalities.

---

### train_real.py

**Purpose:** Train on real VR session data (not synthetic).

**Usage:**

```bash
python scripts/train_real.py --config configs/real_training_config.yaml
```

**Requirements:**

- Real VR session data in `data/sessions/`
- Sessions validated with `validate_session.py`

**When to use:** Training on actual recorded VR interactions.

---

### train_evolution.py

**Purpose:** Evolutionary optimization (genetic algorithms) for model fine-tuning.

**Usage:**

```bash
python scripts/train_evolution.py --config configs/evolution_config.yaml
```

**Options:**

- `--config`: Evolution configuration file
- `--population-size`: Number of variants to maintain
- `--generations`: Number of evolution cycles

**Example:**

```bash
python scripts/train_evolution.py --config configs/evolution_config.yaml --population-size 20 --generations 100
```

**When to use:** Phase 6 evolutionary fine-tuning after backpropagation training.

---

## Inference Scripts

### inference.py

**Purpose:** Run trained model inference on new inputs.

**Usage:**

```bash
python scripts/inference.py --model checkpoints/model_1b.pt --config configs/inference_config.yaml
```

**Options:**

- `--model`: Path to trained model checkpoint (required)
- `--config`: Inference configuration file (required)
- `--input`: Input data path (optional, uses config default if not specified)
- `--output`: Output directory for results (optional)

**Example:**

```bash
python scripts/inference.py \
	--model checkpoints/model_final.pt \
	--config configs/inference_config.yaml \
	--input data/test_sessions/session_001 \
	--output results/inference_001
```

**When to use:** Testing trained model on new data or deploying for VR integration.

---

## Utility Scripts

### validate_session.py

**Purpose:** Validate VR session data quality and integrity.

**Usage:**

```bash
python scripts/validate_session.py <session_dir> [--strict] [--fix] [--json]
```

**Options:**

- `<session_dir>`: Path to session directory (required)
- `--strict`: Fail on warnings (treat warnings as errors)
- `--fix`: Attempt to auto-fix minor issues
- `--json`: Output report as JSON

**Examples:**

```bash
# Validate single session
python scripts/validate_session.py data/sessions/session_001

# Validate with strict mode
python scripts/validate_session.py data/sessions/session_001 --strict

# Get JSON report
python scripts/validate_session.py data/sessions/session_001 --json > report.json

# Validate and auto-fix
python scripts/validate_session.py data/sessions/session_001 --fix
```

**Checks:**

- File existence (metadata.jsonl, audio.wav, vision_left.mp4)
- Timestamp continuity (gaps < 100ms)
- Sensor value ranges (touch: [0,1], proprioception: valid)
- Audio/video synchronization

**Exit codes:**

- 0: Success (passed)
- 1: Failed validation or warnings (with --strict)

---

### audit_docs.py

**Purpose:** Enforce strict documentation standards across Python files.

**Usage:**

```bash
python scripts/audit_docs.py
```

**Checks:**

- File docstrings
- Class docstrings
- Function docstrings (Purpose, Args, Returns, Workflow)
- Missing sections

**Output:**

- Console report of all issues
- `docs/audit_report.txt` with detailed findings

**When to run:** Before commits to ensure all code is properly documented.

---

### analyze_indentation.py

**Purpose:** Analyze and enforce indentation standards (tabs only, max 5 levels).

**Usage:**

```bash
python scripts/analyze_indentation.py
```

**Checks:**

- Tab vs space usage (must use tabs)
- Maximum nesting depth (max 5 levels)
- Average indentation per file

**Output:**

- List of files violating standards
- Indentation statistics

**When to run:** During code reviews or before commits.

---

## Profiling Scripts

### profile_multimodal.py

**Purpose:** Profile multi-modal model performance and identify bottlenecks.

**Usage:**

```bash
python scripts/profile_multimodal.py
```

**Output:**

- Memory usage breakdown
- Time per operation
- TensorBoard profiling trace
- `profiler_output/profile_report.json` with recommendations

**Metrics tracked:**

- Forward pass time
- Backward pass time
- Data loading time
- GPU memory allocation
- Throughput (samples/sec)

**Common bottlenecks identified:**

- Data loading (solution: increase num_workers)
- CPU→GPU transfer (solution: pin_memory=True)
- Model size (solution: gradient checkpointing)

---

### profile_training.py

**Purpose:** Profile training loop performance.

**Usage:**

```bash
python scripts/profile_training.py
```

**When to use:** Debugging slow training or optimizing throughput.

---

## Data Processing Scripts

### tokenize_corpus.py

**Purpose:** Pre-tokenize text corpus for faster training.

**Usage:**

```bash
python scripts/tokenize_corpus.py --input corpus.txt --output tokenized/
```

**Options:**

- `--input`: Input text file or directory
- `--output`: Output directory for tokenized data
- `--vocab-size`: Vocabulary size (default: from config)

**When to use:** Pre-processing large text corpora before training.

---

### prepare_hf_data.py

**Purpose:** Convert HuggingFace datasets (cache or download) to binary format for training.

**Usage:**

```bash
python scripts/prepare_hf_data.py --output data/output.bin --dataset wikimedia/wikipedia
```

**Options:**

- `--dataset`: HuggingFace dataset name (default: wikimedia/wikipedia)
- `--config`: Dataset config (default: 20231101.en)
- `--cache_dir`: HF Cache directory (optional)
- `--output`: Output .bin file path (required)
- `--max_tokens`: Limit processing to N tokens (optional)

**When to use:** Converting external datasets for use with training scripts.

---

## Quick Reference Table

| Script                     | Phase     | Purpose                    | Key Option      |
| -------------------------- | --------- | -------------------------- | --------------- |
| `train.py`                 | 5+        | Full multi-modal training  | `--config`      |
| `train_single_modality.py` | 3         | Single modality validation | `--modality`    |
| `train_text_only.py`       | 3-4       | Text-only LLM mode         | `--config`      |
| `train_real.py`            | 5+        | Real VR data training      | `--config`      |
| `train_evolution.py`       | 6         | Evolutionary optimization  | `--generations` |
| `inference.py`             | 7+        | Model inference            | `--model`       |
| `validate_session.py`      | Data prep | VR session validation      | `--strict`      |
| `audit_docs.py`            | Dev       | Documentation check        | None            |
| `analyze_indentation.py`   | Dev       | Code style check           | None            |
| `profile_multimodal.py`    | Debug     | Performance profiling      | None            |

---

## Common Workflows

### Workflow 1: Initial Pipeline Test

```bash
# 1. Train text-only to validate pipeline
python scripts/train_text_only.py --config configs/text_only_config.yaml

# 2. Profile to check performance
python scripts/profile_multimodal.py

# 3. Run inference test
python scripts/inference.py --model checkpoints/text_model.pt --config configs/inference_config.yaml
```

### Workflow 2: Full Training Pipeline

```bash
# 1. Validate VR data
for session in data/sessions/session_*; do
	python scripts/validate_session.py "$session" --fix
done

# 2. Train baseline
python scripts/train_baseline.py --config configs/model_config.yaml

# 3. Train full model
python scripts/train.py --config configs/training_config.yaml

# 4. Evolutionary fine-tuning
python scripts/train_evolution.py --config configs/evolution_config.yaml
```

### Workflow 3: Code Quality Check

```bash
# 1. Check documentation
python scripts/audit_docs.py

# 2. Check indentation
python scripts/analyze_indentation.py

# 3. Run tests
pytest tests/ -v
```

---

## Best Practices

### Training Scripts

1. **Always specify --config explicitly** - Don't rely on defaults
2. **Use descriptive checkpoint names** - Include model size, step count
3. **Monitor logs in real-time** - Use `tail -f logs/training.log`
4. **Save checkpoints frequently** - Especially for long training runs

### Validation Scripts

1. **Run validate_session.py before training** - Catch data issues early
2. **Use --strict for production data** - Ensure highest quality
3. **Archive validation reports** - Track data quality over time

### Profiling Scripts

1. **Profile on representative data** - Use realistic batch sizes
2. **Profile before scaling up** - Optimize on small model first
3. **Compare before/after changes** - Track optimization impact

---

## Troubleshooting

### Script won't run

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Run from project root directory:

```bash
cd "Multi Dimensional AI"
python scripts/train.py --config configs/training_config.yaml
```

### OOM errors during training

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

1. Reduce batch size in config
2. Enable gradient checkpointing
3. Use gradient accumulation
4. Profile memory with `profile_multimodal.py`

### Slow data loading

**Symptom:** Low samples/sec, CPU at 100%

**Solutions:**

1. Increase `num_workers` in config
2. Enable `pin_memory=True`
3. Pre-process/cache data
4. Use faster storage (SSD)

---

## Adding New Scripts

When creating new scripts:

1. **Add docstring** explaining purpose
2. **Use argparse** for command-line arguments
3. **Add to this reference** with usage examples
4. **Follow project standards** (tabs, max 5 nesting levels)
5. **Run audit tools** before committing:
    ```bash
    python scripts/audit_docs.py
    python scripts/analyze_indentation.py
    ```
