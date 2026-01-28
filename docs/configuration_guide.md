# Configuration Guide

Complete reference for all configuration files in the Multi-Dimensional AI project.

All configuration files are located in the `configs/` directory and use YAML format.

---

## Configuration Files Overview

| File                          | Purpose                     | Use Case                      |
| ----------------------------- | --------------------------- | ----------------------------- |
| `model_config.yaml`           | Model architecture settings | Base model configuration      |
| `training_config.yaml`        | Full training parameters    | Main training runs            |
| `single_modality_config.yaml` | Single-modality training    | Phase 3 validation            |
| `pairwise_config.yaml`        | Pairwise modality training  | Phase 4 integration           |
| `text_only_config.yaml`       | Text-only training          | LLM-like testing              |
| `real_training_config.yaml`   | Real VR data training       | Training on recorded sessions |
| `inference_config.yaml`       | Inference parameters        | Model deployment              |
| `evolution_config.yaml`       | Evolutionary optimization   | Phase 6 fine-tuning           |
| `evolution_dryrun.yaml`       | Evolution testing           | Validate evolution setup      |
| `stage5_config.yaml`          | Phase 5 backprop training   | Full multi-modal training     |

---

## model_config.yaml

**Purpose:** Define model architecture and hyperparameters.

### Key Sections

#### Model Architecture

```yaml
model:
    # Transformer backbone
    hidden_dim: 512 # Hidden layer dimension
    num_layers: 6 # Number of transformer layers
    num_heads: 8 # Attention heads
    context_length: 2048 # Maximum sequence length

    # Dropout
    dropout: 0.1
    attention_dropout: 0.1

    # Embeddings
    vocab_size: 50000 # Text vocabulary size
    embedding_dim: 512 # Token embedding dimension
```

#### Encoder Settings

```yaml
encoders:
    vision:
        type: "resnet" # Architecture: resnet, vit
        pretrained: true # Use pretrained weights
        output_dim: 512

    audio:
        sample_rate: 16000
        hop_length: 320
        codebook_size: 1024
        output_dim: 1536

    proprioception:
        input_dim: 20 # Joint angles + velocities
        hidden_dim: 256
        output_dim: 512

    touch:
        num_sensors: 5 # Per-finger sensors
        hidden_dim: 128
        output_dim: 512
```

#### Decoder Settings

```yaml
decoders:
    text:
        vocab_size: 50000
        hidden_dim: 512

    audio:
        codebook_size: 1024
        sample_rate: 16000

    animation:
        num_joints: 20
        output_type: "continuous" # continuous or discrete
```

### Usage

```bash
# Used as base config for other configs
# Typically imported/extended by training configs
```

---

## training_config.yaml

**Purpose:** Main configuration for full multi-modal training.

### Key Sections

#### Training Parameters

```yaml
training:
    phase: "backprop" # Options: backprop, evolutionary

    optimizer:
        type: "adamw"
        lr: 3.0e-4 # Learning rate
        betas: [0.9, 0.95]
        weight_decay: 0.01
        eps: 1.0e-8

    lr_scheduler:
        type: "cosine_with_warmup"
        warmup_steps: 10000
        max_steps: 1000000
        min_lr: 3.0e-5

    batch_size: 16
    gradient_accumulation_steps: 4 # Effective batch = 64
    max_grad_norm: 1.0

    max_steps: 1000000
    eval_interval: 5000
    save_interval: 10000
    log_interval: 100
```

#### Mixed Precision

```yaml
mixed_precision: true
precision_type: "bf16" # Options: fp16, bf16
```

**Note:** BF16 recommended for better numerical stability.

#### Distributed Training

```yaml
distributed:
    enabled: false
    backend: "nccl" # nccl for NVIDIA GPUs
    world_size: 1 # Number of GPUs
```

**Enable for multi-GPU:**

```yaml
distributed:
    enabled: true
    backend: "nccl"
    world_size: 2 # For 2x RTX 3090
```

#### Data Configuration

```yaml
data:
    synthetic_data_dir: "./data/synthetic"
    cache_dir: "./data/cache"

    num_workers: 4 # DataLoader workers
    prefetch_factor: 2
    pin_memory: true # Faster CPU→GPU transfer

    augmentation:
        audio_noise: 0.01 # Add small noise to audio
        visual_brightness: 0.1 # Random brightness adjustment
        visual_contrast: 0.1 # Random contrast adjustment
```

#### Checkpointing

```yaml
checkpointing:
    save_dir: "./checkpoints"
    keep_last_n: 5 # Keep only 5 most recent checkpoints
    save_optimizer_state: true
    save_on_error: true # Save checkpoint on crash
```

#### Logging

```yaml
logging:
    log_dir: "./logs"
    use_wandb: true # Enable Weights & Biases
    wandb_project: "multi-dimensional-ai"
    wandb_entity: "your_username"
    log_gradients: false # Log gradient histograms (expensive)
    log_learning_rate: true
    log_memory: true
```

#### Validation

```yaml
validation:
    enabled: true
    val_split: 0.05 # 5% of data for validation
    max_val_steps: 100
    metrics:
        - "perplexity"
        - "token_accuracy"
        - "cross_modal_consistency"
```

### Usage

```bash
python scripts/train.py --config configs/training_config.yaml
```

---

## single_modality_config.yaml

**Purpose:** Training individual encoder-decoder pairs in isolation (Phase 3).

### Key Differences from Full Training

```yaml
training:
    modality: "text" # Options: text, audio, vision, touch, proprio

    # Mask other modalities
    active_inputs: ["internal_voice", "external_voice"]
    active_outputs: ["internal_text", "external_text"]

    # Faster training for validation
    max_steps: 50000
    batch_size: 32
```

### Modality Options

**Text modality:**

```yaml
training:
    modality: "text"
    active_inputs: ["internal_voice", "external_voice"]
    active_outputs: ["internal_text", "external_text"]
```

**Audio modality:**

```yaml
training:
    modality: "audio"
    active_inputs: ["audio"]
    active_outputs: ["audio_output"]
```

**Vision modality:**

```yaml
training:
    modality: "vision"
    active_inputs: ["vision_left", "vision_right"]
    active_outputs: ["internal_text"] # Image captioning
```

### Usage

```bash
python scripts/train_single_modality.py --config configs/single_modality_config.yaml
```

---

## pairwise_config.yaml

**Purpose:** Train cross-modal translations between two modalities (Phase 4).

### Configuration

```yaml
training:
    phase: "pairwise"

    # Define modality pairs
    input_modality: "vision"
    output_modality: "text"

    # Active streams
    active_inputs: ["vision_left"]
    active_outputs: ["external_text"]

    max_steps: 100000
    batch_size: 32
```

### Common Pairwise Combinations

**Vision → Text (Image Captioning):**

```yaml
input_modality: "vision"
output_modality: "text"
active_inputs: ["vision_left"]
active_outputs: ["external_text"]
```

**Text → Audio (Speech Synthesis):**

```yaml
input_modality: "text"
output_modality: "audio"
active_inputs: ["external_voice"]
active_outputs: ["audio_output"]
```

**Touch → Vocalization (Reactive Sounds):**

```yaml
input_modality: "touch"
output_modality: "audio"
active_inputs: ["touch"]
active_outputs: ["audio_output"]
```

### Usage

```bash
python scripts/train.py --config configs/pairwise_config.yaml
```

---

## text_only_config.yaml

**Purpose:** Train using only text modalities (LLM-like behavior).

### Configuration

```yaml
training:
    # Only text modalities active
    active_inputs: ["internal_voice", "external_voice"]
    active_outputs: ["internal_text", "external_text"]

    # Loss weights (disable others)
    loss_weights:
        internal_text: 1.0
        external_text: 1.0
        audio: 0.0 # Disabled
        animation: 0.0 # Disabled

    max_steps: 100000
    batch_size: 64 # Can use larger batch
```

### Use Cases

1. **Pipeline validation:** Test training loop with simple text data
2. **LLM baseline:** Establish text-only performance baseline
3. **Fast iteration:** Quick experiments before full multi-modal training

### Usage

```bash
python scripts/train_text_only.py --config configs/text_only_config.yaml
```

---

## real_training_config.yaml

**Purpose:** Training on real VR session data (not synthetic).

### Configuration

```yaml
data:
    root_dir: "data/sessions"
    batch_size: 4 # Smaller due to larger real data
    modalities: ["vision_left", "audio", "touch", "proprio"]

    # Real data specific
    video_fps: 30
    audio_sample_rate: 48000

training:
    max_steps: 100000
    log_interval: 100
    save_interval: 5000

model:
    hidden_dim: 512
    num_layers: 6
```

### Requirements

- VR session data validated with `validate_session.py`
- Directory structure matching `dataset_formats.md`
- Sufficient disk space for video/audio data

### Usage

```bash
python scripts/train_real.py --config configs/real_training_config.yaml
```

---

## inference_config.yaml

**Purpose:** Configuration for model inference and deployment.

### Configuration

```yaml
inference:
    # Model settings
    checkpoint_path: "checkpoints/model_final.pt"
    device: "cuda" # cuda or cpu

    # Generation parameters
    temperature: 0.8 # Sampling temperature (0.0 = greedy)
    top_k: 50 # Top-k sampling
    top_p: 0.9 # Nucleus sampling

    # Performance
    batch_size: 1 # Typically 1 for real-time
    max_length: 512 # Maximum generation length

    # Output settings
    output_dir: "results/"
    save_intermediate: false # Save intermediate states

    # VR integration
    vr_server:
        enabled: false
        host: "localhost"
        port: 5000
```

### Generation Parameters

**Conservative (safe, coherent):**

```yaml
temperature: 0.7
top_k: 40
top_p: 0.9
```

**Creative (diverse, exploratory):**

```yaml
temperature: 1.0
top_k: 100
top_p: 0.95
```

**Deterministic (greedy):**

```yaml
temperature: 0.0
top_k: 1
top_p: 1.0
```

### Usage

```bash
python scripts/inference.py --config configs/inference_config.yaml --model checkpoints/model_1b.pt
```

---

## evolution_config.yaml

**Purpose:** Evolutionary optimization with genetic algorithms (Phase 6).

### Configuration

```yaml
evolution:
    # Population
    population_size: 20 # Number of model variants
    elite_ratio: 0.2 # Top 20% kept unchanged

    # Mutation
    mutation_rate: 0.1 # Probability of mutating a weight
    mutation_scale: 0.01 # Magnitude of mutations

    # Crossover
    crossover_rate: 0.3 # Probability of crossover
    crossover_method: "uniform" # uniform or single_point

    # Evolution
    num_generations: 100
    fitness_metric: "survival_time" # How to evaluate fitness

    # Environment
    vr_env:
        enabled: true
        parallel_instances: 4 # Run 4 creatures simultaneously
        max_episode_length: 1000
```

### Fitness Metrics

**Survival time:**

```yaml
fitness_metric: "survival_time"
```

**Task completion:**

```yaml
fitness_metric: "task_completion"
fitness_config:
    tasks: ["reach_target", "avoid_obstacles"]
    weights: [0.6, 0.4]
```

**Combined metrics:**

```yaml
fitness_metric: "weighted_average"
fitness_config:
    metrics:
        - name: "survival_time"
          weight: 0.4
        - name: "interaction_quality"
          weight: 0.3
        - name: "goal_completion"
          weight: 0.3
```

### Usage

```bash
python scripts/train_evolution.py --config configs/evolution_config.yaml
```

---

## Customizing Configurations

### Creating a New Config

1. **Copy existing config** as template:

    ```bash
    cp configs/training_config.yaml configs/my_experiment.yaml
    ```

2. **Modify parameters** for your experiment

3. **Use with training script:**
    ```bash
    python scripts/train.py --config configs/my_experiment.yaml
    ```

### Common Modifications

#### Adjust Model Size

**Tiny (10M params):**

```yaml
model:
    hidden_dim: 256
    num_layers: 4
    num_heads: 4
```

**Small (100M params):**

```yaml
model:
    hidden_dim: 512
    num_layers: 6
    num_heads: 8
```

**Large (1B params):**

```yaml
model:
    hidden_dim: 1024
    num_layers: 12
    num_heads: 16
```

#### Memory Optimization

**For RTX 3090 24GB:**

```yaml
training:
    batch_size: 16
    gradient_accumulation_steps: 4
    mixed_precision: true
    precision_type: "bf16"

    # Enable gradient checkpointing
    gradient_checkpointing: true
```

#### Speed Optimization

**For fast iteration:**

```yaml
training:
    max_steps: 10000 # Shorter run
    eval_interval: 500 # Less frequent eval
    save_interval: 2000 # Less frequent saves

data:
    num_workers: 8 # More workers
    prefetch_factor: 4 # More prefetching
```

---

## Configuration Best Practices

### Version Control

1. **Never commit credentials** - Use environment variables:

    ```yaml
    logging:
        wandb_api_key: "${WANDB_API_KEY}" # From .env
    ```

2. **Use descriptive names** - Include experiment purpose:
    - ✅ `training_config_1b_text_heavy.yaml`
    - ❌ `config_v3.yaml`

3. **Document changes** - Add comments explaining non-standard values:
    ```yaml
    training:
        lr: 1.0e-3 # Increased for faster convergence on small dataset
    ```

### Testing Configs

Before long training runs:

1. **Dry run** with small steps:

    ```yaml
    training:
        max_steps: 100
    ```

2. **Validate config** structure:

    ```bash
    python -c "import yaml; yaml.safe_load(open('configs/training_config.yaml'))"
    ```

3. **Check resource requirements:**
    ```bash
    python scripts/profile_multimodal.py --config configs/training_config.yaml
    ```

---

## Troubleshooting

### Config Not Loading

**Error:** `FileNotFoundError: configs/training_config.yaml not found`

**Solution:** Run from project root:

```bash
cd "Multi Dimensional AI"
python scripts/train.py --config configs/training_config.yaml
```

### Invalid YAML Syntax

**Error:** `yaml.scanner.ScannerError: mapping values are not allowed here`

**Common causes:**

- Missing quotes around special characters
- Incorrect indentation (use 2 or 4 spaces consistently)
- Tabs instead of spaces in YAML

**Solution:** Validate YAML:

```bash
python -c "import yaml; yaml.safe_load(open('configs/training_config.yaml'))"
```

### Missing Required Fields

**Error:** `KeyError: 'hidden_dim'`

**Solution:** Ensure all required fields are present. Compare with reference configs.

---

## Quick Reference: Common Settings

| Setting           | Conservative | Balanced | Aggressive |
| ----------------- | ------------ | -------- | ---------- |
| Learning Rate     | 1e-4         | 3e-4     | 1e-3       |
| Batch Size (3090) | 8            | 16       | 32         |
| Gradient Accum    | 8            | 4        | 2          |
| Warmup Steps      | 5000         | 10000    | 20000      |
| Save Interval     | 5000         | 10000    | 25000      |
| Dropout           | 0.2          | 0.1      | 0.05       |

**Conservative:** Stable, slower training, good for initial experiments
**Balanced:** Default settings, good for most use cases
**Aggressive:** Faster training, requires monitoring, may be unstable
