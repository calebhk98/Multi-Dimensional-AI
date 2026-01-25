# Training Guide

## Overview

Training the Multi-Dimensional AI Creature is a multi-phase process designed to bootstrap its capabilities from basic understanding to complex, autonomous behavior.

## Environment Setup

Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
cp .env.example .env
# Configure .env with appropriate API keys and paths
```

## Training Phases

### Phase 1 & 2: Planning and Foundation (Done)

- **Goal**: Establish architecture, encoders, and basic loop.
- **Status**: Complete.

### Phase 3: Single-Modality Training (Weeks 5-8)

- **Goal**: Train individual encoders and decoders to ensure they work correctly in isolation.
- **Definition**: Strict 1-to-1 mapping using the current architecture.
    - **Current Inputs**: The 6 currently defined inputs (Vision, Hearing, Touch, Proprioception, Internal Thoughts, External Text).
    - **Current Outputs**: The 4 currently defined outputs (Internal Thoughts, External Speech, Vocalizations, Body Control).
    - **Note**: These modalities are provisional and may evolve.
    - **No Multi-Modal Inputs**: We do NOT mix inputs here (e.g., No Image + Text combinations).
- **Process**:
    - Even in single-modality mode, the model generates tokens for all outputs. Unused inputs are masked. Unused outputs are ignored or supervised to be "neutral".
    - **Thoughts (Internal Monologue)**: The model _may_ generate internal thoughts during this process, or it may not, depending on configuration.
    - **Examples**:
        - **Text -> Text**: Input "Hello", Output "Hello back".
        - **Audio -> Audio**: Input Sound, Output Mimicked Sound (or response).

### Phase 4: Pairwise Integration (Weeks 9-14)

- **Goal**: Combine two modalities to establish cross-modal understanding.
- **Examples**:
    - **Vision -> Text**: Image Captioning. The model sees an image and describes it in text (External Speech or Thoughts).
    - **Text -> Speech**: The model reads text (Input Text) and generates the corresponding Vocalization (or Speech content).
    - **Touch -> Vocalization**: Reacting to a specific touch sensation with a sound (e.g., pain or surprise).
- **Command**:
    ```bash
    python scripts/train.py --config configs/pairwise_config.yaml
    ```

### Phase 5: Backpropagation (Weeks 15-24)

- **Goal**: End-to-end training of the full model using supervised learning on large multi-modal datasets.
- **Data**: Recorded VR sessions, synthetic data, and internet-scale multi-modal data.

### Phase 6: Evolutionary Optimization (Weeks 25-32)

- **Goal**: Fine-tune behavior using evolutionary algorithms (Genetic Algorithms / ES).
- **Process**:
    - Spawn multiple instances of the creature.
    - Evaluate fitness based on survival duration, interaction quality, and goal completion in VR.
    - Mutate and crossover weights of the best performers.

## Running Training

To start a training session:

```bash
python scripts/train.py --config configs/training_config.yaml
```

**Configuration Options:**

- `--model_size`: Select `1b` or `7b`.
- `--batch_size`: Adjust based on GPU VRAM.
- `--modalities`: Specify active modalities for the run.

## Monitoring

Training metrics (loss, sensory alignment, token accuracy) are logged to TensorBoard/WandB.

```bash
tensorboard --logdir runs/
```
