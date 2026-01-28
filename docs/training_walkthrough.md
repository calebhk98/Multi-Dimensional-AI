# Training Walkthrough

Step-by-step guide to training with the Multi-Dimensional AI system.

## Prerequisites

1. Python 3.10+ environment
2. Dependencies installed: `pip install -r requirements.txt`
3. Data prepared per `dataset_formats.md`

## 1. Prepare Configuration

Create or modify `configs/training_config.yaml`:

```yaml
training:
    max_steps: 1000000
    log_interval: 100
    save_interval: 1000
    optimizer:
        lr: 3e-4
        betas: [0.9, 0.95]
        weight_decay: 0.01
    checkpointing:
        save_dir: "checkpoints/"

model:
    hidden_dim: 512
    num_layers: 6
```

## 2. Initialize Dataset

```python
from src.data.real_dataset import RealMultiModalDataset, real_data_collate_fn
from torch.utils.data import DataLoader

dataset = RealMultiModalDataset(root_dir="data/sessions")
loader = DataLoader(dataset, batch_size=4, collate_fn=real_data_collate_fn)
```

## 3. Create Model and Trainer

```python
from src.models.multimodal_creature import MultiModalCreature
from src.training.trainer import Trainer
import yaml

with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

model = MultiModalCreature(config)
trainer = Trainer(model, config, loader, device="cuda")
```

## 4. Train

```python
trainer.train()
```

Training will:

- Log losses every `log_interval` steps
- Save checkpoints every `save_interval` steps
- Save final model at completion

## 5. Monitor Progress

View logs in terminal or use TensorBoard:

```bash
tensorboard --logdir=runs/
```

## 6. Resume from Checkpoint

```python
trainer.load_checkpoint("checkpoints/model_step_5000.pt")
trainer.train()  # Continues from step 5000
```

## 7. Inference

See `scripts/inference.py` for running inference with trained models.

## End-to-End Real Data Example

Complete walkthrough from VR data to trained model:

### Step 1: Prepare VR Data

```bash
# Validate all sessions
for session in data/sessions/session_*; do
	python scripts/validate_session.py "$session" --fix
done

# Check validation summary
python scripts/validate_session.py data/sessions --summary
```

### Step 2: Configure Training

Edit `configs/real_training_config.yaml`:

```yaml
data:
	root_dir: "data/sessions"
	batch_size: 4
	modalities: ["vision_left", "audio", "touch", "proprio"]

training:
	max_steps: 100000
	log_interval: 100
	save_interval: 5000

model:
	hidden_dim: 512
	num_layers: 6
```

### Step 3: Run Training

```bash
python scripts/train_real.py --config configs/real_training_config.yaml
```

### Step 4: Monitor Training

Watch live metrics:

```bash
tensorboard --logdir=runs/
```

or check latest checkpoint:

```python
import torch
ckpt = torch.load("checkpoints/model_step_5000.pt")
print(f"Step: {ckpt['step']}")
```

### Step 5: Evaluate

```bash
python scripts/inference.py \
	--checkpoint checkpoints/model_final.pt \
	--input data/test_sessions/session_001
```

## Monitoring and Debugging

### Key Metrics to Watch

| Metric                  | Good Range | Warning        | Action                            |
| ----------------------- | ---------- | -------------- | --------------------------------- |
| **Total Loss**          | Decreasing | Plateaus early | Check learning rate, data quality |
| **loss_internal_text**  | < 2.0      | > 3.0          | Check text tokenization           |
| **loss_audio**          | < 1.5      | > 2.5          | Check audio normalization         |
| **samples_per_sec**     | > 10       | < 5            | Profile bottlenecks (see below)   |
| **memory_allocated_mb** | < 20000    | > 22000        | Reduce batch size                 |

### Interpreting Per-Modality Losses

The trainer logs modality-specific losses:

- **loss_internal_text**: Loss for internal thought generation
- **loss_external_text**: Loss for external speech generation
- ** loss_audio**: Loss for audio output stream
- **loss_animation**: Loss for body movement prediction

**Normal behavior**:

- All losses decrease together
- Text losses typically lower than audio/animation
- Losses may plateau at different steps

**Problematic behavior**:

- One modality loss increases while others decrease → Check that modality's data
- All losses NaN → Learning rate too high or data corruption
- No decrease after 1000 steps → Model/data mismatch, check shapes

### Profiling for Bottlenecks

If training is slow (< 5 samples/sec):

```bash
python scripts/profile_multimodal.py
```

This generates:

- Memory/time breakdown per operation
- Profiling trace for TensorBoard
- Optimization recommendations in `profiler_output/profile_report.json`

Common bottlenecks:

- **Data loading**: Use `num_workers=4` in DataLoader
- **CPU→GPU transfer**: Enable `pin_memory=True`
- **Model size**: Enable gradient checkpointing for Large+ models

## Troubleshooting

| Issue                  | Solution                                       |
| ---------------------- | ---------------------------------------------- |
| OOM errors             | Reduce batch size or use gradient accumulation |
| Slow training          | Enable mixed precision with `torch.cuda.amp`   |
| Loss not decreasing    | Check learning rate, data quality              |
| NaN loss               | Reduce LR, check for corrupted data            |
| Uneven modality losses | Adjust loss weights in config                  |

## Text-Only Training Mode

For initial validation, train only on text modalities:

```yaml
training:
	loss_weights:
		internal_text: 1.0
		external_text: 1.0
		audio: 0.0  # Disable audio loss
		animation: 0.0  # Disable animation loss
```

This allows testing the pipeline while tracking only text in/out like a normal LLM.
