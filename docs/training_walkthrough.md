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
    max_steps: 10000
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

## Troubleshooting

| Issue               | Solution                                       |
| ------------------- | ---------------------------------------------- |
| OOM errors          | Reduce batch size or use gradient accumulation |
| Slow training       | Enable mixed precision with `torch.cuda.amp`   |
| Loss not decreasing | Check learning rate, data quality              |
