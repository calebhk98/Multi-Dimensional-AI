# Model Scaling Estimates

Rough estimates for training Multi-Dimensional AI models at various scales.

## Baseline Configuration

| Parameter  | Value |
| ---------- | ----- |
| Hidden dim | 512   |
| Layers     | 6     |
| Heads      | 8     |
| Context    | 2048  |

## Scaling Table

| Scale      | Params | Training Samples | GPU Hours (A100) | Notes                |
| ---------- | ------ | ---------------- | ---------------- | -------------------- |
| **Tiny**   | 10M    | 1M               | ~10              | Dev/debugging        |
| **Small**  | 100M   | 10M              | ~100             | Proof of concept     |
| **Medium** | 500M   | 50M              | ~500             | Research experiments |
| **Large**  | 1B     | 100M             | ~1000            | Production baseline  |
| **XL**     | 3B     | 300M             | ~3000            | High capability      |

## Compute Assumptions

- Batch size: 32 per GPU
- Sequence length: 2048
- Precision: BF16
- Hardware: NVIDIA A100 80GB

## Memory Requirements

| Scale  | Model Memory | Activation Memory | Total (per GPU) |
| ------ | ------------ | ----------------- | --------------- |
| Tiny   | 40 MB        | 2 GB              | ~3 GB           |
| Small  | 400 MB       | 8 GB              | ~10 GB          |
| Medium | 2 GB         | 20 GB             | ~24 GB          |
| Large  | 4 GB         | 40 GB             | ~48 GB          |
| XL     | 12 GB        | 60 GB             | ~80 GB          |

## Data Requirements

Multi-modal training requires:

- ~1000 hours of VR session data for Medium scale
- ~5000 hours for Large scale
- Data augmentation can reduce requirements by ~2x

## Cost Estimates (Cloud)

| Scale  | Training Cost | Inference Cost/1M tokens |
| ------ | ------------- | ------------------------ |
| Small  | ~$500         | $0.01                    |
| Medium | ~$2500        | $0.05                    |
| Large  | ~$10000       | $0.10                    |
| XL     | ~$30000       | $0.30                    |

_Estimates based on current cloud GPU pricing. Actual costs vary._

## Recommendations

1. **Start small**: Train Tiny model first to validate pipeline
2. **Scale gradually**: Double model size only after validating previous scale
3. **Monitor closely**: Track loss curves, gradient norms, memory usage
4. **Checkpoint often**: Save every 10% of training for recovery
