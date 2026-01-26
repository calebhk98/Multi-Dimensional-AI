# Dataset Formats

This document describes the expected data format for `RealMultiModalDataset`.

## Directory Structure

```
data_root/
├── session_001/
│   ├── metadata.jsonl      # Per-frame metadata and sensor readings
│   ├── vision_left.mp4      # Left eye video
│   ├── vision_right.mp4     # Right eye video (optional)
│   └── audio.wav            # Audio recording
├── session_002/
│   └── ...
└── session_N/
```

## metadata.jsonl Schema

Each line is a JSON object with:

| Field       | Type      | Description                                     |
| ----------- | --------- | ----------------------------------------------- |
| `timestamp` | float     | Time in seconds from session start              |
| `touch`     | float[5]  | Touch sensor readings (per-finger pressure)     |
| `proprio`   | float[20] | Proprioceptive state (joint angles, velocities) |

### Example

```json
{"timestamp": 0.0, "touch": [0.0, 0.0, 0.0, 0.0, 0.0], "proprio": [0.1, 0.2, ...]}
{"timestamp": 0.033, "touch": [0.1, 0.0, 0.0, 0.0, 0.0], "proprio": [0.11, 0.21, ...]}
```

## Tensor Shapes

After loading and batching with `real_data_collate_fn`:

| Modality       | Shape          | Notes                            |
| -------------- | -------------- | -------------------------------- |
| `vision_left`  | `[B, C, H, W]` | Normalized with ImageNet stats   |
| `vision_right` | `[B, C, H, W]` | Optional                         |
| `audio`        | `[B, T]`       | Variable length, padded in batch |
| `touch`        | `[B, 5]`       | Fixed size                       |
| `proprio`      | `[B, 20]`      | Fixed size                       |
| `timestamp`    | `[B]`          | Scalar per sample                |

## Normalization

Vision uses ImageNet normalization:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

Audio and touch normalization is configurable via `NormalizationConfig`.
