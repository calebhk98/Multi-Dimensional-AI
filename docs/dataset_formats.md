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

## Preprocessing Steps per Modality

### Vision (Left/Right Eye)

1. **Load Frame**: Extract frame at timestamp from video
2. **Resize**: Resize to `[224, 224]` (configurable)
3. **To Tensor**: Convert HWC → CHW, [0, 255] → [0.0, 1.0]
4. **Normalize**: Apply ImageNet stats (mean, std)
5. **Output Shape**: `[3, 224, 224]`

### Audio

1. **Load Segment**: Extract audio chunk from `.wav`
2. **Resample**: Ensure 48kHz sample rate
3. **Mono Conversion**: If stereo, average channels
4. **Normalization**: RMS normalize to [-1, 1] range
5. **Padding**: Pad to max_length if needed
6. **Output Shape**: `[T]` where T varies by segment

### Touch

1. **Extract**: Get touch values from metadata.jsonl
2. **Range Check**: Validate values in [0.0, 1.0]
3. **Normalization**: Optional scaling (configurable)
4. **Output Shape**: `[5]` (5 fingers)

### Proprioception

1. **Extract**: Get proprio values from metadata.jsonl
2. **Joint Angle Normalization**: Scale to [-π, π]
3. **Velocity Normalization**: Clip and scale velocities
4. **Output Shape**: `[20]` (joints + velocities)

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

## Collation Behavior

When batching samples with `real_data_collate_fn`:

1. **Fixed-Size Modalities**: Directly stacked (vision, touch, proprio, timestamp)
2. **Variable-Length Audio**: Padded to max length in batch, creates attention mask
3. **Nested Structures**: Recursively handled (touch_data, animation targets)
4. **Missing Modalities**: Optional modalities set to None if not present

### Example Batch Structure

```python
{
	"inputs": {
		"vision_left": torch.Tensor([B, 3, 224, 224]),
		"audio": torch.Tensor([B, max_audio_len]),
		"audio_mask": torch.Tensor([B, max_audio_len]),  # 1=real, 0=padding
		"touch": torch.Tensor([B, 5]),
		"proprio": torch.Tensor([B, 20]),
		"timestamp": torch.Tensor([B])
	},
	"targets": {...}
}
```

## Data Validation

Use `src.data.validation` module to validate loaded data:

```python
from src.data.validation import validate_input_shapes, validate_value_ranges

# Check batch shapes are consistent
validate_input_shapes(batch["inputs"])

# Check value ranges are valid
validate_value_ranges(batch["inputs"])
```

### Validation Functions

#### `validate_input_shapes(inputs: Dict)`

Ensures all tensors in a batch have matching batch dimensions.

**Raises**: `ValueError` if mismatch detected

#### `validate_value_ranges(data: Dict)`

Checks that values are within expected ranges:

- Images: [0, 1]
- Audio: [-1, 1]
- Touch: [0, 1]
- Proprioception: [-π, π] for angles

**Raises**: `ValueError` if out of range
