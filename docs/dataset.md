# Real-World Data Pipeline Documentation

This document describes the format and structure for real-world multi-modal data used by the `MultiModalCreature`.

## 1. Directory Structure

The `RealMultiModalDataset` expects the following directory structure:

```
dataset_root/
├── session_ID_1/
│   ├── metadata.jsonl      # Timeline and sensor data
│   ├── audio.wav           # Synchronized audio input
│   ├── vision_left.mp4     # Left eye video stream
│   └── vision_right.mp4    # Right eye video stream
├── session_ID_2/
│   └── ...
└── ...
```

## 2. File Formats

### `metadata.jsonl`

Each line is a JSON object representing a single frame/timepoint.
Required fields:

- `timestamp`: (float) Seconds since session start.
- `touch`: (List[Dict]) Touch contact data.
- `proprio`: (List[float]) Joint positions/rotations.

Example:

```json
{"timestamp": 0.033, "touch": [{"id": 1, "force": 0.5}], "proprio": [...]}
```

### `audio.wav`

- Format: WAV
- Sample Rate: 16kHz (default) or 48kHz
- Channels: Mono (1)

### `vision_*.mp4`

- Format: MP4 (H.264/AVC)
- Resolution: 224x224 (recommended) or native VR resolution.
- Function: Synchronized video streams.

## 3. Usage

### Recording

Use `VRRecorder` attached to the `VRServer` to capture data automatically.

### Training

Use `scripts/train_real.py` to train on the captured data.
Configuration can be adjusted in `configs/real_training_config.yaml`.
