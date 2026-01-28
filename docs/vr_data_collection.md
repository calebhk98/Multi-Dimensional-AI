# VR Data Collection Process

This document describes how to collect multi-modal data from VR sessions.

## Overview

Data collection captures synchronized streams from:

- **Vision**: Stereo cameras (left/right eye views)
- **Audio**: Environment microphone
- **Touch**: Finger pressure sensors
- **Proprioception**: Joint angles and velocities

## Hardware Setup

1. VR headset with eye tracking
2. Haptic gloves with pressure sensors
3. External microphone
4. Recording PC with sufficient storage

## Software Requirements

- Unity/Unreal VR application with data export
- Recording daemon for sensor aggregation
- Time synchronization (NTP or local master clock)

## Recording Procedure

1. **Calibrate sensors** - Run calibration sequence before each session
2. **Start recording** - Launch recording daemon, then VR app
3. **Perform task** - User performs target activities
4. **Stop recording** - Daemon aggregates and writes files
5. **Validate** - Run `validate_session.py` to check data integrity

## File Outputs

Each session creates:

```
session_XXX/
├── metadata.jsonl    # Frame timestamps + sensor readings
├── vision_left.mp4   # 30fps, 1080p
├── vision_right.mp4  # Optional
└── audio.wav         # 48kHz mono
```

## Packet Format

Sensor data streamed via UDP:

| Offset | Size | Field                 |
| ------ | ---- | --------------------- |
| 0      | 8    | Timestamp (float64)   |
| 8      | 20   | Touch (5x float32)    |
| 28     | 80   | Proprio (20x float32) |

## Synchronization

All streams are aligned to a common timestamp using:

1. Hardware trigger at session start
2. Software interpolation for slight clock drift
3. Post-processing alignment in `metadata.jsonl`

## Validation

Run after each session:

```bash
python scripts/validate_session.py data_root/session_XXX
```

Checks:

- File existence
- Timestamp continuity
- Reasonable sensor ranges

### Validation Script Parameters

```bash
python scripts/validate_session.py <session_dir> [--strict] [--fix]
```

**Options**:

- `--strict`: Enable strict validation (fail on warnings)
- `--fix`: Attempt to auto-fix minor issues (timestamp gaps, etc.)

### Validation Output

The script outputs a JSON report with:

```json
{
	"session": "session_XXX",
	"status": "passed|failed|warnings",
	"checks": {
		"files_exist": true,
		"timestamp_continuity": true,
		"sensor_ranges": true,
		"video_audio_sync": true
	},
	"warnings": [],
	"errors": []
}
```

## Troubleshooting

### Common Validation Failures

| Issue                       | Cause                          | Solution                                  |
| --------------------------- | ------------------------------ | ----------------------------------------- |
| **Missing vision_left.mp4** | Recording daemon crashed       | Re-record session                         |
| **Timestamp gaps > 100ms**  | Network lag or dropped packets | Check --fix option or mark as low-quality |
| **Touch values > 1.0**      | Sensor calibration drift       | Recalibrate sensors, re-record            |
| **Audio/video desync**      | Clock drift                    | Validate NTP sync before recording        |

### Quality Checks

**Good Session**:

- Timestamp gaps < 35ms (30 FPS ± tolerance)
- All files present
- Sensor values in valid ranges
- Audio RMS > 0.001 (not silent)

**Low Quality Session**:

- Timestamp gaps 35-100ms
- Optional modalities missing (vision_right)
- Sensor saturation < 5%

**Failed Session**:

- Timestamp gaps > 100ms (should discard or fix)
- Required files missing
- Sensor values out of range

## Post-Processing Pipeline

1. **Validate**: Run `validate_session.py` on all sessions
2. **Filter**: Remove failed sessions
3. **Augment**: Generate augmented versions (optional)
4. **Index**: Create dataset index file
5. **Train**: Use with `RealMultiModalDataset`

See [dataset_formats.md](dataset_formats.md) for data loading details.
