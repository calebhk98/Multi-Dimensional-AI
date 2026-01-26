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
