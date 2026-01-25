# Deployment Guide

## System Requirements

To run the Multi-Dimensional AI Creature locally or as a server for a VR client:

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) recommended for 1B model. A100 (40GB/80GB) required for training or 7B inference.
- **RAM**: 64GB+ System RAM.
- **Disk**: 100GB+ SSD space for model checkpoints and buffers.
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+).
- **Software**:
    - Python 3.10+
    - PyTorch 2.0+ (with CUDA support)
    - Unity 2022.3+ (for VR Environment)

## Installation

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/calebhk98/Multi-Dimensional-AI.git
    cd Multi-Dimensional-AI
    ```

2.  **Environment Setup**:

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate

    pip install -r requirements.txt
    ```

## Running Inference

To run the creature in inference mode (connecting to VR or mock environment):

```bash
python scripts/inference.py --model checkpoints/model_1b.pt
```

### VR Integration

The Python backend communicates with the Unity VR environment via TCP/WebSockets.

1.  **Start the Python Server**:
    Run `inference.py` as shown above. It will listen on `localhost:5000` (default).

2.  **Start the Unity Client**:
    - Open the `/vr_integration` project in Unity Hub.
    - Open the `MainScene`.
    - Press **Play**.
    - The creature should connect to the Python backend.

## Troubleshooting

- **Connection Refused**: Ensure the Python script is running and the port corresponds to the one set in Unity's `NetworkManager`.
- **OOM Errors**: Reduce batch size or switch to a quantized model (if available) in `configs/inference_config.yaml`.
- **Latency**: Ensure you are running on a local network or wired connection. Creating 6 streams + rendering VR is bandwidth intensive.
