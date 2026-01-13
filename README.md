# Multi-Dimensional AI Creature

An autonomous AI creature that lives inside VR worlds, using a novel multi-modal transformer architecture to process 6 sensory streams and generate 4 parallel output streams.

## Overview

This project implements an AI creature with:
- **Vision** through stereo eyes (left/right)
- **Hearing** environmental sounds and speech
- **Touch** sensations from VR world
- **Proprioception** of its own body
- **Internal thoughts** and **external speech**
- **Vocalizations** (gasps, laughter, speech)
- **Body control** for movement and expressions

**Think:** Creatures game meets modern transformer AI

## Architecture

- **Unified transformer** (1B params initially, 7B later)
- **Parallel token generation** across all output modalities
- **Hybrid training:** Backpropagation → Evolutionary optimization

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure settings
cp .env.example .env
# Edit .env with your settings

# Run tests
pytest tests/ -v

# Train model (Phase 4+)
python scripts/train.py --config configs/training_config.yaml

# Run inference
python scripts/inference.py --model checkpoints/model_1b.pt
```

## Project Structure

```
multi-dimensional-ai/
├── configs/          # Configuration files
├── src/              # Source code
│   ├── encoders/     # Input encoders (6 modalities)
│   ├── decoders/     # Output decoders (4 modalities)
│   ├── models/       # Core transformer
│   ├── training/     # Training logic
│   ├── evolution/    # Evolutionary training
│   └── vr_integration/  # Unity/VR connection
├── tests/            # Unit and integration tests
├── scripts/          # Training/inference scripts
└── notebooks/        # Analysis notebooks
```

## Implementation Phases

1. ✅ **Planning** - Architecture design complete
2. 🔄 **Foundation** (Weeks 1-4) - Basic structure and encoders
3. ⏳ **Single-Modality** (Weeks 5-8) - Train individual components
4. ⏳ **Pairwise Integration** (Weeks 9-14) - Combine modalities
5. ⏳ **Backpropagation** (Weeks 15-24) - Full model training
6. ⏳ **Evolutionary** (Weeks 25-32) - Fitness-based optimization
7. ⏳ **VR Deployment** (Weeks 33-38) - Real-time VR integration
8. ⏳ **Scaling** (Weeks 39+) - 7B model and advanced features

## Requirements

- **Hardware:** RTX 4090 or A100 GPU
- **Software:** Python 3.10+, PyTorch 2.0+, Unity 2022+
- **Storage:** 100GB+ for datasets and checkpoints

## Documentation

See `docs/` for detailed documentation:
- [Architecture Guide](docs/architecture.md)
- [Training Guide](docs/training_guide.md)
- [Deployment Guide](docs/deployment_guide.md)

## Citation

If you use this work in your research, please cite:
```bibtex
@software{multidimensional_ai_2026,
  title={Multi-Dimensional AI Creature: Autonomous Embodied Intelligence in VR},
  author={Caleb Kirschbaum},
  year={2026},
  url={https://github.com/yourusername/multi-dimensional-ai}
}
```

## License

MIT License - See LICENSE file for details
