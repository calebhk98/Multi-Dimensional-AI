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

## Getting Started

### 1. Installation & Setup

```bash
# Clone repository
git clone https://github.com/calebhk98/Multi-Dimensional-AI.git
cd Multi-Dimensional-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure settings
cp .env.example .env
# Edit .env with your settings (optional for basic training)
```

### 2. Prepare Data

You can either generate dummy data for testing or tokenize your own text corpus.

**Option A: Generate Dummy Data (Fast)**
Useful for checking if the training pipeline works.

```bash
python scripts/generate_dummy_data.py --output data/dummy_corpus.bin --tokens 1000000
```

**Option B: Tokenize Custom Text**
Train on your own text (e.g., books, articles).

```bash
# 1. Place your text file in data/ (e.g., data/my_book.txt)
# 2. Run tokenizer
python scripts/tokenize_corpus.py --input data/my_book.txt --output data/corpus.bin
```

**Option C: Use HuggingFace Datasets**
Download and process datasets like Wikipedia directly.

```bash
# Install datasets library first
pip install datasets

# Run preparation script
python scripts/prepare_hf_data.py --dataset wikimedia/wikipedia --config 20231101.en --output data/wiki.bin
```

### 3. Training

#### Text-to-Text Training (LLM Style)

Train the model just on text input/output, similar to GPT.

```bash
# Normal training
python scripts/train_text_only.py --config configs/training_config.yaml --data data/corpus.bin

# Fast Dry Run (Sanity check)
python scripts/train_text_only.py --config configs/training_config.yaml --data data/corpus.bin --dry-run
```

#### Full Multi-Modal Training (Phase 4+)

Train with all senses enabled (Vision, Audio, etc).

```bash
python scripts/train.py --config configs/training_config.yaml
```

### 4. Inference

Interact with the trained model.

```bash
python scripts/inference.py --model checkpoints/model_1b.pt
```

## Project Structure

````
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

## Development Tools

The project includes strict code quality enforcement scripts:

```bash
# Analyze and enforce indentation (tabs only, max 5 levels)
python scripts/analyze_indentation.py

# Audit documentation (strict docstring requirements)
python scripts/audit_docs.py
````

````

## Implementation Phases

1. ✅ **Planning** - Architecture design complete
2. ✅ **Foundation** - Basic structure and encoders
3. ✅ **Single-Modality** - Train individual components
4. ✅ **Pairwise Integration** - Combine modalities
5. ✅ **Backpropagation** - Full model training implementation complete
6. ✅ **Evolutionary** - Fitness-based optimization
7. ✅ **VR Deployment** - Real-time VR integration
8. ⏳ **Scaling** - 7B model and advanced features

## Requirements

- **Hardware:** RTX 4090 or A100 GPU
- **Software:** Python 3.10+, PyTorch 2.0+, Unity 2022+
- **Storage:** 100GB+ for datasets and checkpoints

### Single Modality Training

When we say single modality, we are still running all of these through the same brain
Like, it should be
Input:
Thoughts: Some internal text,
Text: Some written text,
Image: 1 or 2 images,
...

and
Output:
Thoughts: What it is thinking,
Text: What it is writing,
Voice: What it is saying aloud,
...

Like, even when running in single modality, the only difference is the other inputs or outputs are null. It should still generate a token for all of them every pass through.

Example:
Pass through 1:
Text: What is capital of France?
Thought: None
Image: None
etc.

Output:
Text: Paris
Thought: It (later tokens would be "should be paris") or None (depending if we say thought are single modal or not)
Audio: None
etc.

It still generates all tokens, even if it only has to do 1.
IDK if we should train it to generate None as a token, or if we should just ignore the outputs here.

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
  url={https://github.com/calebhk98/Multi-Dimensional-AI}
}
````

## License

MIT License - See LICENSE file for details
