# Testing Guide

Complete guide for testing the Multi-Dimensional AI project using Test-Driven Development (TDD).

---

## Overview

This project follows strict TDD principles:

1. **Write a failing test** for new functionality
2. **Write minimal code** to make the test pass
3. **Refactor** while keeping tests green
4. **Add ~5 edge case tests** per feature
5. **Run tests frequently** during development

All tests must be **meaningful** - no empty tests or tests that just check `True == True`.

---

## Quick Start

### Running All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_encoders.py -v

# Run specific test function
pytest tests/test_encoders.py::test_audio_encoder_forward -v
```

### Running Tests with Filters

```bash
# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run only encoder tests
pytest tests/test_encoders.py

# Run tests matching pattern
pytest tests/ -k "encoder"
```

---

## Test Structure

### Directory Organization

```
tests/
├── test_encoders.py           # Encoder module tests
├── test_decoders.py           # Decoder module tests
├── test_multimodal_model.py   # Main model tests
├── test_trainer.py            # Training logic tests
├── test_data_pipeline.py      # Data loading tests
├── test_vr_integration.py     # VR server/client tests
├── test_evolution.py          # Evolutionary training tests
└── conftest.py                # Shared fixtures
```

### Test File Naming

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

**Example:**

```python
# File: tests/test_encoders.py

class TestAudioEncoder:
	def test_forward_pass(self):
		# Test basic forward pass
		pass

	def test_quantization(self):
		# Test vector quantization
		pass
```

---

## Writing Tests

### Basic Test Structure

```python
def test_feature_name():
	"""
	Test description explaining what is being tested.
	"""
	# Arrange - Set up test data
	input_data = create_test_input()

	# Act - Execute the functionality
	result = function_under_test(input_data)

	# Assert - Verify the result
	assert result.shape == expected_shape
	assert torch.allclose(result, expected_output)
```

### TDD Workflow Example

**Feature:** Add audio encoder quantization

**Step 1: Write failing test**

```python
def test_audio_encoder_quantization():
	"""Test that audio encoder quantizes features correctly."""
	encoder = AudioEncoder(codebook_size=1024)
	waveform = torch.randn(1, 16000)  # 1 second audio

	# Should return quantized output and indices
	quantized, indices = encoder(waveform, return_indices=True)

	assert quantized.shape[1] == 1024  # Should match codebook size
	assert indices.max() < 1024  # Indices within codebook range
	assert indices.min() >= 0
```

**Result:** Test fails (feature doesn't exist yet)

**Step 2: Write minimal code**

```python
class AudioEncoder(nn.Module):
	def forward(self, waveform, return_indices=False):
		features = self.conv_layers(waveform)
		quantized, indices = self.quantize(features)

		if return_indices:
			return quantized, indices
		return quantized
```

**Result:** Test passes

**Step 3: Add edge case tests**

```python
def test_audio_encoder_empty_input():
	"""Test encoder with empty input."""
	encoder = AudioEncoder()
	waveform = torch.zeros(1, 0)  # Empty

	with pytest.raises(ValueError):
		encoder(waveform)

def test_audio_encoder_batch_processing():
	"""Test encoder handles batches correctly."""
	encoder = AudioEncoder()
	waveform = torch.randn(8, 16000)  # Batch of 8

	output = encoder(waveform)
	assert output.shape[0] == 8  # Batch dimension preserved

def test_audio_encoder_different_lengths():
	"""Test encoder with variable-length inputs."""
	encoder = AudioEncoder()
	short = torch.randn(1, 8000)
	long = torch.randn(1, 32000)

	out_short = encoder(short)
	out_long = encoder(long)

	# Output length should scale with input
	assert out_short.shape[1] < out_long.shape[1]

def test_audio_encoder_codebook_utilization():
	"""Test all codebook vectors are used."""
	encoder = AudioEncoder(codebook_size=128)
	waveform = torch.randn(100, 16000)  # Large batch

	_, indices = encoder(waveform, return_indices=True)
	unique_codes = torch.unique(indices)

	# Should use most of the codebook
	assert len(unique_codes) > 100  # At least 100/128 used

def test_audio_encoder_deterministic():
	"""Test encoder is deterministic with same input."""
	encoder = AudioEncoder()
	encoder.eval()
	waveform = torch.randn(1, 16000)

	output1 = encoder(waveform)
	output2 = encoder(waveform)

	assert torch.allclose(output1, output2)
```

---

## Test Categories

### Unit Tests

**Purpose:** Test individual components in isolation

**Characteristics:**

- Fast (< 100ms per test)
- No external dependencies
- Deterministic

**Example:**

```python
def test_touch_encoder_normalization():
	"""Test touch sensor value normalization."""
	encoder = TouchEncoder()

	# Raw sensor values (0-1 range)
	raw_input = torch.tensor([[0.0, 0.5, 1.0, 0.25, 0.75]])

	# Should normalize to appropriate range
	output = encoder(raw_input)

	assert output.shape == (1, encoder.output_dim)
	assert output.min() >= -1.0
	assert output.max() <= 1.0
```

### Integration Tests

**Purpose:** Test multiple components working together

**Characteristics:**

- Slower (100ms - 1s per test)
- Tests component interactions
- May involve data loading

**Example:**

```python
@pytest.mark.slow
def test_multimodal_forward_all_inputs():
	"""Test model with all input modalities active."""
	config = load_test_config()
	model = MultiModalCreature(config)

	# Create full set of inputs
	batch = create_full_multimodal_batch(batch_size=4)

	# Should process all inputs and generate all outputs
	outputs = model(**batch['inputs'])

	assert 'internal_text' in outputs
	assert 'external_text' in outputs
	assert 'audio' in outputs
	assert 'animation' in outputs
```

### End-to-End Tests

**Purpose:** Test complete workflows

**Characteristics:**

- Slowest (1s - 10s per test)
- Tests full pipelines
- Marked as slow

**Example:**

```python
@pytest.mark.slow
def test_training_one_epoch():
	"""Test one full training epoch completes successfully."""
	config = create_minimal_training_config()

	# Create small dataset
	dataset = SyntheticMultiModalDataset(num_samples=10)
	loader = DataLoader(dataset, batch_size=2)

	# Create model and trainer
	model = MultiModalCreature(config)
	trainer = Trainer(model, config, loader)

	# Train for one epoch
	trainer.train()

	# Should complete without errors
	assert trainer.current_step > 0
```

---

## Common Test Patterns

### Testing Shape Transformations

```python
def test_encoder_output_shape():
	"""Test encoder produces correct output shape."""
	encoder = VisualEncoder(output_dim=512)

	# Input: [batch, channels, height, width]
	input_img = torch.randn(4, 3, 224, 224)

	# Output: [batch, output_dim]
	output = encoder(input_img)

	assert output.shape == (4, 512)
```

### Testing Value Ranges

```python
def test_encoder_output_range():
	"""Test encoder outputs are in valid range."""
	encoder = TouchEncoder()

	# Valid input range [0, 1]
	touch_input = torch.rand(8, 5)  # 8 samples, 5 fingers

	output = encoder(touch_input)

	# Check no NaN or Inf
	assert not torch.isnan(output).any()
	assert not torch.isinf(output).any()

	# Check reasonable range (for embeddings)
	assert output.abs().max() < 10.0
```

### Testing Error Handling

```python
def test_encoder_invalid_input_type():
	"""Test encoder rejects invalid input types."""
	encoder = AudioEncoder()

	# String instead of tensor
	invalid_input = "not a tensor"

	with pytest.raises(TypeError):
		encoder(invalid_input)

def test_encoder_invalid_input_shape():
	"""Test encoder rejects invalid shapes."""
	encoder = AudioEncoder()

	# Wrong number of dimensions
	invalid_input = torch.randn(16000)  # Missing batch dim

	with pytest.raises(ValueError):
		encoder(invalid_input)
```

### Testing Device Placement

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda():
	"""Test model works on CUDA."""
	model = MultiModalCreature(config).cuda()

	# Create CUDA inputs
	batch = create_test_batch()
	batch_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v
	              for k, v in batch.items()}

	# Should run on GPU
	output = model(**batch_cuda)

	assert output['internal_text'].is_cuda
```

---

## Using Fixtures

Fixtures provide reusable test components.

### Common Fixtures

**conftest.py:**

```python
import pytest
import torch
from src.models.multimodal_transformer import MultiModalCreature

@pytest.fixture
def test_config():
	"""Minimal config for testing."""
	return {
		'model': {
			'hidden_dim': 128,
			'num_layers': 2,
			'num_heads': 4,
		},
		'training': {
			'batch_size': 4,
			'lr': 1e-3,
		}
	}

@pytest.fixture
def small_model(test_config):
	"""Small model for fast testing."""
	return MultiModalCreature(test_config)

@pytest.fixture
def sample_batch():
	"""Sample input batch."""
	return {
		'inputs': {
			'internal_voice': torch.randint(0, 1000, (4, 32)),
			'external_voice': torch.randint(0, 1000, (4, 32)),
		},
		'targets': {
			'internal_text': torch.randint(0, 1000, (4, 32)),
			'external_text': torch.randint(0, 1000, (4, 32)),
		}
	}
```

### Using Fixtures in Tests

```python
def test_model_forward(small_model, sample_batch):
	"""Test model forward pass."""
	output = small_model(**sample_batch['inputs'])

	assert 'internal_text' in output
	assert output['internal_text'].shape[0] == 4  # Batch size
```

---

## Test Markers

Mark tests for selective execution.

### Built-in Markers

```python
@pytest.mark.slow
def test_long_running():
	"""Test that takes a long time."""
	pass

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA")
def test_gpu_only():
	"""Test that requires GPU."""
	pass

@pytest.mark.skip(reason="Not yet implemented")
def test_future_feature():
	"""Test for planned feature."""
	pass
```

### Custom Markers

**pytest.ini:**

```ini
[pytest]
markers =
    slow: marks tests as slow (> 1 second)
    gpu: marks tests requiring GPU
    integration: marks integration tests
    unit: marks unit tests
```

**Usage:**

```python
@pytest.mark.unit
def test_simple_function():
	pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline():
	pass
```

**Run specific markers:**

```bash
# Run only unit tests
pytest -m unit

# Run everything except slow tests
pytest -m "not slow"

# Run GPU tests only
pytest -m gpu
```

---

## Coverage

### Generating Coverage Reports

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

### Coverage Goals

| Component         | Target Coverage |
| ----------------- | --------------- |
| Core models       | 90%+            |
| Encoders/Decoders | 85%+            |
| Data pipeline     | 80%+            |
| Training logic    | 85%+            |
| Utilities         | 70%+            |

### Coverage Configuration

**.coveragerc:**

```ini
[run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[report]
precision = 2
skip_empty = True
```

---

## Continuous Testing

### Watch Mode

Use `pytest-watch` for automatic test running:

```bash
# Install
pip install pytest-watch

# Run in watch mode
ptw tests/ -v
```

### Pre-commit Testing

Run tests before commits:

**.git/hooks/pre-commit:**

```bash
#!/bin/bash
echo "Running tests before commit..."
pytest tests/ -m "not slow" -q

if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

---

## Debugging Tests

### Running with Debugger

```python
def test_complex_behavior():
	"""Test complex functionality."""
	model = create_model()

	# Set breakpoint
	import pdb; pdb.set_trace()

	result = model.forward(input_data)
	assert result.shape == expected_shape
```

### Verbose Output

```bash
# Show print statements
pytest tests/ -v -s

# Show full tracebacks
pytest tests/ --tb=long

# Show local variables on failure
pytest tests/ -l
```

### Debugging Failed Tests

```bash
# Re-run only failed tests
pytest --lf

# Re-run failed tests first, then others
pytest --ff

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb
```

---

## Testing Best Practices

### 1. Test One Thing Per Test

❌ **Bad:**

```python
def test_everything():
	"""Test encoder, decoder, and model."""
	encoder = AudioEncoder()
	decoder = AudioDecoder()
	model = MultiModalCreature(config)

	# Testing too many things...
```

✅ **Good:**

```python
def test_audio_encoder_forward():
	"""Test audio encoder forward pass."""
	encoder = AudioEncoder()
	result = encoder(waveform)
	assert result.shape == expected_shape

def test_audio_decoder_forward():
	"""Test audio decoder forward pass."""
	decoder = AudioDecoder()
	result = decoder(codes)
	assert result.shape == expected_shape
```

### 2. Use Descriptive Names

❌ **Bad:**

```python
def test_1():
	pass

def test_model():
	pass
```

✅ **Good:**

```python
def test_audio_encoder_handles_variable_length_input():
	pass

def test_model_generates_all_output_modalities():
	pass
```

### 3. Test Edge Cases

```python
def test_encoder_with_normal_input():
	"""Test with typical input."""
	pass

def test_encoder_with_empty_input():
	"""Test with empty tensor."""
	pass

def test_encoder_with_oversized_input():
	"""Test with very large input."""
	pass

def test_encoder_with_batched_input():
	"""Test with batches."""
	pass

def test_encoder_with_single_sample():
	"""Test with batch size 1."""
	pass
```

### 4. Keep Tests Fast

- Use small models for testing (hidden_dim=128, num_layers=2)
- Use small datasets (10-100 samples)
- Mock expensive operations
- Mark slow tests with `@pytest.mark.slow`

### 5. No Empty Catches

❌ **Bad:**

```python
def test_something():
	try:
		result = risky_function()
	except:
		pass  # Silent failure - BAD!
```

✅ **Good:**

```python
def test_something():
	try:
		result = risky_function()
	except ValueError as e:
		logger.error(f"Expected error occurred: {e}")
		raise
```

---

## Testing Checklist

Before committing code:

- [ ] All tests pass
- [ ] New features have tests
- [ ] Edge cases are tested
- [ ] No empty catch blocks
- [ ] Tests are meaningful (not just `assert True`)
- [ ] Coverage hasn't decreased
- [ ] Ran `pytest tests/ -v`
- [ ] Ran `python scripts/audit_docs.py`
- [ ] Ran `python scripts/analyze_indentation.py`

---

## Troubleshooting Tests

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**

```bash
# Run from project root
cd "Multi Dimensional AI"
pytest tests/
```

### CUDA Out of Memory in Tests

**Solution:**

```python
# Use smaller models in tests
@pytest.fixture
def test_config():
	return {
		'model': {
			'hidden_dim': 64,  # Very small for testing
			'num_layers': 1,
		}
	}
```

### Tests Pass Locally But Fail in CI

**Common causes:**

- Different random seeds
- GPU vs CPU differences
- Timing-dependent tests

**Solution:**

```python
# Set deterministic behavior
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
