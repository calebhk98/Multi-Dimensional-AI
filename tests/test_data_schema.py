"""
Tests for Data Schema definitions.

Purpose:
	Verify that UnifiedSample and other data structures are correctly defined and behave as expected.

History:
	- Created during Real-Data Readiness phase.
"""

import pytest
import torch
from dataclasses import asdict
from src.data.schema import UnifiedSample, ModalityType, NormalizationConfig

def test_unified_sample_initialization():
	"""
	Purpose:
		Test that UnifiedSample can be initialized with valid data.

	Workflow:
		1. Create UnifiedSample with all fields
		2. Verify field values and shapes

	ToDo:
		None
	"""
	sample = UnifiedSample(
		vision_left=torch.randn(3, 224, 224),
		vision_right=torch.randn(3, 224, 224),
		audio=torch.randn(1, 16000),
		touch=torch.randn(10, 5),
		proprio=torch.randn(20),
		timestamp=100.0,
		metadata={"session_id": "test_session"}
	)
	
	assert sample.vision_left.shape == (3, 224, 224)
	assert sample.timestamp == 100.0
	assert sample.metadata["session_id"] == "test_session"

def test_unified_sample_optional_fields():
	"""
	Purpose:
		Test that optional fields can be None.

	Workflow:
		1. Create UnifiedSample with missing optional fields
		2. Verify missing fields are None
		3. Verify present fields are correct

	ToDo:
		None
	"""
	sample = UnifiedSample(
		vision_left=torch.randn(3, 224, 224),
		vision_right=None, # Missing right eye
		audio=torch.randn(1, 16000),
		touch=None, # Missing touch
		proprio=torch.randn(20),
		timestamp=101.0
	)
	
	assert sample.vision_right is None
	assert sample.touch is None
	assert sample.vision_left is not None

def test_modality_type_enum():
	"""
	Purpose:
		Test that ModalityType enum has all required members.

	Workflow:
		1. Assert existence and value of each enum member

	ToDo:
		None
	"""
	assert ModalityType.VISION_LEFT.value == "vision_left"
	assert ModalityType.VISION_RIGHT.value == "vision_right"
	assert ModalityType.AUDIO.value == "audio"
	assert ModalityType.TOUCH.value == "touch"
	assert ModalityType.PROPRIO.value == "proprio"
	assert ModalityType.TEXT.value == "text"

def test_normalization_config():
	"""
	Purpose:
		Test NormalizationConfig default values and overrides.

	Workflow:
		1. Create default config and check defaults
		2. Create config with overrides and check values

	ToDo:
		None
	"""
	config = NormalizationConfig()
	assert config.vision_mean == [0.485, 0.456, 0.406] # ImageNet defaults
	assert config.vision_std == [0.229, 0.224, 0.225]
	
	custom_config = NormalizationConfig(audio_sample_rate=44100)
	assert custom_config.audio_sample_rate == 44100
