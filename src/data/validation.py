"""
Data validation utilities for multi-modal inputs.

Purpose:
	Provides validation functions to check shape, dtype, value ranges,
	and required fields for multi-modal data before training.
	
History:
	- Created for defensive input validation before real data integration.
"""

import torch
from typing import Dict, Any, List, Optional, Union


# Shape validation constants
EXPECTED_SHAPES = {
	"internal_voice_tokens": (None, None),  # [B, seq_len]
	"external_voice_tokens": (None, None),  # [B, seq_len]
	"audio_waveform": None,  # [B, samples] or [B, channels, samples] - flexible
	"left_eye_image": (None, 3, None, None),  # [B, 3, H, W]
	"right_eye_image": (None, 3, None, None),  # [B, 3, H, W]
	"joint_positions": (None, None, None, 3),  # [B, temporal_window, num_joints, 3]
	"joint_rotations": (None, None, None, 4),  # [B, temporal_window, num_joints, 4]
}


def validate_input_shapes(
	inputs: Dict[str, Any],
	expected_batch_size: Optional[int] = None
) -> None:
	"""
	Validate input tensor shapes.
	
	Purpose:
		Checks that all input tensors have expected dimensionality
		and consistent batch sizes.
	
	Args:
		inputs: Dictionary of input tensors.
		expected_batch_size: Optional batch size to enforce (if None, inferred from first tensor).
		
	Raises:
		ValueError: If shapes are invalid or inconsistent.
		
	Workflow:
		1. Check each tensor's dimensionality.
		2. Verify consistent batch size across all tensors.
		3. Validate specific dimension constraints (e.g., quaternions are 4D).
		
	ToDo:
		- Add more specific shape constraints as needed.
	"""
	if not inputs:
		raise ValueError("Empty inputs dictionary")
	
	batch_size = expected_batch_size
	
	for key, value in inputs.items():
		if not isinstance(value, torch.Tensor):
			# Handle nested dicts (e.g., touch_data)
			if isinstance(value, dict):
				validate_input_shapes(value, batch_size)
				continue
			else:
				raise ValueError(f"{key}: Expected torch.Tensor but got {type(value)}")
		
		# Infer batch size from first tensor if not specified
		if batch_size is None:
			batch_size = value.shape[0]
		
		# Check batch size consistency
		if value.shape[0] != batch_size:
			raise ValueError(f"{key}: Batch size mismatch. Expected {batch_size}, got {value.shape[0]}")
		
		# Check specific shape constraints
		if key in EXPECTED_SHAPES:
			expected = EXPECTED_SHAPES[key]
			# Skip validation if expected is None (flexible shapes)
			if expected is None:
				continue
			if len(value.shape) != len(expected):
				raise ValueError(
					f"{key}: Expected {len(expected)} dimensions, got {len(value.shape)}"
				)
			
			# Check fixed dimensions (non-None values in expected)
			for i, (exp_dim, actual_dim) in enumerate(zip(expected, value.shape)):
				if exp_dim is not None and exp_dim != actual_dim:
					raise ValueError(
						f"{key}: Dimension {i} mismatch. Expected {exp_dim}, got {actual_dim}"
					)


def validate_value_ranges(
	data: Dict[str, torch.Tensor],
	ranges: Dict[str, tuple] = None
) -> None:
	"""
	Validate that tensor values are within expected ranges.
	
	Purpose:
		Ensures data values are within valid bounds (e.g., images in [0, 1],
		audio in [-1, 1], probabilities sum to 1).
		
	Args:
		data: Dictionary of tensors to validate.
		ranges: Optional dict mapping keys to (min, max) tuples.
			Default ranges applied if not specified.
			
	Raises:
		ValueError: If values are out of range.
		
	Workflow:
		1. Apply default ranges for known modalities.
		2. Override with custom ranges if provided.
		3. Check min/max values for each tensor.
		
	ToDo:
		- Add probability distribution validation.
	"""
	# Default ranges for common modalities
	default_ranges = {
		"left_eye_image": (0.0, 1.0),
		"right_eye_image": (0.0, 1.0),
		"audio_waveform": (-1.0, 1.0),
	}
	
	if ranges is None:
		ranges = {}
	
	# Merge defaults with custom ranges
	all_ranges = {**default_ranges, **ranges}
	
	for key, tensor in data.items():
		if not isinstance(tensor, torch.Tensor):
			if isinstance(tensor, dict):
				validate_value_ranges(tensor, ranges)
				continue
			else:
				continue
		
		if key in all_ranges:
			min_val, max_val = all_ranges[key]
			actual_min = tensor.min().item()
			actual_max = tensor.max().item()
			
			if actual_min < min_val:
				raise ValueError(
					f"{key}: Values below {min_val} found (min={actual_min})"
				)
			if actual_max > max_val:
				raise ValueError(
					f"{key}: Values above {max_val} found (max={actual_max})"
				)


def validate_required_fields(
	data: Dict[str, Any],
	required_keys: List[str]
) -> None:
	"""
	Validate that required fields are present.
	
	Purpose:
		Ensures that all necessary keys exist in the data dictionary.
		
	Args:
		data: Dictionary to validate.
		required_keys: List of required key names.
		
	Raises:
		ValueError: If required fields are missing.
		
	Workflow:
		1. Check each required key exists.
		2. Report all missing keys at once.
		
	ToDo:
		- Add nested field validation (e.g., "inputs.vision_left").
	"""
	missing = [key for key in required_keys if key not in data]
	
	if missing:
		raise ValueError(f"Missing required fields: {missing}")


def validate_dtypes(
	data: Dict[str, torch.Tensor],
	expected_dtypes: Dict[str, torch.dtype] = None
) -> None:
	"""
	Validate tensor data types.
	
	Purpose:
		Ensures tensors have correct dtypes (e.g., long for tokens, float for embeddings).
		
	Args:
		data: Dictionary of tensors to validate.
		expected_dtypes: Optional dict mapping keys to expected dtypes.
			If not provided, applies common defaults.
			
	Raises:
		ValueError: If dtypes don't match expectations.
		
	Workflow:
		1. Define default dtypes for known modalities.
		2. Override with custom expectations if provided.
		3. Check each tensor's dtype.
		
	ToDo:
		- Add automatic dtype conversion option.
	"""
	# Default dtypes for common modalities
	default_dtypes = {
		"internal_voice_tokens": torch.long,
		"external_voice_tokens": torch.long,
		"audio_waveform": torch.float32,
		"left_eye_image": torch.float32,
		"right_eye_image": torch.float32,
		"joint_positions": torch.float32,
		"joint_rotations": torch.float32,
	}
	
	if expected_dtypes is None:
		expected_dtypes = {}
	
	# Merge defaults with custom dtypes
	all_dtypes = {**default_dtypes, **expected_dtypes}
	
	for key, tensor in data.items():
		if not isinstance(tensor, torch.Tensor):
			if isinstance(tensor, dict):
				validate_dtypes(tensor, expected_dtypes)
				continue
			else:
				continue
		
		if key in all_dtypes:
			expected_dtype = all_dtypes[key]
			if tensor.dtype != expected_dtype:
				raise ValueError(
					f"{key}: Expected dtype {expected_dtype}, got {tensor.dtype}"
				)


def validate_batch(
	batch: Dict[str, Any],
	require_inputs: bool = True,
	require_targets: bool = False
) -> None:
	"""
	Validate a complete batch dictionary.
	
	Purpose:
		Comprehensive validation of a batch including structure, shapes,
		dtypes, and value ranges.
		
	Args:
		batch: Batch dictionary with 'inputs' and optionally 'targets'.
		require_inputs: Whether 'inputs' key is required.
		require_targets: Whether 'targets' key is required.
		
	Raises:
		ValueError: If batch structure or content is invalid.
		
	Workflow:
		1. Validate top-level structure.
		2. Validate inputs shapes and dtypes.
		3. Validate targets if present.
		4. Validate value ranges.
		
	ToDo:
		- Add cross-validation between inputs and targets.
	"""
	# Check top-level structure
	if require_inputs and "inputs" not in batch:
		raise ValueError("Batch missing required 'inputs' key")
	if require_targets and "targets" not in batch:
		raise ValueError("Batch missing required 'targets' key")
	
	# Validate inputs
	if "inputs" in batch:
		validate_input_shapes(batch["inputs"])
		validate_dtypes(batch["inputs"])
		validate_value_ranges(batch["inputs"])
	
	# Validate targets
	if "targets" in batch:
		validate_dtypes(batch["targets"])
