"""
Tests for data validation utilities.
"""

import pytest
import torch
from src.data.validation import (
    validate_input_shapes,
    validate_value_ranges,
    validate_required_fields,
    validate_dtypes,
    validate_batch,
    _check_shape_constraints,
    EXPECTED_SHAPES,
)


class TestValidateInputShapes:
    """Tests for validate_input_shapes function."""

    def test_valid_batch_single_tensor(self):
        """Test validation passes for a single valid tensor."""
        inputs = {"internal_voice_tokens": torch.randint(0, 100, (4, 32))}
        validate_input_shapes(inputs)  # Should not raise

    def test_valid_batch_multiple_tensors(self):
        """Test validation passes for multiple valid tensors with consistent batch size."""
        inputs = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "external_voice_tokens": torch.randint(0, 100, (4, 32)),
            "audio_waveform": torch.randn(4, 16000),
        }
        validate_input_shapes(inputs)  # Should not raise

    def test_empty_inputs_raises(self):
        """Test that empty inputs dictionary raises ValueError."""
        with pytest.raises(ValueError, match="Empty inputs dictionary"):
            validate_input_shapes({})

    def test_batch_size_mismatch_raises(self):
        """Test that inconsistent batch sizes raise ValueError."""
        inputs = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "external_voice_tokens": torch.randint(0, 100, (2, 32)),  # Different batch size
        }
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_input_shapes(inputs)

    def test_expected_batch_size_enforcement(self):
        """Test that expected_batch_size is enforced."""
        inputs = {"internal_voice_tokens": torch.randint(0, 100, (4, 32))}
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_input_shapes(inputs, expected_batch_size=8)

    def test_non_tensor_value_raises(self):
        """Test that non-tensor values raise ValueError."""
        inputs = {"internal_voice_tokens": [1, 2, 3, 4]}  # List instead of tensor
        with pytest.raises(ValueError, match="Expected torch.Tensor"):
            validate_input_shapes(inputs)

    def test_nested_dict_handling(self):
        """Test that nested dictionaries are validated recursively."""
        inputs = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "touch_data": {
                "positions": torch.randn(4, 10, 3),
                "normals": torch.randn(4, 10, 3),
            },
        }
        validate_input_shapes(inputs)  # Should not raise

    def test_nested_dict_batch_mismatch_raises(self):
        """Test that nested dicts with batch mismatch raise ValueError."""
        inputs = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "touch_data": {
                "positions": torch.randn(2, 10, 3),  # Different batch size
            },
        }
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_input_shapes(inputs)

    def test_image_shape_constraint(self):
        """Test that image tensors must have 4 dimensions with 3 channels."""
        inputs = {"left_eye_image": torch.randn(4, 3, 224, 224)}
        validate_input_shapes(inputs)  # Should not raise

    def test_image_wrong_dimensions_raises(self):
        """Test that images with wrong number of dimensions raise ValueError."""
        inputs = {"left_eye_image": torch.randn(4, 224, 224)}  # Missing channel dim
        with pytest.raises(ValueError, match="Expected 4 dimensions"):
            validate_input_shapes(inputs)

    def test_image_wrong_channels_raises(self):
        """Test that images with wrong number of channels raise ValueError."""
        inputs = {"left_eye_image": torch.randn(4, 1, 224, 224)}  # 1 channel instead of 3
        with pytest.raises(ValueError, match="Dimension 1 mismatch"):
            validate_input_shapes(inputs)

    def test_joint_rotations_quaternion_constraint(self):
        """Test that joint rotations must have 4 values (quaternions)."""
        inputs = {"joint_rotations": torch.randn(4, 10, 24, 4)}
        validate_input_shapes(inputs)  # Should not raise

    def test_joint_rotations_wrong_quaternion_size_raises(self):
        """Test that joint rotations with wrong last dimension raise ValueError."""
        inputs = {"joint_rotations": torch.randn(4, 10, 24, 3)}  # 3 instead of 4
        with pytest.raises(ValueError, match="Dimension 3 mismatch"):
            validate_input_shapes(inputs)


class TestCheckShapeConstraints:
    """Tests for _check_shape_constraints helper function."""

    def test_none_expected_skips_validation(self):
        """Test that None expected shape skips validation."""
        tensor = torch.randn(4, 100)
        _check_shape_constraints("audio_waveform", tensor, None)  # Should not raise

    def test_dimension_count_mismatch_raises(self):
        """Test that wrong number of dimensions raises ValueError."""
        tensor = torch.randn(4, 32)  # 2D
        expected = (None, 3, None, None)  # Expects 4D
        with pytest.raises(ValueError, match="Expected 4 dimensions"):
            _check_shape_constraints("left_eye_image", tensor, expected)

    def test_fixed_dimension_mismatch_raises(self):
        """Test that fixed dimension mismatch raises ValueError."""
        tensor = torch.randn(4, 1, 224, 224)  # 1 channel
        expected = (None, 3, None, None)  # Expects 3 channels
        with pytest.raises(ValueError, match="Dimension 1 mismatch"):
            _check_shape_constraints("left_eye_image", tensor, expected)


class TestValidateValueRanges:
    """Tests for validate_value_ranges function."""

    def test_valid_image_range(self):
        """Test that images in [0, 1] pass validation."""
        data = {"left_eye_image": torch.rand(4, 3, 224, 224)}  # [0, 1)
        validate_value_ranges(data)  # Should not raise

    def test_image_below_range_raises(self):
        """Test that images with values below 0 raise ValueError."""
        data = {"left_eye_image": torch.randn(4, 3, 224, 224)}  # Contains negatives
        with pytest.raises(ValueError, match="Values below 0.0 found"):
            validate_value_ranges(data)

    def test_image_above_range_raises(self):
        """Test that images with values above 1 raise ValueError."""
        data = {"left_eye_image": torch.rand(4, 3, 224, 224) + 0.5}  # Some above 1
        with pytest.raises(ValueError, match="Values above 1.0 found"):
            validate_value_ranges(data)

    def test_valid_audio_range(self):
        """Test that audio in [-1, 1] passes validation."""
        data = {"audio_waveform": torch.rand(4, 16000) * 2 - 1}  # [-1, 1]
        validate_value_ranges(data)  # Should not raise

    def test_audio_below_range_raises(self):
        """Test that audio with values below -1 raise ValueError."""
        data = {"audio_waveform": torch.randn(4, 16000) * 2}  # Some below -1
        with pytest.raises(ValueError, match="Values below -1.0 found"):
            validate_value_ranges(data)

    def test_custom_ranges(self):
        """Test that custom ranges override defaults."""
        data = {"custom_field": torch.rand(4, 10) * 100}  # [0, 100)
        custom_ranges = {"custom_field": (0.0, 100.0)}
        validate_value_ranges(data, ranges=custom_ranges)  # Should not raise

    def test_custom_range_override_default(self):
        """Test that custom ranges override default ranges."""
        data = {"left_eye_image": torch.rand(4, 3, 224, 224) * 255}  # [0, 255)
        custom_ranges = {"left_eye_image": (0.0, 255.0)}
        validate_value_ranges(data, ranges=custom_ranges)  # Should not raise

    def test_nested_dict_handling(self):
        """Test that nested dictionaries are validated recursively."""
        data = {
            "left_eye_image": torch.rand(4, 3, 224, 224),
            "touch_data": {
                "left_eye_image": torch.rand(4, 3, 64, 64),  # Nested with same key
            },
        }
        validate_value_ranges(data)  # Should not raise

    def test_non_tensor_values_skipped(self):
        """Test that non-tensor values are skipped."""
        data = {
            "left_eye_image": torch.rand(4, 3, 224, 224),
            "metadata": "some string",  # Non-tensor
        }
        validate_value_ranges(data)  # Should not raise

    def test_unknown_keys_skipped(self):
        """Test that keys without defined ranges are skipped."""
        data = {"unknown_modality": torch.randn(4, 100) * 1000}  # Large values
        validate_value_ranges(data)  # Should not raise (no range defined)


class TestValidateRequiredFields:
    """Tests for validate_required_fields function."""

    def test_all_fields_present(self):
        """Test that validation passes when all required fields are present."""
        data = {"field1": 1, "field2": 2, "field3": 3}
        required = ["field1", "field2"]
        validate_required_fields(data, required)  # Should not raise

    def test_missing_single_field_raises(self):
        """Test that a single missing field raises ValueError."""
        data = {"field1": 1}
        required = ["field1", "field2"]
        with pytest.raises(ValueError, match="Missing required fields.*field2"):
            validate_required_fields(data, required)

    def test_missing_multiple_fields_raises(self):
        """Test that multiple missing fields are reported."""
        data = {"field1": 1}
        required = ["field1", "field2", "field3"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_required_fields(data, required)

    def test_empty_required_list(self):
        """Test that empty required list passes."""
        data = {"field1": 1}
        validate_required_fields(data, [])  # Should not raise

    def test_empty_data_with_requirements_raises(self):
        """Test that empty data with requirements raises ValueError."""
        data = {}
        required = ["field1"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_required_fields(data, required)


class TestValidateDtypes:
    """Tests for validate_dtypes function."""

    def test_valid_token_dtype(self):
        """Test that long dtype for tokens passes validation."""
        data = {"internal_voice_tokens": torch.randint(0, 100, (4, 32))}  # long by default
        validate_dtypes(data)  # Should not raise

    def test_invalid_token_dtype_raises(self):
        """Test that float dtype for tokens raises ValueError."""
        data = {"internal_voice_tokens": torch.randn(4, 32)}  # float32
        with pytest.raises(ValueError, match="Expected dtype.*long.*got.*float"):
            validate_dtypes(data)

    def test_valid_image_dtype(self):
        """Test that float32 dtype for images passes validation."""
        data = {"left_eye_image": torch.randn(4, 3, 224, 224)}  # float32
        validate_dtypes(data)  # Should not raise

    def test_invalid_image_dtype_raises(self):
        """Test that int dtype for images raises ValueError."""
        data = {"left_eye_image": torch.randint(0, 255, (4, 3, 224, 224))}  # long
        with pytest.raises(ValueError, match="Expected dtype.*float32.*got.*int"):
            validate_dtypes(data)

    def test_custom_dtypes(self):
        """Test that custom dtypes are validated."""
        data = {"custom_field": torch.zeros(4, 10, dtype=torch.float16)}
        custom_dtypes = {"custom_field": torch.float16}
        validate_dtypes(data, expected_dtypes=custom_dtypes)  # Should not raise

    def test_custom_dtype_override(self):
        """Test that custom dtypes override defaults."""
        data = {"internal_voice_tokens": torch.randn(4, 32)}  # float32
        custom_dtypes = {"internal_voice_tokens": torch.float32}  # Override to float
        validate_dtypes(data, expected_dtypes=custom_dtypes)  # Should not raise

    def test_nested_dict_handling(self):
        """Test that nested dictionaries are validated recursively."""
        data = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "nested": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 16)),
            },
        }
        validate_dtypes(data)  # Should not raise

    def test_non_tensor_values_skipped(self):
        """Test that non-tensor values are skipped."""
        data = {
            "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            "metadata": {"key": "value"},  # Non-tensor dict
        }
        validate_dtypes(data)  # Should not raise

    def test_unknown_keys_skipped(self):
        """Test that keys without defined dtypes are skipped."""
        data = {"unknown_modality": torch.randint(0, 100, (4, 10))}
        validate_dtypes(data)  # Should not raise


class TestValidateBatch:
    """Tests for validate_batch function."""

    def test_valid_batch_with_inputs(self):
        """Test that a valid batch with inputs passes."""
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            }
        }
        validate_batch(batch)  # Should not raise

    def test_missing_inputs_raises(self):
        """Test that missing inputs key raises ValueError."""
        batch = {"targets": {"internal_text": torch.randint(0, 100, (4, 32))}}
        with pytest.raises(ValueError, match="Batch missing required 'inputs' key"):
            validate_batch(batch, require_inputs=True)

    def test_missing_targets_raises_when_required(self):
        """Test that missing targets raises when require_targets=True."""
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            }
        }
        with pytest.raises(ValueError, match="Batch missing required 'targets' key"):
            validate_batch(batch, require_targets=True)

    def test_valid_batch_with_inputs_and_targets(self):
        """Test that a valid batch with both inputs and targets passes."""
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            },
            "targets": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
            },
        }
        validate_batch(batch, require_targets=True)  # Should not raise

    def test_inputs_not_required(self):
        """Test that inputs can be optional."""
        batch = {"targets": {"internal_voice_tokens": torch.randint(0, 100, (4, 32))}}
        validate_batch(batch, require_inputs=False)  # Should not raise

    def test_batch_validates_shapes(self):
        """Test that batch validates input shapes."""
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randint(0, 100, (4, 32)),
                "external_voice_tokens": torch.randint(0, 100, (2, 32)),  # Mismatch
            }
        }
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_batch(batch)

    def test_batch_validates_dtypes(self):
        """Test that batch validates input dtypes."""
        batch = {
            "inputs": {
                "internal_voice_tokens": torch.randn(4, 32),  # Wrong dtype
            }
        }
        with pytest.raises(ValueError, match="Expected dtype"):
            validate_batch(batch)

    def test_batch_validates_value_ranges(self):
        """Test that batch validates input value ranges."""
        batch = {
            "inputs": {
                "left_eye_image": torch.randn(4, 3, 224, 224),  # Contains negatives
            }
        }
        with pytest.raises(ValueError, match="Values below"):
            validate_batch(batch)


class TestExpectedShapesConstant:
    """Tests for EXPECTED_SHAPES constant."""

    def test_expected_shapes_defined(self):
        """Test that EXPECTED_SHAPES contains expected keys."""
        assert "internal_voice_tokens" in EXPECTED_SHAPES
        assert "external_voice_tokens" in EXPECTED_SHAPES
        assert "left_eye_image" in EXPECTED_SHAPES
        assert "right_eye_image" in EXPECTED_SHAPES
        assert "joint_positions" in EXPECTED_SHAPES
        assert "joint_rotations" in EXPECTED_SHAPES

    def test_image_shape_has_three_channels(self):
        """Test that image shapes expect 3 channels."""
        left_shape = EXPECTED_SHAPES["left_eye_image"]
        right_shape = EXPECTED_SHAPES["right_eye_image"]
        assert left_shape[1] == 3
        assert right_shape[1] == 3

    def test_quaternion_shape_has_four_components(self):
        """Test that rotation shapes expect 4 components (quaternions)."""
        rotation_shape = EXPECTED_SHAPES["joint_rotations"]
        assert rotation_shape[-1] == 4

    def test_position_shape_has_three_components(self):
        """Test that position shapes expect 3 components (xyz)."""
        position_shape = EXPECTED_SHAPES["joint_positions"]
        assert position_shape[-1] == 3
