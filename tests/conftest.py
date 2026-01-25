"""
Shared pytest fixtures for test suite.
Provides common test data, configurations, and utilities.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any


# ===== Device Fixtures =====

@pytest.fixture
def device():
    """
    Get available device (CUDA if available, else CPU).
    
    Returns:
        str: Device string.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cpu_device():
    """
    Force CPU device for tests that need to run on CPU.
    
    Returns:
        str: 'cpu'
    """
    return "cpu"


# ===== Dimension Fixtures =====

@pytest.fixture
def batch_size():
    """
    Standard batch size for tests.
    
    Returns:
        int: Batch size.
    """
    return 4


@pytest.fixture
def small_batch_size():
    """
    Small batch size for quick tests.
    
    Returns:
        int: Small batch size.
    """
    return 2


@pytest.fixture
def seq_len():
    """
    Standard sequence length for tests.
    
    Returns:
        int: Sequence length.
    """
    return 32


@pytest.fixture
def short_seq_len():
    """
    Short sequence length for quick tests.
    
    Returns:
        int: Short sequence length.
    """
    return 10


@pytest.fixture
def embedding_dim():
    """
    Standard embedding dimension for tests.
    
    Returns:
        int: Embedding dim.
    """
    return 1536


@pytest.fixture
def small_embedding_dim():
    """
    Small embedding dimension for quick tests.
    
    Returns:
        int: Small embedding dim.
    """
    return 512


# ===== Configuration Fixtures =====

@pytest.fixture
def minimal_model_config():
    """
    Minimal model configuration for testing.
    
    Returns:
        dict: Config dictionary.
    """
    return {
        "model": {
            "transformer": {
                "hidden_dim": 512,
                "num_layers": 2,
                "num_attention_heads": 8,
                "ffn_dim": 2048,
                "dropout": 0.1,
            },
            "encoders": {
                "internal_voice": {
                    "vocab_size": 1000,
                    "embedding_dim": 512,
                },
                "external_voice": {
                    "vocab_size": 1000,
                    "embedding_dim": 512,
                },
                "audio": {
                    "sample_rate": 16000,
                    "hop_length": 320,
                    "embedding_dim": 512,
                },
                "vision": {
                    "image_size": 224,
                    "patch_size": 16,
                    "embedding_dim": 512,
                    "use_stereo": True,
                },
                "proprioception": {
                    "num_joints": 24,
                    "embedding_dim": 512,
                    "temporal_window": 10,
                },
                "touch": {
                    "max_contacts": 10,
                    "embedding_dim": 512,
                },
            },
            "decoders": {
                "internal_text": {
                    "vocab_size": 1000,
                    "embedding_dim": 512,
                },
                "external_text": {
                    "vocab_size": 1000,
                    "embedding_dim": 512,
                },
                "audio": {
                    "codebook_size": 1024,
                    "embedding_dim": 512,
                },
                "animation": {
                    "num_joints": 24,
                    "num_blend_shapes": 51,
                    "embedding_dim": 512,
                },
            },
            "fusion": {
                "strategy": "concatenate"
            }
        }
    }


@pytest.fixture
def training_config():
    """
    Standard training configuration.
    
    Returns:
        dict: Training config.
    """
    return {
        "training": {
            "optimizer": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "weight_decay": 0.01,
            },
            "max_steps": 100,
            "log_interval": 10,
            "save_interval": 50,
            "gradient_clip": 1.0,
            "checkpointing": {
                "save_dir": "checkpoints",
                "keep_n_checkpoints": 3,
            }
        }
    }


@pytest.fixture
def full_config(minimal_model_config, training_config):
    """
    Full configuration combining model and training configs.
    
    Args:
        minimal_model_config: Model config.
        training_config: Training config.
        
    Returns:
        dict: Full config.
    """
    return {**minimal_model_config, **training_config}


# ===== Input Data Fixtures =====

@pytest.fixture
def dummy_voice_tokens(batch_size, seq_len):
    """
    Generate dummy voice token inputs.
    
    Args:
        batch_size: Batch size fixture.
        seq_len: Sequence length fixture.
        
    Returns:
        torch.Tensor: Random tokens.
    """
    vocab_size = 1000
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def dummy_audio_waveform(batch_size):
    """
    Generate dummy audio waveform (1 second at 16kHz).
    
    Args:
        batch_size: Batch size fixture.
        
    Returns:
        torch.Tensor: Random waveform.
    """
    return torch.randn(batch_size, 16000)


@pytest.fixture
def dummy_images(batch_size):
    """
    Generate dummy stereo images.
    
    Args:
        batch_size: Batch size fixture.
        
    Returns:
        dict: Dict with 'left' and 'right' image tensors.
    """
    return {
        "left": torch.randn(batch_size, 3, 224, 224),
        "right": torch.randn(batch_size, 3, 224, 224),
    }


@pytest.fixture
def dummy_proprioception(batch_size):
    """
    Generate dummy proprioception data.
    
    Args:
        batch_size: Batch size fixture.
        
    Returns:
        dict: Dict with positions and rotations.
    """
    temporal_window = 10
    num_joints = 24
    return {
        "positions": torch.randn(batch_size, temporal_window, num_joints, 3),
        "rotations": torch.randn(batch_size, temporal_window, num_joints, 4),
    }


@pytest.fixture
def dummy_touch_data(batch_size):
    """
    Generate dummy touch contact data.
    
    Args:
        batch_size: Batch size fixture.
        
    Returns:
        dict: Dict with touch data.
    """
    max_contacts = 10
    return {
        "positions": torch.randn(batch_size, max_contacts, 3),
        "normals": torch.randn(batch_size, max_contacts, 3),
        "forces": torch.randn(batch_size, max_contacts, 3),
        "contact_active": torch.ones(batch_size, max_contacts, dtype=torch.bool),
    }


@pytest.fixture
def dummy_multi_modal_inputs(
    dummy_voice_tokens,
    dummy_audio_waveform,
    dummy_images,
    dummy_proprioception,
    dummy_touch_data
):
    """
    Generate complete multi-modal input batch.
    
    Args:
        dummy_voice_tokens: Fixture.
        dummy_audio_waveform: Fixture.
        dummy_images: Fixture.
        dummy_proprioception: Fixture.
        dummy_touch_data: Fixture.
        
    Returns:
        dict: Multi-modal input batch.
    """
    return {
        "internal_voice_tokens": dummy_voice_tokens,
        "external_voice_tokens": dummy_voice_tokens,  # Reuse for simplicity
        "audio_waveform": dummy_audio_waveform,
        "left_eye_image": dummy_images["left"],
        "right_eye_image": dummy_images["right"],
        "joint_positions": dummy_proprioception["positions"],
        "joint_rotations": dummy_proprioception["rotations"],
        "touch_data": dummy_touch_data,
    }


# ===== Target Data Fixtures =====

@pytest.fixture
def dummy_text_targets(batch_size, seq_len):
    """
    Generate dummy text targets.
    
    Args:
        batch_size: Batch size fixture.
        seq_len: Sequence length fixture.
        
    Returns:
        torch.Tensor: Random targets.
    """
    vocab_size = 1000
    return torch.randint(0, vocab_size, (batch_size, seq_len))


@pytest.fixture
def dummy_audio_targets(batch_size, seq_len):
    """
    Generate dummy audio token targets.
    
    Args:
        batch_size: Batch size fixture.
        seq_len: Sequence length fixture.
        
    Returns:
        torch.Tensor: Random targets.
    """
    codebook_size = 1024
    return torch.randint(0, codebook_size, (batch_size, seq_len))


@pytest.fixture
def dummy_animation_targets(batch_size, seq_len):
    """
    Generate dummy animation targets.
    
    Args:
        batch_size: Batch size fixture.
        seq_len: Sequence length fixture.
        
    Returns:
        dict: Animation targets.
    """
    num_joints = 24
    num_blend_shapes = 51
    return {
        "joint_rotations": torch.randn(batch_size, seq_len, num_joints, 4),
        "blend_shapes": torch.rand(batch_size, seq_len, num_blend_shapes),
        "eye_params": torch.rand(batch_size, seq_len, 8),
    }


@pytest.fixture
def dummy_multi_modal_targets(
    dummy_text_targets,
    dummy_audio_targets,
    dummy_animation_targets
):
    """
    Generate complete multi-modal target batch.
    
    Args:
        dummy_text_targets: Fixture.
        dummy_audio_targets: Fixture.
        dummy_animation_targets: Fixture.
        
    Returns:
        dict: Multi-modal targets.
    """
    return {
        "internal_text": dummy_text_targets,
        "external_text": dummy_text_targets,  # Reuse for simplicity
        "audio": dummy_audio_targets,
        "animation": dummy_animation_targets,
    }


# ===== File System Fixtures =====

@pytest.fixture
def temp_dir():
    """
    Create temporary directory for tests.
    
    Returns:
        Path: Path to temp dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """
    Create temporary checkpoint directory.
    
    Args:
        temp_dir: Parent temp dir fixture.
        
    Returns:
        Path: Checkpoint dir path.
    """
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_config_dir(temp_dir):
    """
    Create temporary config directory.
    
    Args:
        temp_dir: Parent temp dir fixture.
        
    Returns:
        Path: Config dir path.
    """
    config_dir = temp_dir / "configs"
    config_dir.mkdir()
    return config_dir


# ===== Encoder Fixtures =====

@pytest.fixture
def internal_voice_encoder(small_embedding_dim):
    """
    Create InternalVoiceEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        InternalVoiceEncoder: Instance.
    """
    from src.encoders.internal_voice_encoder import InternalVoiceEncoder
    return InternalVoiceEncoder(
        vocab_size=1000,
        embedding_dim=small_embedding_dim
    )


@pytest.fixture
def external_voice_encoder(small_embedding_dim):
    """
    Create ExternalVoiceEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        ExternalVoiceEncoder: Instance.
    """
    from src.encoders.external_voice_encoder import ExternalVoiceEncoder
    return ExternalVoiceEncoder(
        vocab_size=1000,
        embedding_dim=small_embedding_dim
    )


@pytest.fixture
def audio_encoder(small_embedding_dim):
    """
    Create AudioEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        AudioEncoder: Instance.
    """
    from src.encoders.audio_encoder import AudioEncoder
    return AudioEncoder(
        sample_rate=16000,
        hop_length=320,
        embedding_dim=small_embedding_dim
    )


@pytest.fixture
def visual_encoder(small_embedding_dim):
    """
    Create VisualEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        VisualEncoder: Instance.
    """
    from src.encoders.visual_encoder import VisualEncoder
    return VisualEncoder(
        image_size=224,
        patch_size=16,
        embedding_dim=small_embedding_dim,
        num_heads=8,
        use_stereo=True
    )


@pytest.fixture
def proprioception_encoder(small_embedding_dim):
    """
    Create ProprioceptionEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        ProprioceptionEncoder: Instance.
    """
    from src.encoders.proprioception_encoder import ProprioceptionEncoder
    return ProprioceptionEncoder(
        num_joints=24,
        embedding_dim=small_embedding_dim,
        temporal_window=10
    )


@pytest.fixture
def touch_encoder(small_embedding_dim):
    """
    Create TouchEncoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        TouchEncoder: Instance.
    """
    from src.encoders.touch_encoder import TouchEncoder
    return TouchEncoder(
        max_contacts=10,
        embedding_dim=small_embedding_dim
    )


# ===== Decoder Fixtures =====

@pytest.fixture
def text_decoder(small_embedding_dim):
    """
    Create InternalTextDecoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        InternalTextDecoder: Instance.
    """
    from src.decoders.text_decoder import InternalTextDecoder
    return InternalTextDecoder(
        vocab_size=1000,
        embedding_dim=small_embedding_dim
    )


@pytest.fixture
def audio_decoder(small_embedding_dim):
    """
    Create AudioDecoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        AudioDecoder: Instance.
    """
    from src.decoders.audio_decoder import AudioDecoder
    return AudioDecoder(
        codebook_size=1024,
        embedding_dim=small_embedding_dim
    )


@pytest.fixture
def animation_decoder(small_embedding_dim):
    """
    Create AnimationDecoder instance.
    
    Args:
        small_embedding_dim: Embedding dim fixture.
        
    Returns:
        AnimationDecoder: Instance.
    """
    from src.decoders.animation_decoder import AnimationDecoder
    return AnimationDecoder(
        embedding_dim=small_embedding_dim,
        num_joints=24,
        num_blend_shapes=51
    )


# ===== Utility Fixtures =====

@pytest.fixture
def set_seed():
    """
    Set random seed for reproducibility.
    
    Returns:
        callable: Function to set seed.
    """
    def _set_seed(seed: int = 42):
        """
        Purpose:
            Set random seed for reproducibility.
            
        Args:
            seed: Seed value.
            
        Returns:
            None
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _set_seed


@pytest.fixture
def assert_shape():
    """
    Utility to assert tensor shapes.
    
    Returns:
        callable: Assertion function.
    """
    def _assert_shape(tensor: torch.Tensor, expected_shape: tuple):
        """
        Purpose:
            Assert tensor has expected shape.
            
        Args:
            tensor: Input tensor.
            expected_shape: Expected shape tuple.
            
        Returns:
            None
        """
        assert tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )
    return _assert_shape


@pytest.fixture
def assert_no_nan():
    """
    Utility to assert no NaN values in tensor.
    
    Returns:
        callable: Assertion function.
    """
    def _assert_no_nan(tensor: torch.Tensor):
        """
        Purpose:
            Assert tensor has no NaN values.
            
        Args:
            tensor: Input tensor.
            
        Returns:
            None
        """
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    return _assert_no_nan


@pytest.fixture
def assert_no_inf():
    """
    Utility to assert no infinite values in tensor.
    
    Returns:
        callable: Assertion function.
    """
    def _assert_no_inf(tensor: torch.Tensor):
        """
        Purpose:
            Assert tensor has no infinite values.
            
        Args:
            tensor: Input tensor.
            
        Returns:
            None
        """
        assert not torch.isinf(tensor).any(), "Tensor contains infinite values"
    return _assert_no_inf


@pytest.fixture
def assert_in_range():
    """
    Utility to assert tensor values are in range.
    
    Returns:
        callable: Assertion function.
    """
    def _assert_in_range(tensor: torch.Tensor, min_val: float, max_val: float):
        """
        Purpose:
            Assert tensor values are within range.
            
        Args:
            tensor: Input tensor.
            min_val: Minimum value.
            max_val: Maximum value.
            
        Returns:
            None
        """
        assert (tensor >= min_val).all(), f"Values below {min_val} found"
        assert (tensor <= max_val).all(), f"Values above {max_val} found"
    return _assert_in_range


# ===== Pytest Configuration =====

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    Args:
        config: Pytest config object.
        
    Returns:
        None
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ===== Parametrized Fixtures =====

@pytest.fixture(params=[1, 2, 4, 8])
def various_batch_sizes(request):
    """
    Parametrized fixture for various batch sizes.
    
    Args:
        request: Pytest request.
        
    Returns:
        int: Batch size.
    """
    return request.param


@pytest.fixture(params=[10, 32, 64, 128])
def various_seq_lens(request):
    """
    Parametrized fixture for various sequence lengths.
    
    Args:
        request: Pytest request.
        
    Returns:
        int: Sequence length.
    """
    return request.param


@pytest.fixture(params=[256, 512, 768, 1536])
def various_embedding_dims(request):
    """
    Parametrized fixture for various embedding dimensions.
    
    Args:
        request: Pytest request.
        
    Returns:
        int: Embedding dimension.
    """
    return request.param


@pytest.fixture(params=[0.0, 0.1, 0.2, 0.5])
def various_dropout_rates(request):
    """
    Parametrized fixture for various dropout rates.
    
    Args:
        request: Pytest request.
        
    Returns:
        float: Dropout rate.
    """
    return request.param
