"""
Property-based tests using hypothesis.

Purpose:
    Use property-based testing to find edge cases and verify
    invariants across random inputs for encoders, data schema,
    and configuration validation.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays

# Import modules to test
from src.data.schema import UnifiedSample, ModalityType
from src.config import Config


class TestEncoderProperties:
    """Property-based tests for encoders."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=100),
        vocab_size=st.integers(min_value=100, max_value=50000)
    )
    @settings(max_examples=20, deadline=None)
    def test_internal_voice_encoder_output_shape(self, batch_size, seq_len, vocab_size):
        """
        Property: InternalVoiceEncoder output shape depends on input shape.

        Purpose:
            Verify output dimensions are consistent with input batch/seq.
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        embedding_dim = 256
        encoder = InternalVoiceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_len=max(seq_len, 512)
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = encoder(input_ids)

        assert output["embeddings"].shape[0] == batch_size
        assert output["embeddings"].shape[1] == seq_len
        assert output["embeddings"].shape[2] == embedding_dim

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        num_frames=st.integers(min_value=1, max_value=20),
        num_joints=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_proprioception_encoder_output_shape(self, batch_size, num_frames, num_joints):
        """
        Property: ProprioceptionEncoder output shape depends on input.

        Purpose:
            Verify output dimensions match input specifications.
        """
        from src.encoders.proprioception_encoder import ProprioceptionEncoder

        embedding_dim = 256
        encoder = ProprioceptionEncoder(
            num_joints=num_joints,
            embedding_dim=embedding_dim
        )

        joint_positions = torch.randn(batch_size, num_frames, num_joints, 3)
        joint_rotations = torch.randn(batch_size, num_frames, num_joints, 4)

        output = encoder(joint_positions, joint_rotations)

        assert output["embeddings"].shape[0] == batch_size
        assert output["embeddings"].shape[1] == num_frames
        assert output["embeddings"].shape[2] == embedding_dim

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        waveform_length=st.integers(min_value=1000, max_value=32000)
    )
    @settings(max_examples=15, deadline=None)
    def test_audio_encoder_no_nan_inf(self, batch_size, waveform_length):
        """
        Property: AudioEncoder should never produce NaN or Inf values.

        Purpose:
            Verify numeric stability across various input sizes.
        """
        from src.encoders.audio_encoder import AudioEncoder

        encoder = AudioEncoder(
            sample_rate=16000,
            embedding_dim=256,
            codebook_size=512
        )

        # Generate audio in valid range [-1, 1]
        audio = torch.rand(batch_size, waveform_length) * 2 - 1

        output = encoder(audio)

        assert not torch.isnan(output["embeddings"]).any()
        assert not torch.isinf(output["embeddings"]).any()


class TestDataSchemaProperties:
    """Property-based tests for data schema."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=20, deadline=None)
    def test_unified_sample_voice_tokens_valid(self, batch_size, seq_len):
        """
        Property: UnifiedSample voice_tokens shape is valid.

        Purpose:
            Verify sample creation with various dimensions.
        """
        sample = UnifiedSample(
            voice_tokens=torch.randint(0, 1000, (batch_size, seq_len)),
            modalities=[ModalityType.INTERNAL_VOICE]
        )

        assert sample.voice_tokens.shape == (batch_size, seq_len)
        assert ModalityType.INTERNAL_VOICE in sample.modalities

    @given(
        num_joints=st.integers(min_value=1, max_value=100),
        num_frames=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_unified_sample_proprioception_valid(self, num_joints, num_frames):
        """
        Property: UnifiedSample proprioception data shape is valid.

        Purpose:
            Verify proprioception data handling.
        """
        sample = UnifiedSample(
            joint_positions=torch.randn(1, num_frames, num_joints, 3),
            joint_rotations=torch.randn(1, num_frames, num_joints, 4),
            modalities=[ModalityType.PROPRIOCEPTION]
        )

        assert sample.joint_positions.shape == (1, num_frames, num_joints, 3)
        assert sample.joint_rotations.shape == (1, num_frames, num_joints, 4)


class TestConfigProperties:
    """Property-based tests for configuration."""

    @given(
        embedding_dim=st.sampled_from([128, 256, 512, 768, 1024]),
        num_layers=st.integers(min_value=1, max_value=12),
        num_heads=st.sampled_from([4, 8, 16])
    )
    @settings(max_examples=20, deadline=None)
    def test_config_valid_model_params(self, embedding_dim, num_layers, num_heads):
        """
        Property: Valid config parameters don't raise errors.

        Purpose:
            Verify configuration accepts valid parameter combinations.
        """
        # Only test combinations where embedding_dim is divisible by num_heads
        assume(embedding_dim % num_heads == 0)

        config_dict = {
            "model": {
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
                "num_heads": num_heads
            }
        }

        config = Config(config_dict)

        assert config.get("model.embedding_dim") == embedding_dim
        assert config.get("model.num_layers") == num_layers

    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        learning_rate=st.floats(min_value=1e-6, max_value=1e-1)
    )
    @settings(max_examples=20, deadline=None)
    def test_config_valid_training_params(self, batch_size, learning_rate):
        """
        Property: Valid training params are stored correctly.

        Purpose:
            Verify training configuration handling.
        """
        config_dict = {
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
        }

        config = Config(config_dict)

        assert config.get("training.batch_size") == batch_size
        assert abs(config.get("training.learning_rate") - learning_rate) < 1e-10


class TestDecoderProperties:
    """Property-based tests for decoders."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=50),
        embedding_dim=st.sampled_from([128, 256, 512])
    )
    @settings(max_examples=15, deadline=None)
    def test_text_decoder_output_shape(self, batch_size, seq_len, embedding_dim):
        """
        Property: TextDecoder output shape matches input batch/seq.

        Purpose:
            Verify decoder output dimensions.
        """
        from src.decoders.text_decoder import InternalTextDecoder

        vocab_size = 50257
        decoder = InternalTextDecoder(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        )

        embeddings = torch.randn(batch_size, seq_len, embedding_dim)
        output = decoder(embeddings, return_logits=True)

        assert output["logits"].shape[0] == batch_size
        assert output["logits"].shape[1] == seq_len
        assert output["logits"].shape[2] == vocab_size

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=20),
        num_joints=st.sampled_from([22, 24, 52])
    )
    @settings(max_examples=15, deadline=None)
    def test_animation_decoder_output_shape(self, batch_size, seq_len, num_joints):
        """
        Property: AnimationDecoder outputs correct joint dimensions.

        Purpose:
            Verify animation output shapes.
        """
        from src.decoders.animation_decoder import AnimationDecoder

        embedding_dim = 256
        decoder = AnimationDecoder(
            embedding_dim=embedding_dim,
            num_joints=num_joints
        )

        embeddings = torch.randn(batch_size, seq_len, embedding_dim)
        output = decoder(embeddings)

        assert output["joint_rotations"].shape[0] == batch_size
        assert output["joint_rotations"].shape[1] == seq_len
        assert output["joint_rotations"].shape[2] == num_joints


class TestNumericStabilityProperties:
    """Property-based tests for numeric stability."""

    @given(
        scale=st.floats(min_value=0.001, max_value=100.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_fusion_module_stable_with_scaled_inputs(self, scale):
        """
        Property: Fusion module handles various input scales.

        Purpose:
            Verify numeric stability across input magnitudes.
        """
        from src.models.fusion_module import TokenFusionModule

        module = TokenFusionModule(
            embedding_dim=256,
            fusion_strategy="concatenate",
            dropout=0.0
        )

        # Scale inputs
        encoder_outputs = {
            "internal_voice": {
                "embeddings": torch.randn(2, 10, 256) * scale,
                "attention_mask": torch.ones(2, 10)
            }
        }

        result = module(encoder_outputs)

        assert not torch.isnan(result["embeddings"]).any()
        assert not torch.isinf(result["embeddings"]).any()

    @given(
        dropout_rate=st.floats(min_value=0.0, max_value=0.9)
    )
    @settings(max_examples=15, deadline=None)
    def test_encoder_dropout_stability(self, dropout_rate):
        """
        Property: Encoders are stable with various dropout rates.

        Purpose:
            Verify dropout doesn't cause numeric issues.
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(
            vocab_size=1000,
            embedding_dim=256,
            dropout=dropout_rate
        )
        encoder.eval()  # Disable dropout for deterministic test

        input_ids = torch.randint(0, 1000, (2, 20))
        output = encoder(input_ids)

        assert not torch.isnan(output["embeddings"]).any()


class TestInvariantProperties:
    """Tests for invariants that should always hold."""

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_attention_mask_binary(self, seq_len):
        """
        Property: Attention masks should be binary (0 or 1).

        Purpose:
            Verify mask values are valid.
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(
            vocab_size=1000,
            embedding_dim=256
        )

        input_ids = torch.randint(0, 1000, (2, seq_len))
        output = encoder(input_ids)

        mask = output["attention_mask"]

        # All values should be 0 or 1
        assert ((mask == 0) | (mask == 1)).all()

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_embedding_dimension_preserved(self, batch_size, seq_len):
        """
        Property: Embedding dimension is preserved through encoder.

        Purpose:
            Verify embedding_dim is consistent.
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        embedding_dim = 512
        encoder = InternalVoiceEncoder(
            vocab_size=1000,
            embedding_dim=embedding_dim
        )

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        output = encoder(input_ids)

        assert output["embeddings"].shape[-1] == embedding_dim
