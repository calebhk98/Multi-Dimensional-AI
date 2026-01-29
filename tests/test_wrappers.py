"""
Tests for training wrappers (auto-encoder modules).

Purpose:
    Verify wrapper forward passes and loss computations
    for AudioAutoEncoder, VoiceAutoEncoder, and MotionAutoEncoder.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from src.training.wrappers import (
    BaseAutoEncoder,
    AudioAutoEncoder,
    VoiceAutoEncoder,
    MotionAutoEncoder
)


class TestBaseAutoEncoder:
    """Tests for BaseAutoEncoder base class."""

    def test_compute_loss_not_implemented(self):
        """
        Test base class compute_loss raises NotImplementedError.

        Purpose:
            Verify abstract interface enforces implementation.
        """
        base = BaseAutoEncoder()

        with pytest.raises(NotImplementedError):
            base.compute_loss({}, {})


class TestAudioAutoEncoder:
    """Tests for AudioAutoEncoder wrapper."""

    @pytest.fixture
    def mock_audio_encoder(self):
        """Create mock audio encoder."""
        encoder = MagicMock()
        encoder.return_value = {
            "embeddings": torch.randn(2, 10, 512),
            "indices": torch.randint(0, 1024, (2, 10))
        }
        return encoder

    @pytest.fixture
    def mock_audio_decoder(self):
        """Create mock audio decoder."""
        decoder = MagicMock()
        decoder.return_value = {
            "logits": torch.randn(2, 10, 1024)
        }
        return decoder

    def test_init(self, mock_audio_encoder, mock_audio_decoder):
        """
        Test AudioAutoEncoder initialization.

        Purpose:
            Verify encoder and decoder are stored.
        """
        wrapper = AudioAutoEncoder(mock_audio_encoder, mock_audio_decoder)

        assert wrapper.encoder is mock_audio_encoder
        assert wrapper.decoder is mock_audio_decoder

    def test_forward(self, mock_audio_encoder, mock_audio_decoder):
        """
        Test AudioAutoEncoder forward pass.

        Purpose:
            Verify forward returns expected structure.
        """
        wrapper = AudioAutoEncoder(mock_audio_encoder, mock_audio_decoder)

        audio_waveform = torch.randn(2, 16000)
        outputs = wrapper(audio_waveform)

        assert "encoder_outputs" in outputs
        assert "decoder_outputs" in outputs
        assert "embeddings" in outputs

        mock_audio_encoder.assert_called_once()
        mock_audio_decoder.assert_called_once()

    def test_compute_loss_matching_lengths(self, mock_audio_encoder, mock_audio_decoder):
        """
        Test compute_loss with matching sequence lengths.

        Purpose:
            Verify loss computation for aligned sequences.
        """
        wrapper = AudioAutoEncoder(mock_audio_encoder, mock_audio_decoder)

        outputs = {
            "decoder_outputs": {
                "logits": torch.randn(2, 10, 1024)  # [B, T, V]
            }
        }
        targets = {
            "target": torch.randint(0, 1024, (2, 10))  # [B, T]
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert "loss" in loss_dict
        assert loss_dict["loss"] == loss.item()

    def test_compute_loss_mismatched_lengths(self, mock_audio_encoder, mock_audio_decoder):
        """
        Test compute_loss handles mismatched sequence lengths.

        Purpose:
            Verify min-length truncation works correctly.
        """
        wrapper = AudioAutoEncoder(mock_audio_encoder, mock_audio_decoder)

        # Logits longer than targets
        outputs = {
            "decoder_outputs": {
                "logits": torch.randn(2, 15, 1024)  # [B, 15, V]
            }
        }
        targets = {
            "target": torch.randint(0, 1024, (2, 10))  # [B, 10]
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        # Should not raise, should truncate to min length
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_compute_loss_targets_longer(self, mock_audio_encoder, mock_audio_decoder):
        """
        Test compute_loss when targets are longer than logits.

        Purpose:
            Verify truncation works when targets longer.
        """
        wrapper = AudioAutoEncoder(mock_audio_encoder, mock_audio_decoder)

        # Targets longer than logits
        outputs = {
            "decoder_outputs": {
                "logits": torch.randn(2, 8, 1024)  # [B, 8, V]
            }
        }
        targets = {
            "target": torch.randint(0, 1024, (2, 12))  # [B, 12]
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)


class TestVoiceAutoEncoder:
    """Tests for VoiceAutoEncoder wrapper."""

    @pytest.fixture
    def mock_voice_encoder(self):
        """Create mock voice encoder."""
        encoder = MagicMock()
        encoder.return_value = {
            "embeddings": torch.randn(2, 20, 512)
        }
        return encoder

    @pytest.fixture
    def mock_text_decoder(self):
        """Create mock text decoder."""
        decoder = MagicMock()
        decoder.return_value = {
            "logits": torch.randn(2, 20, 50257)
        }
        return decoder

    def test_init(self, mock_voice_encoder, mock_text_decoder):
        """
        Test VoiceAutoEncoder initialization.

        Purpose:
            Verify encoder and decoder are stored.
        """
        wrapper = VoiceAutoEncoder(mock_voice_encoder, mock_text_decoder)

        assert wrapper.encoder is mock_voice_encoder
        assert wrapper.decoder is mock_text_decoder

    def test_forward(self, mock_voice_encoder, mock_text_decoder):
        """
        Test VoiceAutoEncoder forward pass.

        Purpose:
            Verify forward returns expected structure.
        """
        wrapper = VoiceAutoEncoder(mock_voice_encoder, mock_text_decoder)

        input_ids = torch.randint(0, 50257, (2, 20))
        outputs = wrapper(input_ids)

        assert "embeddings" in outputs
        assert "decoder_outputs" in outputs

        mock_voice_encoder.assert_called_once()
        mock_text_decoder.assert_called_once()

    def test_compute_loss(self, mock_voice_encoder, mock_text_decoder):
        """
        Test VoiceAutoEncoder compute_loss.

        Purpose:
            Verify cross-entropy loss computation.
        """
        wrapper = VoiceAutoEncoder(mock_voice_encoder, mock_text_decoder)

        outputs = {
            "decoder_outputs": {
                "logits": torch.randn(2, 20, 50257)  # [B, T, V]
            }
        }
        targets = {
            "target": torch.randint(0, 50257, (2, 20))  # [B, T]
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "loss" in loss_dict


class TestMotionAutoEncoder:
    """Tests for MotionAutoEncoder wrapper."""

    @pytest.fixture
    def mock_proprio_encoder(self):
        """Create mock proprioception encoder."""
        encoder = MagicMock()
        encoder.return_value = {
            "embeddings": torch.randn(2, 10, 512)
        }
        return encoder

    @pytest.fixture
    def mock_animation_decoder(self):
        """Create mock animation decoder."""
        decoder = MagicMock()
        decoder.return_value = {
            "joint_rotations": torch.randn(2, 10, 24, 4),
            "blend_shapes": torch.randn(2, 10, 51),
            "eye_params": torch.randn(2, 10, 8)
        }
        return decoder

    def test_init(self, mock_proprio_encoder, mock_animation_decoder):
        """
        Test MotionAutoEncoder initialization.

        Purpose:
            Verify encoder and decoder are stored.
        """
        wrapper = MotionAutoEncoder(mock_proprio_encoder, mock_animation_decoder)

        assert wrapper.encoder is mock_proprio_encoder
        assert wrapper.decoder is mock_animation_decoder

    def test_forward(self, mock_proprio_encoder, mock_animation_decoder):
        """
        Test MotionAutoEncoder forward pass.

        Purpose:
            Verify forward returns expected structure.
        """
        wrapper = MotionAutoEncoder(mock_proprio_encoder, mock_animation_decoder)

        joint_positions = torch.randn(2, 10, 24, 3)
        joint_rotations = torch.randn(2, 10, 24, 4)

        outputs = wrapper(joint_positions, joint_rotations)

        assert "embeddings" in outputs
        assert "rotations" in outputs
        assert "blend_shapes" in outputs
        assert "eye_params" in outputs

    def test_compute_loss_all_targets(self, mock_proprio_encoder, mock_animation_decoder):
        """
        Test compute_loss with all target types.

        Purpose:
            Verify MSE loss for rotations, blend_shapes, and eye_params.
        """
        wrapper = MotionAutoEncoder(mock_proprio_encoder, mock_animation_decoder)

        outputs = {
            "rotations": torch.randn(2, 10, 24, 4),
            "blend_shapes": torch.randn(2, 10, 51),
            "eye_params": torch.randn(2, 10, 8)
        }
        targets = {
            "rotations": torch.randn(2, 10, 24, 4),
            "blend_shapes": torch.randn(2, 10, 51),
            "eye_params": torch.randn(2, 10, 8)
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert "loss_rot" in loss_dict
        assert "loss_blend" in loss_dict
        assert "loss_eyes" in loss_dict
        assert "total_loss" in loss_dict

    def test_compute_loss_rotations_only(self, mock_proprio_encoder, mock_animation_decoder):
        """
        Test compute_loss with only rotation targets.

        Purpose:
            Verify loss works with minimal targets.
        """
        wrapper = MotionAutoEncoder(mock_proprio_encoder, mock_animation_decoder)

        outputs = {
            "rotations": torch.randn(2, 10, 24, 4),
            "blend_shapes": torch.randn(2, 10, 51),
            "eye_params": torch.randn(2, 10, 8)
        }
        targets = {
            "rotations": torch.randn(2, 10, 24, 4)
            # No blend_shapes or eye_params
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert "loss_rot" in loss_dict
        assert "loss_blend" not in loss_dict
        assert "loss_eyes" not in loss_dict

    def test_compute_loss_mismatched_lengths(self, mock_proprio_encoder, mock_animation_decoder):
        """
        Test compute_loss handles sequence length mismatch.

        Purpose:
            Verify truncation to min length.
        """
        wrapper = MotionAutoEncoder(mock_proprio_encoder, mock_animation_decoder)

        outputs = {
            "rotations": torch.randn(2, 15, 24, 4),  # Longer
            "blend_shapes": torch.randn(2, 15, 51),
            "eye_params": torch.randn(2, 15, 8)
        }
        targets = {
            "rotations": torch.randn(2, 10, 24, 4),  # Shorter
            "blend_shapes": torch.randn(2, 10, 51),
            "eye_params": torch.randn(2, 10, 8)
        }

        loss, loss_dict = wrapper.compute_loss(outputs, targets)

        # Should not raise, should use min length
        assert isinstance(loss, torch.Tensor)


class TestWrappersIntegration:
    """Integration tests for wrappers with real encoders/decoders."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cpu")

    def test_audio_autoencoder_with_real_modules(self, device):
        """
        Test AudioAutoEncoder with real encoder/decoder (mocked internals).

        Purpose:
            Verify wrapper works with actual module instances.
        """
        from src.encoders.audio_encoder import AudioEncoder
        from src.decoders.audio_decoder import AudioDecoder

        encoder = AudioEncoder(
            sample_rate=16000,
            embedding_dim=256,
            codebook_size=512
        )
        decoder = AudioDecoder(
            embedding_dim=256,
            codebook_size=512
        )

        wrapper = AudioAutoEncoder(encoder, decoder)
        wrapper.to(device)

        # Test forward
        audio = torch.randn(2, 16000, device=device)
        outputs = wrapper(audio)

        assert "embeddings" in outputs
        assert outputs["embeddings"].shape[0] == 2
        assert outputs["embeddings"].shape[2] == 256

    def test_motion_autoencoder_with_real_modules(self, device):
        """
        Test MotionAutoEncoder with real encoder/decoder.

        Purpose:
            Verify wrapper works with actual module instances.
        """
        from src.encoders.proprioception_encoder import ProprioceptionEncoder
        from src.decoders.animation_decoder import AnimationDecoder

        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=256
        )
        decoder = AnimationDecoder(
            embedding_dim=256,
            num_joints=24
        )

        wrapper = MotionAutoEncoder(encoder, decoder)
        wrapper.to(device)

        # Test forward
        joint_pos = torch.randn(2, 10, 24, 3, device=device)
        joint_rot = torch.randn(2, 10, 24, 4, device=device)

        outputs = wrapper(joint_pos, joint_rot)

        assert "rotations" in outputs
        assert outputs["rotations"].shape[0] == 2
        assert outputs["rotations"].shape[2] == 24
