"""
Additional edge case tests for decoders.
Tests uncovered functionality and edge cases not in test_decoders.py.
"""

import pytest
import torch
import torch.nn as nn
from src.decoders.animation_decoder import AnimationDecoder
from src.decoders.audio_decoder import AudioDecoder
from src.decoders.text_decoder import TextDecoder


class TestAnimationDecoderEdgeCases:
    """Edge cases for AnimationDecoder."""

    def test_normalize_quaternion_method(self):
        """Test normalize_quaternion method directly."""
        decoder = AnimationDecoder(embedding_dim=512)

        # Unnormalized quaternion
        quat = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5]])
        normalized = decoder.normalize_quaternion(quat)

        # Check that magnitude is 1
        magnitudes = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_normalize_quaternion_zero_vector(self):
        """Test normalizing zero quaternion."""
        decoder = AnimationDecoder(embedding_dim=512)

        # Zero quaternion (edge case)
        quat = torch.zeros(2, 4)
        normalized = decoder.normalize_quaternion(quat)

        # Should handle gracefully (F.normalize returns nan for zero vectors)
        # Check that it doesn't crash
        assert normalized.shape == quat.shape

    def test_normalize_quaternion_batched(self):
        """Test normalize_quaternion with batched input."""
        decoder = AnimationDecoder(embedding_dim=512)

        quat = torch.randn(4, 10, 24, 4)
        normalized = decoder.normalize_quaternion(quat)

        # Check all quaternions are normalized
        magnitudes = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)

    def test_to_vrchat_format_method(self):
        """Test to_vrchat_format conversion (currently untested)."""
        decoder = AnimationDecoder(
            embedding_dim=512,
            num_joints=24,
            num_blend_shapes=51
        )

        # Create dummy animation parameters
        joint_rotations = torch.randn(2, 10, 24, 4)
        blend_shapes = torch.rand(2, 10, 51)
        eye_params = torch.rand(2, 10, 8)

        vrchat_format = decoder.to_vrchat_format(
            joint_rotations,
            blend_shapes,
            eye_params
        )

        # Check structure
        assert "Humanoid" in vrchat_format
        assert "FaceTracking" in vrchat_format
        assert "BodyRotations" in vrchat_format["Humanoid"]
        assert "BlendShapes" in vrchat_format["FaceTracking"]
        assert "EyeTracking" in vrchat_format["FaceTracking"]

    def test_blend_shapes_in_valid_range(self):
        """Test that blend shapes are clamped to [0, 1] range."""
        decoder = AnimationDecoder(embedding_dim=512, num_blend_shapes=51)

        hidden_states = torch.randn(2, 10, 512)
        output = decoder(hidden_states)

        blend_shapes = output["blend_shapes"]
        assert (blend_shapes >= 0).all()
        assert (blend_shapes <= 1).all()

    def test_eye_direction_normalized(self):
        """Test that eye directions are normalized."""
        decoder = AnimationDecoder(embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        output = decoder(hidden_states)

        eye_params = output["eye_params"]

        # Extract eye directions (first 3 and next 3 values, skipping openness)
        left_eye_dir = eye_params[..., :3]
        right_eye_dir = eye_params[..., 4:7]

        # Check normalization
        left_magnitudes = torch.norm(left_eye_dir, p=2, dim=-1)
        right_magnitudes = torch.norm(right_eye_dir, p=2, dim=-1)

        assert torch.allclose(left_magnitudes, torch.ones_like(left_magnitudes), atol=1e-5)
        assert torch.allclose(right_magnitudes, torch.ones_like(right_magnitudes), atol=1e-5)

    def test_eye_openness_in_valid_range(self):
        """Test that eye openness is in [0, 1] range."""
        decoder = AnimationDecoder(embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        output = decoder(hidden_states)

        eye_params = output["eye_params"]

        # Extract openness values
        left_eye_open = eye_params[..., 3]
        right_eye_open = eye_params[..., 7]

        assert (left_eye_open >= 0).all()
        assert (left_eye_open <= 1).all()
        assert (right_eye_open >= 0).all()
        assert (right_eye_open <= 1).all()

    def test_custom_loss_weights(self):
        """Test compute_loss with custom loss weights."""
        decoder = AnimationDecoder(embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        target_rotations = torch.randn(2, 10, 24, 4)
        target_rotations = decoder.normalize_quaternion(target_rotations)
        target_blend_shapes = torch.rand(2, 10, 51)
        target_eye_params = torch.rand(2, 10, 8)

        custom_weights = {
            "body": 2.0,
            "face": 1.0,
            "eyes": 0.5,
        }

        loss, loss_dict = decoder.compute_loss(
            hidden_states,
            target_rotations,
            target_blend_shapes,
            target_eye_params,
            loss_weights=custom_weights
        )

        assert "body_loss" in loss_dict
        assert "face_loss" in loss_dict
        assert "eye_loss" in loss_dict
        assert "total_animation_loss" in loss_dict

    def test_geodesic_quaternion_distance(self):
        """Test that quaternion loss uses geodesic distance."""
        decoder = AnimationDecoder(embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        # Create target quaternions that are identical to predictions
        with torch.no_grad():
            predictions = decoder(hidden_states)
            target_rotations = predictions["joint_rotations"].clone()

        target_blend_shapes = predictions["blend_shapes"].clone()
        target_eye_params = predictions["eye_params"].clone()

        loss, loss_dict = decoder.compute_loss(
            hidden_states,
            target_rotations,
            target_blend_shapes,
            target_eye_params
        )

        # Body loss should be very small when predictions match targets
        assert loss_dict["body_loss"] < 0.1

    def test_various_num_joints(self):
        """Test with different number of joints."""
        for num_joints in [10, 24, 32, 52]:
            decoder = AnimationDecoder(
                embedding_dim=512,
                num_joints=num_joints
            )

            hidden_states = torch.randn(2, 10, 512)
            output = decoder(hidden_states)

            assert output["joint_rotations"].shape == (2, 10, num_joints, 4)

    def test_various_num_blend_shapes(self):
        """Test with different number of blend shapes."""
        for num_blend_shapes in [51, 52, 61]:
            decoder = AnimationDecoder(
                embedding_dim=512,
                num_blend_shapes=num_blend_shapes
            )

            hidden_states = torch.randn(2, 10, 512)
            output = decoder(hidden_states)

            assert output["blend_shapes"].shape == (2, 10, num_blend_shapes)


class TestAudioDecoderEdgeCases:
    """Edge cases for AudioDecoder."""

    def test_null_token_enabled(self):
        """Test with use_null_token=True."""
        decoder = AudioDecoder(
            codebook_size=1024,
            embedding_dim=512,
            use_null_token=True
        )

        assert decoder.null_token_id == 1023  # Last token

    def test_null_token_disabled(self):
        """Test with use_null_token=False."""
        decoder = AudioDecoder(
            codebook_size=1024,
            embedding_dim=512,
            use_null_token=False
        )

        assert decoder.null_token_id is None

    def test_temperature_sampling(self):
        """Test sampling with different temperatures."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        # Low temperature (more deterministic)
        output_low = decoder(hidden_states, temperature=0.1)
        assert "tokens" in output_low
        assert output_low["tokens"].shape == (2, 10)

        # High temperature (more random)
        output_high = decoder(hidden_states, temperature=2.0)
        assert "tokens" in output_high
        assert output_high["tokens"].shape == (2, 10)

    def test_top_k_filtering(self):
        """Test top-k sampling."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, temperature=1.0, top_k=50)

        assert "tokens" in output
        assert output["tokens"].shape == (2, 10)
        # Tokens should be valid
        assert (output["tokens"] >= 0).all()
        assert (output["tokens"] < 1024).all()

    def test_top_p_nucleus_sampling(self):
        """Test nucleus (top-p) sampling."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, temperature=1.0, top_p=0.9)

        assert "tokens" in output
        assert output["tokens"].shape == (2, 10)

    def test_combined_top_k_top_p(self):
        """Test combining top-k and top-p filtering."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, temperature=1.0, top_k=100, top_p=0.95)

        assert "tokens" in output
        assert output["tokens"].shape == (2, 10)

    def test_return_logits_mode(self):
        """Test with return_logits=True."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, return_logits=True)

        assert "logits" in output
        assert "probabilities" in output
        assert "tokens" not in output  # Should not sample when returning logits

        assert output["logits"].shape == (2, 10, 1024)
        assert output["probabilities"].shape == (2, 10, 1024)

        # Probabilities should sum to 1
        prob_sums = output["probabilities"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_compute_loss_with_mask(self):
        """Test compute_loss with attention mask."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1024, (2, 10))

        # Create mask where some positions are invalid
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[:, 5:] = 0  # Mask last 5 positions

        loss_with_mask = decoder.compute_loss(
            hidden_states,
            target_tokens,
            attention_mask=attention_mask
        )

        loss_without_mask = decoder.compute_loss(
            hidden_states,
            target_tokens,
            attention_mask=None
        )

        # Losses should be different
        assert not torch.isclose(loss_with_mask, loss_without_mask)

    def test_compute_loss_no_mask(self):
        """Test compute_loss without attention mask."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1024, (2, 10))

        loss = decoder.compute_loss(hidden_states, target_tokens)

        assert loss.dim() == 0  # Scalar
        assert loss > 0

    def test_various_codebook_sizes(self):
        """Test with different codebook sizes."""
        for codebook_size in [256, 512, 1024, 2048]:
            decoder = AudioDecoder(
                codebook_size=codebook_size,
                embedding_dim=512
            )

            hidden_states = torch.randn(2, 10, 512)
            output = decoder(hidden_states, return_logits=True)

            assert output["logits"].shape == (2, 10, codebook_size)

    def test_decode_to_waveform_placeholder(self):
        """Test decode_to_waveform method (placeholder implementation)."""
        decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)

        audio_tokens = torch.randint(0, 1024, (2, 50))

        # Create mock vocoder
        class MockVocoder(nn.Module):
            def decode(self, tokens):
                batch_size, seq_len = tokens.shape
                # Return dummy waveform
                return torch.randn(batch_size, seq_len * 320)

        mock_vocoder = MockVocoder()

        waveform = decoder.decode_to_waveform(audio_tokens, mock_vocoder)

        assert waveform.shape[0] == 2
        assert waveform.dim() == 2


class TestTextDecoderEdgeCases:
    """Edge cases for TextDecoder."""

    def test_null_token_enabled(self):
        """Test with use_null_token=True."""
        decoder = TextDecoder(
            vocab_size=1000,
            embedding_dim=512,
            use_null_token=True
        )

        assert decoder.null_token_id == 999  # Last token

    def test_null_token_disabled(self):
        """Test with use_null_token=False."""
        decoder = TextDecoder(
            vocab_size=1000,
            embedding_dim=512,
            use_null_token=False
        )

        assert decoder.null_token_id is None

    def test_temperature_sampling(self):
        """Test sampling with different temperatures."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        # Low temperature
        output_low = decoder(hidden_states, temperature=0.5)
        assert output_low["tokens"].shape == (2, 10)

        # High temperature
        output_high = decoder(hidden_states, temperature=2.0)
        assert output_high["tokens"].shape == (2, 10)

    def test_top_k_sampling(self):
        """Test top-k filtering."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, top_k=50)

        assert (output["tokens"] >= 0).all()
        assert (output["tokens"] < 1000).all()

    def test_top_p_sampling(self):
        """Test nucleus sampling."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, top_p=0.9)

        assert output["tokens"].shape == (2, 10)

    def test_repetition_penalty(self):
        """Test repetition penalty application."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        # Without repetition penalty
        output_no_penalty = decoder(hidden_states, repetition_penalty=1.0)

        # With repetition penalty
        output_with_penalty = decoder(hidden_states, repetition_penalty=1.5)

        # Both should produce valid tokens
        assert output_no_penalty["tokens"].shape == (2, 10)
        assert output_with_penalty["tokens"].shape == (2, 10)

    def test_return_logits(self):
        """Test return_logits mode."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)

        output = decoder(hidden_states, return_logits=True)

        assert "logits" in output
        assert "probabilities" in output
        assert "tokens" not in output

        assert output["logits"].shape == (2, 10, 1000)

    def test_compute_loss_with_mask(self):
        """Test compute_loss with attention mask."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[:, 7:] = 0

        loss = decoder.compute_loss(
            hidden_states,
            target_tokens,
            attention_mask=attention_mask
        )

        assert loss.dim() == 0
        assert loss > 0

    def test_compute_loss_no_mask(self):
        """Test compute_loss without mask."""
        decoder = TextDecoder(vocab_size=1000, embedding_dim=512)

        hidden_states = torch.randn(2, 10, 512)
        target_tokens = torch.randint(0, 1000, (2, 10))

        loss = decoder.compute_loss(hidden_states, target_tokens)

        assert loss.dim() == 0

    def test_various_vocab_sizes(self):
        """Test with different vocabulary sizes."""
        for vocab_size in [500, 1000, 5000, 50000]:
            decoder = TextDecoder(
                vocab_size=vocab_size,
                embedding_dim=512
            )

            hidden_states = torch.randn(2, 10, 512)
            output = decoder(hidden_states, return_logits=True)

            assert output["logits"].shape == (2, 10, vocab_size)


class TestDecoderOutputConsistency:
    """Test consistency across decoders."""

    def test_all_decoders_have_compute_loss(self):
        """Verify all decoders implement compute_loss method."""
        decoders = [
            TextDecoder(vocab_size=1000, embedding_dim=512),
            AudioDecoder(codebook_size=1024, embedding_dim=512),
            AnimationDecoder(embedding_dim=512),
        ]

        for decoder in decoders:
            assert hasattr(decoder, 'compute_loss')
            assert callable(decoder.compute_loss)

    def test_all_decoders_handle_batched_input(self):
        """Test that all decoders handle batched inputs correctly."""
        hidden_states = torch.randn(4, 20, 512)

        # Text decoder
        text_decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
        text_output = text_decoder(hidden_states)
        assert text_output["tokens"].shape[0] == 4

        # Audio decoder
        audio_decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)
        audio_output = audio_decoder(hidden_states)
        assert audio_output["tokens"].shape[0] == 4

        # Animation decoder
        anim_decoder = AnimationDecoder(embedding_dim=512)
        anim_output = anim_decoder(hidden_states)
        assert anim_output["joint_rotations"].shape[0] == 4

    def test_decoders_produce_valid_outputs(self):
        """Test that decoder outputs are in valid ranges."""
        hidden_states = torch.randn(2, 10, 512)

        # Text decoder - tokens should be in vocab range
        text_decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
        text_output = text_decoder(hidden_states)
        assert (text_output["tokens"] >= 0).all()
        assert (text_output["tokens"] < 1000).all()

        # Audio decoder - tokens should be in codebook range
        audio_decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)
        audio_output = audio_decoder(hidden_states)
        assert (audio_output["tokens"] >= 0).all()
        assert (audio_output["tokens"] < 1024).all()

        # Animation decoder - quaternions should be normalized
        anim_decoder = AnimationDecoder(embedding_dim=512)
        anim_output = anim_decoder(hidden_states)
        quats = anim_output["joint_rotations"]
        quat_norms = torch.norm(quats, p=2, dim=-1)
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)

    def test_decoders_handle_variable_sequence_lengths(self):
        """Test decoders with different sequence lengths."""
        for seq_len in [1, 5, 10, 50]:
            hidden_states = torch.randn(2, seq_len, 512)

            text_decoder = TextDecoder(vocab_size=1000, embedding_dim=512)
            text_output = text_decoder(hidden_states)
            assert text_output["tokens"].shape[1] == seq_len

            audio_decoder = AudioDecoder(codebook_size=1024, embedding_dim=512)
            audio_output = audio_decoder(hidden_states)
            assert audio_output["tokens"].shape[1] == seq_len

            anim_decoder = AnimationDecoder(embedding_dim=512)
            anim_output = anim_decoder(hidden_states)
            assert anim_output["joint_rotations"].shape[1] == seq_len
