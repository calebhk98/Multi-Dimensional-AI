"""
Additional edge case tests for encoders.
Tests uncovered functionality and edge cases not in test_encoders.py.
"""

import pytest
import torch
from src.encoders.visual_encoder import VisualEncoder
from src.encoders.audio_encoder import AudioEncoder
from src.encoders.proprioception_encoder import ProprioceptionEncoder
from src.encoders.touch_encoder import TouchEncoder


class TestVisualEncoderEdgeCases:
    """Edge cases for VisualEncoder."""

    def test_get_output_dim(self):
        """
        Purpose:
            Test get_output_dim returns correct dimension.
            
        Workflow:
            1. Create encoder.
            2. Verify output dim matches embedding dim.
            
        ToDo:
            - None
        """
        encoder = VisualEncoder(embedding_dim=768)
        assert encoder.get_output_dim() == 768

        encoder = VisualEncoder(embedding_dim=1536)
        assert encoder.get_output_dim() == 1536

    def test_mono_vision_only_left(self):
        """
        Purpose:
            Test with only left image (no right image provided).
            
        Workflow:
            1. Create encoder with use_stereo=True.
            2. Pass left image only.
            3. Verify output shape matches mono.
            
        ToDo:
            - None
        """
        encoder = VisualEncoder(
            image_size=224,
            patch_size=16,
            embedding_dim=768,
            num_heads=12,
            use_stereo=True
        )

        batch_size = 2
        left_image = torch.randn(batch_size, 3, 224, 224)

        # Pass only left image (right_image=None)
        output = encoder(left_image, right_image=None)

        # Should only process left eye
        num_patches = (224 // 16) ** 2
        assert output["embeddings"].shape == (batch_size, num_patches, 768)
        assert output["attention_mask"].shape == (batch_size, num_patches)

    def test_use_stereo_false(self):
        """
        Purpose:
            Test with use_stereo=False configuration.
            
        Workflow:
            1. Create encoder with use_stereo=False.
            2. Pass stereo images.
            3. Verify right image is ignored.
            
        ToDo:
            - None
        """
        encoder = VisualEncoder(
            image_size=224,
            patch_size=16,
            embedding_dim=768,
            use_stereo=False
        )

        batch_size = 2
        left_image = torch.randn(batch_size, 3, 224, 224)
        right_image = torch.randn(batch_size, 3, 224, 224)

        # Even with right_image provided, should ignore if use_stereo=False
        output = encoder(left_image, right_image=right_image)

        # Should only have left eye patches
        num_patches = (224 // 16) ** 2
        assert output["embeddings"].shape == (batch_size, num_patches, 768)

    def test_small_embedding_dim_adjusts_num_heads(self):
        """
        Purpose:
            Test that num_heads is adjusted when embedding_dim is small.
            
        Workflow:
            1. Create encoder with small embedding dim.
            2. Run forward.
            3. Verify no crash.
            
        ToDo:
            - None
        """
        # embedding_dim=32 is not divisible by num_heads=12
        encoder = VisualEncoder(
            image_size=32,
            patch_size=8,
            embedding_dim=32,
            num_heads=12,  # Will be adjusted to 8 or 4
            use_stereo=False
        )

        batch_size = 2
        left_image = torch.randn(batch_size, 3, 32, 32)

        # Should not crash
        output = encoder(left_image)
        assert output["embeddings"].shape[2] == 32

    def test_various_image_sizes(self):
        """
        Purpose:
            Test with various image sizes.
            
        Workflow:
            1. Iterate sizes.
            2. Run forward.
            3. Verify output patches count.
            
        ToDo:
            - None
        """
        for image_size in [64, 128, 224, 256]:
            encoder = VisualEncoder(
                image_size=image_size,
                patch_size=16,
                embedding_dim=768,
                use_stereo=False
            )

            left_image = torch.randn(2, 3, image_size, image_size)
            output = encoder(left_image)

            expected_patches = (image_size // 16) ** 2
            assert output["embeddings"].shape[1] == expected_patches

    def test_various_patch_sizes(self):
        """
        Purpose:
            Test with various patch sizes.
            
        Workflow:
            1. Iterate patch sizes.
            2. Run forward.
            3. Verify output patches count.
            
        ToDo:
            - None
        """
        image_size = 224
        for patch_size in [8, 14, 16, 32]:
            encoder = VisualEncoder(
                image_size=image_size,
                patch_size=patch_size,
                embedding_dim=768,
                use_stereo=False
            )

            left_image = torch.randn(2, 3, image_size, image_size)
            output = encoder(left_image)

            expected_patches = (image_size // patch_size) ** 2
            assert output["embeddings"].shape[1] == expected_patches

    def test_grayscale_images(self):
        """
        Purpose:
            Test with grayscale images (1 channel).
            
        Workflow:
            1. Create encoder with in_channels=1.
            2. Run forward with grayscale input.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        encoder = VisualEncoder(
            image_size=224,
            patch_size=16,
            in_channels=1,  # Grayscale
            embedding_dim=768,
            use_stereo=False
        )

        left_image = torch.randn(2, 1, 224, 224)
        output = encoder(left_image)

        num_patches = (224 // 16) ** 2
        assert output["embeddings"].shape == (2, num_patches, 768)

    def test_stereo_concatenation(self):
        """
        Purpose:
            Test that stereo concatenates both eyes correctly.
            
        Workflow:
            1. Create stereo encoder.
            2. Run forward with both images.
            3. Verify output patches = 2 * mono.
            
        ToDo:
            - None
        """
        encoder = VisualEncoder(
            image_size=224,
            patch_size=16,
            embedding_dim=768,
            use_stereo=True
        )

        batch_size = 2
        left_image = torch.randn(batch_size, 3, 224, 224)
        right_image = torch.randn(batch_size, 3, 224, 224)

        output = encoder(left_image, right_image)

        num_patches = (224 // 16) ** 2
        # Should have 2x patches (left + right)
        assert output["embeddings"].shape == (batch_size, num_patches * 2, 768)


class TestAudioEncoderEdgeCases:
    """Edge cases for AudioEncoder."""

    def test_get_output_dim(self):
        """
        Purpose:
            Test get_output_dim returns correct dimension.
            
        Workflow:
            1. Create encoder.
            2. Verify output dim matches embedding dim.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=768)
        assert encoder.get_output_dim() == 768

        encoder = AudioEncoder(embedding_dim=1536)
        assert encoder.get_output_dim() == 1536

    def test_various_hop_lengths(self):
        """
        Purpose:
            Test with various hop_length configurations.
            
        Workflow:
            1. Iterate hop lengths.
            2. Run forward.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        for hop_length in [160, 320, 640]:
            encoder = AudioEncoder(
                sample_rate=16000,
                hop_length=hop_length,
                embedding_dim=512
            )

            waveform = torch.randn(2, 16000)  # 1 second of audio
            output = encoder(waveform, return_indices=False)

            assert output["embeddings"].shape[2] == 512
            assert output["attention_mask"].shape[0] == 2

    def test_various_sample_rates(self):
        """
        Purpose:
            Test with different sample rates.
            
        Workflow:
            1. Iterate sample rates.
            2. Run forward.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        for sample_rate in [8000, 16000, 24000]:
            encoder = AudioEncoder(
                sample_rate=sample_rate,
                hop_length=320,
                embedding_dim=512
            )

            waveform = torch.randn(2, sample_rate)  # 1 second
            output = encoder(waveform)

            assert output["embeddings"].shape[0] == 2

    def test_short_audio_clip(self):
        """
        Purpose:
            Test with very short audio clip.
            
        Workflow:
            1. Create short waveform.
            2. Run forward.
            3. Verify output has frames.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512)

        # Very short clip (0.1 seconds)
        waveform = torch.randn(2, 1600)
        output = encoder(waveform)

        assert output["embeddings"].shape[0] == 2
        assert output["embeddings"].shape[1] > 0  # Should have some frames

    def test_long_audio_clip(self):
        """
        Purpose:
            Test with long audio clip.
            
        Workflow:
            1. Create long waveform.
            2. Run forward.
            3. Verify output limit (if any) or shape.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512)

        # Long clip (10 seconds)
        waveform = torch.randn(2, 160000)
        output = encoder(waveform)

        assert output["embeddings"].shape[0] == 2
        # Should have many frames but less than max (1000)
        assert output["embeddings"].shape[1] < 1000

    def test_silent_audio(self):
        """
        Purpose:
            Test with silent audio (all zeros).
            
        Workflow:
            1. Create zero waveform.
            2. Run forward.
            3. Verify no NaNs.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512)

        # Silent audio
        waveform = torch.zeros(2, 16000)
        output = encoder(waveform)

        assert output["embeddings"].shape[0] == 2
        assert not torch.isnan(output["embeddings"]).any()

    def test_return_indices_true(self):
        """
        Purpose:
            Test with return_indices=True.
            
        Workflow:
            1. Run forward with return_indices=True.
            2. Verify indices present and valid.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512, codebook_size=1024)

        waveform = torch.randn(2, 16000)
        output = encoder(waveform, return_indices=True)

        assert "indices" in output
        assert output["indices"].shape[0] == 2
        # Indices should be in valid range
        assert (output["indices"] >= 0).all()
        assert (output["indices"] < 1024).all()

    def test_return_indices_false(self):
        """
        Purpose:
            Test with return_indices=False.
            
        Workflow:
            1. Run forward with return_indices=False.
            2. Verify indices absent.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512)

        waveform = torch.randn(2, 16000)
        output = encoder(waveform, return_indices=False)

        assert "indices" not in output

    def test_quantize_method(self):
        """
        Purpose:
            Test the quantize method directly.
            
        Workflow:
            1. Create features.
            2. Quantize.
            3. Verify shapes and values.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=128, codebook_size=256)

        features = torch.randn(2, 50, 128)
        quantized, indices = encoder.quantize(features)

        assert quantized.shape == features.shape
        assert indices.shape == (2, 50)
        assert (indices >= 0).all()
        assert (indices < 256).all()

    def test_mono_and_stereo_input(self):
        """
        Purpose:
            Test that encoder handles both 2D and 3D input tensors.
            
        Workflow:
            1. Pass 2D waveform.
            2. Pass 3D waveform (1 channel).
            3. Verify equivalence.
            
        ToDo:
            - None
        """
        encoder = AudioEncoder(embedding_dim=512)

        # 2D input [batch, samples]
        waveform_2d = torch.randn(2, 16000)
        output_2d = encoder(waveform_2d)

        # 3D input [batch, channels=1, samples]
        waveform_3d = torch.randn(2, 1, 16000)
        output_3d = encoder(waveform_3d)

        # Both should produce same shape output
        assert output_2d["embeddings"].shape == output_3d["embeddings"].shape

    def test_various_codebook_sizes(self):
        """
        Purpose:
            Test with different codebook sizes.
            
        Workflow:
            1. Iterate sizes.
            2. Run forward.
            3. Verify indices range.
            
        ToDo:
            - None
        """
        for codebook_size in [256, 512, 1024, 2048]:
            encoder = AudioEncoder(
                embedding_dim=512,
                codebook_size=codebook_size
            )

            waveform = torch.randn(2, 16000)
            output = encoder(waveform, return_indices=True)

            assert (output["indices"] < codebook_size).all()


class TestProprioceptionEncoderEdgeCases:
    """Edge cases for ProprioceptionEncoder."""

    def test_get_output_dim(self):
        """
        Purpose:
            Test get_output_dim returns correct dimension.
            
        Workflow:
            1. Create encoder.
            2. Verify output dim matches embedding dim.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(embedding_dim=768)
        assert encoder.get_output_dim() == 768

    def test_without_velocity_no_previous_state(self):
        """
        Purpose:
            Test with use_velocity=True but no previous state provided.
            
        Workflow:
            1. Create encoder with velocity.
            2. Pass inputs without previous state.
            3. Verify padding/handling.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=512,
            temporal_window=10,
            use_velocity=True
        )

        batch_size = 2
        joint_positions = torch.randn(batch_size, 10, 24, 3)
        joint_rotations = torch.randn(batch_size, 10, 24, 4)

        # No previous states provided - should pad with zeros
        output = encoder(joint_positions, joint_rotations)

        assert output["embeddings"].shape == (batch_size, 10, 512)
        assert not torch.isnan(output["embeddings"]).any()

    def test_with_velocity_and_previous_state(self):
        """
        Purpose:
            Test with velocity computation using previous state.
            
        Workflow:
            1. Create encoder with velocity.
            2. Pass inputs AND previous state.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=512,
            temporal_window=10,
            use_velocity=True
        )

        batch_size = 2
        temporal_len = 10
        joint_positions = torch.randn(batch_size, temporal_len, 24, 3)
        joint_rotations = torch.randn(batch_size, temporal_len, 24, 4)

        # Flatten for previous state
        prev_positions = torch.randn(batch_size * temporal_len, 24, 3)
        prev_rotations = torch.randn(batch_size * temporal_len, 24, 4)

        output = encoder(
            joint_positions,
            joint_rotations,
            previous_positions=prev_positions,
            previous_rotations=prev_rotations
        )

        assert output["embeddings"].shape == (batch_size, temporal_len, 512)

    def test_without_velocity(self):
        """
        Purpose:
            Test with use_velocity=False.
            
        Workflow:
            1. Create encoder without velocity.
            2. Pass inputs.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=512,
            temporal_window=10,
            use_velocity=False
        )

        batch_size = 2
        joint_positions = torch.randn(batch_size, 10, 24, 3)
        joint_rotations = torch.randn(batch_size, 10, 24, 4)

        output = encoder(joint_positions, joint_rotations)

        assert output["embeddings"].shape == (batch_size, 10, 512)

    def test_compute_velocity_method(self):
        """
        Purpose:
            Test compute_velocity method directly.
            
        Workflow:
            1. Create encoder.
            2. Compute velocity manually.
            3. Verify matches formula.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(embedding_dim=512)

        current = torch.tensor([[1.0, 2.0, 3.0]])
        previous = torch.tensor([[0.5, 1.5, 2.5]])
        dt = 0.01

        velocity = encoder.compute_velocity(current, previous, dt)

        expected = (current - previous) / dt
        assert torch.allclose(velocity, expected)

    def test_various_num_joints(self):
        """
        Purpose:
            Test with different number of joints.
            
        Workflow:
            1. Iterate num_joints.
            2. Create encoder.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        for num_joints in [10, 24, 32, 52]:
            encoder = ProprioceptionEncoder(
                num_joints=num_joints,
                embedding_dim=512,
                temporal_window=5
            )

            joint_positions = torch.randn(2, 5, num_joints, 3)
            joint_rotations = torch.randn(2, 5, num_joints, 4)

            output = encoder(joint_positions, joint_rotations)
            assert output["embeddings"].shape == (2, 5, 512)

    def test_various_temporal_windows(self):
        """
        Purpose:
            Test with different temporal window sizes.
            
        Workflow:
            1. Iterate windows.
            2. Create encoder.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        for temporal_window in [1, 5, 10, 20]:
            encoder = ProprioceptionEncoder(
                num_joints=24,
                embedding_dim=512,
                temporal_window=temporal_window
            )

            joint_positions = torch.randn(2, temporal_window, 24, 3)
            joint_rotations = torch.randn(2, temporal_window, 24, 4)

            output = encoder(joint_positions, joint_rotations)
            assert output["embeddings"].shape == (2, temporal_window, 512)

    def test_single_frame_temporal_window(self):
        """
        Purpose:
            Test with temporal_window=1 (no temporal context).
            
        Workflow:
            1. Create encoder with window=1.
            2. Pass 1-frame input.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=512,
            temporal_window=1
        )

        joint_positions = torch.randn(2, 1, 24, 3)
        joint_rotations = torch.randn(2, 1, 24, 4)

        output = encoder(joint_positions, joint_rotations)
        assert output["embeddings"].shape == (2, 1, 512)

    def test_all_joints_same_position(self):
        """
        Purpose:
            Test edge case where all joints are at the same position.
            
        Workflow:
            1. Create encoder.
            2. Pass degenerate input (all zeros).
            3. Verify no NaNs.
            
        ToDo:
            - None
        """
        encoder = ProprioceptionEncoder(
            num_joints=24,
            embedding_dim=512,
            temporal_window=10
        )

        # All joints at origin
        joint_positions = torch.zeros(2, 10, 24, 3)
        joint_rotations = torch.zeros(2, 10, 24, 4)

        output = encoder(joint_positions, joint_rotations)
        assert output["embeddings"].shape == (2, 10, 512)
        assert not torch.isnan(output["embeddings"]).any()


class TestTouchEncoderEdgeCases:
    """Edge cases for TouchEncoder."""

    def test_get_output_dim(self):
        """
        Purpose:
            Test get_output_dim returns correct dimension.
            
        Workflow:
            1. Create encoder.
            2. Verify output dim.
            
        ToDo:
            - None
        """
        from src.encoders.touch_encoder import TouchEncoder
        encoder = TouchEncoder(embedding_dim=768)
        assert encoder.get_output_dim() == 768

    def test_all_contacts_inactive(self):
        """
        Purpose:
            Test with all contact points inactive.
            
        Workflow:
            1. Create encoder.
            2. Pass inputs with all inactive contacts.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        from src.encoders.touch_encoder import TouchEncoder
        encoder = TouchEncoder(
            max_contacts=10,
            embedding_dim=512
        )

        batch_size = 2
        
        # All inactive
        contact_active = torch.zeros(batch_size, 10, dtype=torch.bool)
        
        # Dummy data for other required args
        contact_points = torch.randint(0, 10, (batch_size, 10))
        surface_types = torch.randint(0, 8, (batch_size, 10))
        contact_forces = torch.randn(batch_size, 10, 1)
        contact_positions = torch.randn(batch_size, 10, 3)

        output = encoder(
            contact_active=contact_active,
            contact_points=contact_points,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            surface_types=surface_types
        )
        assert output["embeddings"].shape == (batch_size, 10, 512)

    def test_single_contact_active(self):
        """
        Purpose:
            Test with only one contact point active.
            
        Workflow:
            1. Create encoder.
            2. Activate one contact.
            3. Verify attention mask reflects activity.
            
        ToDo:
            - None
        """
        from src.encoders.touch_encoder import TouchEncoder
        encoder = TouchEncoder(
            max_contacts=10,
            embedding_dim=512
        )

        batch_size = 2
        contact_active = torch.zeros(batch_size, 10, dtype=torch.bool)
        contact_active[:, 0] = True  # Only first contact active

        # Required inputs
        contact_points = torch.randint(0, 10, (batch_size, 10))
        surface_types = torch.randint(0, 8, (batch_size, 10))
        contact_forces = torch.randn(batch_size, 10, 1)  # Scalar force
        contact_positions = torch.randn(batch_size, 10, 3)

        output = encoder(
            contact_active=contact_active,
            contact_points=contact_points,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            surface_types=surface_types
        )
        assert output["embeddings"].shape == (batch_size, 10, 512)
        # Attention mask should reflect active contacts
        assert output["attention_mask"][:, 0].all()

    def test_various_max_contacts(self):
        """
        Purpose:
            Test with different max_contacts values.
            
        Workflow:
            1. Iterate max_contacts.
            2. Create encoder.
            3. Verify output shape.
            
        ToDo:
            - None
        """
        from src.encoders.touch_encoder import TouchEncoder
        for max_contacts in [5, 10, 20, 50]:
            encoder = TouchEncoder(
                max_contacts=max_contacts,
                embedding_dim=512
            )

            contact_active = torch.ones(2, max_contacts, dtype=torch.bool)
            contact_points = torch.randint(0, 10, (2, max_contacts))
            surface_types = torch.randint(0, 8, (2, max_contacts))
            contact_forces = torch.randn(2, max_contacts, 1)
            contact_positions = torch.randn(2, max_contacts, 3)

            output = encoder(
                contact_active=contact_active,
                contact_points=contact_points,
                contact_forces=contact_forces,
                contact_positions=contact_positions,
                surface_types=surface_types
            )
            assert output["embeddings"].shape == (2, max_contacts, 512)

    def test_zero_forces(self):
        """
        Purpose:
            Test with zero force vectors.
            
        Workflow:
            1. Create inputs with zero force.
            2. Run forward.
            3. Verify no NaNs.
            
        ToDo:
            - None
        """
        from src.encoders.touch_encoder import TouchEncoder
        encoder = TouchEncoder(
            max_contacts=10,
            embedding_dim=512
        )

        contact_active = torch.ones(2, 10, dtype=torch.bool)
        contact_points = torch.randint(0, 10, (2, 10))
        surface_types = torch.randint(0, 8, (2, 10))
        contact_forces = torch.zeros(2, 10, 1)  # Zero force
        contact_positions = torch.randn(2, 10, 3)

        output = encoder(
            contact_active=contact_active,
            contact_points=contact_points,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            surface_types=surface_types
        )
        assert output["embeddings"].shape == (2, 10, 512)
        assert not torch.isnan(output["embeddings"]).any()




class TestEncoderOutputConsistency:
    """Test consistency across encoders."""

    def test_all_encoders_have_get_output_dim(self):
        """
        Purpose:
            Verify all encoders implement get_output_dim method.
            
        Workflow:
            1. List encoders.
            2. Check hasattr 'get_output_dim'.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.encoders.external_voice_encoder import ExternalVoiceEncoder

        encoders = [
            InternalVoiceEncoder(vocab_size=1000, embedding_dim=512),
            ExternalVoiceEncoder(vocab_size=1000, embedding_dim=512),
            AudioEncoder(embedding_dim=512),
            VisualEncoder(embedding_dim=512, use_stereo=False),
            ProprioceptionEncoder(embedding_dim=512),
            TouchEncoder(embedding_dim=512),
        ]

        for encoder in encoders:
            assert hasattr(encoder, 'get_output_dim')
            assert encoder.get_output_dim() == 512

    def test_all_encoders_return_attention_mask(self):
        """
        Purpose:
            Verify all encoders return attention masks.
            
        Workflow:
            1. List encoders.
            2. Run forward.
            3. Verify 'attention_mask' in output.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder
        from src.encoders.external_voice_encoder import ExternalVoiceEncoder

        # Internal voice
        encoder1 = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        output1 = encoder1(torch.randint(0, 1000, (2, 10)))
        assert "attention_mask" in output1

        # External voice
        encoder2 = ExternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        output2 = encoder2(torch.randint(0, 1000, (2, 10)))
        assert "attention_mask" in output2

        # Audio
        encoder3 = AudioEncoder(embedding_dim=512)
        output3 = encoder3(torch.randn(2, 16000))
        assert "attention_mask" in output3

        # Visual
        encoder4 = VisualEncoder(embedding_dim=512, use_stereo=False)
        output4 = encoder4(torch.randn(2, 3, 224, 224))
        assert "attention_mask" in output4

        # Proprioception
        encoder5 = ProprioceptionEncoder(embedding_dim=512)
        output5 = encoder5(torch.randn(2, 10, 24, 3), torch.randn(2, 10, 24, 4))
        assert "attention_mask" in output5

        # Touch
        encoder6 = TouchEncoder(embedding_dim=512)
        
        contact_active = torch.ones(2, 10, dtype=torch.bool)
        contact_points = torch.randint(0, 10, (2, 10))
        surface_types = torch.randint(0, 8, (2, 10))
        contact_forces = torch.randn(2, 10, 1)
        contact_positions = torch.randn(2, 10, 3)

        output6 = encoder6(
            contact_active=contact_active,
            contact_points=contact_points,
            contact_forces=contact_forces,
            contact_positions=contact_positions,
            surface_types=surface_types
        )
        assert "attention_mask" in output6

    def test_attention_mask_dtype_and_values(self):
        """
        Purpose:
            Test that attention masks have correct dtype and values.
            
        Workflow:
            1. Run encoder.
            2. Check mask dtype is long.
            3. Check mask valus are 0/1.
            
        ToDo:
            - None
        """
        from src.encoders.internal_voice_encoder import InternalVoiceEncoder

        encoder = InternalVoiceEncoder(vocab_size=1000, embedding_dim=512)
        output = encoder(torch.randint(0, 1000, (2, 10)))

        mask = output["attention_mask"]
        assert mask.dtype == torch.long
        assert mask.shape == (2, 10)
        # Should be 0s or 1s
        assert ((mask == 0) | (mask == 1)).all()
