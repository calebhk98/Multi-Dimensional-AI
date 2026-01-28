"""
Synthetic data generator for mocking VR inputs.
Produces random tensors with correct shapes and ranges for testing.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class SyntheticDataGenerator:
	"""
	Generates synthetic multi-modal data for testing and baseline training.
	
	Produces tensors matching the shapes expected by the encoders:
	- Voice: [seq_len] token IDs
	- Audio: [samples] waveform
	- Vision: [channels, height, width] images
	- Proprioception: [temporal_window, joints, features]
	- Touch: [num_contacts, features]
	"""
	
	def __init__(
		self,
		# Text params
		vocab_size: int = 50257,
		max_seq_length: int = 64,
		
		# Audio params
		sample_rate: int = 16000,
		audio_duration: float = 1.0,  # seconds
		
		# Vision params
		image_size: int = 224,
		
		# Proprioception params
		num_joints: int = 24,
		temporal_window: int = 10,
		
		# Touch params
		max_contacts: int = 5,
		num_contact_points_total: int = 10,
		surface_types: int = 8,
		
		# Audio token params
		codebook_size: int = 1024,
	):
		"""
		Initialize synthetic data generator.

		Args:
			vocab_size: Size of vocabulary for text tokens
			max_seq_length: Maximum sequence length for text
			sample_rate: Audio sample rate in Hz
			audio_duration: Duration of audio clips in seconds
			image_size: Height/Width of images (square)
			num_joints: Number of body joints for proprioception
			temporal_window: Size of temporal window for proprioception sequences
			max_contacts: Maximum number of simultaneous touch contacts
			num_contact_points_total: Total number of possible contact locations on body
			surface_types: Number of distinct surface material types
		"""
		self.vocab_size = vocab_size
		self.max_seq_length = max_seq_length
		self.sample_rate = sample_rate
		self.audio_duration = audio_duration
		self.image_size = image_size
		self.num_joints = num_joints
		self.temporal_window = temporal_window
		self.max_contacts = max_contacts
		self.num_contact_points_total = num_contact_points_total
		self.num_contact_points_total = num_contact_points_total
		self.surface_types = surface_types
		self.codebook_size = codebook_size
		
	def generate_text_tokens(self, seq_len: int = None) -> torch.Tensor:
		"""
		Generate random token IDs.

		Args:
			seq_len: Length of sequence to generate. If None, uses random length.
		
		Returns:
			Tensor of token IDs [seq_len]
		"""
		if seq_len is None:
			# Use max_seq_length for stable batching
			seq_len = self.max_seq_length
		return torch.randint(0, self.vocab_size, (seq_len,))

	def generate_audio_waveform(self) -> torch.Tensor:
		"""
		Generate random audio waveform (white noise).

		Returns:
			Audio waveform tensor [1, num_samples]
		"""
		num_samples = int(self.sample_rate * self.audio_duration)
		return torch.randn(1, num_samples) * 0.1

	def generate_image(self) -> torch.Tensor:
		"""
		Generate random image tensor (RGB).

		Returns:
			Image tensor [3, image_size, image_size]
		"""
		return torch.rand(3, self.image_size, self.image_size)

	def generate_proprioception(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Generate random proprioception sequences.
		Returns: (positions, rotations)
		"""
		# Positions: [temporal_window, num_joints, 3] (xyz)
		positions = torch.randn(self.temporal_window, self.num_joints, 3)
		
		# Rotations: [temporal_window, num_joints, 4] (quaternions)
		# Generate random quaternions and normalize
		rotations = torch.randn(self.temporal_window, self.num_joints, 4)
		rotations = rotations / rotations.norm(dim=-1, keepdim=True)
		
		return positions, rotations

	def generate_touch(self) -> Dict[str, torch.Tensor]:
		"""
		Generate random touch data.

		Returns:
			Dictionary containing touch sensors:
			- contact_active: [max_contacts]
			- contact_points: [max_contacts]
			- contact_forces: [max_contacts, 1]
			- contact_positions: [max_contacts, 3]
			- surface_types: [max_contacts]
			- temperatures: [max_contacts, 1]
		"""
		# Random number of active contacts
		num_active = np.random.randint(0, self.max_contacts + 1)
		
		# Masks [max_contacts]
		contact_active = torch.zeros(self.max_contacts, dtype=torch.bool)
		contact_active[:num_active] = True
		
		# Contact points (indices)
		contact_points = torch.randint(0, self.num_contact_points_total, (self.max_contacts,))
		
		# Forces [max_contacts, 1]
		contact_forces = torch.rand(self.max_contacts, 1)
		
		# Positions relative to body part [max_contacts, 3]
		contact_positions = torch.randn(self.max_contacts, 3) * 0.05
		
		# Surface types
		surface_types = torch.randint(0, self.surface_types, (self.max_contacts,))

		# Temperatures (optional, but good to have)
		temperatures = torch.rand(self.max_contacts, 1)
		
		return {
			"contact_active": contact_active,
			"contact_points": contact_points,
			"contact_forces": contact_forces,
			"contact_positions": contact_positions,
			"surface_types": surface_types,
			"temperatures": temperatures
		}

	def generate_sample(self) -> Dict[str, torch.Tensor]:
		"""
		Generate a complete multi-modal training sample.

		Returns:
			Dictionary containing:
			- inputs: Dict of input tensors
			- targets: Dict of target tensors
		"""
		# Inputs
		inputs = {
			"internal_voice_tokens": self.generate_text_tokens(),
			"external_voice_tokens": self.generate_text_tokens(),
			"audio_waveform": self.generate_audio_waveform(),
			"left_eye_image": self.generate_image(),
			"right_eye_image": self.generate_image(),
		}
		
		pos, rot = self.generate_proprioception()
		inputs["joint_positions"] = pos
		inputs["joint_rotations"] = rot
		
		touch_data = self.generate_touch()
		# Remove batch dim added in helper for simpler dataset logic if needed, 
		# but helper returned 1D/2D tensors. Let's fix helper return in next thought if needed.
		# Fixed touch helper return to be standard [dim] tensors.
		
		# Touch data needs to be unpacked properly later, strictly matching encoder args
		# Encoder expects batch dim. Dataset usually returns single items, DataLoader batches them.
		# So here we return single items.
		inputs["touch_data"] = {
			k: v.squeeze(0) if k == "contact_active" else v 
			for k, v in touch_data.items()
		}

		# Targets (Auto-regressive: predict next token, or reconstruct)
		# For simplicity, targets often look like inputs shifted or specific ground truths.
		# Here we generate separate random targets for demonstration.
		
		targets = {
			"internal_text": self.generate_text_tokens(), # Predict next thought
			"external_text": self.generate_text_tokens(), # Predict next speech
            "audio": torch.randint(0, self.codebook_size, (32,)), # Discrete audio tokens
            "animation": {
                "rotations": torch.randn(self.temporal_window, self.num_joints, 4), # Next pose
                "blend_shapes": torch.rand(self.temporal_window, 51),
                "eye_params": torch.rand(self.temporal_window, 8)
            }
		}
		
		return {"inputs": inputs, "targets": targets}
