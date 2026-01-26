"""
VR Input Processor

Converts raw VR sensor data to encoder-ready tensor format.

Purpose:
	Transforms VRInputMessage data into the exact tensor formats expected
	by the Multi-Dimensional AI encoders (vision, audio, touch, proprioception).

Workflow:
	1. Receive VRInputMessage from server
	2. Decode vision images from base64 to tensor
	3. Convert audio samples to spectrogram/waveform tensor
	4. Parse touch contacts to touch encoder format
	5. Parse joint data to proprioception encoder format

ToDo:
	- Add caching for repeated processing
	- Optimize image decoding performance
	- Add data validation and error recovery
"""

import base64
import io
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch

from src.vr_integration.protocol import VRInputMessage

logger = logging.getLogger(__name__)


# Default processing parameters
DEFAULT_IMAGE_SIZE = 224
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_JOINTS = 24
DEFAULT_MAX_CONTACTS = 10


@dataclass
class InputProcessorConfig:
	"""
	Configuration for input processing.

	Args:
		image_size: Target image size for vision encoder.
		sample_rate: Audio sample rate in Hz.
		num_joints: Number of body joints.
		max_contacts: Maximum touch contacts.
		device: Torch device for tensors.
	"""
	image_size: int = DEFAULT_IMAGE_SIZE
	sample_rate: int = DEFAULT_SAMPLE_RATE
	num_joints: int = DEFAULT_NUM_JOINTS
	max_contacts: int = DEFAULT_MAX_CONTACTS
	device: str = "cpu"


class VRInputProcessor:
	"""
	Processes raw VR input data into encoder-ready tensors.

	Handles conversion of various sensor data formats from VR
	into the specific tensor formats expected by each encoder.

	Args:
		config: Processing configuration.
	"""

	def __init__(self, config: Optional[InputProcessorConfig] = None):
		"""
		Initialize input processor.

		Args:
			config: Processing configuration (uses defaults if None).
		"""
		self.config = config or InputProcessorConfig()

	def process(self, message: VRInputMessage) -> Dict[str, Any]:
		"""
		Process VRInputMessage into encoder-ready format.

		Args:
			message: Input message from VR.

		Returns:
			dict: Encoder-ready data with keys:
				- left_eye_image: Tensor [1, 3, H, W] or None
				- right_eye_image: Tensor [1, 3, H, W] or None
				- audio_waveform: Tensor [1, samples] or None
				- touch_data: Dict with touch tensors or None
				- joint_positions: Tensor [1, T, J, 3] or None
				- joint_rotations: Tensor [1, T, J, 4] or None
		"""
		result = {
			"left_eye_image": self._process_vision(message.vision_left),
			"right_eye_image": self._process_vision(message.vision_right),
			"audio_waveform": self._process_audio(message.audio_samples),
			"touch_data": self._process_touch(message.touch_contacts),
			"joint_positions": self._process_proprioception_positions(
				message.joint_positions
			),
			"joint_rotations": self._process_proprioception_rotations(
				message.joint_rotations
			),
		}
		return result

	def _process_vision(self, image_b64: Optional[str]) -> Optional[torch.Tensor]:
		"""
		Decode base64 image to tensor.

		Purpose:
			Converts base64-encoded image from VR to PyTorch tensor for vision encoder.

		Workflow:
			1. Decode base64 to bytes
			2. Use PIL to load image (or return placeholder)
			3. Resize and normalize to tensor

		ToDo:
			Optimize decoding for stereo pairs

		Args:
			image_b64: Base64 encoded image string.

		Returns:
			torch.Tensor: Image tensor [1, 3, H, W] or None.
		"""
		if not image_b64:
			return None

		try:
			# Decode base64 to bytes
			image_bytes = base64.b64decode(image_b64)

			# Try to use PIL if available, otherwise create placeholder
			try:
				from PIL import Image
				import torchvision.transforms as T

				image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
				transform = T.Compose([
					T.Resize((self.config.image_size, self.config.image_size)),
					T.ToTensor(),
				])
				tensor = transform(image).unsqueeze(0)
				return tensor.to(self.config.device)
			except ImportError:
				logger.warning("PIL/torchvision not available, using placeholder")
				return torch.zeros(
					1, 3, self.config.image_size, self.config.image_size,
					device=self.config.device
				)

		except Exception as e:
			logger.error(f"Failed to process vision data: {e}")
			return None

	def _process_audio(self, samples: list) -> Optional[torch.Tensor]:
		"""
		Convert audio samples to waveform tensor.

		Purpose:
			Converts raw audio sample list from VR to PyTorch tensor for audio encoder.

		Workflow:
			1. Convert list to tensor
			2. Add batch dimension

		ToDo:
			Add sample rate validation

		Args:
			samples: List of audio sample floats.

		Returns:
			torch.Tensor: Audio tensor [1, samples] or None.
		"""
		if not samples:
			return None

		try:
			tensor = torch.tensor(samples, dtype=torch.float32, device=self.config.device)
			return tensor.unsqueeze(0)  # Add batch dimension
		except Exception as e:
			logger.error(f"Failed to process audio data: {e}")
			return None

	def _process_touch(self, contacts: list) -> Optional[Dict[str, torch.Tensor]]:
		"""
		Convert touch contacts to encoder format.

		Purpose:
			Converts touch contact list from VR to structured tensors for touch encoder.

		Workflow:
			1. Initialize zero tensors for max contacts
			2. Fill in data from contact dicts
			3. Mark active contacts

		ToDo:
			Add contact validation and filtering

		Args:
			contacts: List of contact dicts with position, normal, force.

		Returns:
			dict: Touch data tensors or None.
		"""
		if not contacts:
			return None

		try:
			max_c = self.config.max_contacts
			device = self.config.device

			positions = torch.zeros(1, max_c, 3, device=device)
			normals = torch.zeros(1, max_c, 3, device=device)
			forces = torch.zeros(1, max_c, 3, device=device)
			active = torch.zeros(1, max_c, dtype=torch.bool, device=device)

			for i, contact in enumerate(contacts[:max_c]):
				self._extract_contact_data(contact, i, positions, normals, forces, active)

			return {
				"positions": positions,
				"normals": normals,
				"forces": forces,
				"contact_active": active,
			}

		except Exception as e:
			logger.error(f"Failed to process touch data: {e}")
			return None

	def _extract_contact_data(self, contact, index, positions, normals, forces, active):
		"""
		Helper to extract contact data into tensors to reduce nesting.
		
		Args:
			contact: Contact dict.
			index: Contact index.
			positions: Positions tensor.
			normals: Normals tensor.
			forces: Forces tensor.
			active: Active mask tensor.
		"""
		if "position" in contact:
			pos = contact["position"]
			positions[0, index] = torch.tensor(
				[pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
			)
		if "normal" in contact:
			norm = contact["normal"]
			normals[0, index] = torch.tensor(
				[norm.get("x", 0), norm.get("y", 0), norm.get("z", 0)]
			)
		if "force" in contact:
			force = contact["force"]
			forces[0, index] = torch.tensor(
				[force.get("x", 0), force.get("y", 0), force.get("z", 0)]
			)
		active[0, index] = True



	def _process_proprioception_positions(
		self, joint_positions: list
	) -> Optional[torch.Tensor]:
		"""
		Convert joint positions to tensor.

		Purpose:
			Converts joint position list from VR to tensor for proprioception encoder.

		Workflow:
			1. Initialize zero tensor for all joints
			2. Fill in x,y,z positions from list

		ToDo:
			Add joint mapping validation

		Args:
			joint_positions: List of joint position dicts.

		Returns:
			torch.Tensor: Positions tensor [1, 1, J, 3] or None.
		"""
		if not joint_positions:
			return None

		try:
			num_joints = min(len(joint_positions), self.config.num_joints)
			positions = torch.zeros(
				1, 1, self.config.num_joints, 3, device=self.config.device
			)

			for i, joint in enumerate(joint_positions[:num_joints]):
				positions[0, 0, i] = torch.tensor([
					joint.get("x", 0),
					joint.get("y", 0),
					joint.get("z", 0),
				])

			return positions

		except Exception as e:
			logger.error(f"Failed to process joint positions: {e}")
			return None

	def _process_proprioception_rotations(
		self, joint_rotations: list
	) -> Optional[torch.Tensor]:
		"""
		Convert joint rotations to tensor (quaternions).

		Purpose:
			Converts joint rotation list from VR to quaternion tensor for proprioception encoder.

		Workflow:
			1. Initialize tensor with identity quaternions
			2. Fill in x,y,z,w components from list

		ToDo:
			Add quaternion normalization

		Args:
			joint_rotations: List of joint rotation dicts.

		Returns:
			torch.Tensor: Rotations tensor [1, 1, J, 4] or None.
		"""
		if not joint_rotations:
			return None

		try:
			num_joints = min(len(joint_rotations), self.config.num_joints)
			rotations = torch.zeros(
				1, 1, self.config.num_joints, 4, device=self.config.device
			)
			# Initialize to identity quaternion (w=1)
			rotations[:, :, :, 3] = 1.0

			for i, joint in enumerate(joint_rotations[:num_joints]):
				rotations[0, 0, i] = torch.tensor([
					joint.get("x", 0),
					joint.get("y", 0),
					joint.get("z", 0),
					joint.get("w", 1),
				])

			return rotations

		except Exception as e:
			logger.error(f"Failed to process joint rotations: {e}")
			return None
