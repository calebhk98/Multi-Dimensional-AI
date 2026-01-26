"""
VR Output Streamer

Converts model outputs to VR-friendly format for Unity.

Purpose:
	Transforms decoder outputs (vocalizations, body control) into
	VROutputMessage format for transmission to VR client.

Workflow:
	1. Receive model outputs from inference
	2. Extract vocalization tokens/audio from audio decoder
	3. Extract joint rotations and blend shapes from animation decoder
	4. Package into VROutputMessage

ToDo:
	- Add audio codec integration for vocalization
	- Add interpolation for smooth animation
	- Add prediction smoothing to reduce jitter
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch

from src.vr_integration.protocol import VROutputMessage

logger = logging.getLogger(__name__)


# Default output parameters
DEFAULT_NUM_JOINTS = 24
DEFAULT_NUM_BLEND_SHAPES = 51


@dataclass
class OutputStreamerConfig:
	"""
	Configuration for output streaming.

	Args:
		num_joints: Number of body joints.
		num_blend_shapes: Number of facial blend shapes.
	"""
	num_joints: int = DEFAULT_NUM_JOINTS
	num_blend_shapes: int = DEFAULT_NUM_BLEND_SHAPES


class VROutputStreamer:
	"""
	Streams model outputs to VR format.

	Converts model decoder outputs into VROutputMessage for
	transmission to Unity VR client.

	Args:
		config: Streaming configuration.
	"""

	def __init__(self, config: Optional[OutputStreamerConfig] = None):
		"""
		Initialize output streamer.

		Args:
			config: Streaming configuration (uses defaults if None).
		"""
		self.config = config or OutputStreamerConfig()

	def stream(
		self,
		model_outputs: Dict[str, Any],
		timestamp: float
	) -> VROutputMessage:
		"""
		Convert model outputs to VROutputMessage.

		Args:
			model_outputs: Dictionary of model decoder outputs.
			timestamp: Current timestamp in milliseconds.

		Returns:
			VROutputMessage: Formatted output message.
		"""
		# Extract vocalization data
		vocalization_tokens = self._extract_vocalization_tokens(
			model_outputs.get("audio")
		)

		# Extract body control data
		joint_rotations = self._extract_joint_rotations(
			model_outputs.get("animation", {}).get("joint_rotations")
		)
		blend_shapes = self._extract_blend_shapes(
			model_outputs.get("animation", {}).get("blend_shapes")
		)
		eye_params = self._extract_eye_params(
			model_outputs.get("animation", {}).get("eye_params")
		)

		return VROutputMessage(
			timestamp=timestamp,
			vocalization_tokens=vocalization_tokens,
			vocalization_audio=None,  # Future: encode audio bytes
			joint_rotations=joint_rotations,
			blend_shapes=blend_shapes,
			eye_params=eye_params,
		)

	def _extract_vocalization_tokens(
		self, audio_output: Optional[torch.Tensor]
	) -> list:
		"""
		Extract vocalization tokens from audio decoder output.

		Purpose:
			Converts audio decoder tensor output to list of token IDs for VR.

		Workflow:
			1. Extract first batch from tensor
			2. Convert to CPU and list
			3. Handle nested lists

		ToDo:
			Add token validation

		Args:
			audio_output: Audio decoder output tensor.

		Returns:
			list: Token IDs for vocalization.
		"""
		if audio_output is None:
			return []

		try:
			# Assume audio_output is [batch, seq] of token IDs
			if isinstance(audio_output, torch.Tensor):
				# Take first batch, convert to list
				tokens = audio_output[0].detach().cpu().tolist()
				# Handle nested lists
				if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
					tokens = tokens[0]
				return [int(t) for t in tokens]
			return []

		except Exception as e:
			logger.error(f"Failed to extract vocalization: {e}")
			return []

	def _extract_joint_rotations(
		self, joint_rotations: Optional[torch.Tensor]
	) -> list:
		"""
		Extract joint rotations as quaternion dicts.

		Purpose:
			Converts joint rotation tensor to VR-friendly list of quaternion dicts.

		Workflow:
			1. Extract last timestep from tensor
			2. Convert each joint to x,y,z,w dict
			3. Include joint_id for mapping

		ToDo:
			Add quaternion normalization validation

		Args:
			joint_rotations: Joint rotation tensor [batch, seq, joints, 4].

		Returns:
			list: List of joint rotation dicts.
		"""
		if joint_rotations is None:
			return []

		try:
			if isinstance(joint_rotations, torch.Tensor):
				# Take first batch, last timestep
				rotations = joint_rotations[0, -1].detach().cpu()
				result = []
				for i in range(min(rotations.shape[0], self.config.num_joints)):
					q = rotations[i]
					result.append({
						"joint_id": i,
						"x": float(q[0]),
						"y": float(q[1]),
						"z": float(q[2]),
						"w": float(q[3]),
					})
				return result
			return []

		except Exception as e:
			logger.error(f"Failed to extract joint rotations: {e}")
			return []

	def _extract_blend_shapes(
		self, blend_shapes: Optional[torch.Tensor]
	) -> list:
		"""
		Extract blend shape weights.

		Purpose:
			Converts blend shape tensor to VR-friendly list of weight dicts.

		Workflow:
			1. Extract last timestep from tensor
			2. Convert each shape to shape_id and weight dict

		ToDo:
			Add weight clamping to [0, 1]

		Args:
			blend_shapes: Blend shape tensor [batch, seq, shapes].

		Returns:
			list: List of blend shape weight dicts.
		"""
		if blend_shapes is None:
			return []

		try:
			if isinstance(blend_shapes, torch.Tensor):
				# Take first batch, last timestep
				shapes = blend_shapes[0, -1].detach().cpu()
				result = []
				for i in range(min(shapes.shape[0], self.config.num_blend_shapes)):
					result.append({
						"shape_id": i,
						"weight": float(shapes[i]),
					})
				return result
			return []

		except Exception as e:
			logger.error(f"Failed to extract blend shapes: {e}")
			return []

	def _extract_eye_params(
		self, eye_params: Optional[torch.Tensor]
	) -> dict:
		"""
		Extract eye movement parameters.

		Purpose:
			Converts eye parameter tensor to VR-friendly dict.

		Workflow:
			1. Extract last timestep from tensor
			2. Map indices to named eye parameters
			3. Return structured dict

		ToDo:
			Add parameter range validation

		Args:
			eye_params: Eye parameter tensor [batch, seq, 8].

		Returns:
			dict: Eye parameters dict.
		"""
		if eye_params is None:
			return {}

		try:
			if isinstance(eye_params, torch.Tensor):
				# Take first batch, last timestep
				params = eye_params[0, -1].detach().cpu()
				return {
					"left_horizontal": float(params[0]) if len(params) > 0 else 0.0,
					"left_vertical": float(params[1]) if len(params) > 1 else 0.0,
					"right_horizontal": float(params[2]) if len(params) > 2 else 0.0,
					"right_vertical": float(params[3]) if len(params) > 3 else 0.0,
					"left_blink": float(params[4]) if len(params) > 4 else 0.0,
					"right_blink": float(params[5]) if len(params) > 5 else 0.0,
					"left_pupil": float(params[6]) if len(params) > 6 else 0.5,
					"right_pupil": float(params[7]) if len(params) > 7 else 0.5,
				}
			return {}

		except Exception as e:
			logger.error(f"Failed to extract eye params: {e}")
			return {}
