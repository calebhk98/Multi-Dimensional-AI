"""
VR Integration Protocol

Defines message formats for communication between Python backend and Unity VR client.

Purpose:
	Provides dataclasses and serialization utilities for VR input/output messages.
	Uses JSON for simplicity and debuggability.

Workflow:
	1. Unity sends VRInputMessage with sensor data
	2. Python processes and returns VROutputMessage with actuator commands
	3. All messages are length-prefixed for reliable framing

ToDo:
	- Add MessagePack option for better performance
	- Add compression for vision data
"""

import json
import struct
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# Length prefix format: 4-byte unsigned int (big-endian)
LENGTH_PREFIX_FORMAT = ">I"
LENGTH_PREFIX_SIZE = struct.calcsize(LENGTH_PREFIX_FORMAT)


@dataclass
class VRInputMessage:
	"""
	Input message from Unity VR to Python backend.

	Contains sensor data from VR environment:
	- Vision: Stereo camera frames (base64 encoded JPEG/PNG)
	- Hearing: Audio buffer samples
	- Touch: Haptic contact data
	- Proprioception: Joint positions and rotations

	Args:
		timestamp: Message timestamp in milliseconds
		vision_left: Base64 encoded left eye image (optional)
		vision_right: Base64 encoded right eye image (optional)
		audio_samples: List of audio sample floats
		touch_contacts: List of touch contact dicts
		joint_positions: List of joint position dicts
		joint_rotations: List of joint rotation dicts
	"""
	timestamp: float
	vision_left: Optional[str] = None
	vision_right: Optional[str] = None
	audio_samples: List[float] = field(default_factory=list)
	touch_contacts: List[Dict[str, Any]] = field(default_factory=list)
	joint_positions: List[Dict[str, float]] = field(default_factory=list)
	joint_rotations: List[Dict[str, float]] = field(default_factory=list)

	def to_json(self) -> str:
		"""
		Serialize message to JSON string.

		Returns:
			str: JSON representation of message.
		"""
		return json.dumps(asdict(self))

	@classmethod
	def from_json(cls, json_str: str) -> "VRInputMessage":
		"""
		Deserialize message from JSON string.

		Args:
			json_str: JSON string to parse.

		Returns:
			VRInputMessage: Deserialized message.

		Raises:
			ValueError: If JSON is invalid or missing required fields.
		"""
		try:
			data = json.loads(json_str)
			return cls(**data)
		except (json.JSONDecodeError, TypeError) as e:
			logger.error(f"Failed to parse VRInputMessage: {e}")
			raise ValueError(f"Invalid VRInputMessage JSON: {e}") from e


@dataclass
class VROutputMessage:
	"""
	Output message from Python backend to Unity VR.

	Contains actuator commands for VR creature:
	- Vocalizations: Audio tokens or raw audio bytes
	- Body Control: Joint rotation targets and blend shapes

	Args:
		timestamp: Message timestamp in milliseconds
		vocalization_tokens: List of audio codec tokens
		vocalization_audio: Base64 encoded audio bytes (optional)
		joint_rotations: List of joint rotation target dicts
		blend_shapes: List of blend shape weight dicts
		eye_params: Eye movement parameters
	"""
	timestamp: float
	vocalization_tokens: List[int] = field(default_factory=list)
	vocalization_audio: Optional[str] = None
	joint_rotations: List[Dict[str, float]] = field(default_factory=list)
	blend_shapes: List[Dict[str, float]] = field(default_factory=list)
	eye_params: Dict[str, float] = field(default_factory=dict)

	def to_json(self) -> str:
		"""
		Serialize message to JSON string.

		Returns:
			str: JSON representation of message.
		"""
		return json.dumps(asdict(self))

	@classmethod
	def from_json(cls, json_str: str) -> "VROutputMessage":
		"""
		Deserialize message from JSON string.

		Args:
			json_str: JSON string to parse.

		Returns:
			VROutputMessage: Deserialized message.

		Raises:
			ValueError: If JSON is invalid or missing required fields.
		"""
		try:
			data = json.loads(json_str)
			return cls(**data)
		except (json.JSONDecodeError, TypeError) as e:
			logger.error(f"Failed to parse VROutputMessage: {e}")
			raise ValueError(f"Invalid VROutputMessage JSON: {e}") from e


def frame_message(data: bytes) -> bytes:
	"""
	Add length prefix to message for reliable framing.

	Args:
		data: Raw message bytes to frame.

	Returns:
		bytes: Length-prefixed message.
	"""
	length = len(data)
	prefix = struct.pack(LENGTH_PREFIX_FORMAT, length)
	return prefix + data


def unframe_message(data: bytes) -> tuple[int, bytes]:
	"""
	Extract length prefix and message content.

	Args:
		data: Raw bytes containing length prefix and message.

	Returns:
		tuple: (message_length, message_bytes) or (0, b'') if incomplete.

	Raises:
		ValueError: If data is too short for length prefix.
	"""
	if len(data) < LENGTH_PREFIX_SIZE:
		return 0, b''

	length = struct.unpack(LENGTH_PREFIX_FORMAT, data[:LENGTH_PREFIX_SIZE])[0]
	return length, data[LENGTH_PREFIX_SIZE:]
