"""
VR Integration Test Suite

Tests for VR communication infrastructure including protocol, server,
input processing, and output streaming.

Purpose:
	Verify correct functionality of all VR integration components
	without requiring actual VR hardware.

Workflow:
	1. Test protocol serialization/deserialization
	2. Test server connection handling
	3. Test input processor conversions
	4. Test output streamer formatting
	5. Integration test with mock client

ToDo:
	- Add performance benchmarks
	- Add stress tests for high-frequency messages
"""

import json
import time
import socket
import struct
import threading
import pytest
import torch
from typing import Optional

from src.vr_integration.protocol import (
	VRInputMessage,
	VROutputMessage,
	frame_message,
	unframe_message,
	LENGTH_PREFIX_SIZE,
)
from src.vr_integration.server import VRServer, VRServerConfig
from src.vr_integration.input_processor import VRInputProcessor, InputProcessorConfig
from src.vr_integration.output_streamer import VROutputStreamer, OutputStreamerConfig


# ============================================================================
# Protocol Tests
# ============================================================================

class TestVRInputMessage:
	"""Tests for VRInputMessage dataclass."""

	def test_creation_with_defaults(self):
		"""
		Purpose:
			Verify VRInputMessage can be created with minimal args.

		Workflow:
			1. Create message with only timestamp
			2. Verify all optional fields have defaults

		ToDo:
			None
		"""
		msg = VRInputMessage(timestamp=1000.0)

		assert msg.timestamp == 1000.0
		assert msg.vision_left is None
		assert msg.vision_right is None
		assert msg.audio_samples == []
		assert msg.touch_contacts == []
		assert msg.joint_positions == []
		assert msg.joint_rotations == []

	def test_to_json_serialization(self):
		"""
		Purpose:
			Verify message serializes to valid JSON.

		Workflow:
			1. Create message with sample data
			2. Serialize to JSON
			3. Verify JSON structure

		ToDo:
			None
		"""
		msg = VRInputMessage(
			timestamp=1234.5,
			audio_samples=[0.1, 0.2, 0.3],
		)
		json_str = msg.to_json()
		data = json.loads(json_str)

		assert data["timestamp"] == 1234.5
		assert data["audio_samples"] == [0.1, 0.2, 0.3]

	def test_from_json_deserialization(self):
		"""
		Purpose:
			Verify message deserializes from JSON correctly.

		Workflow:
			1. Create JSON string
			2. Deserialize to message
			3. Verify fields match

		ToDo:
			None
		"""
		json_str = '{"timestamp": 5678.0, "vision_left": "abc123"}'
		msg = VRInputMessage.from_json(json_str)

		assert msg.timestamp == 5678.0
		assert msg.vision_left == "abc123"

	def test_from_json_round_trip(self):
		"""
		Purpose:
			Verify serialize → deserialize preserves data.

		Workflow:
			1. Create message with all fields
			2. Serialize and deserialize
			3. Verify equality

		ToDo:
			None
		"""
		original = VRInputMessage(
			timestamp=100.0,
			vision_left="left_img",
			vision_right="right_img",
			audio_samples=[1.0, 2.0],
			touch_contacts=[{"position": {"x": 1}}],
			joint_positions=[{"x": 0, "y": 1, "z": 2}],
			joint_rotations=[{"x": 0, "y": 0, "z": 0, "w": 1}],
		)
		json_str = original.to_json()
		restored = VRInputMessage.from_json(json_str)

		assert restored.timestamp == original.timestamp
		assert restored.vision_left == original.vision_left
		assert restored.audio_samples == original.audio_samples

	def test_from_json_invalid_raises_error(self):
		"""
		Purpose:
			Verify invalid JSON raises ValueError.

		Workflow:
			1. Pass invalid JSON
			2. Assert ValueError raised

		ToDo:
			None
		"""
		with pytest.raises(ValueError):
			VRInputMessage.from_json("not valid json")


class TestVROutputMessage:
	"""Tests for VROutputMessage dataclass."""

	def test_creation_with_defaults(self):
		"""
		Purpose:
			Verify VROutputMessage can be created with minimal args.

		Workflow:
			1. Create message with only timestamp
			2. Verify defaults

		ToDo:
			None
		"""
		msg = VROutputMessage(timestamp=2000.0)

		assert msg.timestamp == 2000.0
		assert msg.vocalization_tokens == []
		assert msg.vocalization_audio is None
		assert msg.joint_rotations == []
		assert msg.blend_shapes == []
		assert msg.eye_params == {}

	def test_to_json_serialization(self):
		"""
		Purpose:
			Verify output message serializes correctly.

		Workflow:
			1. Create message with vocalization tokens
			2. Serialize to JSON
			3. Verify structure

		ToDo:
			None
		"""
		msg = VROutputMessage(
			timestamp=3000.0,
			vocalization_tokens=[1, 2, 3, 4],
		)
		json_str = msg.to_json()
		data = json.loads(json_str)

		assert data["timestamp"] == 3000.0
		assert data["vocalization_tokens"] == [1, 2, 3, 4]

	def test_round_trip(self):
		"""
		Purpose:
			Verify output message round-trip preservation.

		Workflow:
			1. Create message with full data
			2. Serialize and deserialize
			3. Verify match

		ToDo:
			None
		"""
		original = VROutputMessage(
			timestamp=4000.0,
			vocalization_tokens=[100, 200],
			joint_rotations=[{"joint_id": 0, "x": 0.1}],
			blend_shapes=[{"shape_id": 0, "weight": 0.5}],
			eye_params={"left_blink": 0.0},
		)
		json_str = original.to_json()
		restored = VROutputMessage.from_json(json_str)

		assert restored.timestamp == original.timestamp
		assert restored.vocalization_tokens == original.vocalization_tokens


class TestMessageFraming:
	"""Tests for message framing protocol."""

	def test_frame_message_adds_prefix(self):
		"""
		Purpose:
			Verify frame_message adds length prefix.

		Workflow:
			1. Frame a message
			2. Verify prefix is LENGTH_PREFIX_SIZE bytes
			3. Verify content follows

		ToDo:
			None
		"""
		data = b"test message"
		framed = frame_message(data)

		assert len(framed) == LENGTH_PREFIX_SIZE + len(data)
		assert framed[LENGTH_PREFIX_SIZE:] == data

	def test_unframe_message_extracts_length(self):
		"""
		Purpose:
			Verify unframe_message extracts length correctly.

		Workflow:
			1. Frame a message
			2. Unframe it
			3. Verify length and content

		ToDo:
			None
		"""
		data = b"hello world"
		framed = frame_message(data)

		length, content = unframe_message(framed)

		assert length == len(data)
		assert content == data

	def test_unframe_incomplete_data(self):
		"""
		Purpose:
			Verify unframe handles incomplete data gracefully.

		Workflow:
			1. Pass incomplete bytes
			2. Verify returns 0 length

		ToDo:
			None
		"""
		# Less than prefix size
		length, content = unframe_message(b"\x00\x00")
		assert length == 0
		assert content == b''

	def test_frame_unframe_round_trip(self):
		"""
		Purpose:
			Verify frame → unframe preserves data.

		Workflow:
			1. Frame message
			2. Unframe it
			3. Verify match

		ToDo:
			None
		"""
		original = b"complex message with unicode: \xc3\xa9\xc3\xa8"
		framed = frame_message(original)
		length, content = unframe_message(framed)

		assert content[:length] == original


# ============================================================================
# Server Tests
# ============================================================================

class TestVRServerConfig:
	"""Tests for VRServerConfig."""

	def test_default_values(self):
		"""
		Purpose:
			Verify config has sensible defaults.

		Workflow:
			1. Create config with no args
			2. Check defaults

		ToDo:
			None
		"""
		config = VRServerConfig()

		assert config.host == "localhost"
		assert config.port == 5555
		assert config.timeout == 1.0
		assert config.buffer_size == 65536


class TestVRServer:
	"""Tests for VRServer."""

	@pytest.fixture
	def free_port(self):
		"""
		Purpose:
		Find a free port on localhost.
		
		Returns:
			int: A free port number.
		"""
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			return s.getsockname()[1]

	def test_server_start_stop(self, free_port):
		"""
		Purpose:
			Verify server starts and stops cleanly.

		Workflow:
			1. Create server
			2. Start server
			3. Verify running
			4. Stop server
			5. Verify stopped

		ToDo:
			None
		"""
		config = VRServerConfig(port=free_port)
		server = VRServer(config=config)

		assert not server.is_running

		server.start()
		assert server.is_running

		server.stop()
		assert not server.is_running

	def test_server_accept_timeout(self, free_port):
		"""
		Purpose:
			Verify accept returns False on timeout.

		Workflow:
			1. Start server
			2. Call accept (no client)
			3. Verify timeout returns False

		ToDo:
			None
		"""
		config = VRServerConfig(port=free_port, timeout=0.1)
		server = VRServer(config=config)
		server.start()

		try:
			result = server.accept_connection()
			assert result is False
			assert not server.is_connected
		finally:
			server.stop()

	def test_server_client_connection(self, free_port):
		"""
		Purpose:
			Verify server accepts client connection.

		Workflow:
			1. Start server in thread
			2. Connect client
			3. Verify server detects connection

		ToDo:
			None
		"""
		config = VRServerConfig(port=free_port, timeout=2.0)
		server = VRServer(config=config)
		server.start()

		try:
			# Connect client in background
			def connect_client():
				"""
				Purpose:
					Mock VR client for testing server connection.

				Workflow:
					1. Sleep briefly for server setup
					2. Connect to server  
					3. Close immediately

				ToDo:
					None
				"""
				time.sleep(0.1)
				client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				client.connect(('localhost', free_port))
				client.close()

			thread = threading.Thread(target=connect_client)
			thread.start()

			result = server.accept_connection()
			assert result is True
			assert server.is_connected

			thread.join()
		finally:
			server.stop()

	def test_server_message_exchange(self, free_port):
		"""
		Purpose:
			Verify server can receive and respond to messages.

		Workflow:
			1. Start server with handler
			2. Connect client
			3. Send message from client
			4. Verify server receives and responds

		ToDo:
			None
		"""
		def handler(msg: VRInputMessage) -> VROutputMessage:
			"""
			Purpose:
				Mock message handler for testing server communication.

			Workflow:
				1. Receive input message
				2. Return output with incremented timestamp

			ToDo:
				None

			Args:
				msg: VRInputMessage to process.

			Returns:
				VROutputMessage: Response message.
			"""
			return VROutputMessage(timestamp=msg.timestamp + 1)

		config = VRServerConfig(port=free_port, timeout=2.0)
		server = VRServer(config=config, message_handler=handler)
		server.start()

		received_response = []

		try:
			thread = threading.Thread(
				target=self._run_client_exchange_task,
				args=(free_port, received_response)
			)
			thread.start()

			# Server side
			server.accept_connection()
			output = server.run_once()

			thread.join()

			assert output is not None
			assert output.timestamp == 1001.0
			assert len(received_response) == 1
			assert received_response[0].timestamp == 1001.0

		finally:
			server.stop()

	def _run_client_exchange_task(self, port: int, result_list: list):
		"""
		Purpose:
			Mock VR client for testing message exchange.
			Run in a separate thread.

		Args:
			port: Server port to connect to.
			result_list: List to store received response.
		"""
		time.sleep(0.1)
		client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		client.settimeout(2.0)
		client.connect(('localhost', port))

		# Send message
		msg = VRInputMessage(timestamp=1000.0)
		data = msg.to_json().encode('utf-8')
		framed = frame_message(data)
		client.sendall(framed)

		# Receive response
		response_data = client.recv(65536)
		if not response_data:
			client.close()
			return

		length, content = unframe_message(response_data)
		if length > 0:
			response_json = content[:length].decode('utf-8')
			result_list.append(
				VROutputMessage.from_json(response_json)
			)
		client.close()


# ============================================================================
# Input Processor Tests
# ============================================================================

class TestVRInputProcessor:
	"""Tests for VRInputProcessor."""

	@pytest.fixture
	def processor(self):
		"""
		Purpose:
			Create processor instance.

		Workflow:
			1. Create with default config

		ToDo:
			None

		Returns:
			VRInputProcessor: Processor instance for tests.
		"""
		return VRInputProcessor()

	def test_process_empty_message(self, processor):
		"""
		Purpose:
			Verify processing empty message returns None fields.

		Workflow:
			1. Process message with no data
			2. Verify all outputs are None

		ToDo:
			None
		"""
		msg = VRInputMessage(timestamp=0)
		result = processor.process(msg)

		assert result["left_eye_image"] is None
		assert result["right_eye_image"] is None
		assert result["audio_waveform"] is None
		assert result["touch_data"] is None
		assert result["joint_positions"] is None
		assert result["joint_rotations"] is None

	def test_process_audio_samples(self, processor):
		"""
		Purpose:
			Verify audio samples convert to tensor.

		Workflow:
			1. Process message with audio samples
			2. Verify tensor shape and values

		ToDo:
			None
		"""
		samples = [0.1, 0.2, 0.3, 0.4, 0.5]
		msg = VRInputMessage(timestamp=0, audio_samples=samples)
		result = processor.process(msg)

		assert result["audio_waveform"] is not None
		assert result["audio_waveform"].shape == (1, 5)
		assert torch.allclose(
			result["audio_waveform"],
			torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
		)

	def test_process_touch_contacts(self, processor):
		"""
		Purpose:
			Verify touch contacts convert to tensor dict.

		Workflow:
			1. Process message with touch contacts
			2. Verify tensor shapes and values

		ToDo:
			None
		"""
		contacts = [
			{
				"position": {"x": 1.0, "y": 2.0, "z": 3.0},
				"normal": {"x": 0.0, "y": 1.0, "z": 0.0},
				"force": {"x": 0.5, "y": 0.5, "z": 0.0},
			}
		]
		msg = VRInputMessage(timestamp=0, touch_contacts=contacts)
		result = processor.process(msg)

		assert result["touch_data"] is not None
		assert "positions" in result["touch_data"]
		assert "normals" in result["touch_data"]
		assert "forces" in result["touch_data"]
		assert "contact_active" in result["touch_data"]

		assert result["touch_data"]["positions"].shape == (1, 10, 3)
		assert result["touch_data"]["contact_active"][0, 0].item() is True

	def test_process_joint_positions(self, processor):
		"""
		Purpose:
			Verify joint positions convert to tensor.

		Workflow:
			1. Process message with joint positions
			2. Verify tensor shape

		ToDo:
			None
		"""
		joints = [
			{"x": 0.0, "y": 0.0, "z": 0.0},
			{"x": 1.0, "y": 1.0, "z": 1.0},
		]
		msg = VRInputMessage(timestamp=0, joint_positions=joints)
		result = processor.process(msg)

		assert result["joint_positions"] is not None
		assert result["joint_positions"].shape == (1, 1, 24, 3)

	def test_process_joint_rotations(self, processor):
		"""
		Purpose:
			Verify joint rotations convert to quaternion tensor.

		Workflow:
			1. Process message with joint rotations
			2. Verify tensor shape and identity initialization

		ToDo:
			None
		"""
		rotations = [
			{"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
		]
		msg = VRInputMessage(timestamp=0, joint_rotations=rotations)
		result = processor.process(msg)

		assert result["joint_rotations"] is not None
		assert result["joint_rotations"].shape == (1, 1, 24, 4)
		# First joint should have provided rotation
		assert torch.allclose(
			result["joint_rotations"][0, 0, 0],
			torch.tensor([0.0, 0.0, 0.0, 1.0])
		)


# ============================================================================
# Output Streamer Tests
# ============================================================================

class TestVROutputStreamer:
	"""Tests for VROutputStreamer."""

	@pytest.fixture
	def streamer(self):
		"""
		Purpose:
			Create streamer instance.

		Workflow:
			1. Create with default config

		ToDo:
			None

		Returns:
			VROutputStreamer: Streamer instance for tests.
		"""
		return VROutputStreamer()

	def test_stream_empty_outputs(self, streamer):
		"""
		Purpose:
			Verify streaming empty outputs returns message with defaults.

		Workflow:
			1. Stream empty dict
			2. Verify message has empty lists

		ToDo:
			None
		"""
		result = streamer.stream({}, timestamp=5000.0)

		assert result.timestamp == 5000.0
		assert result.vocalization_tokens == []
		assert result.joint_rotations == []
		assert result.blend_shapes == []
		assert result.eye_params == {}

	def test_stream_audio_tokens(self, streamer):
		"""
		Purpose:
			Verify audio tokens are extracted correctly.

		Workflow:
			1. Create audio output tensor
			2. Stream and verify tokens

		ToDo:
			None
		"""
		audio_output = torch.tensor([[100, 200, 300]])
		outputs = {"audio": audio_output}

		result = streamer.stream(outputs, timestamp=6000.0)

		assert result.vocalization_tokens == [100, 200, 300]

	def test_stream_joint_rotations(self, streamer):
		"""
		Purpose:
			Verify joint rotations are extracted as dicts.

		Workflow:
			1. Create animation output with rotations
			2. Stream and verify format

		ToDo:
			None
		"""
		# Shape: [batch, seq, joints, 4]
		joint_rotations = torch.zeros(1, 2, 24, 4)
		joint_rotations[0, -1, 0] = torch.tensor([0.1, 0.2, 0.3, 0.9])

		outputs = {
			"animation": {
				"joint_rotations": joint_rotations,
			}
		}

		result = streamer.stream(outputs, timestamp=7000.0)

		assert len(result.joint_rotations) == 24
		assert result.joint_rotations[0]["joint_id"] == 0
		assert abs(result.joint_rotations[0]["x"] - 0.1) < 0.01

	def test_stream_blend_shapes(self, streamer):
		"""
		Purpose:
			Verify blend shapes are extracted correctly.

		Workflow:
			1. Create animation output with blend shapes
			2. Stream and verify format

		ToDo:
			None
		"""
		# Shape: [batch, seq, shapes]
		blend_shapes = torch.zeros(1, 1, 51)
		blend_shapes[0, 0, 0] = 0.75

		outputs = {
			"animation": {
				"blend_shapes": blend_shapes,
			}
		}

		result = streamer.stream(outputs, timestamp=8000.0)

		assert len(result.blend_shapes) == 51
		assert result.blend_shapes[0]["shape_id"] == 0
		assert abs(result.blend_shapes[0]["weight"] - 0.75) < 0.01

	def test_stream_eye_params(self, streamer):
		"""
		Purpose:
			Verify eye parameters are extracted.

		Workflow:
			1. Create animation output with eye params
			2. Stream and verify dict format

		ToDo:
			None
		"""
		# Shape: [batch, seq, 8]
		eye_params = torch.tensor([[[0.1, 0.2, -0.1, -0.2, 0.0, 0.0, 0.5, 0.5]]])

		outputs = {
			"animation": {
				"eye_params": eye_params,
			}
		}

		result = streamer.stream(outputs, timestamp=9000.0)

		assert "left_horizontal" in result.eye_params
		assert abs(result.eye_params["left_horizontal"] - 0.1) < 0.01
		assert abs(result.eye_params["left_vertical"] - 0.2) < 0.01


# ============================================================================
# Integration Tests
# ============================================================================

class TestVRIntegration:
	"""Integration tests for full VR message flow."""

	def test_input_output_flow(self):
		"""
		Purpose:
			Test complete input → process → output flow.

		Workflow:
			1. Create VRInputMessage
			2. Process through input processor
			3. Mock model outputs
			4. Stream through output streamer
			5. Verify VROutputMessage

		ToDo:
			None
		"""
		# Input
		input_msg = VRInputMessage(
			timestamp=1000.0,
			audio_samples=[0.1, 0.2] * 100,
			joint_positions=[{"x": 0, "y": 0, "z": 0}] * 24,
		)

		# Process input
		processor = VRInputProcessor()
		processed = processor.process(input_msg)

		assert processed["audio_waveform"] is not None
		assert processed["joint_positions"] is not None

		# Mock model output (simulating model inference)
		model_outputs = {
			"audio": torch.tensor([[50, 60, 70]]),
			"animation": {
				"joint_rotations": torch.zeros(1, 1, 24, 4),
				"blend_shapes": torch.zeros(1, 1, 51),
				"eye_params": torch.zeros(1, 1, 8),
			}
		}

		# Stream output
		streamer = VROutputStreamer()
		output_msg = streamer.stream(model_outputs, timestamp=input_msg.timestamp + 50)

		assert output_msg.timestamp == 1050.0
		assert output_msg.vocalization_tokens == [50, 60, 70]
		assert len(output_msg.joint_rotations) == 24

	def test_message_serialization_in_flow(self):
		"""
		Purpose:
			Test serialization at each step of the flow.

		Workflow:
			1. Serialize input message
			2. Frame for transmission
			3. Unframe and deserialize
			4. Verify data integrity

		ToDo:
			None
		"""
		# Original message
		original = VRInputMessage(
			timestamp=2000.0,
			audio_samples=[1.0, 2.0, 3.0],
		)

		# Serialize and frame (like server send)
		json_str = original.to_json()
		data = json_str.encode('utf-8')
		framed = frame_message(data)

		# Unframe and deserialize (like client receive)
		length, content = unframe_message(framed)
		restored_json = content[:length].decode('utf-8')
		restored = VRInputMessage.from_json(restored_json)

		assert restored.timestamp == original.timestamp
		assert restored.audio_samples == original.audio_samples
