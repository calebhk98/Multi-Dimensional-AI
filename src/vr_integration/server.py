"""
VR Server

TCP server for real-time communication with Unity VR client.

Purpose:
	Handles connection lifecycle, message receiving/sending, and
	coordinates with the inference loop.

Workflow:
	1. Server binds to configured host:port
	2. Accepts single client connection (one VR instance)
	3. Receives VRInputMessages, processes through callback
	4. Sends VROutputMessages back to client

ToDo:
	- Add WebSocket support for browser-based VR
	- Support multiple concurrent VR clients
	- Add authentication/handshake protocol
"""

import socket
import asyncio
import logging
from typing import Callable, Optional, Any
from dataclasses import dataclass
from typing import Callable, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.vr_integration.recorder import VRRecorder

from src.vr_integration.protocol import (
	VRInputMessage,
	VROutputMessage,
	frame_message,
	unframe_message,
	LENGTH_PREFIX_SIZE,
)

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT = 1.0
DEFAULT_BUFFER_SIZE = 65536  # 64KB buffer for vision data


@dataclass
class VRServerConfig:
	"""
	Configuration for VR server.

	Args:
		host: Host address to bind to.
		port: Port number to listen on.
		timeout: Socket timeout in seconds.
		buffer_size: Receive buffer size in bytes.
	"""
	host: str = DEFAULT_HOST
	port: int = DEFAULT_PORT
	timeout: float = DEFAULT_TIMEOUT
	buffer_size: int = DEFAULT_BUFFER_SIZE


class VRServer:
	"""
	TCP server for VR integration.

	Manages connection to Unity VR client and handles message exchange.
	Designed for single-client operation (one AI creature per server).

	Args:
		config: Server configuration.
		message_handler: Callback function to process incoming messages.
		message_handler: Callback function to process incoming messages.
			Signature: (VRInputMessage) -> VROutputMessage
		recorder: Optional VRRecorder instance for capturing session data.
	"""

	def __init__(
		self,
		config: Optional[VRServerConfig] = None,
		message_handler: Optional[Callable[[VRInputMessage], VROutputMessage]] = None,
		recorder: Optional["VRRecorder"] = None,
	):
		"""
		Initialize VR server.

		Args:
			config: Server configuration (uses defaults if None).
			message_handler: Callback for processing messages.
			recorder: recorder instance.
		"""
		self.config = config or VRServerConfig()
		self.message_handler = message_handler
		self.recorder = recorder
		self._socket: Optional[socket.socket] = None
		self._client_socket: Optional[socket.socket] = None
		self._running = False
		self._receive_buffer = b''

	@property
	def is_running(self) -> bool:
		"""
		Check if server is currently running.

		Returns:
			bool: True if server running.
		"""
		return self._running

	@property
	def is_connected(self) -> bool:
		"""
		Check if a client is currently connected.

		Returns:
			bool: True if client connected.
		"""
		return self._client_socket is not None

	def start(self) -> None:
		"""
		Start the server and begin listening for connections.

		Raises:
			OSError: If port is already in use or binding fails.
		"""
		if self._running:
			logger.warning("Server already running")
			return

		try:
			self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			# Allow port reuse to avoid "Address already in use" errors
			self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self._socket.bind((self.config.host, self.config.port))
			self._socket.listen(1)
			self._socket.settimeout(self.config.timeout)
			self._running = True
			logger.info(
				f"VR Server listening on {self.config.host}:{self.config.port}"
			)
		except OSError as e:
			logger.error(f"Failed to start VR server: {e}")
			self._cleanup()
			raise

	def stop(self) -> None:
		"""Stop the server and close all connections."""
		self._running = False
		self._cleanup()
		logger.info("VR Server stopped")

	def _cleanup(self) -> None:
		"""
		Clean up socket resources.

		Purpose:
			Safely close client and server sockets and reset buffers.

		Workflow:
			1. Close client socket if open
			2. Close server socket if open
			3. Clear receive buffer

		ToDo:
			None
		"""
		if self._client_socket:
			try:
				self._client_socket.close()
			except OSError as e:
				logger.debug(f"Error closing client socket: {e}")
			self._client_socket = None

		if self._socket:
			try:
				self._socket.close()
			except OSError as e:
				logger.debug(f"Error closing server socket: {e}")
			self._socket = None

		self._receive_buffer = b''

	def accept_connection(self) -> bool:
		"""
		Accept a pending client connection.

		Returns:
			bool: True if connection accepted, False if timeout or error.
		"""
		if not self._running or not self._socket:
			return False

		try:
			client, address = self._socket.accept()
			client.settimeout(self.config.timeout)
			self._client_socket = client
			logger.info(f"VR client connected from {address}")
			return True
		except socket.timeout:
			return False
		except OSError as e:
			logger.error(f"Error accepting connection: {e}")
			return False

	def receive_message(self) -> Optional[VRInputMessage]:
		"""
		Receive and parse a message from connected client.

		Returns:
			VRInputMessage if complete message received, None otherwise.

		Raises:
			ConnectionError: If client disconnected.
		"""
		if not self._client_socket:
			return None

		try:
			# Read available data into buffer
			data = self._client_socket.recv(self.config.buffer_size)
			if not data:
				raise ConnectionError("Client disconnected")

			self._receive_buffer += data

			# Try to extract complete message
			if len(self._receive_buffer) < LENGTH_PREFIX_SIZE:
				return None

			msg_length, remaining = unframe_message(self._receive_buffer)
			if msg_length == 0:
				return None

			if len(remaining) < msg_length:
				# Haven't received complete message yet
				return None

			# Extract complete message and update buffer
			message_data = remaining[:msg_length]
			self._receive_buffer = remaining[msg_length:]

			message = VRInputMessage.from_json(message_data.decode('utf-8'))
			
			# Record validation hook
			if self.recorder:
				self.recorder.record_frame(message)
				
			return message

		except socket.timeout:
			return None
		except ConnectionError:
			logger.info("Client disconnected")
			self._disconnect_client()
			raise
		except Exception as e:
			logger.error(f"Error receiving message: {e}")
			return None

	def send_message(self, message: VROutputMessage) -> bool:
		"""
		Send a message to connected client.

		Args:
			message: Output message to send.

		Returns:
			bool: True if sent successfully, False otherwise.
		"""
		if not self._client_socket:
			logger.warning("Cannot send message: no client connected")
			return False

		try:
			data = message.to_json().encode('utf-8')
			framed = frame_message(data)
			self._client_socket.sendall(framed)
			return True
		except OSError as e:
			logger.error(f"Error sending message: {e}")
			self._disconnect_client()
			return False

	def _disconnect_client(self) -> None:
		"""
		Handle client disconnection.

		Purpose:
			Cleanly disconnect the VR client and reset connection state.

		Workflow:
			1. Close client socket
			2. Clear receive buffer
			3. Log disconnection

		ToDo:
			None
		"""
		if self._client_socket:
			try:
				self._client_socket.close()
			except OSError:
				pass
			self._client_socket = None
			self._receive_buffer = b''
			logger.info("Client disconnected, waiting for new connection")

	def run_once(self) -> Optional[VROutputMessage]:
		"""
		Run one iteration of the server loop.

		Accepts connection if needed, receives message, processes through
		handler, and sends response.

		Returns:
			VROutputMessage if message was processed, None otherwise.
		"""
		if not self._running:
			return None

		# Accept connection if needed
		if not self.is_connected:
			self.accept_connection()
			return None

		# Receive and process message
		try:
			input_msg = self.receive_message()
			if input_msg is None:
				return None

			if self.message_handler:
				output_msg = self.message_handler(input_msg)
				self.send_message(output_msg)
				return output_msg
			else:
				logger.warning("No message handler configured")
				return None

		except ConnectionError:
			return None
