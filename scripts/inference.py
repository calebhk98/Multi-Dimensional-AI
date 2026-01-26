#!/usr/bin/env python
"""
VR Inference Server

Main entry point for running the Multi-Dimensional AI creature
in inference mode with VR integration.

Purpose:
	Loads trained model, starts VR server, and runs continuous
	inference loop processing VR sensor data and returning actions.

Workflow:
	1. Load configuration from YAML
	2. Initialize model from checkpoint
	3. Start VR server
	4. Accept VR client connection
	5. Run inference loop:
		- Receive sensor data
		- Process through encoders
		- Generate outputs
		- Stream back to VR

ToDo:
	- Add graceful shutdown handling
	- Add performance monitoring/logging
	- Add checkpoint hot-reloading

Usage:
	python scripts/inference.py --config configs/inference_config.yaml
	python scripts/inference.py --dry-run  # Test server without model
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vr_integration.server import VRServer, VRServerConfig
from src.vr_integration.protocol import VRInputMessage, VROutputMessage
from src.vr_integration.input_processor import VRInputProcessor, InputProcessorConfig
from src.vr_integration.output_streamer import VROutputStreamer, OutputStreamerConfig


# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configuration paths
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference_config.yaml"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"


class InferenceServer:
	"""
	Manages VR inference loop.

	Coordinates between VR server, model, and data processors
	to provide real-time creature behavior in VR.

	Args:
		config_path: Path to configuration YAML file.
		dry_run: If True, run without loading model.
	"""

	def __init__(
		self,
		config_path: Optional[Path] = None,
		dry_run: bool = False

	):
		"""
		Initialize inference server.

		Purpose:
			Sets up inference server with configuration and components.

		Workflow:
			1. Load or initialize config
			2. Set running flags
			3. Initialize component placeholders

		ToDo:
			None

		Args:
			config_path: Path to config file.
			dry_run: If True, skip model loading.
		"""
		self.config_path = config_path or DEFAULT_CONFIG_PATH
		self.dry_run = dry_run
		self.config = self._load_config()

		self._running = False
		self._shutdown_requested = False

		# Components initialized in start()
		self.model = None
		self.vr_server: Optional[VRServer] = None
		self.input_processor: Optional[VRInputProcessor] = None
		self.output_streamer: Optional[VROutputStreamer] = None

	def _load_config(self) -> dict:
		"""
		Load configuration from YAML file.

		Purpose:
			Loads YAML configuration or returns defaults if file not found.

		Workflow:
			1. Check if config file exists
			2. Load YAML if exists, else use defaults

		ToDo:
			None

		Returns:
			dict: Configuration dictionary.

		Raises:
			FileNotFoundError: If config file doesn't exist.
		"""
		if not self.config_path.exists():
			logger.warning(f"Config file not found: {self.config_path}")
			return self._default_config()

		with open(self.config_path, 'r') as f:
			config = yaml.safe_load(f)
			logger.info(f"Loaded config from {self.config_path}")
			return config

	def _default_config(self) -> dict:
		"""
		Provide default configuration.

		Purpose:
			Generates fallback configuration when file not available.

		Workflow:
			1. Create dict with inference settings
			2. Add VR server settings

		ToDo:
			None

		Returns:
			dict: Default config values.
		"""
		return {
			"inference": {
				"checkpoint_path": str(DEFAULT_CHECKPOINT_PATH),
				"device": "cuda" if torch.cuda.is_available() else "cpu",
			},
			"vr": {
				"unity": {
					"host": "localhost",
					"port": 5555,
					"timeout": 1.0,
				},
				"input_processing": {
					"image_size": 224,
					"sample_rate": 16000,
					"num_joints": 24,
					"max_contacts": 10,
				},
			},
		}

	def _load_model(self) -> Optional[torch.nn.Module]:
		"""
		Load model from checkpoint.

		Purpose:
			Loads trained model from checkpoint file or returns None in dry-run mode.

		Workflow:
			1. Skip if dry run
			2. Find checkpoint path
			3. Load model and state dict
			4. Move to device and eval mode

		ToDo:
			Add checkpoint validation

		Returns:
			Module: Loaded model or None if dry run.
		"""
		if self.dry_run:
			logger.info("Dry run mode - skipping model load")
			return None

		checkpoint_path = Path(
			self.config.get("inference", {}).get(
				"checkpoint_path",
				str(DEFAULT_CHECKPOINT_PATH)
			)
		)

		if not checkpoint_path.exists():
			logger.warning(f"Checkpoint not found: {checkpoint_path}")
			logger.info("Running in dry-run mode without model")
			return None

		try:
			device = self.config.get("inference", {}).get("device", "cpu")
			logger.info(f"Loading model from {checkpoint_path}")

			# Import model class
			from src.models.multimodal_transformer import MultiModalCreature

			# Load checkpoint
			checkpoint = torch.load(checkpoint_path, map_location=device)

			# Build model from config
			model_config = checkpoint.get("config", {})
			model = MultiModalCreature(model_config)
			model.load_state_dict(checkpoint["model_state_dict"])
			model.to(device)
			model.eval()

			logger.info(f"Model loaded successfully on {device}")
			return model

		except Exception as e:
			logger.error(f"Failed to load model: {e}")
			return None

	def _init_components(self) -> None:
		"""
		Initialize VR server and processors.

		Purpose:
			Sets up VR server, input processor, and output streamer components.

		Workflow:
			1. Create VR server config
			2. Initialize input processor
			3. Initialize output streamer

		ToDo:
			None
		"""
		vr_config = self.config.get("vr", {})
		unity_config = vr_config.get("unity", {})
		input_config = vr_config.get("input_processing", {})

		# VR Server
		server_config = VRServerConfig(
			host=unity_config.get("host", "localhost"),
			port=unity_config.get("port", 5555),
			timeout=unity_config.get("timeout", 1.0),
		)
		self.vr_server = VRServer(
			config=server_config,
			message_handler=self._handle_message,
		)

		# Input processor
		device = self.config.get("inference", {}).get("device", "cpu")
		processor_config = InputProcessorConfig(
			image_size=input_config.get("image_size", 224),
			sample_rate=input_config.get("sample_rate", 16000),
			num_joints=input_config.get("num_joints", 24),
			max_contacts=input_config.get("max_contacts", 10),
			device=device,
		)
		self.input_processor = VRInputProcessor(config=processor_config)

		# Output streamer
		self.output_streamer = VROutputStreamer()

	def _handle_message(self, input_msg: VRInputMessage) -> VROutputMessage:
		"""
		Process VR input and generate output.

		This is the core inference handler called for each VR frame.

		Purpose:
			Main inference callback processing VR sensor data into creature actions.

		Workflow:
			1. Process input through encoders
			2. Run model inference (or dummy  outputs)
			3. Stream outputs to VR format
			4. Log latency warnings if needed

		ToDo:
			Add caching for repeated inputs

		Args:
			input_msg: VR sensor data.

		Returns:
			VROutputMessage: Creature action commands.
		"""
		start_time = time.time()

		# Process input through encoders
		processed_inputs = self.input_processor.process(input_msg)

		if self.model is not None:
			# Run model inference
			try:
				with torch.no_grad():
					model_outputs = self.model(processed_inputs)
			except Exception as e:
				logger.error(f"Model inference failed: {e}")
				model_outputs = {}
		else:
			# Dry run - return dummy outputs
			model_outputs = self._generate_dummy_outputs()

		# Stream outputs to VR format
		output_msg = self.output_streamer.stream(
			model_outputs,
			timestamp=input_msg.timestamp + (time.time() - start_time) * 1000
		)

		# Log latency
		latency_ms = (time.time() - start_time) * 1000
		if latency_ms > 50:  # Log if above target
			logger.warning(f"High latency: {latency_ms:.1f}ms")

		return output_msg

	def _generate_dummy_outputs(self) -> dict:
		"""
		Generate placeholder outputs for dry-run mode.

		Purpose:
			Creates zero tensors for testing server without model.

		Workflow:
			1. Create zero audio tokens
			2. Create zero animation tensors

		ToDo:
			None

		Returns:
			dict: Mock model outputs.
		"""
		return {
			"audio": torch.zeros(1, 10, dtype=torch.long),
			"animation": {
				"joint_rotations": torch.zeros(1, 1, 24, 4),
				"blend_shapes": torch.zeros(1, 1, 51),
				"eye_params": torch.zeros(1, 1, 8),
			}
		}

	def start(self) -> None:
		"""Start the inference server."""
		logger.info("Starting inference server...")

		# Load model
		self.model = self._load_model()

		# Initialize components
		self._init_components()

		# Start VR server
		self.vr_server.start()
		self._running = True

		logger.info("Inference server ready, waiting for VR client...")

	def stop(self) -> None:
		"""Stop the inference server."""
		logger.info("Stopping inference server...")
		self._running = False

		if self.vr_server:
			self.vr_server.stop()

		logger.info("Inference server stopped")

	def run(self) -> None:
		"""
		Run the main inference loop.

		Blocks until shutdown is requested.
		"""
		self.start()

		try:
			while self._running and not self._shutdown_requested:
				# Run one iteration of server loop
				self.vr_server.run_once()

		except KeyboardInterrupt:
			logger.info("Keyboard interrupt received")
		finally:
			self.stop()

	def request_shutdown(self) -> None:
		"""Request graceful shutdown."""
		self._shutdown_requested = True


def setup_signal_handlers(server: InferenceServer) -> None:
	"""
	Set up signal handlers for graceful shutdown.

	Args:
		server: Server instance to shut down.
	"""
	def signal_handler(signum, frame):
		"""
		Purpose:
			Signal handler for graceful shutdown.

		Workflow:
			1. Log received signal
			2. Request server shutdown

		ToDo:
			None

		Args:
			signum: Signal number.
			frame: Current stack frame.
		"""
		logger.info(f"Received signal {signum}")
		server.request_shutdown()

	signal.signal(signal.SIGINT, signal_handler)
	signal.signal(signal.SIGTERM, signal_handler)


def parse_args() -> argparse.Namespace:
	"""
	Parse command line arguments.

	Returns:
		Namespace: Parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description="Run Multi-Dimensional AI VR inference server"
	)

	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG_PATH,
		help="Path to inference config YAML file",
	)

	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Run server without loading model (for testing)",
	)

	parser.add_argument(
		"--port",
		type=int,
		default=None,
		help="Override VR server port",
	)

	parser.add_argument(
		"--verbose",
		"-v",
		action="store_true",
		help="Enable verbose logging",
	)

	return parser.parse_args()


def main() -> None:
	"""Main entry point."""
	args = parse_args()

	if args.verbose:
		logging.getLogger().setLevel(logging.DEBUG)

	logger.info("=" * 50)
	logger.info("Multi-Dimensional AI VR Inference Server")
	logger.info("=" * 50)

	# Create server
	server = InferenceServer(
		config_path=args.config,
		dry_run=args.dry_run,
	)

	# Override port if specified
	if args.port:
		server.config.setdefault("vr", {}).setdefault("unity", {})["port"] = args.port

	# Set up signal handlers
	setup_signal_handlers(server)

	# Run server
	server.run()

	logger.info("Server exited cleanly")


if __name__ == "__main__":
	main()
