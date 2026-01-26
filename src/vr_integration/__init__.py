"""
VR Integration Module

Provides real-time communication between the Multi-Dimensional AI
and Unity VR environments via TCP.

Purpose:
	Handles bidirectional streaming of sensory inputs from VR and
	model outputs back to VR for actuation.

Workflow:
	1. VRServer accepts TCP connections from Unity client
	2. VRInputProcessor converts raw VR data to encoder-ready tensors
	3. VROutputStreamer converts model outputs to VR-friendly format
	4. Protocol module handles serialization/deserialization

ToDo:
	- Add WebSocket support as alternative to TCP
	- Implement connection pooling for multiple VR clients
"""

from src.vr_integration.server import VRServer
from src.vr_integration.protocol import VRInputMessage, VROutputMessage
from src.vr_integration.input_processor import VRInputProcessor
from src.vr_integration.output_streamer import VROutputStreamer

__all__ = [
	"VRServer",
	"VRInputMessage",
	"VROutputMessage",
	"VRInputProcessor",
	"VROutputStreamer",
]
