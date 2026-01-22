"""
Multi-Dimensional AI Creature Package

An autonomous AI creature with multi-modal sensory perception and parallel action generation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.models.multimodal_transformer import MultiModalCreature
from src.config import Config

__all__ = [
	"MultiModalCreature",
	"Config",
]
