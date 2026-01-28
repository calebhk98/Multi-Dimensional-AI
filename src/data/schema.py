"""
Data Schema Definitions

Defines the types and structures for data samples used across the pipeline.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch

class ModalityType(Enum):
    """Enumeration of supported modalities."""
    VISION_LEFT = "vision_left"
    VISION_RIGHT = "vision_right"
    AUDIO = "audio"
    TOUCH = "touch"
    PROPRIO = "proprio"
    TEXT = "text" # For future or metadata use

@dataclass
class NormalizationConfig:
    """
    Configuration for data normalization.
    
    Args:
        vision_mean: Mean for image normalization (default: ImageNet).
        vision_std: Std for image normalization (default: ImageNet).
        audio_sample_rate: Expected sample rate for audio.
        touch_max_force: Maximum expected force value for scaling.
    """
    vision_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    vision_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    audio_sample_rate: int = 16000
    touch_max_force: float = 1.0

@dataclass
class UnifiedSample:
    """
    A single synchronized multi-modal sample.
    
    All tensor fields are optional to support missing modalities.
    
    Args:
        vision_left: Tensor [C, H, W]
        vision_right: Tensor [C, H, W]
        audio: Tensor [Samples] or [Channels, Samples]
        touch: Tensor [Points, Features]
        proprio: Tensor [Features]
        text: Tokenized text tensor [SeqLen]
        timestamp: Float timestamp in seconds
        metadata: Any additional metadata
    """
    vision_left: Optional[torch.Tensor] = None
    vision_right: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    touch: Optional[torch.Tensor] = None
    proprio: Optional[torch.Tensor] = None
    text: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
