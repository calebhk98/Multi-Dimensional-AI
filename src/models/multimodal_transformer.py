"""
Main MultiModalCreature model - ties everything together.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.encoders import (
	VoiceTokenizer,
	InternalVoiceEncoder,
	ExternalVoiceEncoder,
	AudioEncoder,
	VisualEncoder,
	ProprioceptionEncoder,
	TouchEncoder,
)
from src.decoders import (
	InternalTextDecoder,
	ExternalTextDecoder,
	AudioDecoder,
	AnimationDecoder,
)
from src.models.transformer_backbone import TransformerBackbone
from src.models.fusion_module import TokenFusionModule


class MultiModalCreature(nn.Module):
	"""
	Complete multi-modal AI creature model.
	
	Processes 6 input modalities and generates 4 output streams in parallel:
	
	Inputs (senses):
	- Internal voice (thoughts)
	- External voice (heard speech)
	- Audio (environmental sounds)
	- Vision (stereo - left/right eyes)
	- Proprioception (body state)
	- Touch (tactile sensations)
	
	Outputs (actions):
	- Internal text (private thoughts)
	- External text (speech)
	- Audio (vocalizations)
	- Animation (body movements)
	"""
	
	def __init__(self, config: Dict):
		"""
		Initialize multi-modal creature.
		
		Args:
			config: Configuration dictionary (from config.yaml)
		"""
		super().__init__()
		
		self.config = config
		model_config = config.get("model", {})
		
		# Extract dimensions
		vocab_size = model_config.get("encoders", {}).get("internal_voice", {}).get("vocab_size", 50257)
		embedding_dim = model_config.get("transformer", {}).get("hidden_dim", 1536)
		
		# === ENCODERS ===
		self.internal_voice_encoder = InternalVoiceEncoder(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
		)
		
		self.external_voice_encoder = ExternalVoiceEncoder(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
		)
		
		audio_config = model_config.get("encoders", {}).get("audio", {}).copy()
		if "embedding_dim" in audio_config:
			del audio_config["embedding_dim"]
			
		self.audio_encoder = AudioEncoder(
			embedding_dim=embedding_dim,
			**audio_config
		)
		
		vision_config = model_config.get("encoders", {}).get("vision", {}).copy()
		if "embedding_dim" in vision_config:
			del vision_config["embedding_dim"]

		self.visual_encoder = VisualEncoder(
			embedding_dim=embedding_dim,
			**vision_config
		)
		
		proprio_config = model_config.get("encoders", {}).get("proprioception", {}).copy()
		if "embedding_dim" in proprio_config:
			del proprio_config["embedding_dim"]
			
		self.proprioception_encoder = ProprioceptionEncoder(
			embedding_dim=embedding_dim,
			**proprio_config
		)
		
		touch_config = model_config.get("encoders", {}).get("touch", {}).copy()
		if "embedding_dim" in touch_config:
			del touch_config["embedding_dim"]
			
		self.touch_encoder = TouchEncoder(
			embedding_dim=embedding_dim,
			**touch_config
		)
		
		# === TOKEN FUSION ===
		fusion_config = model_config.get("fusion", {})
		self.fusion_module = TokenFusionModule(
			embedding_dim=embedding_dim,
			fusion_strategy=fusion_config.get("strategy", "concatenate"),
			use_modality_embeddings=fusion_config.get("modality_embeddings", True),
		)
		
		# === TRANSFORMER BACKBONE ===
		transformer_config = model_config.get("transformer", {})
		self.transformer = TransformerBackbone(
			num_layers=transformer_config.get("num_layers", 24),
			hidden_dim=transformer_config.get("hidden_dim", 1536),
			num_heads=transformer_config.get("num_attention_heads", 16),
			ffn_dim=transformer_config.get("ffn_dim", 6144),
			dropout=transformer_config.get("dropout", 0.1),
		)
		
		# === DECODERS ===
		self.internal_text_decoder = InternalTextDecoder(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
		)
		
		self.external_text_decoder = ExternalTextDecoder(
			vocab_size=vocab_size,
			embedding_dim=embedding_dim,
		)
		
		audio_decoder_config = model_config.get("decoders", {}).get("audio", {}).copy()
		if "embedding_dim" in audio_decoder_config:
			del audio_decoder_config["embedding_dim"]

		self.audio_decoder = AudioDecoder(
			embedding_dim=embedding_dim,
			**audio_decoder_config
		)
		
		anim_decoder_config = model_config.get("decoders", {}).get("animation", {}).copy()
		if "embedding_dim" in anim_decoder_config:
			del anim_decoder_config["embedding_dim"]

		self.animation_decoder = AnimationDecoder(
			embedding_dim=embedding_dim,
			**anim_decoder_config
		)
	
	def encode_inputs(
		self,
		internal_voice_tokens: Optional[torch.Tensor] = None,
		external_voice_tokens: Optional[torch.Tensor] = None,
		audio_waveform: Optional[torch.Tensor] = None,
		left_eye_image: Optional[torch.Tensor] = None,
		right_eye_image: Optional[torch.Tensor] = None,
		joint_positions: Optional[torch.Tensor] = None,
		joint_rotations: Optional[torch.Tensor] = None,
		touch_data: Optional[Dict[str, torch.Tensor]] = None,
	) -> Dict[str, Dict[str, torch.Tensor]]:
		"""
		Encode all input modalities.
		
		Args:
			internal_voice_tokens: Internal thought tokens
			external_voice_tokens: Heard speech tokens
			audio_waveform: Raw audio input
			left_eye_image: Left eye vision
			right_eye_image: Right eye vision
			joint_positions: Body joint positions
			joint_rotations: Body joint rotations
			touch_data: Touch sensor data
			
		Returns:
			Dictionary of encoder outputs
		"""
		encoder_outputs = {}
		
		# Internal voice
		if internal_voice_tokens is not None:
			encoder_outputs["internal_voice"] = self.internal_voice_encoder(
				internal_voice_tokens
			)
		
		# External voice
		if external_voice_tokens is not None:
			encoder_outputs["external_voice"] = self.external_voice_encoder(
				external_voice_tokens
			)
		
		# Audio
		if audio_waveform is not None:
			encoder_outputs["audio"] = self.audio_encoder(audio_waveform)
		
		# Vision
		if left_eye_image is not None:
			encoder_outputs["vision"] = self.visual_encoder(
				left_eye_image,
				right_eye_image,
			)
		
		# Proprioception
		if joint_positions is not None and joint_rotations is not None:
			encoder_outputs["proprioception"] = self.proprioception_encoder(
				joint_positions,
				joint_rotations,
			)
		
		# Touch
		if touch_data is not None:
			encoder_outputs["touch"] = self.touch_encoder(**touch_data)
		
		return encoder_outputs
	
	def forward(
		self,
		# Input modalities
		internal_voice_tokens: Optional[torch.Tensor] = None,
		external_voice_tokens: Optional[torch.Tensor] = None,
		audio_waveform: Optional[torch.Tensor] = None,
		left_eye_image: Optional[torch.Tensor] = None,
		right_eye_image: Optional[torch.Tensor] = None,
		joint_positions: Optional[torch.Tensor] = None,
		joint_rotations: Optional[torch.Tensor] = None,
		touch_data: Optional[Dict[str, torch.Tensor]] = None,
		# Generation parameters
		temperature: float = 0.8,
		top_k: int = 50,
		top_p: float = 0.9,
		return_hidden_states: bool = False,
	) -> Dict[str, torch.Tensor]:
		"""
		Forward pass - parallel generation across all output modalities.
		
		Args:
			internal_voice_tokens: Internal thought tokens [batch, seq_len]
			external_voice_tokens: Heard speech tokens [batch, seq_len]
			audio_waveform: Raw audio input [batch, samples]
			left_eye_image: Left eye vision [batch, 3, H, W]
			right_eye_image: Right eye vision [batch, 3, H, W]
			joint_positions: Body joint positions [batch, time, joints, 3]
			joint_rotations: Body joint rotations [batch, time, joints, 4]
			touch_data: Dictionary of touch sensor data
			temperature: Sampling temperature for text/audio generation
			top_k: Top-k sampling parameter
			top_p: Top-p sampling parameter
			return_hidden_states: Whether to return transformer hidden states

		Returns:
			Dictionary with all 4 output streams
		"""
		# 1. Encode all inputs
		encoder_outputs = self.encode_inputs(
			internal_voice_tokens=internal_voice_tokens,
			external_voice_tokens=external_voice_tokens,
			audio_waveform=audio_waveform,
			left_eye_image=left_eye_image,
			right_eye_image=right_eye_image,
			joint_positions=joint_positions,
			joint_rotations=joint_rotations,
			touch_data=touch_data,
		)
		
		# 2. Fuse multi-modal tokens
		fusion_output = self.fusion_module(encoder_outputs)
		fused_embeddings = fusion_output["embeddings"]
		attention_mask = fusion_output["attention_mask"]
		modality_ranges = fusion_output["modality_ranges"]
		
		# 3. Pass through transformer
		hidden_states = self.transformer(
			fused_embeddings,
			attention_mask=attention_mask,
		)
		
		# 4. Generate from all decoders in parallel
		outputs = {}
		
		# Internal text
		outputs["internal_text"] = self.internal_text_decoder(
			hidden_states,
			temperature=temperature,
			top_k=top_k,
			top_p=top_p,
		)
		
		# External text
		outputs["external_text"] = self.external_text_decoder(
			hidden_states,
			temperature=temperature,
			top_k=top_k,
			top_p=top_p,
		)
		
		# Audio
		outputs["audio"] = self.audio_decoder(
			hidden_states,
			temperature=temperature,
			top_k=top_k,
			top_p=top_p,
		)
		
		# Animation
		outputs["animation"] = self.animation_decoder(hidden_states)
		
		if return_hidden_states:
			outputs["hidden_states"] = hidden_states
			outputs["modality_ranges"] = modality_ranges
		
		return outputs
	
	def compute_loss(
		self,
		outputs: Dict[str, torch.Tensor],
		targets: Dict[str, torch.Tensor],
		loss_weights: Optional[Dict[str, float]] = None,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Compute multi-task loss.
		
		Args:
			outputs: Model outputs (from forward with return_hidden_states=True)
			targets: Target values for each modality
			loss_weights: Optional weights for each loss
			
		Returns:
			Tuple of (total_loss, loss_dict)
		"""
		if loss_weights is None:
			loss_weights = self.config.get("loss_weights", {
				"internal_text": 1.0,
				"external_text": 1.0,
				"audio": 0.8,
				"animation": 0.6,
			})
		
		hidden_states = outputs["hidden_states"]
		# Safely get modality ranges, fallback to empty if generic forward/old code
		modality_ranges = outputs.get("modality_ranges", {})
		
		losses = {}
		
		# Helper to get relevant hidden states
		def get_modality_states(name: str):
			if name in modality_ranges:
				start, end = modality_ranges[name]
				# Check bounds
				if start < hidden_states.shape[1] and end <= hidden_states.shape[1]:
					return hidden_states[:, start:end, :]
			# Fallback: return full hidden states (legacy behavior or if no range found)
			# WARNING: This might cause shape mismatches if not aligned
			return hidden_states

		# Internal text loss
		if "internal_text" in targets:
			# Maps to internal_voice inputs
			states = get_modality_states("internal_voice")
			losses["internal_text"] = self.internal_text_decoder.compute_loss(
				states,
				targets["internal_text"]
			)
		
		# External text loss
		if "external_text" in targets:
			# Maps to external_voice inputs
			states = get_modality_states("external_voice")
			losses["external_text"] = self.external_text_decoder.compute_loss(
				states,
				targets["external_text"]
			)
		
		# Audio loss
		if "audio" in targets:
			# Maps to audio inputs
			states = get_modality_states("audio")
			losses["audio"] = self.audio_decoder.compute_loss(
				states,
				targets["audio"]
			)
		
		# Animation loss
		if "animation" in targets:
			# Maps to proprioception inputs
			states = get_modality_states("proprioception")
			anim_loss, anim_loss_dict = self.animation_decoder.compute_loss(
				states,
				targets["animation"]["rotations"],
				targets["animation"]["blend_shapes"],
				targets["animation"]["eye_params"],
			)
			losses["animation"] = anim_loss
			losses.update(anim_loss_dict)
		
		# Compute total loss
		total_loss = sum(
			loss_weights.get(key, 1.0) * value
			for key, value in losses.items()
			if key in loss_weights
		)
		
		losses["total_loss"] = total_loss
		
		return total_loss, losses
