"""
Training wrappers for single-modality tasks.
Combines Encoder and Decoder into a single module that computes loss.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

# Encoders
from src.encoders.audio_encoder import AudioEncoder
from src.encoders.internal_voice_encoder import InternalVoiceEncoder
from src.encoders.external_voice_encoder import ExternalVoiceEncoder
from src.encoders.proprioception_encoder import ProprioceptionEncoder

# Decoders
from src.decoders.audio_decoder import AudioDecoder
from src.decoders.text_decoder import InternalTextDecoder, ExternalTextDecoder
from src.decoders.animation_decoder import AnimationDecoder


class BaseAutoEncoder(nn.Module):
    """Base class for auto-encoders."""
    
    def compute_loss(self, outputs: Dict[str, Any], targets: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss from outputs and targets."""
        raise NotImplementedError


class AudioAutoEncoder(BaseAutoEncoder):
    """
    Wraps AudioEncoder + AudioDecoder.
    Goal: Reconstruct audio tokens/embeddings.
    """
    def __init__(self, encoder: AudioEncoder, decoder: AudioDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(
        self,
        audio_waveform: torch.Tensor,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Encode
        enc_out = self.encoder(audio_waveform, return_indices=True)
        embeddings = enc_out["embeddings"]
        
        # Decode
        # Pass embeddings to decoder. AudioDecoder returns {'logits': ...}
        dec_out = self.decoder(embeddings, return_logits=True)
        
        return {
            "encoder_outputs": enc_out,
            "decoder_outputs": dec_out,
            "embeddings": embeddings
        }

    def compute_loss(self, outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets: Dict containing 'target' tokens [batch, seq_len]
        """
        target_tokens = targets["target"]
        
        # Decoder output contains logits [batch, seq_len, codebook_size]
        logits = outputs["decoder_outputs"]["logits"]
        
        # Simple Cross Entropy
        loss_fct = nn.CrossEntropyLoss()
        
        # Reshape for loss
        # logits: [B, T, V], targets: [B, T]
        B, T, V = logits.shape
        T_target = target_tokens.shape[1]
        min_T = min(T, T_target)
        
        logits = logits[:, :min_T, :].reshape(-1, V)
        target_tokens = target_tokens[:, :min_T].reshape(-1)
        
        loss = loss_fct(logits, target_tokens)
        
        return loss, {"loss": loss.item()}


class VoiceAutoEncoder(BaseAutoEncoder):
    """
    Wraps VoiceEncoder (Internal/External) + TextDecoder.
    Goal: Reconstruct text (masked language modeling or causal output).
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        enc_out = self.encoder(input_ids)
        embeddings = enc_out["embeddings"]
        
        # Pass to decoder
        dec_out = self.decoder(embeddings, return_logits=True)
        
        return {
            "embeddings": embeddings,
            "decoder_outputs": dec_out # Changed to match the new compute_loss
        }
    
    def compute_loss(self, outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets: Dict containing 'target' tensor [batch, seq_len]
        """
        target_tokens = targets["target"]
        
        # Decoder output contains logits [batch, seq_len, codebook_size]
        logits = outputs["decoder_outputs"]["logits"]
        
        # Simple Cross Entropy
        loss_fct = nn.CrossEntropyLoss()
        
        # Reshape for loss
        # logits: [B, T, V], targets: [B, T]
        B, T, V = logits.shape
        T_target = target_tokens.shape[1]
        min_T = min(T, T_target)
        
        logits = logits[:, :min_T, :].reshape(-1, V)
        target_tokens = target_tokens[:, :min_T].reshape(-1)
        
        loss = loss_fct(logits, target_tokens)
        
        return loss, {"loss": loss.item()}


class MotionAutoEncoder(BaseAutoEncoder):
    """
    Wraps ProprioceptionEncoder + AnimationDecoder.
    Goal: Reconstruct motion/pose.
    """
    def __init__(self, encoder: ProprioceptionEncoder, decoder: AnimationDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(
        self,
        joint_positions: torch.Tensor,
        joint_rotations: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        enc_out = self.encoder(joint_positions, joint_rotations)
        embeddings = enc_out["embeddings"]
        
        dec_out = self.decoder(embeddings)
        
        return {
            "embeddings": embeddings,
            "rotations": dec_out["joint_rotations"],
            "blend_shapes": dec_out["blend_shapes"],
            "eye_params": dec_out["eye_params"]
        }
    
    def compute_loss(self, outputs: Dict[str, Any], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets: Dict with 'rotations', 'blend_shapes', 'eye_params'
        """
        # Targets
        target_rot = targets["rotations"]
        target_blend = targets.get("blend_shapes")
        target_eyes = targets.get("eye_params")
        
        # Outputs
        pred_rot = outputs["rotations"]
        pred_blend = outputs["blend_shapes"]
        pred_eyes = outputs["eye_params"]
        
        # MSE Losses
        mse = nn.MSELoss()
        
        # Align lengths
        min_T = min(pred_rot.shape[1], target_rot.shape[1])
        
        loss_rot = mse(pred_rot[:, :min_T], target_rot[:, :min_T])
        
        total_loss = loss_rot
        loss_dict = {"loss_rot": loss_rot.item()}
        
        if target_blend is not None:
             loss_blend = mse(pred_blend[:, :min_T], target_blend[:, :min_T])
             total_loss += loss_blend
             loss_dict["loss_blend"] = loss_blend.item()
             
        if target_eyes is not None:
             loss_eyes = mse(pred_eyes[:, :min_T], target_eyes[:, :min_T])
             total_loss += loss_eyes
             loss_dict["loss_eyes"] = loss_eyes.item()
             
        loss_dict["total_loss"] = total_loss.item()
        
        return total_loss, loss_dict
