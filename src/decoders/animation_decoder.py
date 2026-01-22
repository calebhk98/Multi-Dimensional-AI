"""
Animation decoder for generating creature body movements.
Outputs continuous values for joint angles and facial expressions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class AnimationDecoder(nn.Module):
	"""
	Animation decoder for creature body control.
	
	Generates continuous values for:
	- Joint rotations (24 joints)
	- Facial blend shapes (51 ARKit standard)
	- Eye tracking
	
	Unlike other decoders, outputs continuous values, not discrete tokens.
	"""
	
	def __init__(
		self,
		embedding_dim: int = 1536,
		num_joints: int = 24,
		num_blend_shapes: int = 51,  # ARKit facial blend shapes
		dropout: float = 0.1,
	):
		"""
		Initialize animation decoder.
		
		Args:
			embedding_dim: Input embedding dimension
			num_joints: Number of body joints
			num_blend_shapes: Number of facial blend shapes
			dropout: Dropout probability
		"""
		super().__init__()
		
		self.embedding_dim = embedding_dim
		self.num_joints = num_joints
		self.num_blend_shapes = num_blend_shapes
		
		# Body animation head
		# Each joint: quaternion (4 values)
		self.body_head = nn.Sequential(
			nn.Linear(embedding_dim, 512),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(512, num_joints * 4),  # 4 for quaternion
		)
		
		# Facial animation head
		# Blend shapes: 0.0-1.0 values
		self.face_head = nn.Sequential(
			nn.Linear(embedding_dim, 256),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(256, num_blend_shapes),
		)
		
		# Eye tracking head
		# 2 eyes × (3D direction + openness)
		self.eye_head = nn.Sequential(
			nn.Linear(embedding_dim, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 8),  # 2 eyes × 4 values each
		)
		
		# Layer norm
		self.layer_norm = nn.LayerNorm(embedding_dim)
		
		# Dropout
		self.dropout = nn.Dropout(dropout)
	
	def normalize_quaternion(self, quat: torch.Tensor) -> torch.Tensor:
		"""
		Normalize quaternions to unit length.
		
		Args:
			quat: Quaternions [..., 4]
			
		Returns:
			Normalized quaternions [..., 4]
		"""
		return F.normalize(quat, p=2, dim=-1)
	
	def forward(
		self,
		hidden_states: torch.Tensor,
	) -> Dict[str, torch.Tensor]:
		"""
		Generate animation parameters.
		
		Args:
			hidden_states: Hidden states [batch, seq_len, dim]
			
		Returns:
			Dictionary containing:
				- joint_rotations: Quaternions [batch, seq_len, num_joints, 4]
				- blend_shapes: Blend shape weights [batch, seq_len, num_blend_shapes]
				- eye_params: Eye parameters [batch, seq_len, 8]
		"""
		batch_size, seq_len, _ = hidden_states.shape
		
		# Normalize and dropout
		hidden_states = self.layer_norm(hidden_states)
		hidden_states = self.dropout(hidden_states)
		
		# Generate body rotations
		body_output = self.body_head(hidden_states)
		# [batch, seq_len, num_joints * 4]
		
		joint_rotations = body_output.view(
			batch_size, seq_len, self.num_joints, 4
		)
		# Normalize quaternions
		joint_rotations = self.normalize_quaternion(joint_rotations)
		
		# Generate facial blend shapes
		face_output = self.face_head(hidden_states)
		# [batch, seq_len, num_blend_shapes]
		
		# Apply sigmoid to keep in [0, 1] range
		blend_shapes = torch.sigmoid(face_output)
		
		# Generate eye parameters
		eye_output = self.eye_head(hidden_states)
		# [batch, seq_len, 8]
		
		# Split into direction (3D normalized) and openness (sigmoid)
		left_eye_dir = F.normalize(eye_output[..., :3], p=2, dim=-1)
		left_eye_open = torch.sigmoid(eye_output[..., 3:4])
		right_eye_dir = F.normalize(eye_output[..., 4:7], p=2, dim=-1)
		right_eye_open = torch.sigmoid(eye_output[..., 7:8])
		
		eye_params = torch.cat([
			left_eye_dir, left_eye_open,
			right_eye_dir, right_eye_open
		], dim=-1)
		
		return {
			"joint_rotations": joint_rotations,
			"blend_shapes": blend_shapes,
			"eye_params": eye_params,
		}
	
	def compute_loss(
		self,
		hidden_states: torch.Tensor,
		target_rotations: torch.Tensor,
		target_blend_shapes: torch.Tensor,
		target_eye_params: torch.Tensor,
		loss_weights: Optional[Dict[str, float]] = None,
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Compute loss for animation parameters.
		
		Args:
			hidden_states: Hidden states
			target_rotations: Target joint rotations [batch, seq_len, num_joints, 4]
			target_blend_shapes: Target blend shapes [batch, seq_len, num_blend_shapes]
			target_eye_params: Target eye parameters [batch, seq_len, 8]
			loss_weights: Optional weights for each loss component
			
		Returns:
			Tuple of (total_loss, loss_dict)
		"""
		if loss_weights is None:
			loss_weights = {
				"body": 1.0,
				"face": 0.5,
				"eyes": 0.3,
			}
		
		# Generate predictions
		predictions = self.forward(hidden_states)
		
		# Body loss (quaternion distance)
		# Use geodesic distance for quaternions
		pred_rots = predictions["joint_rotations"]
		
		# Dot product between quaternions
		dot_product = (pred_rots * target_rotations).sum(dim=-1)
		# Clamp to avoid numerical issues
		dot_product = torch.clamp(dot_product, -1.0, 1.0)
		# Geodesic distance
		body_loss = (1.0 - torch.abs(dot_product)).mean()
		
		# Face loss (MSE for blend shapes)
		face_loss = F.mse_loss(
			predictions["blend_shapes"],
			target_blend_shapes
		)
		
		# Eye loss (MSE)
		eye_loss = F.mse_loss(
			predictions["eye_params"],
			target_eye_params
		)
		
		# Combine losses
		total_loss = (
			loss_weights["body"] * body_loss +
			loss_weights["face"] * face_loss +
			loss_weights["eyes"] * eye_loss
		)
		
		loss_dict = {
			"body_loss": body_loss,
			"face_loss": face_loss,
			"eye_loss": eye_loss,
			"total_animation_loss": total_loss,
		}
		
		return total_loss, loss_dict
	
	def to_vrchat_format(
		self,
		joint_rotations: torch.Tensor,
		blend_shapes: torch.Tensor,
		eye_params: torch.Tensor,
	) -> Dict[str, torch.Tensor]:
		"""
		Convert to VRChat-compatible format.
		
		Args:
			joint_rotations: Joint rotations
			blend_shapes: Facial blend shapes
			eye_params: Eye parameters
			
		Returns:
			Dictionary with VRChat-formatted parameters
		"""
		# This would convert to VRChat's specific format
		# Placeholder - actual implementation depends on VRChat API
		
		return {
			"Humanoid": {
				"BodyRotations": joint_rotations,
			},
			"FaceTracking": {
				"BlendShapes": blend_shapes,
				"EyeTracking": eye_params,
			},
		}
