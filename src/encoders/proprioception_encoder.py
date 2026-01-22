"""
Proprioception encoder - processes creature's body state.
Encodes joint positions, rotations, and velocities.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ProprioceptionEncoder(nn.Module):
	"""
	Encoder for proprioceptive information (body state).
	
	Processes full-body tracking data including:
	- Joint positions (3D coordinates)
	- Joint rotations (quaternions)
	- Optional: velocities, accelerations
	
	For VR creatures with full body tracking.
	"""
	
	def __init__(
		self,
		num_joints: int = 24,  # Full body: head, hands, elbows, shoulders, hips, knees, feet, etc.
		position_dim: int = 3,  # x, y, z
		rotation_dim: int = 4,  # quaternion: qw, qx, qy, qz
		embedding_dim: int = 1536,
		temporal_window: int = 10,  # Number of past frames to consider
		use_velocity: bool = True,
		dropout: float = 0.1,
	):
		"""
		Initialize proprioception encoder.
		
		Args:
			num_joints: Number of tracked body joints
			position_dim: Dimensionality of position (3 for 3D)
			rotation_dim: Dimensionality of rotation (4 for quaternion)
			embedding_dim: Output embedding dimension
			temporal_window: Number of frames in temporal context
			use_velocity: Whether to include velocities
			dropout: Dropout probability
		"""
		super().__init__()
		
		self.num_joints = num_joints
		self.position_dim = position_dim
		self.rotation_dim = rotation_dim
		self.embedding_dim = embedding_dim
		self.temporal_window = temporal_window
		self.use_velocity = use_velocity
		
		# Calculate input dimension per joint
		joint_features = position_dim + rotation_dim
		if use_velocity:
			joint_features += position_dim  # Position velocity
			joint_features += rotation_dim  # Rotation velocity (angular velocity)
		
		# Per-joint encoder
		self.joint_encoder = nn.Sequential(
			nn.Linear(joint_features, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
		)
		
		# Combine all joints
		self.joints_combiner = nn.Linear(num_joints * 256, embedding_dim)
		
		# Temporal encoding (for sequence of body states)
		self.temporal_encoder = nn.LSTM(
			input_size=embedding_dim,
			hidden_size=embedding_dim,
			num_layers=2,
			batch_first=True,
			dropout=dropout if temporal_window > 1 else 0,
		)
		
		# Positional embedding for temporal sequence
		self.temporal_position_embedding = nn.Parameter(
			torch.randn(1, temporal_window, embedding_dim)
		)
		
		# Modality embedding
		self.modality_embedding = nn.Parameter(
			torch.randn(1, 1, embedding_dim)
		)
		
		# Layer norm and dropout
		self.layer_norm = nn.LayerNorm(embedding_dim)
		self.dropout = nn.Dropout(dropout)
	
	def compute_velocity(
		self,
		current: torch.Tensor,
		previous: torch.Tensor,
		dt: float = 1/90.0,  # 90 Hz VR tracking
	) -> torch.Tensor:
		"""
		Compute velocity from current and previous states.
		
		Args:
			current: Current state [batch, features]
			previous: Previous state [batch, features]
			dt: Time delta
			
		Returns:
			Velocity [batch, features]
		"""
		return (current - previous) / dt
	
	def forward(
		self,
		joint_positions: torch.Tensor,
		joint_rotations: torch.Tensor,
		previous_positions: Optional[torch.Tensor] = None,
		previous_rotations: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		"""
		Encode proprioceptive data.
		
		Args:
			joint_positions: Joint positions [batch, temporal_window, num_joints, 3]
			joint_rotations: Joint rotations [batch, temporal_window, num_joints, 4]
			previous_positions: Previous positions for velocity calc (optional)
			previous_rotations: Previous rotations for velocity calc (optional)
			
		Returns:
			Dictionary containing:
				- embeddings: Encoded embeddings [batch, temporal_window, embed_dim]
				- attention_mask: All ones [batch, temporal_window]
		"""
		batch_size = joint_positions.shape[0]
		temporal_len = joint_positions.shape[1]
		
		# Flatten temporal dimension temporarily
		positions_flat = joint_positions.reshape(
			batch_size * temporal_len, self.num_joints, self.position_dim
		)
		rotations_flat = joint_rotations.reshape(
			batch_size * temporal_len, self.num_joints, self.rotation_dim
		)
		
		# Concatenate position and rotation per joint
		joint_features = torch.cat([positions_flat, rotations_flat], dim=-1)
		# [B*T, num_joints, pos_dim + rot_dim]
		
		# Add velocities if enabled
		if self.use_velocity:
			if previous_positions is not None:
				pos_vel = self.compute_velocity(positions_flat, previous_positions)
				rot_vel = self.compute_velocity(rotations_flat, previous_rotations)
				joint_features = torch.cat([joint_features, pos_vel, rot_vel], dim=-1)
			else:
				# Pad with zeros if no previous state provided
				# Shape: [B*T, num_joints, pos_dim + rot_dim]
				zeros_pos = torch.zeros_like(positions_flat)
				zeros_rot = torch.zeros_like(rotations_flat)
				joint_features = torch.cat([joint_features, zeros_pos, zeros_rot], dim=-1)
		
		# Encode each joint
		encoded_joints = self.joint_encoder(joint_features)
		# [B*T, num_joints, 256]
		
		# Flatten joints dimension
		combined = encoded_joints.reshape(
			batch_size * temporal_len, -1
		)  # [B*T, num_joints * 256]
		
		# Combine all joints
		embeddings = self.joints_combiner(combined)
		# [B*T, embedding_dim]
		
		# Reshape back to temporal sequence
		embeddings = embeddings.reshape(
			batch_size, temporal_len, self.embedding_dim
		)
		
		# Add temporal positional embedding
		embeddings = embeddings + self.temporal_position_embedding[:, :temporal_len, :]
		
		# Apply LSTM for temporal encoding
		embeddings, _ = self.temporal_encoder(embeddings)
		
		# Add modality embedding
		modal_embeds = self.modality_embedding.expand(batch_size, temporal_len, -1)
		embeddings = embeddings + modal_embeds
		
		# Apply layer norm and dropout
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)
		
		# Create attention mask (all valid)
		attention_mask = torch.ones(
			batch_size, temporal_len,
			dtype=torch.long,
			device=joint_positions.device
		)
		
		return {
			"embeddings": embeddings,
			"attention_mask": attention_mask,
		}
	
	def get_output_dim(self) -> int:
		"""Get output embedding dimension."""
		return self.embedding_dim
