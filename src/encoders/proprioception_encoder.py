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
		**kwargs,  # Absorb extra config args
	):
		"""
		==============================================================================
		Function: __init__
		==============================================================================
		Purpose:  Initializes the ProprioceptionEncoder module. Sets up the dimensions,
		          layers (joint encoder, combiner, temporal encoder), and embeddings
		          (temporal position and modality) required to process body state data.

		Parameters:
		    - num_joints: int
		        Number of tracked body joints (default: 24).
		    - position_dim: int
		        Dimensionality of position (default: 3 for x, y, z).
		    - rotation_dim: int
		        Dimensionality of rotation (default: 4 for quaternion).
		    - embedding_dim: int
		        Output embedding dimension (default: 1536).
		    - temporal_window: int
		        Number of past frames to consider (default: 10).
		    - use_velocity: bool
		        Whether to include velocities in the input features (default: True).
		    - dropout: float
		        Dropout probability (default: 0.1).

		Returns:
		    None

		Dependencies:
		    - torch.nn.Sequential
		    - torch.nn.Linear
		    - torch.nn.LSTM
		    - torch.nn.LayerNorm
		    - torch.nn.Dropout

		Processing Workflow:
		    1.  Store configuration parameters.
		    2.  Calculate input dimension per joint based on position, rotation, and velocities.
		    3.  Define `joint_encoder` (MLP) to process individual joint features.
		    4.  Define `joints_combiner` (Linear) to aggregate all joints.
		    5.  Define `temporal_encoder` (LSTM) for sequence processing.
		    6.  Initialize `temporal_position_embedding` and `modality_embedding`.
		    7.  Initialize normalization and dropout layers.

		ToDo:
		    - None

		Usage:
		    model = ProprioceptionEncoder(num_joints=24, embedding_dim=512)
		==============================================================================
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
		==============================================================================
		Function: compute_velocity
		==============================================================================
		Purpose:  Computes the velocity between the current and previous states.
		          Used for both position and rotation velocities.

		Parameters:
		    - current: torch.Tensor
		        Current state tensor [batch, features].
		    - previous: torch.Tensor
		        Previous state tensor [batch, features].
		    - dt: float
		        Time delta between frames (default: 1/90.0 for 90 Hz VR).

		Returns:
		    torch.Tensor - Velocity tensor [batch, features].

		Dependencies:
		    - None

		Processing Workflow:
		    1.  Subtract previous state from current state.
		    2.  Divide by time delta `dt`.

		ToDo:
		    - None

		Usage:
		    vel = self.compute_velocity(curr_pos, prev_pos)
		==============================================================================
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
		==============================================================================
		Function: forward
		==============================================================================
		Purpose:  Processes the proprioceptive input data (positions, rotations)
		          through the encoding layers to produce embeddings.

		Parameters:
		    - joint_positions: torch.Tensor
		        Joint positions [batch, temporal_window, num_joints, 3].
		    - joint_rotations: torch.Tensor
		        Joint rotations [batch, temporal_window, num_joints, 4].
		    - previous_positions: Optional[torch.Tensor]
		        Previous positions for velocity calculation (optional).
		    - previous_rotations: Optional[torch.Tensor]
		        Previous rotations for velocity calculation (optional).

		Returns:
		    Dict[str, torch.Tensor] - Dictionary containing:
		        - "embeddings": Encoded embeddings [batch, temporal_window, embed_dim]
		        - "attention_mask": Attention mask [batch, temporal_window]

		Dependencies:
		    - self.joint_encoder
		    - self.joints_combiner
		    - self.temporal_encoder
		    - self.compute_velocity (if use_velocity is True)
		    - self.modality_embedding
		    - self.temporal_position_embedding

		Processing Workflow:
		    1.  Flatten temporal dimension.
		    2.  Concatenate position and rotation per joint.
		    3.  Compute and append velocities if `use_velocity` is True (or pad with zeros).
		    4.  Pass through `joint_encoder` (MLP) for each joint.
		    5.  Flatten and pass through `joints_combiner` to get single vector per frame.
		    6.  Reshape back to temporal sequence.
		    7.  Add `temporal_position_embedding`.
		    8.  Pass through `temporal_encoder` (LSTM).
		    9.  Add `modality_embedding`.
		    10. Apply layer norm and dropout.
		    11. Generate attention mask.

		ToDo:
		    - None

		Usage:
		    output = model(positions, rotations)
		==============================================================================
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
		"""
		==============================================================================
		Function: get_output_dim
		==============================================================================
		Purpose:  Returns the size of the output embeddings produced by this encoder.

		Parameters:
		    - None

		Returns:
		    int - Embedding dimension size (e.g., 1536).

		Dependencies:
		    - None

		Processing Workflow:
		    1.  Return `self.embedding_dim`.

		ToDo:
		    - None

		Usage:
		    dim = model.get_output_dim()
		==============================================================================
		"""
		return self.embedding_dim
