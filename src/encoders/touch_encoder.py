"""
Touch encoder - processes tactile sensations from VR world.
Encodes collision data, surface properties, and contact forces.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class TouchEncoder(nn.Module):
	"""
	Encoder for touch/tactile input.
	
	Processes collision and surface contact information from VR physics engine:
	- Contact points (hands, feet, body parts)
	- Contact forces
	- Surface materials
	- Temperature (if simulated)
	"""
	
	def __init__(
		self,
		num_contact_points: int = 10,  # hands, feet, etc.
		embedding_dim: int = 1536,
		surface_types: int = 8,  # wood, metal, soft, liquid, etc.
		max_contacts: int = 20,  # Max simultaneous contacts
		dropout: float = 0.1,
	):
		"""
		==============================================================================
		Function: __init__
		==============================================================================
		Purpose:  Initializes the TouchEncoder module. Sets up embeddings for contact
		          points, surface types, and continuous features (force, position,
		          temperature), along with attention mechanisms to aggregate multiple
		          contacts.

		Parameters:
		    - num_contact_points: int
		        Number of potential contact points on body (default: 10, e.g., hands, feet).
		    - embedding_dim: int
		        Output embedding dimension (default: 1536).
		    - surface_types: int
		        Number of surface material types (default: 8, e.g., wood, metal).
		    - max_contacts: int
		        Maximum simultaneous contacts to process (default: 20).
		    - dropout: float
		        Dropout probability (default: 0.1).

		Returns:
		    None

		Dependencies:
		    - torch.nn.Embedding
		    - torch.nn.Sequential
		    - torch.nn.Linear
		    - torch.nn.ReLU
		    - torch.nn.MultiheadAttention
		    - torch.nn.LayerNorm
		    - torch.nn.Dropout

		Processing Workflow:
		    1.  Store configuration parameters.
		    2.  Initialize `contact_point_embed` and `surface_type_embed` layers.
		    3.  Define `continuous_encoder` (MLP) for scalar/vector features.
		    4.  Define `contact_combiner` (MLP) to merge all contact features.
		    5.  Define `contact_aggregator` (MultiheadAttention) for global contact context.
		    6.  Initialize `position_embedding` and `modality_embedding`.
		    7.  Initialize `dropout` and `layer_norm`.

		ToDo:
		    - None

		Usage:
		    model = TouchEncoder(num_contact_points=10, embedding_dim=512)
		==============================================================================
		"""
		super().__init__()
		
		self.num_contact_points = num_contact_points
		self.embedding_dim = embedding_dim
		self.surface_types = surface_types
		self.max_contacts = max_contacts
		
		# Contact point embedding (which body part is touching)
		self.contact_point_embed = nn.Embedding(
			num_embeddings=num_contact_points,
			embedding_dim=128,
		)
		
		# Surface type embedding
		self.surface_type_embed = nn.Embedding(
			num_embeddings=surface_types,
			embedding_dim=128,
		)
		
		# Continuous features encoder (force, position, temperature)
		# Input: force magnitude (1) + contact point 3D (3) + temperature (1) = 5
		self.continuous_encoder = nn.Sequential(
			nn.Linear(5, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
		)
		
		# Combine all contact features
		self.contact_combiner = nn.Sequential(
			nn.Linear(128 * 3, embedding_dim),  # 3 parts: point, surface, continuous
			nn.ReLU(),
			nn.Linear(embedding_dim, embedding_dim),
		)
		
		# Aggregate multiple contacts
		# Calculate safe num_heads
		num_heads = 8
		if embedding_dim % num_heads != 0:
			for h in [4, 2, 1]:
				if embedding_dim % h == 0:
					num_heads = h
					break

		self.contact_aggregator = nn.MultiheadAttention(
			embed_dim=embedding_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)
		
		# Positional embedding for contact sequence
		self.position_embedding = nn.Parameter(
			torch.randn(1, max_contacts, embedding_dim)
		)
		
		# Modality embedding
		self.modality_embedding = nn.Parameter(
			torch.randn(1, 1, embedding_dim)
		)
		
		# Layer norm and dropout
		self.layer_norm = nn.LayerNorm(embedding_dim)
		self.dropout = nn.Dropout(dropout)
	
	def forward(
		self,
		contact_active: torch.Tensor,  # [batch, num_contacts] bool
		contact_points: torch.Tensor,  # [batch, num_contacts] int (which body part)
		contact_forces: torch.Tensor,  # [batch, num_contacts, 1] float
		contact_positions: torch.Tensor,  # [batch, num_contacts, 3] float
		surface_types: torch.Tensor,  # [batch, num_contacts] int
		temperatures: Optional[torch.Tensor] = None,  # [batch, num_contacts, 1] float
	) -> Dict[str, torch.Tensor]:
		"""
		==============================================================================
		Function: forward
		==============================================================================
		Purpose:  Processes tactile sensations (contacts) into embeddings.

		Parameters:
		    - contact_active: torch.Tensor
		        Boolean mask of active contacts [batch, num_contacts].
		    - contact_points: torch.Tensor
		        IDs for which body part is touching [batch, num_contacts].
		    - contact_forces: torch.Tensor
		        Contact force magnitudes [batch, num_contacts, 1].
		    - contact_positions: torch.Tensor
		        3D positions of contact points relative to body [batch, num_contacts, 3].
		    - surface_types: torch.Tensor
		        Surface material type IDs [batch, num_contacts].
		    - temperatures: Optional[torch.Tensor]
		        Surface temperatures [batch, num_contacts, 1] (optional).

		Returns:
		    Dict[str, torch.Tensor] - Dictionary containing:
		        - "embeddings": Encoded embeddings [batch, num_contacts, embed_dim].
		        - "attention_mask": Contact active mask [batch, num_contacts].

		Dependencies:
		    - self.contact_point_embed
		    - self.surface_type_embed
		    - self.continuous_encoder
		    - self.contact_combiner
		    - self.contact_aggregator
		    - self.modality_embedding
		    - self.position_embedding

		Processing Workflow:
		    1.  Initialize temperatures to zero if not provided.
		    2.  Embed contact points (which body part).
		    3.  Embed surface types.
		    4.  Concatenate continuous features (force, position, temperature).
		    5.  Encode continuous features using `continuous_encoder`.
		    6.  Concatenate all features (point, surface, continuous).
		    7.  project to embedding dimension using `contact_combiner`.
		    8.  Add `position_embedding`.
		    9.  Aggregate contacts using self-attention (`contact_aggregator`) with masking.
		    10. Add `modality_embedding`.
		    11. Apply layer norm and dropout.
		    12. Create attention mask from `contact_active`.

		ToDo:
		    - None

		Usage:
		    output = model(active_mask, points, forces, positions, surfaces)
		==============================================================================
		"""
		batch_size, num_contacts = contact_active.shape
		
		# Default temperatures if not provided
		if temperatures is None:
			temperatures = torch.zeros(
				batch_size, num_contacts, 1,
				device=contact_active.device
			)
		
		# Embed contact points (which body part)
		point_embeds = self.contact_point_embed(contact_points)
		# [batch, num_contacts, 128]
		
		# Embed surface types
		surface_embeds = self.surface_type_embed(surface_types)
		# [batch, num_contacts, 128]
		
		# Encode continuous features
		continuous_features = torch.cat([
			contact_forces,
			contact_positions,
			temperatures,
		], dim=-1)  # [batch, num_contacts, 5]
		
		continuous_embeds = self.continuous_encoder(continuous_features)
		# [batch, num_contacts, 128]
		
		# Combine all features
		combined = torch.cat([
			point_embeds,
			surface_embeds,
			continuous_embeds,
		], dim=-1)  # [batch, num_contacts, 384]
		
		# Project to embedding dimension
		embeddings = self.contact_combiner(combined)
		# [batch, num_contacts, embedding_dim]
		
		# Add positional embedding
		embeddings = embeddings + self.position_embedding[:, :num_contacts, :]
		
		# Self-attention to aggregate contacts
		embeddings, _ = self.contact_aggregator(
			query=embeddings,
			key=embeddings,
			value=embeddings,
			key_padding_mask=~contact_active,  # Mask inactive contacts
		)
		
		# Add modality embedding
		modal_embeds = self.modality_embedding.expand(batch_size, num_contacts, -1)
		embeddings = embeddings + modal_embeds
		
		# Apply layer norm and dropout
		embeddings = self.layer_norm(embeddings)
		embeddings = self.dropout(embeddings)
		
		# Use contact_active as attention mask
		attention_mask = contact_active.long()
		
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
