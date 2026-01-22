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
        Initialize touch encoder.
        
        Args:
            num_contact_points: Number of potential contact points on body
            embedding_dim: Output embedding dimension
            surface_types: Number of surface material types
            max_contacts: Maximum simultaneous contacts to process
            dropout: Dropout probability
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
        Encode touch sensations.
        
        Args:
            contact_active: Boolean mask of active contacts
            contact_points: Body part IDs for each contact
            contact_forces: Contact force magnitudes
            contact_positions: 3D positions of contact points (relative to body)
            surface_types: Surface material type IDs
            temperatures: Surface temperatures (optional)
            
        Returns:
            Dictionary containing:
                - embeddings: Encoded embeddings [batch, num_contacts, embed_dim]
                - attention_mask: Contact active mask [batch, num_contacts]
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
        """Get output embedding dimension."""
        return self.embedding_dim
