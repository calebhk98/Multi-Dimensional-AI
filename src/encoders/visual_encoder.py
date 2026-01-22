"""
Visual encoder - processes stereo vision (left/right eye).
Uses Vision Transformer (ViT) architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from einops import rearrange


class VisualEncoder(nn.Module):
	"""
	Vision encoder using Vision Transformer (ViT) architecture.
	
	Processes stereo vision from creature's left and right eyes.
	Images are split into patches and processed as sequences.
	"""
	
	def __init__(
		self,
		image_size: int = 224,
		patch_size: int = 16,
		in_channels: int = 3,
		embedding_dim: int = 1536,
		num_layers: int = 12,
		num_heads: int = 12,
		mlp_ratio: int = 4,
		dropout: float = 0.1,
		use_stereo: bool = True,
	):
		"""
		Initialize visual encoder.
		
		Args:
			image_size: Input image size (assumes square)
			patch_size: Size of image patches
			in_channels: Number of input channels (3 for RGB)
			embedding_dim: Embedding dimension
			num_layers: Number of transformer layers
			num_heads: Number of attention heads (will be adjusted if not divisible by embedding_dim)
			mlp_ratio: MLP hidden dim ratio
			dropout: Dropout probability
			use_stereo: Whether to process left/right eyes separately
		"""
		super().__init__()
		
		# Ensure num_heads divides embedding_dim
		if embedding_dim % num_heads != 0:
			# Try to find a valid num_heads
			original_heads = num_heads
			# If embedding_dim is small (e.g. 32), 12 won't work. 8 or 4 might.
			# If embedding_dim is large (e.g. 1536), 12 works.
			# Let's try to match user intent or find largest divisor <= num_heads
			candidates = [h for h in range(num_heads, 0, -1) if embedding_dim % h == 0]
			if candidates:
				num_heads = candidates[0]
			else:
				num_heads = 1 # Fallback
			
			# Warn if possible? No logger set up.

		
		self.image_size = image_size
		self.patch_size = patch_size
		self.embedding_dim = embedding_dim
		self.use_stereo = use_stereo
		
		# Calculate number of patches
		self.num_patches = (image_size // patch_size) ** 2
		
		# Patch embedding (conv layer)
		self.patch_embed = nn.Conv2d(
			in_channels=in_channels,
			out_channels=embedding_dim,
			kernel_size=patch_size,
			stride=patch_size,
		)
		
		# Positional embedding
		self.position_embedding = nn.Parameter(
			torch.randn(1, self.num_patches, embedding_dim)
		)
		
		# Eye-specific embeddings (left vs right)
		if use_stereo:
			self.left_eye_embedding = nn.Parameter(
				torch.randn(1, 1, embedding_dim)
			)
			self.right_eye_embedding = nn.Parameter(
				torch.randn(1, 1, embedding_dim)
			)
		
		# Modality embedding
		self.modality_embedding = nn.Parameter(
			torch.randn(1, 1, embedding_dim)
		)
		
		# Transformer encoder layers
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embedding_dim,
			nhead=num_heads,
			dim_feedforward=embedding_dim * mlp_ratio,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers,
		)
		
		# Layer norm and dropout
		self.layer_norm = nn.LayerNorm(embedding_dim)
		self.dropout = nn.Dropout(dropout)
	
	def forward(
		self,
		left_image: torch.Tensor,
		right_image: Optional[torch.Tensor] = None,
	) -> Dict[str, torch.Tensor]:
		"""
		Encode stereo images.
		
		Args:
			left_image: Left eye image [batch, channels, height, width]
			right_image: Right eye image [batch, channels, height, width] (optional)
			
		Returns:
			Dictionary containing:
				- embeddings: Encoded embeddings [batch, num_patches*2, embed_dim] if stereo
				- attention_mask: All ones [batch, num_patches*2]
		"""
		batch_size = left_image.shape[0]
		
		# Process left eye
		left_patches = self.patch_embed(left_image)  # [B, D, H', W']
		left_patches = rearrange(left_patches, 'b d h w -> b (h w) d')  # [B, N, D]
		
		# Add positional embedding
		left_patches = left_patches + self.position_embedding
		
		if self.use_stereo:
			# Add eye-specific embedding
			left_patches = left_patches + self.left_eye_embedding
		
		# Add modality embedding
		left_patches = left_patches + self.modality_embedding
		
		# If stereo, process right eye
		if right_image is not None and self.use_stereo:
			right_patches = self.patch_embed(right_image)
			right_patches = rearrange(right_patches, 'b d h w -> b (h w) d')
			right_patches = right_patches + self.position_embedding
			right_patches = right_patches + self.right_eye_embedding
			right_patches = right_patches + self.modality_embedding
			
			# Concatenate left and right
			patches = torch.cat([left_patches, right_patches], dim=1)  # [B, 2N, D]
		else:
			patches = left_patches
		
		# Apply layer norm
		patches = self.layer_norm(patches)
		patches = self.dropout(patches)
		
		# Pass through transformer
		embeddings = self.transformer(patches)
		
		# Create attention mask (all valid)
		seq_len = embeddings.shape[1]
		attention_mask = torch.ones(
			batch_size, seq_len,
			dtype=torch.long,
			device=left_image.device
		)
		
		return {
			"embeddings": embeddings,
			"attention_mask": attention_mask,
		}
	
	def get_output_dim(self) -> int:
		"""Get output embedding dimension."""
		return self.embedding_dim
