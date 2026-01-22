"""
Generic Trainer class for MultiModalCreature.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import logging
from tqdm import tqdm
from pathlib import Path

class Trainer:
	"""
	Main training class.
	Handles training loop, checkpointing, and logging.
	"""
	
	def __init__(
		self,
		model: nn.Module,
		config: Dict[str, Any],
		train_loader: DataLoader,
		val_loader: Optional[DataLoader] = None,
		device: str = "cuda" if torch.cuda.is_available() else "cpu",
	):
		self.model = model.to(device)
		self.config = config
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		
		# Optimizer
		train_cfg = config.get("training", {})
		opt_cfg = train_cfg.get("optimizer", {})
		self.optimizer = optim.AdamW(
			model.parameters(),
			lr=float(opt_cfg.get("lr", 3e-4)),
			betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
			weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
		)
		
		self.max_steps = train_cfg.get("max_steps", 1000)
		self.log_interval = train_cfg.get("log_interval", 10)
		self.save_interval = train_cfg.get("save_interval", 100)
		self.save_dir = Path(train_cfg.get("checkpointing", {}).get("save_dir", "checkpoints"))
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)

	def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""Execute one training step."""
		# Move inputs to device
		inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
				for k, v in batch["inputs"].items()}
		
		# Touch data is nested, handle separately
		if "touch_data" in inputs:
			inputs["touch_data"] = {k: v.to(self.device) for k, v in inputs["touch_data"].items()}
		
		# Move targets to device
		targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
				for k, v in batch["targets"].items()}
		# Handle nested animation targets
		if "animation" in targets:
			targets["animation"] = {k: v.to(self.device) for k, v in targets["animation"].items()}
		
		# Forward pass returning hidden states for loss computation
		outputs = self.model(
			internal_voice_tokens=inputs.get("internal_voice_tokens"),
			external_voice_tokens=inputs.get("external_voice_tokens"),
			audio_waveform=inputs.get("audio_waveform"),
			left_eye_image=inputs.get("left_eye_image"),
			right_eye_image=inputs.get("right_eye_image"),
			joint_positions=inputs.get("joint_positions"),
			joint_rotations=inputs.get("joint_rotations"),
			touch_data=inputs.get("touch_data"),
			return_hidden_states=True
		)
		
		# Compute loss
		loss, loss_dict = self.model.compute_loss(outputs, targets)
		
		# Backward pass
		self.optimizer.zero_grad()
		loss.backward()
		
		# Clip gradients (placeholder)
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		
		self.optimizer.step()
		
		return loss, loss_dict

	def train(self):
		"""Main training loop."""
		self.model.train()
		step = 0
		progress_bar = tqdm(total=self.max_steps)
		
		while step < self.max_steps:
			for batch in self.train_loader:
				if step >= self.max_steps:
					break
				
				loss, loss_dict = self.train_step(batch)
				
				# Logging
				if step % self.log_interval == 0:
					loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
					progress_bar.set_description(f"Step {step} | Loss: {loss.item():.4f}")
					# self.logger.info(f"Step {step}: {loss_str}") # Too verbose for tqdm
				
				# Save checkpoint
				if step % self.save_interval == 0 and step > 0:
					self.save_checkpoint(step)
				
				step += 1
				progress_bar.update(1)
				
		progress_bar.close()
		self.logger.info("Training complete.")
		self.save_checkpoint(step, final=True)

	def save_checkpoint(self, step: int, final: bool = False):
		name = "model_final.pt" if final else f"model_step_{step}.pt"
		path = self.save_dir / name
		torch.save({
			'step': step,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'config': self.config,
		}, path)
		self.logger.info(f"Saved checkpoint to {path}")
