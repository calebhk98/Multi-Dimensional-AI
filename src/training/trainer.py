"""
Generic Trainer class for MultiModalCreature.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List, Tuple, Union
import logging
from tqdm import tqdm
from pathlib import Path
import time

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
		"""
		Initialize trainer.

		Args:
			model: Multi-modal creature model to train
			config: Configuration dictionary
			train_loader: DataLoader for training data
			val_loader: DataLoader for validation data (optional)
			device: Device string ('cuda' or 'cpu') to train on
		"""
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
		
		# Metrics tracking
		self.metrics_history = []
		self.step_start_time = None
		
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)

	def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
		"""
		Execute one training step.

		Args:
			batch: Batch of data containing 'inputs' and 'targets'

		Returns:
			Tuple of (loss_tensor, loss_dict_for_logging)
		"""
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
			**inputs,
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
		
		# Extract per-modality losses if available
		modality_losses = self._extract_modality_losses(loss_dict)
		
		return loss, {**loss_dict, **modality_losses}

	def _extract_modality_losses(self, loss_dict: Dict[str, Any]) -> Dict[str, float]:
		"""
		Purpose:
			Extract per-modality losses from the loss dictionary.
			
		Workflow:
			1. Check for modality-specific keys in loss_dict
			2. Convert tensor values to floats
			3. Return dict with modality losses
			
		ToDo:
			None
			
		Args:
			loss_dict: Dictionary of losses from model.compute_loss
			
		Returns:
			Dict mapping modality names to loss values
		"""
		modality_keys = ["internal_text", "external_text", "audio", "animation"]
		modality_losses = {}
		
		for key in modality_keys:
			if key in loss_dict:
				value = loss_dict[key]
				modality_losses[f"loss_{key}"] = value.item() if isinstance(value, torch.Tensor) else value
				
		return modality_losses
	
	def _compute_throughput_metrics(self, batch_size: int, step_time: float) -> Dict[str, float]:
		"""
		Purpose:
			Compute throughput metrics for monitoring training performance.
			
		Workflow:
			1. Calculate samples/sec from batch size and step time
			2. Get current memory usage if on CUDA
			3. Return metrics dict
			
		ToDo:
			None
			
		Args:
			batch_size: Number of samples in batch
			step_time: Time taken for step in seconds
			
		Returns:
			Dict with throughput and memory metrics
		"""
		metrics = {}
		
		if step_time > 0:
			metrics["samples_per_sec"] = batch_size / step_time
			
		if torch.cuda.is_available() and self.device == "cuda":
			metrics["memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
			metrics["memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
			
		return metrics
	
	def _process_training_step(self, batch, step, progress_bar):
		"""
		Process a single training iteration including forward, backward, logging, and checkpointing.
		
		Args:
			batch: Current batch of data
			step: Current step number
			progress_bar: tqdm progress bar for visualization
			
		Returns:
			None
		"""
		# Track step timing
		step_start = time.time()
		
		loss, loss_dict = self.train_step(batch)
		
		step_time = time.time() - step_start
		
		# Get batch size from inputs
		batch_size = self._get_batch_size(batch)
		
		# Compute throughput metrics
		throughput_metrics = self._compute_throughput_metrics(batch_size, step_time)
		
		# Combine all metrics
		all_metrics = {**loss_dict, **throughput_metrics, "step_time_sec": step_time}
		
		# Store metrics history
		self.metrics_history.append({"step": step, **all_metrics})
		
		# Logging - only log at specified intervals
		if step % self.log_interval == 0:
			loss_str = ", ".join([f"{k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}" for k, v in loss_dict.items()])
			progress_bar.set_description(f"Step {step} | Loss: {loss.item():.4f} | {throughput_metrics.get('samples_per_sec', 0):.1f} samples/s")
			# self.logger.info(f"Step {step}: {loss_str}") # Too verbose for tqdm
		
		# Save checkpoint at specified intervals (but not at step 0)
		if step % self.save_interval == 0 and step > 0:
			self.save_checkpoint(step)
		
		progress_bar.update(1)
	
	def _get_batch_size(self, batch: Dict[str, Any]) -> int:
		"""
		Purpose:
			Extract batch size from batch data.
			
		Workflow:
			1. Check inputs dict for any tensor
			2. Return its first dimension
			3. Default to 1 if not found
			
		ToDo:
			None
			
		Args:
			batch: Batch dictionary
			
		Returns:
			Batch size as integer
		"""
		if "inputs" not in batch:
			return 1
		
		for key, value in batch["inputs"].items():
			if isinstance(value, torch.Tensor):
				return value.shape[0]
			
			# Handle nested dictionaries (e.g., touch_data)
			if not isinstance(value, dict):
				continue
			
			for subvalue in value.values():
				if isinstance(subvalue, torch.Tensor):
					return subvalue.shape[0]
		
		return 1

	def train(self):
		"""Main training loop."""
		self.model.train()
		step = 0
		progress_bar = tqdm(total=self.max_steps)
		
		while step < self.max_steps:
			for batch in self.train_loader:
				if step >= self.max_steps:
					break
				
				self._process_training_step(batch, step, progress_bar)
				step += 1
				
		progress_bar.close()
		self.logger.info("Training complete.")
		self.save_checkpoint(step, final=True)

	def save_checkpoint(self, step: int, final: bool = False):
		"""
		Save model checkpoint.

		Args:
			step: Current training step
			final: Whether this is the final model save after training completes
		"""
		name = "model_final.pt" if final else f"model_step_{step}.pt"
		path = self.save_dir / name
		torch.save({
			'step': step,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'config': self.config,
		}, path)
		self.logger.info(f"Saved checkpoint to {path}")

	def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
		"""
		Load a checkpoint and restore model/optimizer state.

		Args:
			checkpoint_path: Path to the .pt file.

		Returns:
			Dict containing the loaded checkpoint data (step, config, etc).
		"""
		path = Path(checkpoint_path)
		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found at {path}")

		checkpoint = torch.load(path, map_location=self.device)
		
		# Restore model
		self.model.load_state_dict(checkpoint['model_state_dict'])
		
		# Restore optimizer
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
		self.logger.info(f"Loaded checkpoint from {path} (Step {checkpoint.get('step', 'unknown')})")
		
		return checkpoint
