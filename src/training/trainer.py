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
import math
import contextlib
from torch.cuda.amp import GradScaler, autocast

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
		
		# Scheduler components
		self.scheduler_cfg = train_cfg.get("scheduler", {})
		self.warmup_steps = self.scheduler_cfg.get("warmup_steps", 1000)
		self.min_lr = float(self.scheduler_cfg.get("min_lr", 6e-5))
		# Max steps needed for scheduler
		self.max_steps = train_cfg.get("max_steps", 100000)
		self.scheduler = self._get_cosine_schedule_with_warmup(
			self.optimizer, 
			num_warmup_steps=self.warmup_steps, 
			num_training_steps=self.max_steps,
			min_lr=self.min_lr
		)
		
		self.log_interval = train_cfg.get("log_interval", 10)
		self.save_interval = train_cfg.get("save_interval", 1000)
		self.save_dir = Path(config.get("checkpointing", {}).get("save_dir", "checkpoints"))
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		# Optimization flags
		self.mixed_precision = train_cfg.get("mixed_precision", "no")
		self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
		self.enable_flash_attention = train_cfg.get("enable_flash_attention", True)
		self.gradient_checkpointing = train_cfg.get("gradient_checkpointing", False)
		self.compile_model = train_cfg.get("compile_model", False)
		
		# Setup scaler for mixed precision
		self.scaler = GradScaler(enabled=(self.mixed_precision != "no"))
		self.dtype = torch.bfloat16 if self.mixed_precision == "bf16" else (torch.float16 if self.mixed_precision == "fp16" else torch.float32)

		# Apply optimizations
		if self.gradient_checkpointing:
			if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "enable_gradient_checkpointing"):
				self.model.transformer.enable_gradient_checkpointing(True)
		
		if self.compile_model and torch.cuda.is_available():
			print("Compiling model via torch.compile...")
			self.model = torch.compile(self.model)

		# Metrics tracking
		self.metrics_history = []
		self.step_start_time = None
		self.total_tokens_processed = 0
		
		# MFU Estimation CONSTANTS (Approximate)
		# 3090 Peak TFLOPS: ~71 (FP16 Tensor), ~35 (FP32)
		# A100 Peak TFLOPS: ~312 (BF16 Tensor)
		# We'll assume a user configurable peak or default to 3090-ish level for MFU calc (70 TFLOPS)
		self.peak_flops = 70e12 
		self.model_params = sum(p.numel() for p in self.model.parameters())
		
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger(__name__)

	def _get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
		"""
		Simple cosine scheduler with warmup.
		
		Args:
			optimizer: Optimizer instance.
			num_warmup_steps: Number of warmup steps.
			num_training_steps: Total training steps.
			min_lr: Minimum learning rate.
			
		Returns:
			LambdaLR scheduler.
		"""
		def lr_lambda(current_step):
			"""
			Lambda function for LR schedule.
			
			Args:
				current_step: Current training step.
				
			Returns:
				LR multiplier.
			"""
			if current_step < num_warmup_steps:
				return float(current_step) / float(max(1, num_warmup_steps))
			progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
			return max(min_lr / optimizer.defaults['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
		return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
		
		# Flash Attention Context
		# sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
		# We primarily enable flash, fall back to mem_efficient or math if needed.
		# If user wants strict flash, we can disable others.
		ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True) if (torch.cuda.is_available() and self.enable_flash_attention) else contextlib.nullcontext()
		
		with ctx:
			with autocast(dtype=self.dtype, enabled=(self.mixed_precision != "no")):
				# Forward pass returning hidden states for loss computation
				outputs = self.model(
					**inputs,
					return_hidden_states=True
				)
				
				# Compute loss
				loss, loss_dict = self.model.compute_loss(outputs, targets)
				# Divide loss by gradient accumulation steps
				loss = loss / self.gradient_accumulation_steps
		
		# Backward pass with scaler
		self.scaler.scale(loss).backward()
		
		# Extract per-modality losses if available (unscaled for logging)
		modality_losses = self._extract_modality_losses(loss_dict)
		
		# Return scaled loss for tracking? Standard approach is to log the true loss.
		# loss_dict values are already detached and unscaled usually, but if compute_loss returns tensor...
		# self.model.compute_loss usually computes mean.
		
		return loss * self.gradient_accumulation_steps, {**loss_dict, **modality_losses}

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
		
		# Estimate FLOPS and MFU
		# 6 * params * tokens (approx for training)
		tokens_processed = batch_size * 512 # Placeholder, ideally use actual seq len
		# We can get actual seq len from inputs
		if "inputs" in batch and "internal_voice" in batch["inputs"]:
			tokens_processed = batch["inputs"]["internal_voice"].numel()
		
		flops = 6 * self.model_params * tokens_processed
		tflops = flops / step_time / 1e12
		mfu = tflops / (self.peak_flops / 1e12)
		
		throughput_metrics["tflops"] = tflops
		throughput_metrics["mfu"] = mfu
		
		# Combine all metrics
		all_metrics = {
			**loss_dict, 
			**throughput_metrics, 
			"step_time_sec": step_time,
			"lr": self.optimizer.param_groups[0]['lr']
		}
		
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
		
		curriculum_cfg = self.config.get("training", {}).get("curriculum", {})
		curriculum_stages = curriculum_cfg.get("stages", [])
		use_curriculum = curriculum_cfg.get("enabled", False)
		
		current_seq_len = 0 # Default (full)
		
		# Initialize gradients
		self.optimizer.zero_grad()
		
		while step < self.max_steps:
			for batch in self.train_loader:
				if step >= self.max_steps:
					break
					
				# Curriculum Logic
				if use_curriculum:
					current_seq_len = self._get_curriculum_seq_len(step, curriculum_stages)

					# Slicing logic would go here:
					# Recursively slice tensors in batch to [:tgt_len] along seq dimension?
					# For now, assuming batch is already compliant or we verify usage.
					# Note: Implementing robust recursive slicing for nested dicts:
					pass # TODO: Implement slice_batch(batch, current_seq_len)
				
				# Training Step
				# The logic for gradient accumulation is inside train_step? No, train_step returns loss.
				# We moved optimizer step OUT of train_step to handle accumulation here properly?
				# Wait, my previous edit kept optimizer.step logic inside train_step? 
				# No, I removed optimizer.step() from the replacement chunk for train_step!
				# So I must add it here.
				
				loss, metrics = self.train_step(batch)
				
				# Gradient Accumulation Step
				if (step + 1) % self.gradient_accumulation_steps == 0:
					# Clip gradients
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("training", {}).get("max_grad_norm", 1.0))
					
					# Step optimizer
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.optimizer.zero_grad()
					self.scheduler.step()
				
				self._process_training_step_logging(metrics, step, progress_bar)
				step += 1
				
		progress_bar.close()
		self.logger.info("Training complete.")
		self.save_checkpoint(step, final=True)

	def _process_training_step_logging(self, metrics, step, progress_bar):
		"""
		Handle logging and checkpoint saving during training.
		
		Args:
			metrics: Dict of metrics from train_step.
			step: Current training step.
			progress_bar: tqdm progress bar instance.
		"""
		# Store metrics history
		self.metrics_history.append({"step": step, **metrics})
		
		# Logging
		if step % self.log_interval == 0:
			loss_val = metrics.get('total_loss', 0)
			mfu_val = metrics.get('mfu', 0)
			tokens_sec = metrics.get('samples_per_sec', 0) * 512 # approx
			
			desc = f"Step {step} | Loss: {loss_val:.4f} | MFU: {mfu_val:.1%} | LR: {metrics.get('lr', 0):.2e}"
			progress_bar.set_description(desc)
			
		# Save checkpoint
		if step % self.save_interval == 0 and step > 0:
			self.save_checkpoint(step)
		
		progress_bar.update(1)

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

	def _get_curriculum_seq_len(self, step: int, stages: List[Dict[str, int]]) -> int:
		"""
		Determine target sequence length based on training step.

		Args:
			step: Current training step.
			stages: List of stage dicts with 'step' and 'seq_len'.

		Returns:
			Target sequence length.
		"""
		tgt_len = 10000 # Default max
		for stage in stages:
			if step >= stage["step"]:
				tgt_len = stage["seq_len"]
		return tgt_len
