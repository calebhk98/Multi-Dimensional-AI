"""
Profiling script for multi-modal training.
Uses torch.profiler to capture memory and compute metrics.
"""

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config constants
PROFILE_STEPS = 5
WARMUP_STEPS = 2
OUTPUT_DIR = "profiler_output"


def create_mock_batch(batch_size=4):
	"""
	Create a mock batch for profiling.
	
	Args:
		batch_size: Number of samples in batch.
		
	Returns:
		Dict with inputs and targets.
	"""
	return {
		"inputs": {
			"vision_left": torch.randn(batch_size, 3, 224, 224),
			"audio": torch.randn(batch_size, 16000),
			"touch": torch.randn(batch_size, 5),
		},
		"targets": {}
	}


def profile_training():
	"""
	Run profiling on training loop.
	
	Workflow:
		1. Import Trainer and model.
		2. Create mock data.
		3. Profile train_step calls.
		4. Export trace to OUTPUT_DIR.

	Returns:
		None
	"""
	try:
		from src.training.trainer import Trainer
		from src.models.multimodal_creature import MultiModalCreature
		model_class = MultiModalCreature
	except ImportError:
		logger.warning("MultiModalCreature not found, using mock model.")
		import torch.nn as nn
		
		class MockModel(nn.Module):
			def __init__(self):
				"""Initialize MockModel."""
				super().__init__()
				self.param = nn.Parameter(torch.randn(1))
				
			def forward(self, **kwargs):
				"""
				Forward pass.

				Returns:
					dict: Output.
				"""
				return {"hidden": self.param}
				
			def compute_loss(self, outputs, targets):
				"""
				Compute loss.
				
				Args:
					outputs: Outputs.
					targets: Targets.
					
				Returns:
					tuple: Loss and dict.
				"""
				return torch.tensor(0.1, requires_grad=True), {"total": 0.1}
		
		model_class = MockModel
	
	# Create model and trainer
	config = {"training": {"max_steps": PROFILE_STEPS}}
	model = model_class() if model_class.__name__ == "MockModel" else model_class(config={})
	
	# Mock dataset
	dataset = TensorDataset(torch.randn(100, 10))
	loader = DataLoader(dataset, batch_size=4)
	
	from src.training.trainer import Trainer
	trainer = Trainer(model, config, loader, device="cpu")
	
	# Profiling
	output_path = Path(OUTPUT_DIR)
	output_path.mkdir(exist_ok=True)
	
	with profile(
		activities=[ProfilerActivity.CPU],
		schedule=torch.profiler.schedule(wait=1, warmup=WARMUP_STEPS, active=PROFILE_STEPS),
		on_trace_ready=tensorboard_trace_handler(str(output_path)),
		record_shapes=True,
		profile_memory=True,
		with_stack=True
	) as prof:
		for step in range(WARMUP_STEPS + PROFILE_STEPS + 1):
			batch = create_mock_batch()
			try:
				loss, _ = trainer.train_step(batch)
				logger.info(f"Step {step}: loss={loss.item():.4f}")
			except Exception as e:
				logger.error(f"Step {step} failed: {e}")
				break
			prof.step()
	
	# Print summary
	logger.info("Profiling complete. Trace saved to %s", output_path)
	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == "__main__":
	profile_training()
