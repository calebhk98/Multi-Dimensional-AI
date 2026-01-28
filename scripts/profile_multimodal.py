"""
Enhanced profiling script for multi-modal training batches.
Profiles memory, time, and throughput for realistic workloads.
"""

import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config constants
PROFILE_STEPS = 10
WARMUP_STEPS = 2
OUTPUT_DIR = "profiler_output"
REPORT_FILE = "profile_report.json"


def profile_multimodal_training():
	"""
	Purpose:
		Profile multi-modal training with realistic batch shapes.
		
	Workflow:
		1. Import dataset and model
		2. Create profiler
		3. Run training steps
		4. Generate profiling report
		5. Save trace and summary
		
	ToDo:
		None
		
	Returns:
		None
	"""
	try:
		from src.data.synthetic_generator import SyntheticDataGenerator
		from src.data.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
		from src.training.trainer import Trainer
		logger.info("Successfully imported real dataset components")
	except ImportError as e:
		logger.error(f"Failed to import required modules: {e}")
		return
	
	# Try to import real model
	try:
		from src.models.multimodal_transformer import MultiModalCreature
		model_class = MultiModalCreature
		logger.info("Using MultiModalCreature model")
	except ImportError:
		logger.warning("MultiModalCreature not found, using mock model")
		import torch.nn as nn
		
		class MockModel(nn.Module):
			"""
			Purpose:
				Mock model for profiling when real model unavailable.
				
			Workflow:
				Simple forward/backward pass
				
			ToDo:
				None
			"""
			
			def __init__(self):
				"""Initialize mock model."""
				super().__init__()
				self.param = nn.Parameter(torch.randn(512, 512))
				
			def forward(self, **kwargs):
				"""
				Forward pass.
				
				Args:
					**kwargs: Various inputs.
					
				Returns:
					Dict with outputs.
				"""
				return {"hidden": self.param.unsqueeze(0).expand(2, -1, -1)}
				
			def compute_loss(self, outputs, targets):
				"""
				Compute loss.
				
				Args:
					outputs: Model outputs.
					targets: Targets.
					
				Returns:
					Tuple of (loss, loss_dict).
				"""
				loss = torch.tensor(0.5, requires_grad=True)
				loss_dict = {"total": loss}
				return loss, loss_dict
		
		model_class = MockModel
	
	# Create dataset
	generator = SyntheticDataGenerator()
	dataset = MultiModalDataset(generator, length=100)
	loader = DataLoader(dataset, batch_size=4, collate_fn=multimodal_collate_fn)
	
	# Create model and trainer
	config = {
		"training": {
			"max_steps": WARMUP_STEPS + PROFILE_STEPS + 1,
			"log_interval": 5
		}
	}
	
	model = model_class() if model_class.__name__ == "MockModel" else model_class(config={})
	
	# Determine device
	device = "cuda" if torch.cuda.is_available() else "cpu"
	logger.info(f"Profiling on device: {device}")
	
	trainer = Trainer(model, config, loader, device=device)
	
	# Setup profiling output
	output_path = Path(OUTPUT_DIR)
	output_path.mkdir(exist_ok=True)
	
	# Profile activities
	activities = [ProfilerActivity.CPU]
	if torch.cuda.is_available():
		activities.append(ProfilerActivity.CUDA)
	
	# Profiling metrics
	profile_metrics = {
		"device": device,
		"batch_size": 4,
		"num_modalities": 6,
		"steps_profiled": PROFILE_STEPS,
		"per_step_metrics": []
	}
	
	logger.info("Starting profiling...")
	
	with profile(
		activities=activities,
		schedule=torch.profiler.schedule(
			wait=1, 
			warmup=WARMUP_STEPS, 
			active=PROFILE_STEPS,
			repeat=1
		),
		on_trace_ready=tensorboard_trace_handler(str(output_path)),
		record_shapes=True,
		profile_memory=True,
		with_stack=True
	) as prof:
		for step in range(WARMUP_STEPS + PROFILE_STEPS + 1):
			batch = next(iter(loader))
			
			# Track memory before step
			mem_before = _get_memory_mb()
			
			try:
				import time
				step_start = time.time()
				loss, loss_dict = trainer.train_step(batch)
				step_time = time.time() - step_start
				
				# Track memory after step
				mem_after = _get_memory_mb()
				mem_peak = _get_peak_memory_mb()
				
				logger.info(
					f"Step {step}: loss={loss.item():.4f}, "
					f"time={step_time:.3f}s, "
					f"mem={mem_after:.1f}MB"
				)
				
				# Store metrics for active profiling steps
				if step >= WARMUP_STEPS:
					_store_step_metrics(profile_metrics, step, loss, step_time, mem_after, mem_peak, mem_before)
			except Exception as e:
				logger.error(f"Step {step} failed: {e}")
				break
			
			prof.step()
	
	# Compute aggregate statistics
	_compute_summary_statistics(profile_metrics)
	
	# Save report
	report_path = output_path / REPORT_FILE
	with open(report_path, "w") as f:
		json.dump(profile_metrics, f, indent=2)
	
	logger.info(f"Profile report saved to {report_path}")
	
	# Print summary table
	print("\n" + "="*60)
	print("PROFILING SUMMARY")
	print("="*60)
	print(prof.key_averages().table(
		sort_by="cpu_time_total" if device == "cpu" else "cuda_time_total",
		row_limit=20
	))
	print("="*60)
	
	if "summary" in profile_metrics:
		print(f"\nAverage step time: {profile_metrics['summary']['avg_step_time_sec']:.3f}s")
		print(f"Throughput: {profile_metrics['summary']['samples_per_sec']:.1f} samples/sec")
		if torch.cuda.is_available():
			print(f"Peak memory: {profile_metrics['summary']['max_memory_mb']:.1f} MB")
	
	print(f"\nDetailed traces saved to: {output_path}")
	print(f"View with: tensorboard --logdir={output_path}")
	
	# Generate recommendations
	generate_recommendations(profile_metrics)


def _get_memory_mb():
	"""
	Purpose:
		Get current memory usage in MB.
		
	Returns:
		Memory in MB, or 0 if CUDA unavailable.
	"""
	if torch.cuda.is_available():
		return torch.cuda.memory_allocated() / 1024 / 1024
	return 0


def _get_peak_memory_mb():
	"""
	Purpose:
		Get peak memory usage and reset stats.
		
	Returns:
		Peak memory in MB, or 0 if CUDA unavailable.
	"""
	if not torch.cuda.is_available():
		return 0
	
	peak = torch.cuda.max_memory_allocated() / 1024 / 1024
	torch.cuda.reset_peak_memory_stats()
	return peak


def _store_step_metrics(profile_metrics, step, loss, step_time, mem_after, mem_peak, mem_before):
	"""
	Purpose:
		Store metrics for a profiling step.
		
	Args:
		profile_metrics: Metrics dictionary.
		step: Step number.
		loss: Loss tensor.
		step_time: Time in seconds.
		mem_after: Memory after step.
		mem_peak: Peak memory.
		mem_before: Memory before step.
	"""
	profile_metrics["per_step_metrics"].append({
		"step": step,
		"loss": loss.item(),
		"time_sec": step_time,
		"memory_mb": mem_after,
		"memory_peak_mb": mem_peak,
		"memory_delta_mb": mem_after - mem_before
	})


def _compute_summary_statistics(profile_metrics):
	"""
	Purpose:
		Compute aggregate statistics from collected metrics.
		
	Args:
		profile_metrics: Metrics dictionary to update.
	"""
	if not profile_metrics["per_step_metrics"]:
		return
	
	step_times = [m["time_sec"] for m in profile_metrics["per_step_metrics"]]
	memory_peaks = [m["memory_peak_mb"] for m in profile_metrics["per_step_metrics"]]
	
	profile_metrics["summary"] = {
		"avg_step_time_sec": sum(step_times) / len(step_times),
		"min_step_time_sec": min(step_times),
		"max_step_time_sec": max(step_times),
		"avg_memory_mb": sum(memory_peaks) / len(memory_peaks) if memory_peaks else 0,
		"max_memory_mb": max(memory_peaks) if memory_peaks else 0,
		"samples_per_sec": 4 / (sum(step_times) / len(step_times)) if step_times else 0
	}


def generate_recommendations(metrics):
	"""
	Purpose:
		Generate optimization recommendations based on profiling data.
		
	Workflow:
		1. Analyze metrics
		2. Identify bottlenecks
		3. Print recommendations
		
	ToDo:
		None
		
	Args:
		metrics: Profile metrics dictionary.
		
	Returns:
		None
	"""
	print("\n" + "="*60)
	print("RECOMMENDATIONS")
	print("="*60)
	
	if "summary" not in metrics:
		print("Insufficient data for recommendations")
		return
	
	summary = metrics["summary"]
	device = metrics["device"]
	
	# Throughput recommendations
	throughput = summary["samples_per_sec"]
	if throughput < 10:
		print("⚠ Low throughput detected (<10 samples/sec)")
		print("  - Consider using mixed precision (torch.cuda.amp)")
		print("  - Reduce model size or batch size")
		print("  - Check for CPU bottlenecks in data loading")
	
	# Memory recommendations
	if device == "cuda":
		max_mem = summary["max_memory_mb"]
		if max_mem > 20000:  # >20GB
			print(f"⚠ High memory usage: {max_mem:.0f}MB")
			print("  - Consider gradient checkpointing")
			print("  - Reduce batch size")
			print("  - Use gradient accumulation for effective larger batches")
		elif max_mem < 5000:  # <5GB
			print(f"✓ Memory usage is reasonable: {max_mem:.0f}MB")
			print("  - You may be able to increase batch size")
	
	# Step time recommendations
	avg_time = summary["avg_step_time_sec"]
	if avg_time > 1.0:
		print(f"⚠ Slow step time: {avg_time:.2f}s")
		print("  - Profile CPU vs GPU time to identify bottleneck")
		print("  - Ensure data is pinned for CUDA transfers")
	
	print("="*60)


if __name__ == "__main__":
	profile_multimodal_training()
