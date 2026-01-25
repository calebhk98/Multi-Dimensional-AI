"""
Evolutionary Training Script for Multi-Dimensional AI Creature.

Purpose:
	Run evolutionary optimization (ES or GA) to fine-tune model behavior.
	
Workflow:
	1. Load base model and configuration
	2. Initialize evolution manager with strategy
	3. Run training for N generations
	4. Save final checkpoint
	
Usage:
	python scripts/train_evolution.py --config configs/evolution_config.yaml
"""
import argparse
import yaml
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.multimodal_creature import MultiModalCreature
from src.evolution.manager import EvolutionManager
from src.evolution.strategies import OpenAIES, SimpleGA, ParameterProxy
from src.evolution.fitness import SurvivalFitness, TaskCompletionFitness

def load_config(config_path: str) -> dict:
	"""
	Load YAML configuration.
	
	Args:
		config_path: Path to config file
		
	Returns:
		Configuration dictionary
	"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def create_strategy(config: dict):
	"""
	Create evolution strategy from config.
	
	Args:
		config: Strategy configuration
		
	Returns:
		EvolutionStrategy instance
	"""
	strategy_type = config.get('type', 'es')
	
	if strategy_type == 'es':
		return OpenAIES(
			sigma=config.get('sigma', 0.1),
			learning_rate=config.get('learning_rate', 0.01)
		)
	elif strategy_type == 'ga':
		return SimpleGA(
			pop_size=config.get('pop_size', 20),
			mutation_rate=config.get('mutation_rate', 0.1),
			mutation_power=config.get('mutation_power', 0.02),
			elite_ratio=config.get('elite_ratio', 0.2)
		)
	else:
		raise ValueError(f"Unknown strategy type: {strategy_type}")

def create_fitness_evaluator(config: dict):
	"""
	Create fitness evaluator from config.
	
	Args:
		config: Fitness configuration
		
	Returns:
		FitnessEvaluator instance
	"""
	fitness_type = config.get('type', 'survival')
	
	if fitness_type == 'survival':
		return SurvivalFitness(
			timestep_reward=config.get('timestep_reward', 1.0),
			failure_penalty=config.get('failure_penalty', -100.0)
		)
	elif fitness_type == 'task':
		return TaskCompletionFitness(
			completion_reward=config.get('completion_reward', 100.0),
			progress_weight=config.get('progress_weight', 0.5)
		)
	else:
		raise ValueError(f"Unknown fitness type: {fitness_type}")

def main():
	"""
	Main training loop.
	
	Purpose:
		Entry point for evolutionary training.
		
	Workflow:
		1. Parse arguments
		2. Load config and model
		3. Initialize manager
		4. Train
		5. Save final checkpoint
	"""
	parser = argparse.ArgumentParser(description='Evolutionary Training')
	parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
	parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
	args = parser.parse_args()
	
	# Load configuration
	config = load_config(args.config)
	
	# Create or load model
	model_config = config['model']
	model = MultiModalCreature(
		d_model=model_config['d_model'],
		nhead=model_config['nhead'],
		num_layers=model_config['num_layers'],
		dim_feedforward=model_config['dim_feedforward']
	)
	
	if args.checkpoint:
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		print(f"Loaded checkpoint from {args.checkpoint}")
	
	# Move model to device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	
	# Create components
	strategy = create_strategy(config['strategy'])
	fitness_evaluator = create_fitness_evaluator(config['fitness'])
	proxy = ParameterProxy(target=config.get('param_target', 'weights'))
	
	# Create manager
	manager = EvolutionManager(
		model=model,
		strategy=strategy,
		proxy=proxy,
		fitness_evaluator=fitness_evaluator,
		num_workers=config.get('num_workers', 4),
		max_steps_per_episode=config.get('max_steps', 1000)
	)
	
	# Train
	num_generations = config.get('num_generations', 100)
	checkpoint_every = config.get('checkpoint_every', 10)
	
	print(f"Starting evolutionary training for {num_generations} generations...")
	print(f"Strategy: {type(strategy).__name__}")
	print(f"Fitness: {type(fitness_evaluator).__name__}")
	print(f"Workers: {config.get('num_workers', 4)}")
	
	manager.train(num_generations, checkpoint_every)
	
	# Save final checkpoint
	final_checkpoint_path = config.get('final_checkpoint', 'checkpoints/evolution_final.pt')
	manager._save_checkpoint(final_checkpoint_path)
	print(f"Training complete. Final checkpoint: {final_checkpoint_path}")

if __name__ == '__main__':
	main()
