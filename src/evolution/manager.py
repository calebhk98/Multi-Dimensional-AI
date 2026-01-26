"""
Evolution Manager for coordinating training.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from multiprocessing import Pool
from src.evolution.strategies import EvolutionStrategy, OpenAIES, SimpleGA, ParameterProxy
from src.evolution.worker import EvolutionWorker
from src.evolution.fitness import FitnessEvaluator

class EvolutionManager:
	"""
	Orchestrates the evolutionary training loop.
	
	Purpose:
		Manage population/seeds, coordinate workers, update parameters.
		
	Workflow:
		1. Initialize base parameters
		2. For each generation:
			a. Generate evaluation tasks (seeds or population)
			b. Distribute to workers
			c. Collect results
			d. Update parameters using strategy
		3. Save checkpoints
	"""
	def __init__(
		self,
		model: torch.nn.Module,
		strategy: EvolutionStrategy,
		proxy: ParameterProxy,
		fitness_evaluator: FitnessEvaluator,
		num_workers: int = 4,
		max_steps_per_episode: int = 1000
	):
		"""
		Args:
			model: The model to evolve
			strategy: Evolution strategy (OpenAIES or SimpleGA)
			proxy: Parameter proxy for model interface
			fitness_evaluator: Fitness function
			num_workers: Number of parallel workers
			max_steps_per_episode: Max timesteps per evaluation
		"""
		self.model = model
		self.strategy = strategy
		self.proxy = proxy
		self.fitness_evaluator = fitness_evaluator
		self.num_workers = num_workers
		self.max_steps_per_episode = max_steps_per_episode
		
		# Initialize base parameters
		self.current_params = self.proxy.get_params(model)
		self.generation = 0
		
		# Population (for GA)
		self.population: Optional[List[np.ndarray]] = None
		if isinstance(strategy, SimpleGA):
			self._init_population()
	
	def _init_population(self):
		"""
		Initialize population for GA.
		
		Purpose:
			Create initial population around current parameters.
		"""
		pop_size = self.strategy.pop_size
		self.population = []
		
		for _ in range(pop_size):
			# Add small random variation to base params
			noise = np.random.randn(*self.current_params.shape) * 0.01
			individual = self.current_params + noise
			self.population.append(individual)
	
	def train(self, num_generations: int, checkpoint_every: int = 10):
		"""
		Run evolutionary training for N generations.
		
		Purpose:
			Main training loop.
			
		Args:
			num_generations: Number of generations to run
			checkpoint_every: Save checkpoint every N generations
			
		Workflow:
			1. For each generation: evaluate → update
			2. Log best fitness
			3. Checkpoint periodically
		"""
		for gen in range(num_generations):
			self.generation = gen
			
			if isinstance(self.strategy, OpenAIES):
				best_fitness = self._train_es_generation()
			elif isinstance(self.strategy, SimpleGA):
				best_fitness = self._train_ga_generation()
			else:
				raise ValueError(f"Unknown strategy type: {type(self.strategy)}")
			
			print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}")
			
			if (gen + 1) % checkpoint_every == 0:
				self._save_checkpoint(f"checkpoint_gen_{gen+1}.pt")
	
	def _train_es_generation(self) -> float:
		"""
		Train one generation using Evolution Strategies.
		
		Purpose:
			Generate seeds, evaluate, update params.
			
		Returns:
			Best fitness this generation
		"""
		# Generate random seeds
		seeds = [np.random.randint(0, 2**31) for _ in range(self.num_workers)]
		
		# Evaluate in parallel
		results = []
		worker = EvolutionWorker(self.model, self.proxy, self.fitness_evaluator, self.max_steps_per_episode)
		
		for seed in seeds:
			fitness = worker.evaluate_seed(self.current_params, seed, self.strategy.sigma)
			results.append((seed, fitness))
		
		# Update parameters
		self.current_params = self.strategy.update(self.current_params, results)
		self.proxy.set_params(self.model, self.current_params)
		
		# Return best fitness
		best_fitness = max(f for _, f in results)
		return best_fitness
	
	def _train_ga_generation(self) -> float:
		"""
		Train one generation using Genetic Algorithm.
		
		Purpose:
			Evaluate population, evolve to next generation.
			
		Returns:
			Best fitness this generation
		"""
		# Evaluate population
		results = []
		worker = EvolutionWorker(self.model, self.proxy, self.fitness_evaluator, self.max_steps_per_episode)
		
		for individual in self.population:
			fitness = worker.evaluate_params(individual)
			results.append((individual, fitness))
		
		# Evolve to next generation
		self.population = self.strategy.evolve(results)
		
		# Update current params to best individual
		best_individual, best_fitness = max(results, key=lambda x: x[1])
		self.current_params = best_individual
		self.proxy.set_params(self.model, self.current_params)
		
		return best_fitness
	
	def _save_checkpoint(self, filename: str):
		"""
		Save training checkpoint.
		
		Purpose:
			Persist model state and evolution progress.
			
		Args:
			filename: Checkpoint filename
		"""
		checkpoint = {
			'generation': self.generation,
			'model_state_dict': self.model.state_dict(),
			'current_params': self.current_params,
			'population': self.population
		}
		torch.save(checkpoint, filename)
		print(f"Saved checkpoint: {filename}")
