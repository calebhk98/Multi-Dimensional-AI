import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional

class ParameterProxy:
	"""
	Interface to extract and inject parameters into a model.
	Supports targeting full weights, LoRA adapters, or hyperparameters.
	"""
	def __init__(self, target: str = "weights"):
		"""
		Args:
			target: "weights", "lora", or "hyperparams" (not fully impl yet)
		"""
		self.target = target

	def get_params(self, model: torch.nn.Module) -> np.ndarray:
		"""
		Extracts parameters as a single flattened numpy array.
		
		Args:
			model: PyTorch model to extract parameters from
			
		Returns:
			Flattened numpy array of parameters
		"""
		params = []
		for name, param in model.named_parameters():
			if self._should_include(name):
				params.append(param.data.cpu().numpy().flatten())
		
		if not params:
			return np.array([])
			
		return np.concatenate(params)

	def set_params(self, model: torch.nn.Module, flat_params: np.ndarray):
		"""
		Injects a flattened numpy array of parameters back into the model.
		
		Args:
			model: PyTorch model to update
			flat_params: Flattened parameter array
		"""
		offset = 0
		for name, param in model.named_parameters():
			if self._should_include(name):
				numel = param.numel()
				# Slice and reshape
				param_data = flat_params[offset:offset+numel]
				param.data.copy_(torch.from_numpy(param_data).view(param.shape))
				offset += numel

	def _should_include(self, name: str) -> bool:
		"""
		Determine if parameter should be included based on target.
		
		Args:
			name: Parameter name
			
		Returns:
			True if parameter should be included
		"""
		if self.target == "weights":
			return True
		elif self.target == "lora":
			return "lora" in name.lower()
		return False

class EvolutionStrategy(ABC):
	"""
	Base class for evolution strategies.
	"""
	@abstractmethod
	def ask(self):
		"""Request new parameters/seeds to evaluate."""
		pass

	@abstractmethod
	def tell(self, results):
		"""
		Update based on evaluation results.
		
		Args:
			results: Evaluation results
		"""
		pass

class OpenAIES(EvolutionStrategy):
	"""
	OpenAI-ES (Evolution Strategies) implementation.
	Uses noise seeds to avoid storing full parameter copies.
	"""
	def __init__(self, sigma: float = 0.1, learning_rate: float = 0.01):
		"""
		Args:
			sigma: Noise standard deviation
			learning_rate: Update step size
		"""
		self.sigma = sigma
		self.learning_rate = learning_rate

	def perturb(self, params: np.ndarray, seed: int) -> np.ndarray:
		"""
		Generate perturbed parameters: w' = w + sigma * noise(seed)
		
		Args:
			params: Base parameter vector
			seed: Random seed for noise generation
			
		Returns:
			Perturbed parameter vector
		"""
		np.random.seed(seed)
		noise = np.random.randn(*params.shape)
		return params + self.sigma * noise

	def update(self, current_params: np.ndarray, results: List[Tuple[int, float]]) -> np.ndarray:
		"""
		Refine parameters based on results: w = w + lr * (1/n*sigma) * sum(reward * noise)
		
		Args:
			current_params: Current parameter vector
			results: List of (seed, reward) tuples
			
		Returns:
			Updated parameter vector
		"""
		if not results:
			return current_params

		# Normalize rewards (optional but good practice)
		rewards = np.array([r for _, r in results])
		# Simple whitening if variance is not 0
		if rewards.std() > 1e-6:
			rewards = (rewards - rewards.mean()) / rewards.std()

		weighted_noise = np.zeros_like(current_params)
		
		for (seed, _), reward in zip(results, rewards):
			np.random.seed(seed)
			noise = np.random.randn(*current_params.shape)
			weighted_noise += reward * noise

		step = (self.learning_rate / (len(results) * self.sigma)) * weighted_noise
		return current_params + step
		
	# Stub methods to satisfy ABC if needed, though we use specific flows
	def ask(self): pass
	def tell(self, results): pass

class SimpleGA(EvolutionStrategy):
	"""
	Simple Genetic Algorithm.
	Evolution operates on a population of parameter vectors.
	"""
	def __init__(self, pop_size: int = 20, mutation_rate: float = 0.1, mutation_power: float = 0.02, elite_ratio: float = 0.2):
		"""
		Args:
			pop_size: Population size
			mutation_rate: Probability of mutation per gene
			mutation_power: Standard deviation of mutation noise
			elite_ratio: Fraction of population to preserve
		"""
		self.pop_size = pop_size
		self.mutation_rate = mutation_rate
		self.mutation_power = mutation_power
		self.elite_count = int(pop_size * elite_ratio)

	def evolve(self, population_with_fitness: List[Tuple[np.ndarray, float]]) -> List[np.ndarray]:
		"""
		Evolves the population to the next generation.
		
		Args:
			population_with_fitness: List of (genome, fitness) tuples
			
		Returns:
			List of new genomes (np.ndarray)
		"""
		# Sort by fitness (descending)
		sorted_pop = sorted(population_with_fitness, key=lambda x: x[1], reverse=True)
		
		# Elitism
		new_pop = [ind for ind, _ in sorted_pop[:self.elite_count]]
		
		# Fill rest with crossover/mutation
		while len(new_pop) < self.pop_size:
			parent1 = self._select(sorted_pop)
			parent2 = self._select(sorted_pop)
			
			child = self._crossover(parent1, parent2)
			child = self._mutate(child)
			new_pop.append(child)
			
		return new_pop
	
	def _select(self, sorted_pop):
		"""
		Tournament selection from top half of population.
		
		Args:
			sorted_pop: Population sorted by fitness (descending)
			
		Returns:
			Selected genome
		"""
		# Tournament selection or simple random from top half
		idx = np.random.randint(0, len(sorted_pop) // 2 + 1)
		return sorted_pop[idx][0]

	def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
		"""
		Uniform crossover of two parents.
		
		Args:
			p1: First parent genome
			p2: Second parent genome
			
		Returns:
			Child genome
		"""
		# Uniform crossover
		mask = np.random.rand(*p1.shape) > 0.5
		child = np.where(mask, p1, p2)
		return child

	def _mutate(self, genome: np.ndarray) -> np.ndarray:
		"""
		Mutate genome with Gaussian noise.
		
		Args:
			genome: Genome to mutate
			
		Returns:
			Mutated genome
		"""
		mask = np.random.rand(*genome.shape) < self.mutation_rate
		noise = np.random.randn(*genome.shape) * self.mutation_power
		return genome + mask * noise

	def ask(self): pass
	def tell(self): pass
