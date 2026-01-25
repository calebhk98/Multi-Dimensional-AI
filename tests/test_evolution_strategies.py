
import pytest
import torch
import numpy as np
from src.evolution.strategies import OpenAIES, SimpleGA, ParameterProxy, EvolutionStrategy

class MockModel(torch.nn.Module):
	"""
	Simulated model for testing strategies.
	"""
	def __init__(self):
		"""
		Initialize mock model with fixed weights.
		"""
		super().__init__()
		self.layer = torch.nn.Linear(2, 2)
		# Fix weights for deterministic testing
		with torch.no_grad():
			self.layer.weight.fill_(1.0)
			self.layer.bias.fill_(0.0)

def test_parameter_proxy_full_weights():
	"""
	Test extraction and injection of full model weights.
	
	Purpose:
		Verify ParameterProxy can correctly extract and inject model parameters.
		
	Workflow:
		1. Create a mock model
		2. Extract parameters as numpy array
		3. Verify shape and values
		4. Inject new parameters
		5. Verify model weights updated
		
	ToDo:
		None.
	"""
	model = MockModel()
	proxy = ParameterProxy(target="weights")
	
	# Test extraction
	params = proxy.get_params(model)
	assert isinstance(params, np.ndarray)
	assert params.shape == (6,)  # 4 weights + 2 biases
	# First 4 should be weights (1.0), last 2 should be biases (0.0)
	assert np.allclose(params[:4], 1.0)
	assert np.allclose(params[4:], 0.0)
	
	# Test injection
	new_params = np.ones(6) * 0.5
	proxy.set_params(model, new_params)
	
	assert torch.allclose(model.layer.weight, torch.tensor(0.5))
	assert torch.allclose(model.layer.bias, torch.tensor(0.5))

def test_openai_es_perturbation():
	"""
	Test that OpenAIES generates perturbations correctly.
	
	Purpose:
		Verify OpenAIES can perturb parameters deterministically using seeds.
		
	Workflow:
		1. Create base parameters
		2. Perturb with a seed
		3. Verify perturbation is different from base
		4. Verify same seed produces same perturbation
		
	ToDo:
		None.
	"""
	model = MockModel()
	proxy = ParameterProxy(target="weights")
	base_params = proxy.get_params(model)
	
	strategy = OpenAIES(sigma=0.1, learning_rate=0.01)
	
	# Generate a seed
	seed = 42
	
	# Get perturbed parameters
	perturbed_params = strategy.perturb(base_params, seed)
	
	assert perturbed_params.shape == base_params.shape
	assert not np.allclose(perturbed_params, base_params)
	
	# Check deterministic behavior
	perturbed_params_2 = strategy.perturb(base_params, seed)
	assert np.allclose(perturbed_params, perturbed_params_2)

def test_openai_es_update():
	"""
	Test OpenAIES weight update step.
	
	Purpose:
		Verify ES updates parameters in direction of higher fitness.
		
	Workflow:
		1. Create base parameters
		2. Mock results with different fitnesses
		3. Update parameters
		4. Verify parameters changed
		
	ToDo:
		None.
	"""
	base_params = np.zeros(10)
	strategy = OpenAIES(sigma=0.1, learning_rate=0.1)
	
	# Mock results: seed 1 performed well, seed 2 performed poorly
	results = [
		(1, 10.0), # Positive reward
		(2, -10.0) # Negative reward
	]
	
	new_params = strategy.update(base_params, results)
	
	# Expect params to move away from 0
	assert not np.allclose(new_params, base_params)

def test_simple_ga_crossover_mutation():
	"""
	Test SimpleGA population evolution.
	
	Purpose:
		Verify GA can evolve a population through selection, crossover, and mutation.
		
	Workflow:
		1. Initialize random population
		2. Calculate fitness for each individual
		3. Evolve to next generation
		4. Verify population size maintained
		5. Verify population changed
		
	ToDo:
		None.
	"""
	np.random.seed(123)  # Fix seed for reproducibility
	pop_size = 10
	param_size = 5
	strategy = SimpleGA(pop_size=pop_size, mutation_rate=0.1, mutation_power=0.1)
	
	# Initialize random population
	population = [np.random.randn(param_size) for _ in range(pop_size)]
	
	# Mock fitness: simply sum of params
	fitnesses = [np.sum(ind) for ind in population]
	results = list(zip(population, fitnesses))
	
	# Evolve
	new_population = strategy.evolve(results)
	
	assert len(new_population) == pop_size
	assert isinstance(new_population[0], np.ndarray)
	assert new_population[0].shape == (param_size,)
	
	# Ensure population changed (highly likely with mutation)
	assert not np.allclose(new_population[0], population[0])
