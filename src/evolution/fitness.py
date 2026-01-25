import numpy as np
from typing import Any, Dict
import torch

class FitnessEvaluator:
	"""
	Base fitness evaluator interface.
	Subclasses implement specific fitness calculations for VR tasks.
	
	Purpose:
		Provide a standardized interface for evaluating creature performance.
		
	Workflow:
		1. Call evaluate() with model outputs and environment state
		2. Returns a fitness score (higher is better)
	"""
	def evaluate(self, outputs: Dict[str, Any], env_state: Dict[str, Any]) -> float:
		"""
		Evaluates fitness based on model outputs and environment.
		
		Args:
			outputs: Model predictions/actions
			env_state: Current environment state
			
		Returns:
			Fitness score (float)
		"""
		raise NotImplementedError

class SurvivalFitness(FitnessEvaluator):
	"""
	Simple survival-based fitness: reward for staying alive.
	
	Purpose:
		Baseline fitness for initial testing. Creatures are rewarded for duration.
		
	Workflow:
		Count timesteps survived, penalize for catastrophic failures.
	"""
	def __init__(self, timestep_reward: float = 1.0, failure_penalty: float = -100.0):
		self.timestep_reward = timestep_reward
		self.failure_penalty = failure_penalty
	
	def evaluate(self, outputs: Dict[str, Any], env_state: Dict[str, Any]) -> float:
		"""
		Purpose:
			Calculate fitness based on survival.
			
		Args:
			outputs: Not used in this simple evaluator
			env_state: Must contain 'timesteps' and optionally 'alive'
			
		Returns:
			Fitness score
		"""
		timesteps = env_state.get('timesteps', 0)
		alive = env_state.get('alive', True)
		
		if not alive:
			return self.failure_penalty
			
		return timesteps * self.timestep_reward

class TaskCompletionFitness(FitnessEvaluator):
	"""
	Task-oriented fitness: reward for completing objectives.
	
	Purpose:
		Evaluate creature performance on specific goals (e.g., pick up object, navigate).
		
	Workflow:
		Check env_state for task completion flags and partial progress.
	"""
	def __init__(self, completion_reward: float = 100.0, progress_weight: float = 0.5):
		self.completion_reward = completion_reward
		self.progress_weight = progress_weight
	
	def evaluate(self, outputs: Dict[str, Any], env_state: Dict[str, Any]) -> float:
		"""
		Purpose:
			Calculate fitness based on task completion.
			
		Args:
			outputs: Model actions
			env_state: Must contain 'task_completed' (bool) and 'task_progress' (0-1)
			
		Returns:
			Fitness score
		"""
		completed = env_state.get('task_completed', False)
		progress = env_state.get('task_progress', 0.0)
		
		if completed:
			return self.completion_reward
		
		return progress * self.progress_weight * self.completion_reward
