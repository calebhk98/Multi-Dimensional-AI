"""
Tests for evolution worker module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

from src.evolution.worker import EvolutionWorker
from src.evolution.strategies import ParameterProxy
from src.evolution.fitness import SurvivalFitness, TaskCompletionFitness


class SimpleModel(nn.Module):
    """Simple model for testing evolution worker."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)


@pytest.fixture
def parameter_proxy():
    """Create parameter proxy."""
    return ParameterProxy(target="weights")


@pytest.fixture
def survival_fitness():
    """Create survival fitness evaluator."""
    return SurvivalFitness(timestep_reward=1.0, failure_penalty=-100.0)


@pytest.fixture
def task_fitness():
    """Create task completion fitness evaluator."""
    return TaskCompletionFitness(completion_reward=100.0, progress_weight=0.5)


@pytest.fixture
def worker(simple_model, parameter_proxy, survival_fitness):
    """Create evolution worker."""
    return EvolutionWorker(
        model=simple_model,
        proxy=parameter_proxy,
        fitness_evaluator=survival_fitness,
        max_steps=100,
    )


@pytest.fixture
def base_params(simple_model, parameter_proxy):
    """Get base parameters from model."""
    return parameter_proxy.get_params(simple_model)


class TestEvolutionWorkerInit:
    """Tests for EvolutionWorker initialization."""

    def test_init_stores_model(self, worker, simple_model):
        """Test that worker stores model reference."""
        assert worker.model is simple_model

    def test_init_stores_proxy(self, worker, parameter_proxy):
        """Test that worker stores proxy reference."""
        assert worker.proxy is parameter_proxy

    def test_init_stores_fitness_evaluator(self, worker, survival_fitness):
        """Test that worker stores fitness evaluator reference."""
        assert worker.fitness_evaluator is survival_fitness

    def test_init_stores_max_steps(self, worker):
        """Test that worker stores max_steps."""
        assert worker.max_steps == 100

    def test_init_default_max_steps(self, simple_model, parameter_proxy, survival_fitness):
        """Test default max_steps value."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
        )
        assert worker.max_steps == 1000


class TestEvaluateSeed:
    """Tests for evaluate_seed method."""

    def test_evaluate_seed_returns_float(self, worker, base_params):
        """Test that evaluate_seed returns a float fitness value."""
        fitness = worker.evaluate_seed(base_params, seed=42, sigma=0.1)
        assert isinstance(fitness, float)

    def test_evaluate_seed_applies_perturbation(self, worker, base_params, simple_model, parameter_proxy):
        """Test that evaluate_seed applies perturbation to params."""
        original_params = parameter_proxy.get_params(simple_model).copy()

        # Evaluate with seed - this will set perturbed params on model
        worker.evaluate_seed(base_params, seed=42, sigma=0.1)

        # Model params should now be perturbed
        new_params = parameter_proxy.get_params(simple_model)
        assert not np.allclose(new_params, original_params), "Params not perturbed"

    def test_evaluate_seed_reproducible_with_same_seed(self, worker, base_params):
        """Test that same seed produces same perturbation."""
        fitness1 = worker.evaluate_seed(base_params, seed=42, sigma=0.1)
        fitness2 = worker.evaluate_seed(base_params, seed=42, sigma=0.1)
        assert fitness1 == fitness2, "Same seed should produce same fitness"

    def test_evaluate_seed_different_seeds_differ(self, worker, base_params):
        """Test that different seeds produce different perturbations."""
        # Run multiple times to account for potential random coincidence
        fitnesses_seed1 = [worker.evaluate_seed(base_params, seed=42, sigma=0.1) for _ in range(3)]
        fitnesses_seed2 = [worker.evaluate_seed(base_params, seed=123, sigma=0.1) for _ in range(3)]

        # The fitness values themselves may be similar since they're based on timesteps,
        # but the perturbations should differ
        # Let's verify by checking the model params after each evaluation
        worker.evaluate_seed(base_params, seed=42, sigma=0.1)
        params_42 = worker.proxy.get_params(worker.model).copy()

        worker.evaluate_seed(base_params, seed=123, sigma=0.1)
        params_123 = worker.proxy.get_params(worker.model).copy()

        assert not np.allclose(params_42, params_123), "Different seeds should produce different params"

    def test_evaluate_seed_sigma_affects_perturbation(self, worker, base_params):
        """Test that sigma controls perturbation magnitude."""
        worker.evaluate_seed(base_params, seed=42, sigma=0.01)
        small_perturbation = worker.proxy.get_params(worker.model).copy()

        worker.evaluate_seed(base_params, seed=42, sigma=1.0)
        large_perturbation = worker.proxy.get_params(worker.model).copy()

        small_diff = np.abs(small_perturbation - base_params).mean()
        large_diff = np.abs(large_perturbation - base_params).mean()

        assert large_diff > small_diff, "Larger sigma should produce larger perturbation"

    def test_evaluate_seed_calls_evaluate_params(self, worker, base_params):
        """Test that evaluate_seed calls evaluate_params."""
        with patch.object(worker, 'evaluate_params', return_value=1.0) as mock:
            worker.evaluate_seed(base_params, seed=42, sigma=0.1)
            mock.assert_called_once()


class TestEvaluateParams:
    """Tests for evaluate_params method."""

    def test_evaluate_params_returns_float(self, worker, base_params):
        """Test that evaluate_params returns a float fitness value."""
        fitness = worker.evaluate_params(base_params)
        assert isinstance(fitness, float)

    def test_evaluate_params_sets_model_params(self, worker, base_params, parameter_proxy):
        """Test that evaluate_params sets model parameters."""
        # Create modified params
        modified_params = base_params + 0.1

        worker.evaluate_params(modified_params)

        actual_params = parameter_proxy.get_params(worker.model)
        np.testing.assert_array_almost_equal(actual_params, modified_params)

    def test_evaluate_params_calls_run_episode(self, worker, base_params):
        """Test that evaluate_params calls _run_episode."""
        with patch.object(worker, '_run_episode', return_value=50.0) as mock:
            fitness = worker.evaluate_params(base_params)
            mock.assert_called_once()
            assert fitness == 50.0


class TestRunEpisode:
    """Tests for _run_episode method."""

    def test_run_episode_returns_float(self, worker):
        """Test that _run_episode returns a float fitness value."""
        fitness = worker._run_episode()
        assert isinstance(fitness, float)

    def test_run_episode_sets_model_eval(self, worker, simple_model):
        """Test that _run_episode sets model to eval mode."""
        simple_model.train()  # Ensure it's in train mode first
        worker._run_episode()
        assert not simple_model.training, "Model should be in eval mode"

    def test_run_episode_respects_max_steps(self, simple_model, parameter_proxy, survival_fitness):
        """Test that _run_episode runs for at most max_steps."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            max_steps=50,
        )
        fitness = worker._run_episode()
        # With survival fitness and alive=True, fitness = timesteps * reward
        # Should be max_steps * 1.0 = 50.0
        assert fitness == 50.0

    def test_run_episode_returns_fitness_score(self, worker):
        """Test that _run_episode returns fitness from evaluator."""
        fitness = worker._run_episode()
        # With default survival fitness (timestep_reward=1.0) and max_steps=100
        assert fitness == 100.0

    def test_run_episode_uses_fitness_evaluator(
        self, simple_model, parameter_proxy, task_fitness
    ):
        """Test that _run_episode uses the configured fitness evaluator."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=task_fitness,
            max_steps=100,
        )
        fitness = worker._run_episode()
        # TaskCompletionFitness returns progress * weight * reward
        # With task_completed=False and task_progress=0.0, fitness should be 0
        assert fitness == 0.0

    def test_run_episode_with_torch_no_grad(self, worker):
        """Test that _run_episode runs without gradient computation."""
        # This is a behavioral test - if gradients were computed,
        # it would be slower and use more memory
        # We can verify by checking that no gradients are accumulated
        for param in worker.model.parameters():
            param.grad = None

        worker._run_episode()

        # No gradients should be accumulated
        for param in worker.model.parameters():
            assert param.grad is None


class TestIntegration:
    """Integration tests for EvolutionWorker."""

    def test_full_evaluation_cycle_es(self, simple_model, parameter_proxy, survival_fitness):
        """Test a full ES-style evaluation cycle."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            max_steps=50,
        )
        base_params = parameter_proxy.get_params(simple_model)

        # Evaluate multiple seeds
        fitnesses = []
        for seed in range(5):
            fitness = worker.evaluate_seed(base_params, seed=seed, sigma=0.1)
            fitnesses.append(fitness)

        # All should return valid fitness values
        assert all(isinstance(f, float) for f in fitnesses)
        assert all(f >= 0 for f in fitnesses)  # Survival fitness is positive when alive

    def test_full_evaluation_cycle_ga(self, simple_model, parameter_proxy, survival_fitness):
        """Test a full GA-style evaluation cycle."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            max_steps=50,
        )
        base_params = parameter_proxy.get_params(simple_model)

        # Create a population
        population = [base_params + np.random.randn(*base_params.shape) * 0.01 for _ in range(10)]

        # Evaluate each individual
        fitnesses = []
        for individual in population:
            fitness = worker.evaluate_params(individual)
            fitnesses.append(fitness)

        # All should return valid fitness values
        assert all(isinstance(f, float) for f in fitnesses)
        assert len(fitnesses) == 10

    def test_different_fitness_evaluators(self, simple_model, parameter_proxy):
        """Test worker with different fitness evaluators."""
        base_params = ParameterProxy().get_params(simple_model)

        # Survival fitness
        survival = SurvivalFitness(timestep_reward=2.0)
        worker_survival = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival,
            max_steps=50,
        )
        fitness_survival = worker_survival.evaluate_params(base_params)

        # Task completion fitness
        task = TaskCompletionFitness(completion_reward=100.0)
        worker_task = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=task,
            max_steps=50,
        )
        fitness_task = worker_task.evaluate_params(base_params)

        # Should return different values based on evaluator
        assert fitness_survival == 100.0  # 50 * 2.0
        assert fitness_task == 0.0  # No task progress

    def test_worker_preserves_model_after_evaluation(
        self, simple_model, parameter_proxy, survival_fitness
    ):
        """Test that model can still be used after evaluation."""
        worker = EvolutionWorker(
            model=simple_model,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            max_steps=10,
        )
        base_params = parameter_proxy.get_params(simple_model)

        # Run evaluation
        worker.evaluate_params(base_params)

        # Model should still work
        x = torch.randn(1, 10)
        output = simple_model(x)
        assert output.shape == (1, 5)
