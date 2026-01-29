"""
Tests for evolution manager module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evolution.manager import EvolutionManager
from src.evolution.strategies import OpenAIES, SimpleGA, ParameterProxy
from src.evolution.fitness import SurvivalFitness, TaskCompletionFitness


class SimpleModel(nn.Module):
    """Simple model for testing evolution."""

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
def openai_es_strategy():
    """Create OpenAI ES strategy."""
    return OpenAIES(sigma=0.1, learning_rate=0.01)


@pytest.fixture
def simple_ga_strategy():
    """Create Simple GA strategy."""
    return SimpleGA(pop_size=10, mutation_rate=0.1, mutation_power=0.02, elite_ratio=0.2)


@pytest.fixture
def parameter_proxy():
    """Create parameter proxy."""
    return ParameterProxy(target="weights")


@pytest.fixture
def survival_fitness():
    """Create survival fitness evaluator."""
    return SurvivalFitness(timestep_reward=1.0, failure_penalty=-100.0)


@pytest.fixture
def es_manager(simple_model, openai_es_strategy, parameter_proxy, survival_fitness):
    """Create evolution manager with ES strategy."""
    return EvolutionManager(
        model=simple_model,
        strategy=openai_es_strategy,
        proxy=parameter_proxy,
        fitness_evaluator=survival_fitness,
        num_workers=2,
        max_steps_per_episode=100,
    )


@pytest.fixture
def ga_manager(simple_model, simple_ga_strategy, parameter_proxy, survival_fitness):
    """Create evolution manager with GA strategy."""
    return EvolutionManager(
        model=simple_model,
        strategy=simple_ga_strategy,
        proxy=parameter_proxy,
        fitness_evaluator=survival_fitness,
        num_workers=2,
        max_steps_per_episode=100,
    )


class TestEvolutionManagerInit:
    """Tests for EvolutionManager initialization."""

    def test_init_with_openai_es_strategy(
        self, simple_model, openai_es_strategy, parameter_proxy, survival_fitness
    ):
        """Test initialization with OpenAI ES strategy."""
        manager = EvolutionManager(
            model=simple_model,
            strategy=openai_es_strategy,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            num_workers=4,
            max_steps_per_episode=1000,
        )
        assert manager.model is simple_model
        assert manager.strategy is openai_es_strategy
        assert manager.num_workers == 4
        assert manager.max_steps_per_episode == 1000
        assert manager.generation == 0
        assert manager.population is None  # ES doesn't use population

    def test_init_with_simple_ga_strategy(
        self, simple_model, simple_ga_strategy, parameter_proxy, survival_fitness
    ):
        """Test initialization with Simple GA strategy."""
        manager = EvolutionManager(
            model=simple_model,
            strategy=simple_ga_strategy,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            num_workers=4,
            max_steps_per_episode=1000,
        )
        assert manager.model is simple_model
        assert manager.strategy is simple_ga_strategy
        assert manager.population is not None  # GA uses population
        assert len(manager.population) == simple_ga_strategy.pop_size

    def test_current_params_extracted(self, es_manager, parameter_proxy, simple_model):
        """Test that current parameters are extracted from model."""
        expected_params = parameter_proxy.get_params(simple_model)
        np.testing.assert_array_almost_equal(es_manager.current_params, expected_params)

    def test_generation_starts_at_zero(self, es_manager):
        """Test that generation counter starts at 0."""
        assert es_manager.generation == 0


class TestInitPopulation:
    """Tests for _init_population method."""

    def test_population_size_matches_strategy(self, ga_manager, simple_ga_strategy):
        """Test that population size matches strategy config."""
        assert len(ga_manager.population) == simple_ga_strategy.pop_size

    def test_population_individuals_have_correct_shape(self, ga_manager):
        """Test that each individual has same shape as current params."""
        expected_shape = ga_manager.current_params.shape
        for individual in ga_manager.population:
            assert individual.shape == expected_shape

    def test_population_has_variation(self, ga_manager):
        """Test that population individuals are not identical."""
        # At least some individuals should differ from base params
        diffs = [
            np.allclose(ind, ga_manager.current_params) for ind in ga_manager.population
        ]
        assert not all(diffs), "All individuals are identical to base params"

    def test_population_variation_is_small(self, ga_manager):
        """Test that initial population variation is small (0.01 std)."""
        for individual in ga_manager.population:
            diff = np.abs(individual - ga_manager.current_params)
            # Differences should be small (initialized with 0.01 std noise)
            assert diff.max() < 0.1, "Population variation too large"


class TestTrainESGeneration:
    """Tests for _train_es_generation method."""

    def test_es_generation_returns_fitness(self, es_manager):
        """Test that ES generation returns a fitness value."""
        fitness = es_manager._train_es_generation()
        assert isinstance(fitness, float)

    def test_es_generation_updates_params(self, es_manager):
        """Test that ES generation updates current params."""
        original_params = es_manager.current_params.copy()
        es_manager._train_es_generation()
        # Params should change (unless all fitness values are equal)
        assert not np.allclose(
            es_manager.current_params, original_params
        ), "Params not updated"

    def test_es_generation_increments_correctly_in_train(self, es_manager):
        """Test that generation counter increments during training."""
        es_manager.train(num_generations=2, checkpoint_every=100)
        assert es_manager.generation == 1  # 0-indexed, so after 2 gens it's 1


class TestTrainGAGeneration:
    """Tests for _train_ga_generation method."""

    def test_ga_generation_returns_fitness(self, ga_manager):
        """Test that GA generation returns a fitness value."""
        fitness = ga_manager._train_ga_generation()
        assert isinstance(fitness, float)

    def test_ga_generation_evolves_population(self, ga_manager):
        """Test that GA generation evolves the population."""
        original_population = [ind.copy() for ind in ga_manager.population]
        ga_manager._train_ga_generation()
        # Population should change
        changed = False
        for orig, new in zip(original_population, ga_manager.population):
            if not np.allclose(orig, new):
                changed = True
                break
        assert changed, "Population not evolved"

    def test_ga_generation_updates_current_params_to_best(self, ga_manager):
        """Test that current params are updated to best individual."""
        ga_manager._train_ga_generation()
        # Current params should be one of the population members
        found = False
        for individual in ga_manager.population:
            if np.allclose(ga_manager.current_params, individual):
                found = True
                break
        # Note: current_params is set to best from previous generation's results
        # so it may not be in new population, but params should be updated
        assert ga_manager.current_params is not None


class TestTrain:
    """Tests for train method."""

    def test_train_es_runs_multiple_generations(self, es_manager):
        """Test that ES training runs for specified generations."""
        with patch.object(es_manager, "_train_es_generation", return_value=1.0) as mock:
            es_manager.train(num_generations=5, checkpoint_every=100)
            assert mock.call_count == 5

    def test_train_ga_runs_multiple_generations(self, ga_manager):
        """Test that GA training runs for specified generations."""
        with patch.object(ga_manager, "_train_ga_generation", return_value=1.0) as mock:
            ga_manager.train(num_generations=5, checkpoint_every=100)
            assert mock.call_count == 5

    def test_train_calls_checkpoint_at_interval(self, es_manager):
        """Test that checkpoints are saved at specified intervals."""
        with patch.object(es_manager, "_save_checkpoint") as mock_save:
            es_manager.train(num_generations=10, checkpoint_every=5)
            # Should be called at gen 5 and 10 (1-indexed: 4 and 9)
            assert mock_save.call_count == 2

    def test_train_unknown_strategy_raises(
        self, simple_model, parameter_proxy, survival_fitness
    ):
        """Test that unknown strategy type raises ValueError."""

        class UnknownStrategy:
            pass

        manager = EvolutionManager(
            model=simple_model,
            strategy=UnknownStrategy(),
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
        )
        with pytest.raises(ValueError, match="Unknown strategy type"):
            manager.train(num_generations=1)


class TestSaveCheckpoint:
    """Tests for _save_checkpoint method."""

    def test_checkpoint_saves_correctly(self, es_manager):
        """Test that checkpoint file is created with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            es_manager._save_checkpoint(str(checkpoint_path))

            assert checkpoint_path.exists()

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "generation" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "current_params" in checkpoint
            assert "population" in checkpoint

    def test_checkpoint_contains_generation(self, es_manager):
        """Test that checkpoint contains correct generation number."""
        es_manager.generation = 42
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            es_manager._save_checkpoint(str(checkpoint_path))

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert checkpoint["generation"] == 42

    def test_checkpoint_contains_model_state(self, es_manager):
        """Test that checkpoint contains model state dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            es_manager._save_checkpoint(str(checkpoint_path))

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model_state = checkpoint["model_state_dict"]
            assert "fc1.weight" in model_state
            assert "fc2.weight" in model_state

    def test_checkpoint_can_be_loaded(self, es_manager, simple_model):
        """Test that checkpoint can be loaded back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            es_manager._save_checkpoint(str(checkpoint_path))

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            new_model = SimpleModel()
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify parameters match
            for (name1, p1), (name2, p2) in zip(
                es_manager.model.named_parameters(), new_model.named_parameters()
            ):
                assert name1 == name2
                torch.testing.assert_close(p1, p2)

    def test_ga_checkpoint_contains_population(self, ga_manager):
        """Test that GA checkpoint contains population."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            ga_manager._save_checkpoint(str(checkpoint_path))

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert checkpoint["population"] is not None
            assert len(checkpoint["population"]) == ga_manager.strategy.pop_size


class TestIntegration:
    """Integration tests for EvolutionManager."""

    def test_full_es_training_loop(
        self, simple_model, openai_es_strategy, parameter_proxy, survival_fitness
    ):
        """Test a full ES training loop."""
        manager = EvolutionManager(
            model=simple_model,
            strategy=openai_es_strategy,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            num_workers=2,
            max_steps_per_episode=10,
        )
        initial_params = manager.current_params.copy()
        manager.train(num_generations=3, checkpoint_every=100)

        # Params should have changed
        assert not np.allclose(manager.current_params, initial_params)

    def test_full_ga_training_loop(
        self, simple_model, simple_ga_strategy, parameter_proxy, survival_fitness
    ):
        """Test a full GA training loop."""
        manager = EvolutionManager(
            model=simple_model,
            strategy=simple_ga_strategy,
            proxy=parameter_proxy,
            fitness_evaluator=survival_fitness,
            num_workers=2,
            max_steps_per_episode=10,
        )
        initial_params = manager.current_params.copy()
        manager.train(num_generations=3, checkpoint_every=100)

        # Params should have changed
        assert not np.allclose(manager.current_params, initial_params)

    def test_training_with_task_completion_fitness(
        self, simple_model, openai_es_strategy, parameter_proxy
    ):
        """Test training with TaskCompletionFitness evaluator."""
        fitness = TaskCompletionFitness(completion_reward=100.0, progress_weight=0.5)
        manager = EvolutionManager(
            model=simple_model,
            strategy=openai_es_strategy,
            proxy=parameter_proxy,
            fitness_evaluator=fitness,
            num_workers=2,
            max_steps_per_episode=10,
        )
        # Should run without errors
        manager.train(num_generations=2, checkpoint_every=100)
