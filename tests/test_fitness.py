"""
Tests for evolution fitness evaluation module.

Purpose:
    Verify fitness evaluators correctly compute fitness scores
    for various environment states and model outputs.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from src.evolution.fitness import (
    FitnessEvaluator,
    SurvivalFitness,
    TaskCompletionFitness
)


class TestFitnessEvaluatorBase:
    """Tests for FitnessEvaluator base class."""

    def test_base_evaluator_not_implemented(self):
        """
        Test base class raises NotImplementedError.

        Purpose:
            Verify abstract interface enforces implementation.

        Workflow:
            1. Create base evaluator
            2. Call evaluate
            3. Expect NotImplementedError
        """
        evaluator = FitnessEvaluator()

        with pytest.raises(NotImplementedError):
            evaluator.evaluate({}, {})


class TestSurvivalFitness:
    """Tests for SurvivalFitness evaluator."""

    def test_init_default_params(self):
        """
        Test default initialization parameters.

        Purpose:
            Verify default reward and penalty values.
        """
        evaluator = SurvivalFitness()

        assert evaluator.timestep_reward == 1.0
        assert evaluator.failure_penalty == -100.0

    def test_init_custom_params(self):
        """
        Test custom initialization parameters.

        Purpose:
            Verify custom reward/penalty values are stored.
        """
        evaluator = SurvivalFitness(
            timestep_reward=2.5,
            failure_penalty=-50.0
        )

        assert evaluator.timestep_reward == 2.5
        assert evaluator.failure_penalty == -50.0

    def test_evaluate_alive_positive_timesteps(self):
        """
        Test fitness for alive creature with positive timesteps.

        Purpose:
            Verify reward scales with timesteps survived.

        Workflow:
            1. Create evaluator
            2. Evaluate with alive=True and timesteps
            3. Check fitness = timesteps * reward
        """
        evaluator = SurvivalFitness(timestep_reward=1.0)

        env_state = {"timesteps": 100, "alive": True}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 100.0

    def test_evaluate_alive_zero_timesteps(self):
        """
        Test fitness for alive creature with zero timesteps.

        Purpose:
            Verify zero timesteps gives zero fitness.
        """
        evaluator = SurvivalFitness(timestep_reward=1.0)

        env_state = {"timesteps": 0, "alive": True}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 0.0

    def test_evaluate_dead_returns_penalty(self):
        """
        Test fitness for dead creature returns penalty.

        Purpose:
            Verify death penalty is applied regardless of timesteps.
        """
        evaluator = SurvivalFitness(
            timestep_reward=1.0,
            failure_penalty=-100.0
        )

        env_state = {"timesteps": 500, "alive": False}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == -100.0

    def test_evaluate_missing_alive_defaults_true(self):
        """
        Test fitness when alive key is missing.

        Purpose:
            Verify alive defaults to True when not specified.
        """
        evaluator = SurvivalFitness(timestep_reward=2.0)

        env_state = {"timesteps": 50}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 100.0  # 50 * 2.0

    def test_evaluate_missing_timesteps_defaults_zero(self):
        """
        Test fitness when timesteps key is missing.

        Purpose:
            Verify timesteps defaults to 0 when not specified.
        """
        evaluator = SurvivalFitness(timestep_reward=1.0)

        env_state = {"alive": True}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 0.0

    def test_evaluate_custom_reward_scales(self):
        """
        Test custom timestep reward scaling.

        Purpose:
            Verify fitness scales with custom reward.
        """
        evaluator = SurvivalFitness(timestep_reward=0.5)

        env_state = {"timesteps": 200, "alive": True}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 100.0  # 200 * 0.5

    def test_evaluate_empty_env_state(self):
        """
        Test fitness with empty environment state.

        Purpose:
            Verify defaults are used for empty state.
        """
        evaluator = SurvivalFitness()

        fitness = evaluator.evaluate({}, {})

        assert fitness == 0.0  # 0 timesteps * 1.0


class TestTaskCompletionFitness:
    """Tests for TaskCompletionFitness evaluator."""

    def test_init_default_params(self):
        """
        Test default initialization parameters.

        Purpose:
            Verify default completion reward and progress weight.
        """
        evaluator = TaskCompletionFitness()

        assert evaluator.completion_reward == 100.0
        assert evaluator.progress_weight == 0.5

    def test_init_custom_params(self):
        """
        Test custom initialization parameters.

        Purpose:
            Verify custom values are stored correctly.
        """
        evaluator = TaskCompletionFitness(
            completion_reward=200.0,
            progress_weight=0.75
        )

        assert evaluator.completion_reward == 200.0
        assert evaluator.progress_weight == 0.75

    def test_evaluate_task_completed(self):
        """
        Test fitness when task is completed.

        Purpose:
            Verify full completion reward is given.

        Workflow:
            1. Create evaluator
            2. Evaluate with task_completed=True
            3. Check full reward returned
        """
        evaluator = TaskCompletionFitness(completion_reward=100.0)

        env_state = {"task_completed": True, "task_progress": 1.0}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 100.0

    def test_evaluate_partial_progress(self):
        """
        Test fitness for partial task progress.

        Purpose:
            Verify partial progress gives weighted reward.
        """
        evaluator = TaskCompletionFitness(
            completion_reward=100.0,
            progress_weight=0.5
        )

        env_state = {"task_completed": False, "task_progress": 0.6}
        fitness = evaluator.evaluate({}, env_state)

        expected = 0.6 * 0.5 * 100.0  # progress * weight * reward
        assert fitness == expected

    def test_evaluate_zero_progress(self):
        """
        Test fitness when no progress made.

        Purpose:
            Verify zero progress gives zero fitness.
        """
        evaluator = TaskCompletionFitness()

        env_state = {"task_completed": False, "task_progress": 0.0}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 0.0

    def test_evaluate_missing_completed_defaults_false(self):
        """
        Test fitness when task_completed key is missing.

        Purpose:
            Verify task_completed defaults to False.
        """
        evaluator = TaskCompletionFitness(
            completion_reward=100.0,
            progress_weight=1.0
        )

        env_state = {"task_progress": 0.5}
        fitness = evaluator.evaluate({}, env_state)

        # Should use partial progress calculation
        assert fitness == 50.0  # 0.5 * 1.0 * 100

    def test_evaluate_missing_progress_defaults_zero(self):
        """
        Test fitness when task_progress key is missing.

        Purpose:
            Verify task_progress defaults to 0.0.
        """
        evaluator = TaskCompletionFitness()

        env_state = {"task_completed": False}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 0.0

    def test_evaluate_completed_ignores_progress(self):
        """
        Test that completion overrides partial progress.

        Purpose:
            Verify completed task gets full reward regardless of progress value.
        """
        evaluator = TaskCompletionFitness(completion_reward=100.0)

        # Even with low progress, completion should give full reward
        env_state = {"task_completed": True, "task_progress": 0.1}
        fitness = evaluator.evaluate({}, env_state)

        assert fitness == 100.0

    def test_evaluate_empty_env_state(self):
        """
        Test fitness with empty environment state.

        Purpose:
            Verify defaults are used for empty state.
        """
        evaluator = TaskCompletionFitness()

        fitness = evaluator.evaluate({}, {})

        assert fitness == 0.0  # Not completed, 0 progress

    def test_evaluate_high_progress_not_completed(self):
        """
        Test high progress but not completed.

        Purpose:
            Verify high progress still gets partial reward.
        """
        evaluator = TaskCompletionFitness(
            completion_reward=100.0,
            progress_weight=0.5
        )

        env_state = {"task_completed": False, "task_progress": 0.99}
        fitness = evaluator.evaluate({}, env_state)

        expected = 0.99 * 0.5 * 100.0
        assert abs(fitness - expected) < 0.001


class TestFitnessIntegration:
    """Integration tests for fitness evaluators."""

    def test_survival_fitness_realistic_scenario(self):
        """
        Test survival fitness in realistic training scenario.

        Purpose:
            Verify fitness works in typical training loop context.
        """
        evaluator = SurvivalFitness(timestep_reward=0.01)

        # Simulate evolution episode results
        results = []
        for episode in range(5):
            timesteps = (episode + 1) * 100
            alive = episode < 4  # Last episode dies

            env_state = {"timesteps": timesteps, "alive": alive}
            fitness = evaluator.evaluate({}, env_state)
            results.append(fitness)

        # First 4 should be positive, last should be penalty
        assert all(r > 0 for r in results[:4])
        assert results[4] == -100.0

    def test_task_completion_progressive_improvement(self):
        """
        Test task completion fitness shows progressive improvement.

        Purpose:
            Verify fitness increases with task progress.
        """
        evaluator = TaskCompletionFitness(
            completion_reward=100.0,
            progress_weight=1.0
        )

        # Simulate training with improving progress
        fitnesses = []
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            completed = progress >= 1.0
            env_state = {"task_completed": completed, "task_progress": progress}
            fitness = evaluator.evaluate({}, env_state)
            fitnesses.append(fitness)

        # Fitness should monotonically increase
        for i in range(len(fitnesses) - 1):
            assert fitnesses[i+1] >= fitnesses[i]

        # Final should be full completion reward
        assert fitnesses[-1] == 100.0
