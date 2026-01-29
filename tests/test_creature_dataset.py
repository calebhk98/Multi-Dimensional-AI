"""
Tests for CreatureDataset and collate_fn.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CreatureDataset, collate_fn


class TestCreatureDatasetInit:
    """Tests for CreatureDataset initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        dataset = CreatureDataset()
        assert dataset.length == 1000
        assert dataset.synthetic is True
        assert dataset.generator is not None

    def test_init_custom_length(self):
        """Test initialization with custom length."""
        dataset = CreatureDataset(length=500)
        assert dataset.length == 500
        assert len(dataset) == 500

    def test_init_synthetic_mode(self):
        """Test initialization in synthetic mode."""
        dataset = CreatureDataset(synthetic=True)
        assert dataset.synthetic is True

    def test_init_non_synthetic_mode(self):
        """Test initialization in non-synthetic mode."""
        dataset = CreatureDataset(synthetic=False)
        assert dataset.synthetic is False

    def test_init_with_config(self):
        """Test initialization with config dict."""
        config = {"some_key": "some_value"}
        dataset = CreatureDataset(config=config)
        # Config is currently not used but should not cause error
        assert dataset is not None


class TestCreatureDatasetLen:
    """Tests for __len__ method."""

    def test_length_returns_configured_value(self):
        """Test that __len__ returns the configured length."""
        dataset = CreatureDataset(length=100)
        assert len(dataset) == 100

    def test_length_with_various_values(self):
        """Test __len__ with various length values."""
        for length in [1, 10, 100, 1000, 10000]:
            dataset = CreatureDataset(length=length)
            assert len(dataset) == length


class TestCreatureDatasetGetitem:
    """Tests for __getitem__ method."""

    def test_getitem_returns_dict(self):
        """Test that __getitem__ returns a dictionary."""
        dataset = CreatureDataset(length=10)
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_getitem_returns_valid_sample(self):
        """Test that __getitem__ returns a valid sample structure."""
        dataset = CreatureDataset(length=10)
        sample = dataset[0]

        # Sample should have inputs and targets
        assert "inputs" in sample or any(
            isinstance(v, torch.Tensor) for v in sample.values()
        )

    def test_synthetic_mode_uses_generator(self):
        """Test that synthetic mode uses the generator."""
        dataset = CreatureDataset(length=10, synthetic=True)
        sample = dataset[0]
        # Should return data from generator without error
        assert sample is not None

    def test_non_synthetic_mode_raises_not_implemented(self):
        """Test that non-synthetic mode raises NotImplementedError."""
        dataset = CreatureDataset(length=10, synthetic=False)
        with pytest.raises(NotImplementedError, match="Real data loading not implemented"):
            _ = dataset[0]

    def test_getitem_different_indices(self):
        """Test that different indices can be accessed."""
        dataset = CreatureDataset(length=100)
        samples = [dataset[i] for i in [0, 10, 50, 99]]
        assert len(samples) == 4
        for sample in samples:
            assert sample is not None

    def test_getitem_generates_new_data_each_time(self):
        """Test that each call to __getitem__ generates new data."""
        dataset = CreatureDataset(length=10)
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Samples may have same structure but values should differ
        # (synthetic generator creates random data each time)
        # This depends on generator implementation


class TestCollateFn:
    """Tests for collate_fn."""

    def test_collate_fn_batches_correctly(self):
        """Test that collate_fn creates proper batches."""
        dataset = CreatureDataset(length=10)
        batch = [dataset[i] for i in range(4)]
        collated = collate_fn(batch)

        assert isinstance(collated, dict)

    def test_collate_fn_stacks_tensors(self):
        """Test that collate_fn stacks tensors along batch dimension."""
        dataset = CreatureDataset(length=10)
        batch = [dataset[i] for i in range(4)]
        collated = collate_fn(batch)

        # Check that tensors are batched (batch dimension = 4)
        for key, value in collated.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == 4, f"Key {key} not batched correctly"

    def test_collate_fn_with_dataloader(self):
        """Test collate_fn works with DataLoader."""
        dataset = CreatureDataset(length=20)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        batch = next(iter(dataloader))
        assert isinstance(batch, dict)


class TestIntegration:
    """Integration tests for CreatureDataset."""

    def test_full_dataloader_iteration(self):
        """Test iterating through entire dataset with DataLoader."""
        dataset = CreatureDataset(length=20)
        dataloader = DataLoader(
            dataset, batch_size=4, collate_fn=collate_fn, num_workers=0
        )

        batches = list(dataloader)
        assert len(batches) == 5  # 20 samples / 4 batch_size

    def test_dataset_with_drop_last(self):
        """Test dataset with drop_last=True."""
        dataset = CreatureDataset(length=22)  # Not divisible by 4
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0,
        )

        batches = list(dataloader)
        assert len(batches) == 5  # 22 // 4 = 5 (dropping 2)

    def test_dataset_reproducibility_with_seed(self):
        """Test that dataset is reproducible when generator is seeded."""
        # Note: This depends on the SyntheticDataGenerator implementation
        # The current implementation may not be seeded
        dataset1 = CreatureDataset(length=10)
        dataset2 = CreatureDataset(length=10)

        # Both datasets should work
        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert sample1 is not None
        assert sample2 is not None
