"""
Tests for TextDataset and TextOnlyDataset.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.text_dataset import TextDataset
from src.data.text_only_dataset import TextOnlyDataset, text_only_collate_fn


class TestTextDatasetInit:
    """Tests for TextDataset initialization."""

    @pytest.fixture
    def temp_token_file_bin(self):
        """Create a temporary .bin token file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tokens = np.arange(10000, dtype=np.uint16)
            tokens.tofile(f)
            yield f.name
        Path(f.name).unlink()

    @pytest.fixture
    def temp_token_file_npy(self):
        """Create a temporary .npy token file."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tokens = np.arange(10000, dtype=np.int64)
            np.save(f, tokens)
            yield f.name
        Path(f.name).unlink()

    @pytest.fixture
    def temp_token_file_pt(self):
        """Create a temporary .pt token file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tokens = torch.arange(10000, dtype=torch.long)
            torch.save(tokens, f.name)
            yield f.name
        Path(f.name).unlink()

    def test_init_with_bin_file(self, temp_token_file_bin):
        """Test initialization with .bin file."""
        dataset = TextDataset(temp_token_file_bin, seq_length=64)
        assert dataset is not None
        assert dataset.seq_length == 64

    def test_init_with_npy_file(self, temp_token_file_npy):
        """Test initialization with .npy file."""
        dataset = TextDataset(temp_token_file_npy, seq_length=64)
        assert dataset is not None

    def test_init_with_pt_file(self, temp_token_file_pt):
        """Test initialization with .pt file."""
        dataset = TextDataset(temp_token_file_pt, seq_length=64)
        assert dataset is not None

    def test_init_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            TextDataset("/nonexistent/path/tokens.bin")

    def test_init_unsupported_format(self):
        """Test that ValueError is raised for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some text")
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                TextDataset(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_init_default_seq_length(self, temp_token_file_bin):
        """Test default sequence length is 512."""
        dataset = TextDataset(temp_token_file_bin)
        assert dataset.seq_length == 512

    def test_num_samples_calculated_correctly(self, temp_token_file_bin):
        """Test that num_samples is calculated correctly."""
        dataset = TextDataset(temp_token_file_bin, seq_length=100)
        # 10000 tokens / 100 seq_length = 100 samples
        assert dataset.num_samples == 100


class TestTextDatasetLen:
    """Tests for TextDataset __len__ method."""

    @pytest.fixture
    def dataset(self):
        """Create a TextDataset with known size."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tokens = np.arange(10000, dtype=np.uint16)
            tokens.tofile(f)
            temp_path = f.name
        yield TextDataset(temp_path, seq_length=100)
        Path(temp_path).unlink()

    def test_len_returns_num_samples(self, dataset):
        """Test that __len__ returns num_samples."""
        assert len(dataset) == dataset.num_samples
        assert len(dataset) == 100


class TestTextDatasetGetitem:
    """Tests for TextDataset __getitem__ method."""

    @pytest.fixture
    def dataset(self):
        """Create a TextDataset with sequential tokens."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Create sequential tokens for predictable testing
            tokens = np.arange(10000, dtype=np.uint16)
            tokens.tofile(f)
            temp_path = f.name
        yield TextDataset(temp_path, seq_length=64)
        Path(temp_path).unlink()

    def test_getitem_returns_dict(self, dataset):
        """Test that __getitem__ returns a dictionary."""
        sample = dataset[0]
        assert isinstance(sample, dict)

    def test_getitem_has_input_and_target(self, dataset):
        """Test that sample has 'input' and 'target' keys."""
        sample = dataset[0]
        assert "input" in sample
        assert "target" in sample

    def test_input_shape(self, dataset):
        """Test that input has correct shape."""
        sample = dataset[0]
        assert sample["input"].shape == (64,)

    def test_target_shape(self, dataset):
        """Test that target has correct shape."""
        sample = dataset[0]
        assert sample["target"].shape == (64,)

    def test_input_target_offset(self, dataset):
        """Test that target is shifted by 1 from input."""
        sample = dataset[0]
        # For sequential tokens, target should be input shifted by 1
        # input: tokens[0:64], target: tokens[1:65]
        assert sample["input"][0].item() == 0
        assert sample["target"][0].item() == 1

    def test_getitem_dtype(self, dataset):
        """Test that tensors have correct dtype."""
        sample = dataset[0]
        assert sample["input"].dtype == torch.long
        assert sample["target"].dtype == torch.long

    def test_getitem_different_indices(self, dataset):
        """Test accessing different indices."""
        samples = [dataset[i] for i in [0, 10, 50]]
        # Check each sample starts at expected position
        assert samples[0]["input"][0].item() == 0
        assert samples[1]["input"][0].item() == 640  # 10 * 64
        assert samples[2]["input"][0].item() == 3200  # 50 * 64

    def test_getitem_edge_case_last_sample(self, dataset):
        """Test accessing the last sample."""
        last_idx = len(dataset) - 1
        sample = dataset[last_idx]
        assert sample["input"].shape == (64,)
        assert sample["target"].shape == (64,)


class TestTextOnlyDatasetInit:
    """Tests for TextOnlyDataset initialization."""

    @pytest.fixture
    def mock_text_dataset(self):
        """Create a mock TextDataset."""

        class MockTextDataset:
            def __init__(self):
                self._data = [
                    {"input": torch.arange(64), "target": torch.arange(1, 65)}
                    for _ in range(100)
                ]

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        return MockTextDataset()

    def test_init_wraps_text_dataset(self, mock_text_dataset):
        """Test that TextOnlyDataset wraps a TextDataset."""
        adapter = TextOnlyDataset(mock_text_dataset)
        assert adapter.text_dataset is mock_text_dataset


class TestTextOnlyDatasetLen:
    """Tests for TextOnlyDataset __len__ method."""

    @pytest.fixture
    def adapter(self):
        """Create a TextOnlyDataset adapter."""

        class MockTextDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"input": torch.arange(64), "target": torch.arange(1, 65)}

        return TextOnlyDataset(MockTextDataset())

    def test_len_delegates_to_wrapped_dataset(self, adapter):
        """Test that __len__ returns length of wrapped dataset."""
        assert len(adapter) == 100


class TestTextOnlyDatasetGetitem:
    """Tests for TextOnlyDataset __getitem__ method."""

    @pytest.fixture
    def adapter(self):
        """Create a TextOnlyDataset adapter."""

        class MockTextDataset:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {
                    "input": torch.arange(64) + idx * 64,
                    "target": torch.arange(1, 65) + idx * 64,
                }

        return TextOnlyDataset(MockTextDataset())

    def test_getitem_returns_model_format(self, adapter):
        """Test that __getitem__ returns model-compatible format."""
        sample = adapter[0]
        assert "inputs" in sample
        assert "targets" in sample

    def test_getitem_has_internal_voice_tokens(self, adapter):
        """Test that inputs contain internal_voice_tokens."""
        sample = adapter[0]
        assert "internal_voice_tokens" in sample["inputs"]

    def test_getitem_has_internal_text(self, adapter):
        """Test that targets contain internal_text."""
        sample = adapter[0]
        assert "internal_text" in sample["targets"]

    def test_getitem_preserves_data(self, adapter):
        """Test that data is preserved correctly."""
        sample = adapter[0]
        assert sample["inputs"]["internal_voice_tokens"][0].item() == 0
        assert sample["targets"]["internal_text"][0].item() == 1

    def test_getitem_different_indices(self, adapter):
        """Test accessing different indices."""
        sample0 = adapter[0]
        sample10 = adapter[10]

        assert sample0["inputs"]["internal_voice_tokens"][0].item() == 0
        assert sample10["inputs"]["internal_voice_tokens"][0].item() == 640


class TestTextOnlyCollateFn:
    """Tests for text_only_collate_fn."""

    @pytest.fixture
    def batch(self):
        """Create a batch of samples."""
        return [
            {
                "inputs": {"internal_voice_tokens": torch.arange(64)},
                "targets": {"internal_text": torch.arange(1, 65)},
            }
            for _ in range(4)
        ]

    def test_collate_returns_dict(self, batch):
        """Test that collate returns a dictionary."""
        collated = text_only_collate_fn(batch)
        assert isinstance(collated, dict)

    def test_collate_has_inputs_and_targets(self, batch):
        """Test that collated batch has inputs and targets."""
        collated = text_only_collate_fn(batch)
        assert "inputs" in collated
        assert "targets" in collated

    def test_collate_stacks_inputs(self, batch):
        """Test that inputs are stacked correctly."""
        collated = text_only_collate_fn(batch)
        assert collated["inputs"]["internal_voice_tokens"].shape == (4, 64)

    def test_collate_stacks_targets(self, batch):
        """Test that targets are stacked correctly."""
        collated = text_only_collate_fn(batch)
        assert collated["targets"]["internal_text"].shape == (4, 64)


class TestIntegration:
    """Integration tests for text datasets."""

    @pytest.fixture
    def text_dataset(self):
        """Create a real TextDataset."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tokens = np.arange(10000, dtype=np.uint16)
            tokens.tofile(f)
            temp_path = f.name
        yield TextDataset(temp_path, seq_length=64)
        Path(temp_path).unlink()

    def test_text_only_with_real_text_dataset(self, text_dataset):
        """Test TextOnlyDataset with a real TextDataset."""
        adapter = TextOnlyDataset(text_dataset)
        assert len(adapter) == len(text_dataset)

        sample = adapter[0]
        assert "inputs" in sample
        assert "targets" in sample

    def test_full_dataloader_pipeline(self, text_dataset):
        """Test full DataLoader pipeline."""
        adapter = TextOnlyDataset(text_dataset)
        dataloader = DataLoader(
            adapter, batch_size=4, collate_fn=text_only_collate_fn, num_workers=0
        )

        batch = next(iter(dataloader))
        assert batch["inputs"]["internal_voice_tokens"].shape == (4, 64)
        assert batch["targets"]["internal_text"].shape == (4, 64)

    def test_iterate_full_dataset(self, text_dataset):
        """Test iterating through entire dataset."""
        adapter = TextOnlyDataset(text_dataset)
        dataloader = DataLoader(
            adapter,
            batch_size=8,
            collate_fn=text_only_collate_fn,
            drop_last=True,
            num_workers=0,
        )

        batches = list(dataloader)
        # 156 samples (10000 // 64) / 8 batch_size = 19 batches (with drop_last)
        assert len(batches) >= 1

        for batch in batches:
            assert batch["inputs"]["internal_voice_tokens"].shape[0] == 8
            assert batch["targets"]["internal_text"].shape[0] == 8
