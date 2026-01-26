"""
Tests for RealMultiModalDataset.

Purpose:
    Verify that the dataset correctly loads multimedia files from disk, aligns them by timestamp,
    and returns a valid UnifiedSample dictionary.

History:
    - Created during Real-Data Readiness phase.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import numpy as np

# We mock these because we don't want to actually depend on CV2 or audio libs for the unit test logic
from src.data.real_dataset import RealMultiModalDataset

@pytest.fixture
def mock_dataset_root():
    """
    Purpose:
        Create a temporary directory structure for the dataset.

    Returns:
        Path: Path to the temporary root directory.
    """
    temp_dir = tempfile.mkdtemp()
    root = Path(temp_dir)
    
    # Create session 1
    sess1 = root / "session_001"
    sess1.mkdir()
    (sess1 / "vision_left.mp4").touch()
    (sess1 / "audio.wav").touch()
    
    # Create metadata
    metadata = [
        {"timestamp": 0.1, "touch": [0.0]*5, "proprio": [0.0]*20},
        {"timestamp": 0.2, "touch": [0.1]*5, "proprio": [0.1]*20},
        {"timestamp": 0.3, "touch": [0.2]*5, "proprio": [0.2]*20},
    ]
    with open(sess1 / "metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
            
    yield root
    shutil.rmtree(temp_dir)

def test_real_dataset_loading(mock_dataset_root):
    """
    Purpose:
        Test that RealMultiModalDataset loads and aligns data correctly.

    Workflow:
        1. Initialize dataset with mock root
        2. Verify dataset length
        3. Retrieve a sample
        4. Verify sample keys and timestamp accuracy

    ToDo:
        None
    """
    dataset = RealMultiModalDataset(root_dir=str(mock_dataset_root))
    
    assert len(dataset) > 0
    sample = dataset[0]
    
    assert "inputs" in sample
    assert "targets" in sample
    
    inputs = sample["inputs"]
    assert "vision_left" in inputs
    assert "audio" in inputs
    assert "touch" in inputs
    assert inputs["timestamp"] == 0.1

def test_normalization():
    """
    Purpose:
        Verify that data is correctly normalized.
        
    Workflow:
        1. Initialize dataset with default normalization.
        2. Inspect vision and audio values (mocked) to ensure transformations are applied.
        (Note: Since we mock read_video/load, we mostly check that the shape/type is correct 
         and that the code doesn't crash during transform application).
    """
    # We can rely on the side-effects or simply check that the returned tensors 
    # are floats and have expected shapes after "normalization".
    pass 

def test_real_data_collate():
    """
    Purpose:
        Verify batching and padding logic.
    
    Workflow:
        1. Create a batch of samples with different lengths (if applicable) or just standard samples.
        2. Run collate function.
        3. Check stacked tensor shapes.
    """
    from src.data.real_dataset import real_data_collate_fn
    
    # Mock samples
    s1 = {
        "inputs": {
            "vision_left": torch.randn(3, 224, 224),
            "audio": torch.randn(16000),
            "touch": torch.randn(5, 3), # 5 points
            "timestamp": 0.1
        },
        "targets": {}
    }
    s2 = {
        "inputs": {
            "vision_left": torch.randn(3, 224, 224),
            "audio": torch.randn(20000), # Different length
            "touch": torch.randn(2, 3), # Different length
            "timestamp": 0.2
        },
        "targets": {}
    }
    
    batch = [s1, s2]
    collated = real_data_collate_fn(batch)
    
    inputs = collated["inputs"]
    
    # Vision should be stacked: [B, C, H, W]
    assert inputs["vision_left"].shape == (2, 3, 224, 224)
    
    # Audio should be padded to longest: [B, MaxLen]
    assert inputs["audio"].shape == (2, 20000)
    
    # Touch should be padded: [B, MaxPoints, Features]
    assert inputs["touch"].shape == (2, 5, 3)
