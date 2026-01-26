"""
Tests for the VRRecorder class.

Purpose:
    Verify that VRRecorder correctly initializes, records frames to disk (videos, audio, metadata),
    and handles session lifecycle events.

History:
    - Created during Real-Data Readiness phase.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.vr_integration.recorder import VRRecorder
from src.vr_integration.protocol import VRInputMessage

@pytest.fixture
def temp_storage():
    """
    Purpose:
        Create a temporary directory for testing storage.

    Returns:
        str: Path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_recorder_lifecycle(temp_storage):
    """
    Purpose:
        Test start and stop session creates directories.

    Workflow:
        1. Start session
        2. Check directory creation
        3. Stop session
        4. Check session clear

    ToDo:
        None
    """
    recorder = VRRecorder(storage_root=temp_storage)
    
    recorder.start_session("test_sess")
    assert (Path(temp_storage) / "test_sess").exists()
    assert (Path(temp_storage) / "test_sess" / "metadata.jsonl").exists()
    
    recorder.stop_session()
    assert recorder.current_session_dir is None

@patch("src.vr_integration.recorder.cv2")
def test_record_frame(mock_cv2, temp_storage):
    """
    Purpose:
        Test recording a single frame writes to metadata and audio.

    Workflow:
        1. Start session
        2. Create dummy message
        3. Record frame
        4. Verify metadata and audio written

    ToDo:
        None
    """
    recorder = VRRecorder(storage_root=temp_storage)
    recorder.start_session("test_sess")
    
    # Create dummy message
    msg = VRInputMessage(
        timestamp=1.0,
        audio_samples=[0.1, -0.1, 0.5],
        touch_contacts=[{"f": 1.0}],
        vision_left=None # Skip vision logic for simple test
    )
    
    recorder.record_frame(msg)
    
    # Check metadata
    with open(Path(temp_storage) / "test_sess" / "metadata.jsonl", "r") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["timestamp"] == 1.0
        assert data["touch"][0]["f"] == 1.0
        assert data["has_audio"] == True
        
    recorder.stop_session()
