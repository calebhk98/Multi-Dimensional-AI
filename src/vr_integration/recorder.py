
import json
import logging
import base64
import time
import struct
import wave
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import queue
import threading

# Use OpenCV if available, otherwise mock or warn
try:
    import cv2
except ImportError:
    cv2 = None

from src.vr_integration.protocol import VRInputMessage

logger = logging.getLogger(__name__)

class VRRecorder:
    """
    Records VR session data to disk.
    
    Handlers synchronized saving of:
    - Vision (Left/Right) -> .mp4
    - Audio -> .wav
    - Metadata (Touch, Proprio, etc.) -> .jsonl
    """
    
    def __init__(self, storage_root: str = "dataset_raw"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        self.current_session_dir: Optional[Path] = None
        self._is_recording = False
        
        # Writers
        self.meta_file = None
        self.video_writer_l = None
        self.video_writer_r = None
        self.audio_file = None
        
        # Audio buffer
        self.audio_buffer = []
        self.audio_sample_rate = 16000 # Default alignment
        
        # Threading for non-blocking I/O (optional, but good for realtime)
        self.queue = queue.Queue()
        self.worker_thread = None

    def start_session(self, session_id: str = None):
        """Start a new recording session."""
        if not session_id:
            session_id = f"session_{int(time.time())}"
            
        self.current_session_dir = self.storage_root / session_id
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.meta_file = open(self.current_session_dir / "metadata.jsonl", "w")
        
        # Video - initialized on first frame when we know resolution
        self.video_writer_l = None
        self.video_writer_r = None
        
        # Audio
        self.audio_path = self.current_session_dir / "audio.wav"
        self.audio_file = wave.open(str(self.audio_path), 'wb')
        self.audio_file.setnchannels(1)
        self.audio_file.setsampwidth(2) # 16-bit
        self.audio_file.setframerate(self.audio_sample_rate)
        
        self._is_recording = True
        logger.info(f"Started recording session: {session_id}")

    def stop_session(self):
        """Stop current session and close files."""
        if not self._is_recording:
            return
            
        self._is_recording = False
        
        if self.meta_file:
            self.meta_file.close()
            self.meta_file = None
            
        if self.video_writer_l:
            self.video_writer_l.release()
        if self.video_writer_r:
            self.video_writer_r.release()
            
        if self.audio_file:
            self.audio_file.close()
            self.audio_file = None
            
        logger.info(f"Stopped recording session: {self.current_session_dir.name}")
        self.current_session_dir = None

    def record_frame(self, message: VRInputMessage):
        """
        Record a single frame from input message.
        Writes immediately (blocking) for simplicity, or queue if needed.
        """
        if not self._is_recording:
            return

        # 1. Metadata (Proprio, Touch, Timestamp)
        meta_entry = {
            "timestamp": message.timestamp,
            "proprio": message.joint_positions, # Simplified
            "touch": message.touch_contacts,
            "has_vision": bool(message.vision_left),
            "has_audio": bool(message.audio_samples)
        }
        self.meta_file.write(json.dumps(meta_entry) + "\n")
        self.meta_file.flush()
        
        # 2. Audio
        if message.audio_samples:
            # Float list -> int16 bytes
            audio_data = np.array(message.audio_samples, dtype=np.float32)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            self.audio_file.writeframes(audio_int16.tobytes())
            
        # 3. Vision
        if cv2 and message.vision_left:
            self._write_video_frame(message.vision_left, message.vision_right)
            
    def _write_video_frame(self, left_b64: str, right_b64: str):
        """Decode and write video frames."""
        try:
            # Decode Left
            left_bytes = base64.b64decode(left_b64)
            left_np = np.frombuffer(left_bytes, dtype=np.uint8)
            left_img = cv2.imdecode(left_np, cv2.IMREAD_COLOR)
            
            if self.video_writer_l is None:
                h, w = left_img.shape[:2]
                self.video_writer_l = self._init_writer("vision_left.mp4", w, h)
            self.video_writer_l.write(left_img)
            
            # Decode Right (Optional)
            if right_b64:
                right_bytes = base64.b64decode(right_b64)
                right_np = np.frombuffer(right_bytes, dtype=np.uint8)
                right_img = cv2.imdecode(right_np, cv2.IMREAD_COLOR)
                
                if self.video_writer_r is None:
                    h, w = right_img.shape[:2]
                    self.video_writer_r = self._init_writer("vision_right.mp4", w, h)
                self.video_writer_r.write(right_img)
                
    def _init_writer(self, filename: str, w: int, h: int):
        """Helper to initialize video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(
            str(self.current_session_dir / filename), 
            fourcc, 30, (w, h)
        )
                
        except Exception as e:
            logger.error(f"Error writing video frame: {e}")
