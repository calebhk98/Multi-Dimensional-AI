"""
Session validation script for VR data collection.

Validates recorded VR sessions for data quality and integrity.
Checks file existence, timestamp continuity, sensor ranges, and sync.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Third-party imports (conditional)
try:
	import cv2
	HAS_CV2 = True
except ImportError:
	HAS_CV2 = False

try:
	import numpy as np
	HAS_NUMPY = True
except ImportError:
	HAS_NUMPY = False

try:
	import wave
	HAS_WAVE = True
except ImportError:
	HAS_WAVE = False

# Constants
MAX_TIMESTAMP_GAP_MS = 100  # Maximum gap before warning
GOOD_TIMESTAMP_GAP_MS = 35  # Expected gap for 30 FPS
TOUCH_MIN = 0.0
TOUCH_MAX = 1.0
AUDIO_RMS_THRESHOLD = 0.001  # Minimum RMS to not be silent

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_files_exist(session_dir: Path) -> Tuple[bool, List[str]]:
	"""
	Check if required files exist in session directory.
	
	Args:
		session_dir: Path to session directory
		
	Returns:
		Tuple of (passed, errors)
	"""
	errors = []
	required_files = ['metadata.jsonl', 'audio.wav', 'vision_left.mp4']
	
	for filename in required_files:
		filepath = session_dir / filename
		if not filepath.exists():
			errors.append(f"Missing required file: {filename}")
	
	return len(errors) == 0, errors


def validate_metadata_continuity(session_dir: Path, strict: bool = False) -> Tuple[bool, List[str], List[str]]:
	"""
	Validate timestamp continuity in metadata.jsonl.
	
	Args:
		session_dir: Path to session directory
		strict: Enable strict validation (fail on warnings)
		
	Returns:
		Tuple of (passed, errors, warnings)
	"""
	metadata_path = session_dir / 'metadata.jsonl'
	if not metadata_path.exists():
		return False, ["metadata.jsonl not found"], []
	
	errors = []
	warnings = []
	timestamps = []
	
	try:
		with open(metadata_path, 'r') as f:
			for line_num, line in enumerate(f, 1):
				if not line.strip():
					continue
				
				try:
					data = json.loads(line)
				except json.JSONDecodeError as e:
					errors.append(f"Line {line_num}: Invalid JSON - {e}")
					continue
				
				# Check required fields
				if 'timestamp' not in data:
					errors.append(f"Line {line_num}: Missing 'timestamp' field")
					continue
				
				timestamps.append(data['timestamp'])
		
		# Check timestamp continuity
		if len(timestamps) > 1:
			for i in range(1, len(timestamps)):
				gap_ms = (timestamps[i] - timestamps[i-1]) * 1000
				
				if gap_ms > MAX_TIMESTAMP_GAP_MS:
					errors.append(f"Timestamp gap > {MAX_TIMESTAMP_GAP_MS}ms at index {i}: {gap_ms:.1f}ms")
				elif gap_ms > GOOD_TIMESTAMP_GAP_MS:
					warnings.append(f"Timestamp gap {gap_ms:.1f}ms at index {i} (expected ~{GOOD_TIMESTAMP_GAP_MS}ms)")
	
	except Exception as e:
		errors.append(f"Error reading metadata.jsonl: {e}")
	
	passed = len(errors) == 0
	if strict and len(warnings) > 0:
		passed = False
	
	return passed, errors, warnings


def validate_sensor_ranges(session_dir: Path) -> Tuple[bool, List[str]]:
	"""
	Validate sensor value ranges in metadata.
	
	Args:
		session_dir: Path to session directory
		
	Returns:
		Tuple of (passed, errors)
	"""
	metadata_path = session_dir / 'metadata.jsonl'
	if not metadata_path.exists():
		return False, ["metadata.jsonl not found"]
	
	errors = []
	
	try:
		with open(metadata_path, 'r') as f:
			for line_num, line in enumerate(f, 1):
				if not line.strip():
					continue
				
				try:
					data = json.loads(line)
				except json.JSONDecodeError:
					continue  # Already reported in continuity check
				
				# Validate touch values
				if 'touch' in data:
					touch = data['touch']
					if isinstance(touch, list):
						for idx, val in enumerate(touch):
							if not (TOUCH_MIN <= val <= TOUCH_MAX):
								errors.append(f"Line {line_num}: Touch[{idx}] value {val} out of range [{TOUCH_MIN}, {TOUCH_MAX}]")
	
	except Exception as e:
		errors.append(f"Error validating sensor ranges: {e}")
	
	return len(errors) == 0, errors


def validate_audio_video_sync(session_dir: Path) -> Tuple[bool, List[str], List[str]]:
	"""
	Validate audio and video synchronization.
	
	Args:
		session_dir: Path to session directory
		
	Returns:
		Tuple of (passed, errors, warnings)
	"""
	if not HAS_CV2 or not HAS_WAVE:
		return True, [], ["Skipping A/V sync check (missing opencv-python or wave)"]
	
	errors = []
	warnings = []
	
	audio_path = session_dir / 'audio.wav'
	video_path = session_dir / 'vision_left.mp4'
	
	if not audio_path.exists() or not video_path.exists():
		return True, [], ["Skipping A/V sync (files missing)"]
	
	try:
		# Get audio duration
		with wave.open(str(audio_path), 'r') as wav:
			frames = wav.getnframes()
			rate = wav.getframerate()
			audio_duration = frames / float(rate)
		
		# Get video duration
		cap = cv2.VideoCapture(str(video_path))
		fps = cap.get(cv2.CAP_PROP_FPS)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		video_duration = frame_count / fps if fps > 0 else 0
		cap.release()
		
		# Check sync
		duration_diff = abs(audio_duration - video_duration)
		if duration_diff > 1.0:  # More than 1 second difference
			errors.append(f"Audio/video desync: {duration_diff:.2f}s difference")
		elif duration_diff > 0.1:  # More than 100ms
			warnings.append(f"Minor A/V desync: {duration_diff:.2f}s")
	
	except Exception as e:
		warnings.append(f"Could not validate A/V sync: {e}")
	
	return len(errors) == 0, errors, warnings


def validate_audio_quality(session_dir: Path) -> Tuple[bool, List[str], List[str]]:
	"""
	Check if audio is not silent.
	
	Args:
		session_dir: Path to session directory
		
	Returns:
		Tuple of (passed, errors, warnings)
	"""
	if not HAS_WAVE or not HAS_NUMPY:
		return True, [], ["Skipping audio quality check (missing dependencies)"]
	
	audio_path = session_dir / 'audio.wav'
	if not audio_path.exists():
		return True, [], []
	
	errors = []
	warnings = []
	
	try:
		with wave.open(str(audio_path), 'r') as wav:
			frames = wav.readframes(wav.getnframes())
			audio_data = np.frombuffer(frames, dtype=np.int16)
			rms = np.sqrt(np.mean(audio_data.astype(float)**2)) / 32768.0  # Normalize
			
			if rms < AUDIO_RMS_THRESHOLD:
				warnings.append(f"Audio appears silent (RMS: {rms:.6f})")
	
	except Exception as e:
		warnings.append(f"Could not validate audio quality: {e}")
	
	return len(errors) == 0, errors, warnings


def validate_session(session_dir: Path, strict: bool = False, fix: bool = False) -> Dict[str, Any]:
	"""
	Run all validation checks on a session.
	
	Args:
		session_dir: Path to session directory
		strict: Enable strict validation
		fix: Attempt to auto-fix issues (not yet implemented)
		
	Returns:
		Validation report dictionary
	"""
	session_name = session_dir.name
	logger.info(f"Validating session: {session_name}")
	
	report = {
		"session": session_name,
		"status": "passed",
		"checks": {},
		"warnings": [],
		"errors": []
	}
	
	# File existence check
	files_passed, files_errors = validate_files_exist(session_dir)
	report["checks"]["files_exist"] = files_passed
	report["errors"].extend(files_errors)
	
	# If files missing, skip other checks
	if not files_passed:
		report["status"] = "failed"
		return report
	
	# Timestamp continuity
	continuity_passed, continuity_errors, continuity_warnings = validate_metadata_continuity(session_dir, strict)
	report["checks"]["timestamp_continuity"] = continuity_passed
	report["errors"].extend(continuity_errors)
	report["warnings"].extend(continuity_warnings)
	
	# Sensor ranges
	ranges_passed, ranges_errors = validate_sensor_ranges(session_dir)
	report["checks"]["sensor_ranges"] = ranges_passed
	report["errors"].extend(ranges_errors)
	
	# A/V sync
	sync_passed, sync_errors, sync_warnings = validate_audio_video_sync(session_dir)
	report["checks"]["video_audio_sync"] = sync_passed
	report["errors"].extend(sync_errors)
	report["warnings"].extend(sync_warnings)
	
	# Audio quality
	audio_passed, audio_errors, audio_warnings = validate_audio_quality(session_dir)
	report["errors"].extend(audio_errors)
	report["warnings"].extend(audio_warnings)
	
	# Determine final status
	if len(report["errors"]) > 0:
		report["status"] = "failed"
	elif len(report["warnings"]) > 0:
		report["status"] = "warnings"
	
	return report


def print_report(report: Dict[str, Any]) -> None:
	"""
	Print validation report to console.
	
	Args:
		report: Validation report dictionary
	"""
	print(f"\n{'='*60}")
	print(f"Session: {report['session']}")
	print(f"Status: {report['status'].upper()}")
	print(f"{'='*60}\n")
	
	print("Checks:")
	for check, passed in report["checks"].items():
		status = "✓ PASS" if passed else "✗ FAIL"
		print(f"  {check}: {status}")
	
	if report["errors"]:
		print(f"\nErrors ({len(report['errors'])}):")
		for error in report["errors"]:
			print(f"  ✗ {error}")
	
	if report["warnings"]:
		print(f"\nWarnings ({len(report['warnings'])}):")
		for warning in report["warnings"]:
			print(f"  ⚠ {warning}")
	
	print()


def main():
	"""
	Main entry point for session validation.
	"""
	parser = argparse.ArgumentParser(description="Validate VR session data quality")
	parser.add_argument("session_dir", type=Path, help="Path to session directory")
	parser.add_argument("--strict", action="store_true", help="Fail on warnings")
	parser.add_argument("--fix", action="store_true", help="Attempt to auto-fix issues")
	parser.add_argument("--json", action="store_true", help="Output JSON report")
	
	args = parser.parse_args()
	
	if not args.session_dir.exists():
		logger.error(f"Session directory not found: {args.session_dir}")
		sys.exit(1)
	
	if not args.session_dir.is_dir():
		logger.error(f"Path is not a directory: {args.session_dir}")
		sys.exit(1)
	
	# Run validation
	report = validate_session(args.session_dir, strict=args.strict, fix=args.fix)
	
	# Output report
	if args.json:
		print(json.dumps(report, indent=2))
	else:
		print_report(report)
	
	# Exit with appropriate code
	if report["status"] == "failed":
		sys.exit(1)
	elif report["status"] == "warnings" and args.strict:
		sys.exit(1)
	else:
		sys.exit(0)


if __name__ == "__main__":
	main()
