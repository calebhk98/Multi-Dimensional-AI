"""
Automated test to enforce indentation limits (max 5 levels).
"""
import pytest
import os
import sys

# Ensure scripts dir is on python path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from analyze_indentation import analyze_indentation
except ImportError:
    pytest.fail(f"Could not import analyze_indentation from {SCRIPTS_DIR}")

def test_indentation_depth():
    """
    ==============================================================================
    Test: test_indentation_depth
    ==============================================================================
    Purpose:  
        Scans the codebase (`src`, `scripts`, `tests`) and asserts that no
        file contains lines with indentation >= 5 levels (20 spaces/tabs).

    Workflow:
        1. Define directories to walk.
        2. Recursively check each directory using helper.
        3. Accumulate violations.
        4. Fail if any violations found.

    ToDo:
        - None
    ==============================================================================
    """
    walk_dirs = ["src", "scripts", "tests"]
    violations = {}

    for start_dir_name in walk_dirs:
        start_dir = os.path.join(PROJECT_ROOT, start_dir_name)
        _check_directory(start_dir, violations)

    if violations:
        error_msg = "Indentation violations found (>= 5 levels / 20 spaces):\n"
        for fpath, lines in violations.items():
            error_msg += f"  {fpath}: lines {lines}\n"
        pytest.fail(error_msg)

def _check_directory(start_dir, violations):
    """
    Helper to check all python files in a directory.

    Args:
        start_dir: Directory to scan.
        violations: Dictionary to collect violations.
    """
    if not os.path.exists(start_dir):
        return

    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
                
            path = os.path.join(root, file)
            _check_file(path, violations)

def _check_file(path, violations):
    """
    Helper to analyze a single file and record violations.
    
    Args:
        path: File path to analyze.
        violations: Dictionary to collect violations.
    """
    result = analyze_indentation(path)
    
    if result and result.get("deep_lines"):
        rel_path = os.path.relpath(path, PROJECT_ROOT)
        violations[rel_path] = result["deep_lines"]
