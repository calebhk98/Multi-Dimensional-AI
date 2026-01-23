
"""
Automated test to audit codebase for missing documentation.
Replaces/Uses logic from scripts/audit_docs.py
"""
import pytest
import os
import sys
import json

# Add scripts directory to path to allow importing audit_docs
# We assume the test is run from project root or checks relative to this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

try:
    from audit_docs import check_file
except ImportError:
    # Fallback or error if audit_docs cannot be imported
    pytest.fail(f"Could not import audit_docs from {SCRIPTS_DIR}")

def test_codebase_documentation():
    """
    ==============================================================================
    Test: test_codebase_documentation
    ==============================================================================
    Purpose:  Scans 'src', 'scripts', and 'tests' directories for Python files
              and verifies they have proper file headers and function docstrings
              using the logic from scripts/audit_docs.py.

    Assertions:
        - The report dictionary must be empty (no missing documentation).
    ==============================================================================
    """
    report = {}
    walk_dirs = ["src", "scripts", "tests"]
    
    # 1. Check directories
    for start_dir_name in walk_dirs:
        start_dir = os.path.join(PROJECT_ROOT, start_dir_name)
        if not os.path.exists(start_dir):
            continue
            
        for root, dirs, files in os.walk(start_dir):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    # We pass the absolute path to check_file
                    result = check_file(path)
                    
                    if result["file_header"] or result["functions"]:
                        # Convert to relative path for readability in report
                        rel_path = os.path.relpath(path, PROJECT_ROOT)
                        report[rel_path] = result

    # 2. Check root files
    for file in os.listdir(PROJECT_ROOT):
        path = os.path.join(PROJECT_ROOT, file)
        if os.path.isfile(path) and file.endswith(".py"):
            result = check_file(path)
            if result["file_header"] or result["functions"]:
                report[file] = result

    # If report is not empty, fail the test with detailed output
    if report:
        error_msg = json.dumps(report, indent=2)
        pytest.fail(f"Documentation missing in the following files:\n{error_msg}")
