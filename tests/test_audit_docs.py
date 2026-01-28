
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
	Purpose:  
		Scans 'src', 'scripts', and 'tests' directories for Python files
		and verifies they have proper file headers and function docstrings
		using the logic from scripts/audit_docs.py.

	Workflow:
		1. Define directories to scan.
		2. Scan directories recursively.
		3. Scan root files.
		4. Accumulate failures in report.
		5. Assert report is empty.

	ToDo:
		- None
	==============================================================================
	"""
	report = {}
	walk_dirs = ["src", "scripts", "tests"]
	
	# 1. Check directories
	for start_dir_name in walk_dirs:
		start_dir = os.path.join(PROJECT_ROOT, start_dir_name)
		_scan_directory_for_docs(start_dir, report)

	# 2. Check root files
	_scan_root_files_for_docs(PROJECT_ROOT, report)

	# If report is not empty, fail the test with detailed output
	if report:
		error_msg = json.dumps(report, indent=2)
		pytest.fail(f"Documentation missing in the following files:\n{error_msg}")

def _scan_directory_for_docs(start_dir, report):
	"""
	Helper to scan directory for docs.
	
	Args:
		start_dir: Directory to scan.
		report: Dict to collect results.
		
	Returns:
		None
	"""
	if not os.path.exists(start_dir):
		return
		
	for root, dirs, files in os.walk(start_dir):
		for file in files:
			if not file.endswith(".py"):
				continue

			path = os.path.join(root, file)
			_check_and_report(path, report)

def _scan_root_files_for_docs(root_dir, report):
	"""
	Helper to scan root files for docs.
	
	Args:
		root_dir: Root directory to scan.
		report: Dict to collect results.
		
	Returns:
		None
	"""
	for file in os.listdir(root_dir):
		path = os.path.join(root_dir, file)
		if os.path.isfile(path) and file.endswith(".py"):
			_check_and_report(path, report)

def _check_and_report(path, report):
	"""
	Checks a single file and adds to report if issues found.
	
	Args:
		path: Path to file.
		report: Dict to collect results.
		
	Returns:
		None
	"""
	result = check_file(path)
	if result["file_header"] or result["functions"]:
		rel_path = os.path.relpath(path, PROJECT_ROOT)
		report[rel_path] = result
