r"""
Environment Diagnostic Script

Quick script to verify the Python environment is correctly set up
for training the Multi-Dimensional AI model.

Usage:
	python scripts/check_environment.py
	.\venv\Scripts\python.exe scripts/check_environment.py
"""

import sys
from pathlib import Path


def check_python_version():
	"""
	Check Python version.
	
	Purpose:
		Verify Python version and virtual environment status.
	"""
	print("=" * 80)
	print("Python Version")
	print("=" * 80)
	print(f"Python {sys.version}")
	print(f"Executable: {sys.executable}")
	
	# Check if we're in venv
	in_venv = hasattr(sys, 'real_prefix') or (
		hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
	)
	
	if in_venv:
		print("✓ Running in virtual environment")
	else:
		print("⚠ NOT running in virtual environment (this is OK if using full path)")
	
	print()


def check_packages():
	"""
	Check required packages are installed.
	
	Purpose:
		Verify all required packages are available.
	
	Returns:
		True if all packages are installed.
	"""
	print("=" * 80)
	print("Required Packages")
	print("=" * 80)
	
	required_packages = {
		"torch": "PyTorch",
		"transformers": "HuggingFace Transformers",
		"numpy": "NumPy",
		"yaml": "PyYAML",
		"tqdm": "Progress bars",
	}
	
	all_ok = True
	
	for package, description in required_packages.items():
		try:
			if package == "yaml":
				# Special case for PyYAML
				import yaml
				module = yaml
			else:
				module = __import__(package)
			
			version = getattr(module, "__version__", "unknown")
			print(f"✓ {description:30s} {version}")
		except ImportError:
			print(f"✗ {description:30s} NOT INSTALLED")
			all_ok = False
	
	print()
	return all_ok


def check_cuda():
	"""
	Check CUDA availability.
	
	Purpose:
		Verify CUDA/GPU is available for training.
	"""
	print("=" * 80)
	print("CUDA / GPU")
	print("=" * 80)
	
	try:
		import torch
		
		if torch.cuda.is_available():
			print(f"✓ CUDA available")
			print(f"  Version: {torch.version.cuda}")
			print(f"  Device count: {torch.cuda.device_count()}")
			for i in range(torch.cuda.device_count()):
				print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
				props = torch.cuda.get_device_properties(i)
				print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
		else:
			print("⚠ CUDA not available (CPU-only mode)")
	except Exception as e:
		print(f"✗ Error checking CUDA: {e}")
	
	print()


def check_project_structure():
	"""
	Check project directories exist.
	
	Purpose:
		Verify required project directories are present.
	
	Returns:
		True if all directories exist.
	"""
	print("=" * 80)
	print("Project Structure")
	print("=" * 80)
	
	project_root = Path(__file__).parent.parent
	
	required_dirs = {
		"src": "Source code",
		"configs": "Configuration files",
		"scripts": "Training scripts",
		"tests": "Test suite",
		"docs": "Documentation",
	}
	
	all_ok = True
	
	for dir_name, description in required_dirs.items():
		dir_path = project_root / dir_name
		if dir_path.exists() and dir_path.is_dir():
			print(f"✓ {description:30s} {dir_path}")
		else:
			print(f"✗ {description:30s} MISSING: {dir_path}")
			all_ok = False
	
	print()
	return all_ok


def check_config_files():
	"""
	Check required config files exist.
	
	Purpose:
		Verify all required configuration files are present.
	
	Returns:
		True if all config files exist.
	"""
	print("=" * 80)
	print("Configuration Files")
	print("=" * 80)
	
	project_root = Path(__file__).parent.parent
	configs_dir = project_root / "configs"
	
	required_configs = [
		"text_only_config.yaml",
		"training_config.yaml",
		"model_config.yaml",
	]
	
	all_ok = True
	
	for config_file in required_configs:
		config_path = configs_dir / config_file
		if config_path.exists():
			print(f"✓ {config_file:30s} exists")
		else:
			print(f"✗ {config_file:30s} MISSING")
			all_ok = False
	
	print()
	return all_ok


def check_imports():
	"""
	Check src modules can be imported.
	
	Purpose:
		Verify project modules are importable.
	
	Returns:
		True if all modules import successfully.
	"""
	print("=" * 80)
	print("Module Imports")
	print("=" * 80)
	
	# Add project root to path
	project_root = Path(__file__).parent.parent
	sys.path.insert(0, str(project_root))
	
	modules_to_check = [
		"src.models.multimodal_transformer",
		"src.training.trainer",
		"src.data.text_dataset",
		"src.data.text_only_dataset",
	]
	
	all_ok = True
	
	for module_name in modules_to_check:
		try:
			__import__(module_name)
			print(f"✓ {module_name}")
		except ImportError as e:
			print(f"✗ {module_name}")
			print(f"  Error: {e}")
			all_ok = False
		except Exception as e:
			print(f"⚠ {module_name}")
			print(f"  Warning: {e}")
	
	print()
	return all_ok


def main():
	"""
	Run all checks.
	
	Purpose:
		Execute all environment checks and report status.
	
	Returns:
		0 if all checks pass, 1 otherwise.
	"""
	print("\n" + "=" * 80)
	print("Multi-Dimensional AI - Environment Diagnostic")
	print("=" * 80 + "\n")
	
	results = []
	
	check_python_version()
	
	results.append(("Packages", check_packages()))
	check_cuda()
	results.append(("Project Structure", check_project_structure()))
	results.append(("Config Files", check_config_files()))
	results.append(("Module Imports", check_imports()))
	
	# Summary
	print("=" * 80)
	print("Summary")
	print("=" * 80)
	
	all_passed = all(result for _, result in results)
	
	for name, passed in results:
		status = "✓ PASS" if passed else "✗ FAIL"
		print(f"{status:10s} {name}")
	
	print()
	
	if all_passed:
		print("✓ All checks passed! Environment is ready for training.")
	else:
		print("⚠ Some checks failed. Please fix the issues above.")
		print("\nCommon fixes:")
		print("  - Missing packages: pip install -r requirements.txt")
		print("  - Module import errors: Check that you're running from project root")
		print("  - CUDA issues: Reinstall PyTorch with CUDA support")
	
	print("=" * 80 + "\n")
	
	return 0 if all_passed else 1


if __name__ == "__main__":
	sys.exit(main())
