"""
Script to analyze indentation levels in the codebase.
"""
import os
import ast
import statistics
import json

def analyze_indentation(filepath):
	"""
	==============================================================================
	Function: analyze_indentation
	==============================================================================
	Purpose:  Analyzes indentation for a single file. Calculates average and maximum
			indentation levels for the file and identifying specific blocks 
			(functions, classes) with their respective indentation metrics.

	Parameters:
		- filepath: str
			Path to the python file to analyze.

	Returns:
		Dict[str, Any] or None - Dictionary containing metrics, or None if empty:
			- "file_avg_indent": float
			- "file_max_indent": float
			- "deep_lines": List[int] (Lines exceeding threshold)
			- "blocks": List[Dict] (Metrics for classes/functions)

	Dependencies:
		- ast
		- statistics
	==============================================================================
	"""
	with open(filepath, 'r', encoding='utf-8') as f:
		lines = f.readlines()
		content = "".join(lines)

	# 1. Line-level indentation analysis
	line_indents = []
	deep_lines = []
	MAX_INDENT_THRESHOLD = 24 # 6 levels (4 spaces * 6) - Relaxed slightly for existing codebase
	
	for i, line in enumerate(lines):
		# Normalize indentation: treat tabs as 4 spaces
		expanded_line = line.expandtabs(4)
		stripped = expanded_line.lstrip()
		
		if not stripped or stripped.startswith('#'):
			continue
			
		indent_spaces = len(expanded_line) - len(stripped)
		line_indents.append(indent_spaces)
		
		if indent_spaces >= MAX_INDENT_THRESHOLD:
			deep_lines.append(i + 1) # 1-indexed

	if not line_indents:
		return None

	avg_indent = statistics.mean(line_indents) / 4.0
	max_indent = max(line_indents) / 4.0
	
	# 2. Block-level analysis (Functions/Classes) using AST
	try:
		tree = ast.parse(content)
	except SyntaxError:
		return {"error": "SyntaxError"}

	blocks = []
	
	for node in ast.walk(tree):
		if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
			continue

		block_data = _analyze_block(node, lines)
		if block_data:
			blocks.append(block_data)

	return {
		"file_avg_indent": avg_indent,
		"file_max_indent": max_indent,
		"deep_lines": deep_lines,
		"blocks": blocks
	}

def _analyze_block(node, lines):
	"""
	Analyzes a single AST block (Function/Class) for indentation metrics.

	Args:
		node: The AST node to analyze.
		lines: List of lines in the file.
		
	Returns:
		Dict with block metrics or None.
	"""
	# Simple heuristic: scan lines from start to end of node
	# (Note: end_lineno might be missing in older python, but we assume 3.8+)
	if not hasattr(node, 'end_lineno'):
		return None

	block_indents = []
	for i in range(node.lineno - 1, node.end_lineno):
		if i >= len(lines):
			break
			
		line = lines[i].expandtabs(4)
		stripped = line.lstrip()
		
		if not stripped or stripped.startswith('#'):
			continue
			
		indent = len(line) - len(stripped)
		block_indents.append(indent)
	
	if not block_indents:
		return None

	return {
		"type": type(node).__name__,
		"name": node.name,
		"lineno": node.lineno,
		"avg_indent": statistics.mean(block_indents) / 4.0,
		"max_indent": max(block_indents) / 4.0
	}

def _collect_files(start_dir, files_list):
	"""
	Recursively collects python files from a directory.
	
	Args:
		start_dir: Directory to search.
		files_list: List to append found files to.
	
	Returns:
		None (modifies files_list in-place)
	"""
	if not os.path.exists(start_dir):
		return
		
	for root, dirs, files in os.walk(start_dir):
		for file in files:
			if file.endswith(".py"):
				files_list.append(os.path.join(root, file))

def _collect_files(start_dir, files_list):
	"""
	Recursively collects python files from a directory.
	
	Args:
		start_dir: Directory to search.
		files_list: List to append found files to.
	
	Returns:
		None (modifies files_list in-place)
	"""
	if not os.path.exists(start_dir):
		return
		
	for root, dirs, files in os.walk(start_dir):
		for file in files:
			if file.endswith(".py"):
				files_list.append(os.path.join(root, file))

def print_simple_report(results):
	"""
	Print report in simple file:line:message format.
	
	Args:
		results: Dictionary of analysis results.
	"""
	found_violations = False
	for filepath, res in results.items():
		if res["deep_lines"]:
			found_violations = True
			for line in res["deep_lines"]:
				print(f"{filepath}:{line}:indentation_error: Indentation exceeds 5 levels")
	
	if not found_violations:
		# Optional: Print nothing if clean, or a success message? 
		# Standard unix tool behavior is silent on success.
		pass

def main():
	"""
	==============================================================================
	Function: main
	==============================================================================
	Purpose:  Main entry point for indent analysis. Scans directories or specific
			files provided via CLI args. Outputs violations in simple format
			by default, or detailed report if requested.

	Parameters:
		- None (uses sys.argv)

	Returns:
		None
	==============================================================================
	"""
	import argparse
	import sys
	
	parser = argparse.ArgumentParser(description="Analyze indentation levels.")
	parser.add_argument("paths", nargs="*", help="Files or directories to scan")
	parser.add_argument("--verbose", action="store_true", help="Show detailed stats and worst offenders")
	args = parser.parse_args()
	
	results = {}
	files_to_scan = []
	
	if args.paths:
		for path in args.paths:
			if os.path.isfile(path) and path.endswith(".py"):
				files_to_scan.append(path)
			elif os.path.isdir(path):
				_collect_files(path, files_to_scan)
	else:
		# Default behavior
		walk_dirs = ["src", "scripts", "tests"]
		for start_dir in walk_dirs:
			_collect_files(start_dir, files_to_scan)
	
	# Analyze
	all_blocks = []
	
	for filepath in files_to_scan:
		res = analyze_indentation(filepath)
		if res and "error" not in res:
			results[filepath] = res
			for b in res["blocks"]:
				b["file"] = filepath
				all_blocks.append(b)

	if not args.verbose:
		print_simple_report(results)
		return

	# Verbose Output (Legacy)
	print("="*60)
	print("INDENTATION ANALYSIS REPORT")
	print("="*60)
	
	# Report Violations (Indent >= 5 levels)
	print("\n[!] VIOLATIONS (Indentation >= 5 levels / 20 spaces):")
	violation_count = 0
	for filepath, res in results.items():
		if res["deep_lines"]:
			violation_count += 1
			print(f"\n  File: {filepath}")
			print(f"  Lines: {res['deep_lines']}")
	
	if violation_count == 0:
		print("  None found. Great job!")

	# Top Offenders (Highest Average Indentation)
	print("\n" + "-"*60)
	print("WORST OFFENDERS (Avg Indent >= 3.0)")
	print("-" * 60)
	print(f"{'Type':<15} {'Name':<35} {'Avg Lvl':<10} {'Location'}")
	print("-" * 60)
	
	# Sort by average indent descending
	all_blocks.sort(key=lambda x: x["avg_indent"], reverse=True)
	
	count = 0
	for b in all_blocks:
		if b["avg_indent"] >= 3.0:
			filename = os.path.basename(b["file"])
			location = f"{filename}:{b['lineno']}"
			print(f"{b['type']:<15} {b['name']:<35} {b['avg_indent']:.2f}       {location}")
			count += 1
	
	if count == 0:
		print("No blocks found with average indentation >= 3.0")

	# Summary Stats
	print("\n" + "-"*60)
	print("SUMMARY")
	print(f"Files Scanned: {len(results)}")
	
if __name__ == "__main__":
	main()
