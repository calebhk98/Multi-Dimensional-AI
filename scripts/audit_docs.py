
"""
Script to audit codebase for missing documentation.
"""
import ast
import os
import argparse
import sys

def check_file(filepath):
    """
    ==============================================================================
    Function: check_file
    ==============================================================================
    Purpose:  Parses a single Python file using the AST module to identify missing
              file headers or function docstrings. Also performs basic checks for
              missing proper 'Args' or 'Returns' sections in docstrings.

    Parameters:
        - filepath: str
            Path to the python file to check.

    Returns:
        Dict[str, Any] - Dictionary containing:
            - "file_header": bool (True if missing)
            - "functions": List[Dict] (Details of functions with issues)
            - "error": str (Optional error message, e.g., SyntaxError)

    Dependencies:
        - ast.parse
        - ast.get_docstring
        - ast.walk

    Processing Workflow:
        1.  Open and read file content using utf-8 encoding.
        2.  Parse content into an AST tree; handle SyntaxError.
        3.  Check for module-level docstring (file header).
        4.  Walk through all nodes in the tree:
            a. Identify `FunctionDef` and `AsyncFunctionDef` nodes.
            b. Check if docstring exists.
            c. If docstring exists, check for "Args:" if args exist (excluding self).
            d. Check for "Returns:" if return/yield statements exist.
        5.  Collect all issues found.
        6.  Return report dictionary.

    ToDo:
        - Enhance regex checks for more strict formatting.

    Usage:
        result = check_file("src/my_script.py")
    ==============================================================================
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return {"file_header": False, "functions": [], "error": "SyntaxError"}

    missing_header = ast.get_docstring(tree) is None
    missing_functions = []
    
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        docstring = ast.get_docstring(node)
        if docstring is None:
            missing_functions.append({
                "name": node.name,
                "lineno": node.lineno,
                "reason": "Missing docstring"
            })
            continue

        # Content checks
        issues = _validate_docstring_content(node, docstring)
        if issues:
            missing_functions.append({
                "name": node.name,
                "lineno": node.lineno,
                "reason": ", ".join(issues)
            })

    return {
        "file_header": missing_header,
        "functions": missing_functions
    }

def _validate_docstring_content(node, docstring):
    """
    ==============================================================================
    Function: _validate_docstring_content
    ==============================================================================
    Purpose:  Validates docstring content based on function type.
              - Test functions: Must have 'Purpose', 'Workflow', 'ToDo'.
              - Regular functions: Must have 'Args' (if params exist) and 'Returns' (if returns).

    Parameters:
        - node: ast.FunctionDef
            The function node being checked.
        - docstring: str
            The docstring of the function.

    Returns:
        List[str] - List of issue descriptions found suitable for reporting.
    ==============================================================================
    """
    issues = []
    
    # Check if it's a test function
    is_test = node.name.startswith("test_")

    if is_test:
        # Enforce Test Documentation Standard
        if "Purpose:" not in docstring and "Purpose" not in docstring:
             issues.append("Missing 'Purpose' section")
        if "Workflow:" not in docstring and "Workflow" not in docstring:
             issues.append("Missing 'Workflow' section")
        if "ToDo:" not in docstring and "ToDo" not in docstring:
             issues.append("Missing 'ToDo' section")
    else:
        # Enforce Standard Documentation Standard for regular AND internal functions
        
        # Allow one-liners for simple no-arg/single-arg funcs? User said "anything we make should have the comments".
        # Sticking to the previous logic for Args/Returns, but removing the skip for `_`
        
        has_args = "Args:" in docstring or "Parameters:" in docstring
        has_returns = "Returns:" in docstring or "Yields:" in docstring
        
        # Check Args (exclude self/cls)
        args_count = len([a for a in node.args.args if a.arg not in ('self', 'cls')])
        if args_count > 0 and not has_args:
            issues.append("Missing 'Args:' section")
        
        # Check Returns
        if _has_return_statement(node) and not has_returns:
            issues.append("Missing 'Returns:' section")

    return issues

def _has_return_statement(node):
    """
    ==============================================================================
    Function: _has_return_statement
    ==============================================================================
    Purpose:  Checks if a function node contains a non-empty return or yield
              statement. Traverses the function body recursively.

    Parameters:
        - node: ast.FunctionDef
            The function node to inspect.

    Returns:
        bool - True if a return/yield statement is found, False otherwise.
    ==============================================================================
    """
    for child in ast.walk(node):
        if not isinstance(child, (ast.Return, ast.Yield)):
            continue
            
        if isinstance(child, ast.Return) and child.value is None:
             continue # ignore empty returns
        return True
        
    return False

def main():
    """
    ==============================================================================
    Function: main
    ==============================================================================
    Purpose:  Main entry point for the documentation audit script. Scans specified
              directories ("src", "scripts", "tests") and root files for Python
              files, checks them for documentation issues, and prints a JSON report.

    Parameters:
        - None

    Returns:
        None

    Dependencies:
        - os.walk
        - os.path
        - check_file (local function)
        - json.dumps

    Processing Workflow:
        1.  Define list of directories to scan (`walk_dirs`).
        2.  Iterate through directories and walk through them recursively.
        3.  For each `.py` file found, call `check_file`.
        4.  If issues are found, add to `report` dictionary.
        5.  Iterate through files in proper valid root directory.
        6.  Check root `.py` files.
        7.  Print final `report` as formatted JSON.

    ToDo:
        - Add command line arguments for target directories.

    Usage:
        Run directly: `python scripts/audit_docs.py`
    ==============================================================================
    """
    print(json.dumps(report, indent=2))

def print_simple_report(report):
    """
    Print report in a simple line-by-line format easier for agents/grep to parse.
    Format: relative_path:lineno:function_name: reason

    Args:
        report: Dictionary containing audit results.
    """
    for filepath, data in report.items():
        if "functions" in data:
            for func in data["functions"]:
                print(f"{filepath}:{func['lineno']}:{func['name']}: {func['reason']}")
        if data.get("file_header"):
            print(f"{filepath}:1:module_header: Missing file header docstring")

def main():
    """
    Main entry point. Supports CLI args for targeting specific paths.
    """
    parser = argparse.ArgumentParser(description="Audit codebase for documentation.")
    parser.add_argument("paths", nargs="*", help="Specific files or directories to scan. Defaults to standard dirs.")
    parser.add_argument("--json", action="store_true", help="Output in JSON format (default is line-by-line).")
    args = parser.parse_args()

    report = {}
    
    # Determine paths to scan
    if args.paths:
        search_paths = args.paths
    else:
        search_paths = ["src", "scripts", "tests"]

    for path in search_paths:
        if os.path.isfile(path) and path.endswith(".py"):
            result = check_file(path)
            if result["file_header"] or result["functions"]:
                 report[path] = result
        elif os.path.isdir(path):
             _scan_directory(path, report)
    
    # If no paths args provided, also check root py files (legacy behavior)
    if not args.paths:
        for file in os.listdir("."):
            if os.path.isfile(file) and file.endswith(".py"):
                result = check_file(file)
                if result["file_header"] or result["functions"]:
                    report[file] = result

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_simple_report(report)

def _scan_directory(start_dir, report):
    """
    Helper to scan a directory recursively for python files.
    
    Args:
        start_dir: Directory to start scanning from.
        report: Dictionary to collect results.
        
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
            result = check_file(path)
            if result["file_header"] or result["functions"]:
                report[path] = result

if __name__ == "__main__":
    main()
