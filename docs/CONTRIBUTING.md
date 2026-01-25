# Contributing to Multi-Dimensional AI

We welcome contributions to the Multi-Dimensional AI project! This document outlines the process for contributing code, documentation, and other resources.

Repository: [https://github.com/calebhk98/Multi-Dimensional-AI](https://github.com/calebhk98/Multi-Dimensional-AI)

## Code of Conduct

Please adhere to the project's Code of Conduct in all interactions.

## Getting Started

1.  **Fork the Repository**: Create a fork of the project to your own GitHub account.
2.  **Clone the Fork**: `git clone https://github.com/calebhk98/Multi-Dimensional-AI.git`
3.  **Create a Branch**: `git checkout -b feature/your-feature-name`

## Coding Standards

### Indentation

- **Use TABS**, not spaces.
- Do not use spaces for indentation.

### Logic and Complexity

- **Never Nesting**: Avoid deep nesting of code blocks.
    - Use guard clauses (early returns) to handle edge cases.
    - Extract logic into helper functions.
    - Ideal nesting depth should not exceed 5 tabs.
- **File Size**:
    - Files > 250 lines: Warning (consider refactoring).
    - Files > 500 lines: Critical (refactor recommended).
    - Files > 1,000 lines: Forbidden (no reason for this).
- **TDD (Test Driven Development)**:
    1.  Create a minimal failing test.
    2.  Write minimal code to pass the test.
    3.  Repeat.
    4.  Add ~5 additional tests for edge cases per feature.
    5.  No empty catch blocks. Always log errors.

### Documentation & Comments

Strict documentation standards are enforced.

**File Headers**
Every file must start with a header comment explaining its purpose.

**Classes**

- Must have a docstring/comment explaining what the class does.
- Must list important variables/fields it stores.

**Functions**
Every function must have a comment block detailing:

1.  **Purpose**: What the function limits/does.
2.  **Args**: Parameters and their types/descriptions.
3.  **Returns**: Return value and type.
4.  **Errors**: Any exceptions raised.
5.  **Workflow**: Brief description of the logic flow.

**Example:**

```python
"""
file: utils/math_ops.py
Provides basic mathematical operations for the neural modules.
"""

class TensorCalculator:
	"""
	Handles tensor operations for the custom layers.

	Fields:
		- device: The cuda device to run on.
		- precision: Float32 or Float16.
	"""

	def add_tensors(self, t1, t2):
		"""
		Purpose: Adds two tensors together safely.

		Args:
			t1 (Tensor): First input tensor.
			t2 (Tensor): Second input tensor.

		Returns:
			Tensor: The sum of t1 and t2.

		Errors:
			ValueError: If tensor shapes do not match.

		Workflow:
			1. Check shapes of t1 and t2.
			2. If mismatch, raise error.
			3. Return t1 + t2.
		"""
		if t1.shape != t2.shape:
			raise ValueError("Shapes mismatch")
		return t1 + t2
```

## Development Workflow

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Tests**:
    Ensure all tests pass before submitting your changes.
    ```bash
    pytest tests/
    ```
3.  **Code Style**:
    - Follow PEP 8 guidelines.
    - Add comments for complex logic.
    - Ensure type hints are used where appropriate.

## Pull Requests

1.  Push your branch to your fork.
2.  Open a Pull Request (PR) against the `main` branch.
3.  Provide a clear description of your changes and link to any relevant issues.
4.  Wait for review and address any feedback.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub repository with detailed reproduction steps or clear requirements.
