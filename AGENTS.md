# AGENTS.md

## Project Overview

ASHGF is a research repository implementing gradient-free optimization algorithms (GD, SGES, ASEBO, ASGF, ASHGF) for continuous optimization and reinforcement learning. The project is structured with modular algorithms, problems, and experiment scripts.

## Build/Lint/Test Commands

### Dependencies
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
# Unified experiment runner
python scripts/run_experiment.py --mode profiles --dim 100 --functions sphere rastrigin --algorithms GD SGES --seeds 5

# Performance profiles
python scripts/run_profiles.py --dim 10 100 1000 --functions all --algorithms all --seeds 1

# RL experiments
python scripts/run_rl.py --env Pendulum --seeds 10 --algorithms all

# Statistical analysis
python scripts/run_stats.py --dim 100 --functions sphere levy rastrigin ackley --algorithms all --seeds 100 --iters 10000
```

### Testing
The project uses standard Python `unittest` or `pytest` (based on standard conventions). To run tests:

```bash
# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_algorithms.py -v

# Run single test function
python -m pytest tests/test_algorithms.py::TestGD::test_optimize -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Linting with ruff (if configured)
ruff check src/ scripts/
ruff format --check src/ scripts/

# Type checking with mypy (if configured)
mypy src/ scripts/
```

## Code Style Guidelines

### Imports
- Use absolute imports from the project root
- Group imports: standard library, third-party, local modules
- Use `from src.algorithms import BaseOptimizer` style
- Avoid wildcard imports (`from module import *`)
- One import per line

### Formatting
- Use 4 spaces for indentation
- Line length: 88 characters (Black default) or 100 characters
- Use double quotes for strings, single quotes only for embedding
- No trailing commas in single-line collections
- Use trailing commas in multiline collections and function signatures

### Types
- Use type hints for all function parameters and return values
- Import `typing` module for complex types (List, Dict, Optional, etc.)
- Use `numpy.ndarray` type hints for arrays
- Avoid `Any` type unless absolutely necessary
- Document complex types in docstrings

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `BaseOptimizer`, `ASHGF`)
- **Functions/Methods**: `snake_case` (e.g., `optimize`, `compute_gradient`)
- **Variables**: `snake_case` (e.g., `population_size`, `learning_rate`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Private members**: Prefix with underscore (e.g., `_gradient`, `_population`)
- **Module names**: `snake_case` (e.g., `base.py`, `ashgf.py`)

### Error Handling
- Use specific exception types (e.g., `ValueError`, `TypeError`)
- Raise descriptive error messages with context
- Use try-except blocks for external library calls
- Log errors appropriately using Python's `logging` module
- Don't use bare `except:` or `except Exception:` unless necessary

### Documentation
- Docstrings using Google style or NumPy style
- Public functions and classes must have docstrings
- Parameter descriptions with types
- Return value descriptions
- Example usage in docstrings for complex functions

### Project Structure Conventions
- `src/algorithms/`: Base class and optimizer implementations
- `src/problems/`: Problem definitions and RL environment wrappers
- `src/utils/`: Helper functions and utilities
- `src/experiments/`: Experiment scripts
- `scripts/`: Command-line entry points
- `config/`: Configuration files
- `results/`: Generated experiment outputs (gitignored)

### Algorithm Implementation Guidelines
1. All optimizers inherit from `BaseOptimizer`
2. Implement `optimize()` method with clear signature
3. Use numpy for numerical operations
4. Maintain reproducibility via seed handling
5. Store results in structured format under `results/`

### RL Environment Guidelines
- Use `gym` for RL environments
- Handle seeds separately for algorithm and environment
- Support standard environments (Pendulum, etc.)

### Configuration Guidelines
- Use argparse for command-line arguments
- Store hyperparameters in algorithm classes
- Support configuration files in `config/` directory

### Git Workflow
- Commit messages: imperative mood ("Add feature", "Fix bug")
- Branch naming: `feature/name`, `bugfix/name`, `experiment/name`
- Never commit generated files (results/, __pycache__, .pyc)
- Use `.gitignore` for temporary files

### Performance Considerations
- Vectorize operations with numpy
- Avoid unnecessary copies of large arrays
- Profile code with `cProfile` for bottlenecks
- Cache expensive computations where appropriate

### Testing Guidelines
- Test algorithms on standard benchmarks (sphere, rastrigin, etc.)
- Verify reproducibility with fixed seeds
- Test edge cases (zero gradients, boundary conditions)
- Use parameterized tests for multiple algorithm configurations

### Memory Management
- Clean up large arrays after use
- Use context managers for file operations
- Avoid global variables for state

### Notes
- Results are saved to `results/{mode}/{dim}/{function}/{algorithm}/seed_{seed}/descent.csv`
- Seeds are handled separately for algorithm and environment
- Benchmark functions are defined in `src/problems/function.py`
- The project uses numpy, pandas, matplotlib, scipy, scikit-learn, and gym
