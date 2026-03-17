# AGENTS.md

## Project Overview

ASHGF is a research repository implementing gradient-free optimization algorithms (GD, SGES, ASEBO, ASGF, ASHGF) for continuous optimization and reinforcement learning. The project uses a flat structure with all source code in the `src/` directory.

## Build/Lint/Test Commands

### Dependencies
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run performance profiles
cd src
python profiles.py --n-runs 10 --workers 4

# Generate plots from results
cd src
python stat_plots.py --dim 100 --plot-comparison --summary

# Run RL experiments
cd src
python RL_problems.py

# Quick testing
cd src
python testing_stuffs.py
```

### Testing
The project uses simple test scripts in `src/testing_stuffs.py`. To run:
```bash
cd src
python testing_stuffs.py
```

To modify or add a test, edit the `src/testing_stuffs.py` script. For example, to test a new function:
```python
from functions import Function
f = Function('sphere', seed_env=0)
result = f([1, 2, 3])
```

### Code Quality
```bash
# Linting with ruff (if configured)
ruff check src/
ruff format --check src/

# Type checking with mypy (if configured)
mypy src/
```

## Code Style Guidelines

### Imports
- **Current practice**: Wildcard imports used (e.g., `from functions import *`)
- **Recommended**: Avoid wildcard imports; import specific items
- Group imports: standard library, third-party, local modules
- One import per line

### Formatting
- Use 4 spaces for indentation
- Line length: 88 characters (Black default)
- Use double quotes for strings, single quotes only for embedding
- No trailing commas in single-line collections
- Use trailing commas in multiline collections and function signatures

### Types
- **Current practice**: No type hints used in existing code
- **Recommended**: Add type hints for new code
- Use `numpy.ndarray` type hints for arrays
- Avoid `Any` type unless absolutely necessary

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `GD`, `ASHGF`, `Function`)
- **Functions/Methods**: `snake_case` (e.g., `optimize`, `sphere`)
- **Variables**: `snake_case` (e.g., `learning_rate`, `dim`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
- **Private members**: Prefix with underscore (e.g., `_gradient`)
- **Module names**: `snake_case` (e.g., `gd.py`, `ashgf.py`)

### Error Handling
- Use specific exception types (e.g., `ValueError`, `TypeError`)
- Raise descriptive error messages with context
- Example from code:
  ```python
  if lr < 0:
      raise ValueError("Error: learning rate < 0")
  ```

### Documentation
- **Current practice**: Basic docstrings in NumPy style
- **Recommended**: Consistent docstrings with parameters and returns
- Public functions and classes must have docstrings
- Parameter descriptions with types
- Return value descriptions

### Project Structure
- `src/`: All source code
  - `optimizers/`: Algorithm implementations
    - `gd.py`, `sges.py`, `asebo.py`, `asgf.py`, `ashgf.py`: Algorithm classes
    - `base.py`: Base optimizer class
  - `functions/`: Benchmark function definitions
    - `benchmarks.py`: 78 benchmark functions
  - `profiles.py`: Performance profile runner
  - `stat_plots.py`: Plot generation from results
  - `RL_problems.py`: RL experiment runner
  - `testing_stuffs.py`: Test/demo script
- `results/`: Generated experiment outputs (gitignored)

### Algorithm Implementation Guidelines
1. Each algorithm is a class with an `optimize()` method
2. Common signature: `optimize(function, dim, it, x_init, debug, itprint)`
3. Use numpy for numerical operations
4. Maintain reproducibility via seed handling
5. Return `(best_values, all_values)` tuples

### RL Environment Guidelines
- Use `gym` for RL environments
- Handle seeds separately for algorithm (`seed`) and environment (`seed_env`)
- Support standard environments (Pendulum, CartPole)

### Configuration Guidelines
- Hyperparameters are set in class `__init__` methods
- Default values defined in class data attributes
- No external configuration files used

### Git Workflow
- Commit messages: imperative mood ("Add feature", "Fix bug")
- Branch naming: `feature/name`, `bugfix/name`, `experiment/name`
- Never commit generated files (results/, __pycache__, .pyc)
- Use `.gitignore` for temporary files

### Performance Considerations
- Vectorize operations with numpy
- Avoid unnecessary copies of large arrays
- Use `np.random.seed()` for reproducibility

### Testing Guidelines
- Test algorithms on standard benchmarks (sphere, rastrigin, etc.)
- Verify reproducibility with fixed seeds
- Test edge cases (zero gradients, boundary conditions)

### Memory Management
- Clean up large arrays after use
- Use context managers for file operations
- Avoid global variables for state

### Notes
- Results are saved to `results/profiles/dim=<dim>/results.parquet`
- Plots are saved to `results/plots/dim=<dim>/<function>/`
- Seeds are handled separately for algorithm and environment
- Benchmark functions are defined in `src/functions/benchmarks.py`
- The project uses numpy, pandas, matplotlib, scipy, scikit-learn, and gym
