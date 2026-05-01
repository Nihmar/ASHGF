# ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization

Repository for the master's thesis: [Link](https://thesis.unipd.it/handle/20.500.12608/21569)

A Python package for **derivative-free optimization** implementing algorithms based on
Gaussian smoothing and directional derivative estimation.

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **GD** | Vanilla Gradient Descent with Central Gaussian Smoothing |
| **SGES** | Self-Guided Evolution Strategies — adaptively mixes gradient-history directions |
| **ASGF** | Adaptive Stochastic Gradient-Free — Gauss-Hermite quadrature, adaptive sigma |
| **ASHGF** | Adaptive Stochastic Historical Gradient-Free — ASGF + gradient history |
| **ASEBO** | Adaptive ES with Active Subspaces — PCA-based covariance adaptation |

## Installation

Requires **Python 3.10+**.

### Recommended: `uv` (virtual environment isolato)

[`uv`](https://docs.astral.sh/uv/) è un package manager veloce scritto in Rust che gestisce
ambienti virtuali e dipendenze senza toccare i pacchetti di sistema.

```bash
# Installa uv (se non presente)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clona il repository
git clone <repo-url> && cd ASHGF

# Crea l'ambiente virtuale e installa le dipendenze core
uv sync

# Con dipendenze di sviluppo (test)
uv sync --group dev

# Con supporto RL (gymnasium)
uv sync --group rl

# Tutto insieme
uv sync --group dev --group rl
```

Dopo `uv sync`, tutti i comandi vanno prefixati con `uv run`:

```bash
uv run pytest tests/ -v
uv run python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000
```

L'ambiente virtuale si trova in `.venv/`. Per attivare la shell manualmente:

```bash
source .venv/bin/activate
```

### Alternativa: `pip` tradizionale

```bash
# Basic installation
pip install -e .

# With development dependencies (tests)
pip install -e ".[dev]"

# With RL support (gymnasium)
pip install -e ".[rl]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```python
from ashgf.algorithms import ASHGF
from ashgf.functions import get_function

f = get_function("sphere")
algo = ASHGF(seed=42)
# optimize restituisce (best_values, all_values)
# best_values: list[(x, f(x))] — migliori punti trovati
# all_values:  list[float]        — valore di f a ogni iterazione
best_values, all_values = algo.optimize(f, dim=100, max_iter=1000)

print(f"Final value: {all_values[-1]:.6e}")
```

Eseguire lo script con `uv run`:

```bash
uv run python quickstart.py
```

## CLI Usage

```bash
# Con uv (prefisso uv run)
uv run python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000
uv run python -m ashgf list
uv run python -m ashgf compare --algos gd,sges,asgf,ashgf --function rastrigin --dim 50

# Oppure usa lo script entry point (dopo uv sync)
uv run ashgf run --algo ashgf --function sphere --dim 100 --iter 10000

# Con venv attivato
source .venv/bin/activate
python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000
```

## Running Tests

```bash
# Con uv
uv run pytest tests/ -v
uv run pytest tests/ -v -m "not slow"
uv run pytest tests/ --cov=ashgf --cov-report=html

# Con venv attivato
source .venv/bin/activate
pytest tests/ -v
pytest tests/ -v -m "not slow"
pytest tests/ --cov=ashgf --cov-report=html
```

## Project Structure

```
ashgf/
├── __init__.py
├── algorithms/
│   ├── __init__.py
│   ├── base.py          # BaseOptimizer (template method)
│   ├── gd.py            # Vanilla Gradient Descent
│   ├── sges.py          # Self-Guided Evolution Strategies
│   ├── asgf.py          # Adaptive Stochastic Gradient-Free
│   ├── ashgf.py         # Adaptive Stochastic Historical Gradient-Free
│   └── asebo.py         # Adaptive ES with Active Subspaces
├── functions/
│   ├── __init__.py      # Registry
│   ├── classic.py       # sphere, rastrigin, ackley, ...
│   ├── extended.py      # extended_rosenbrock, extended_*, ...
│   ├── benchmark.py     # diagonal_*, perturbed_*, ...
│   └── rl.py            # RL environments (Pendulum, CartPole)
├── gradient/
│   ├── __init__.py
│   ├── estimators.py    # Gauss-Hermite, Gaussian Smoothing
│   └── sampling.py      # Direction sampling strategies
├── utils/
│   ├── __init__.py
│   └── logging.py       # Logging configuration
└── cli/
    ├── __init__.py
    └── run.py           # CLI entry point
tests/
├── test_functions.py
├── test_gradient_estimator.py
├── test_algorithms.py
├── test_sampling.py
└── regression/
src_old/                 # Original codebase (preserved for reference)
thesis/                  # LaTeX thesis sources and PDF
```

## Bug Fixes (v0.2.0)

This version fixes several critical bugs identified in the original code:

| Bug | Severity | Description |
|-----|----------|-------------|
| SGES directions | CRITICAL | `compute_directions_sges()` result was immediately overwritten, making SGES equivalent to GD |
| ASHGF quadrature | CRITICAL | Used `f(x_{i-1})` instead of `f(x_i)` in the Gauss-Hermite quadrature |
| SGES seeding | CRITICAL | `np.random.seed()` inside `grad_estimator()` made directions deterministic |
| Alpha update | HIGH | Logic for alpha (probability of random directions) was inverted |
| ASEBO gradient | HIGH | Missing division by `n_samples` in gradient formula |
| Duplicate `liarwhd` | HIGH | Function defined twice with different implementations |
| Global state | MEDIUM | Class-level mutable `data` dict replaced with instance attributes |
| Wildcard imports | MEDIUM | `from functions import *` replaced with explicit imports |
| `print()` usage | MEDIUM | Replaced with `logging` module |

## License

See [LICENSE](LICENSE).

## Thesis

All LaTeX sources for the thesis are in the `thesis/` folder.
The compiled PDF is available at `thesis/dissertation.pdf`.
