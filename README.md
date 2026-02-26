# ASHGF
Repository for this master degree: https://thesis.unipd.it/handle/20.500.12608/21569

---

This repository contains the implementation and comparison of several gradient-free optimization algorithms for continuous optimization and reinforcement learning problems. The algorithms are:

- **GD**: Vanilla Gradient Descent with finite-difference gradient estimation.
- **SGES**: Self-Guided Evolution Strategies.
- **ASEBO**: Adaptive ES with Active Subspaces.
- **ASGF**: Adaptive Stochastic Gradient-Free method.
- **ASHGF**: Adaptive Stochastic Historical Gradient-Free method.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config/                  # Configuration files for experiments
├── results/                 # All experiment outputs (created automatically)
├── src/                     # Source code
│   ├── algorithms/          # Optimizer implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base class for all optimizers
│   │   ├── gd.py
│   │   ├── sges.py
│   │   ├── asebo.py
│   │   ├── asgf.py
│   │   └── ashgf.py
│   ├── problems/            # Problem definitions
│   │   ├── __init__.py
│   │   ├── function.py      # Function class for synthetic benchmarks
│   │   └── rl_envs.py       # RL environment wrappers
│   ├── utils/               # Helper functions
│   │   ├── __init__.py
│   │   └── io_utils.py      # Result saving, function list loading
│   └── experiments/         # Experiment scripts (used by main runners)
│       ├── __init__.py
│       ├── profiles.py      # Data for performance profiles
│       ├── rl.py            # RL experiments
│       └── stats.py         # Statistical summaries over seeds
└── scripts/                 # Command-line entry points
    ├── run_experiment.py    # Unified experiment runner
    ├── run_profiles.py      # Run performance profile experiments
    ├── run_rl.py            # Run RL experiments
    └── run_stats.py         # Run statistical analysis
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

All experiments can be launched via the unified script `scripts/run_experiment.py` with appropriate arguments. Alternatively, use the dedicated scripts for each experiment type.

### Unified runner

```bash
python scripts/run_experiment.py --mode profiles --dim 100 --functions sphere rastrigin --algorithms GD SGES --seeds 5
```

Available modes: `profiles`, `rl`, `stats`.

### Performance profiles

```bash
python scripts/run_profiles.py --dim 10 100 1000 --functions all --algorithms all --seeds 1
```

### RL experiments

```bash
python scripts/run_rl.py --env Pendulum --seeds 10 --algorithms all
```

### Statistical analysis (mean/std over seeds)

```bash
python scripts/run_stats.py --dim 100 --functions sphere levy rastrigin ackley --algorithms all --seeds 100 --iters 10000
```

All results are saved under `results/` in a structured format:  
`results/{mode}/{dim}/{function}/{algorithm}/seed_{seed}/descent.csv` and optional pickled best values.

## Configuration

Default hyperparameters are set inside each algorithm class. To change them, modify the class initializer or pass arguments via the experiment scripts (future improvement).

## Adding a new algorithm

1. Create a new file in `src/algorithms/` with a class inheriting from `BaseOptimizer`.
2. Implement `optimize()` and any helper methods.
3. Add the algorithm to the `ALGORITHMS` dictionary in `src/algorithms/__init__.py`.

## Notes

- The RL environments use `gym`. Make sure to have the required backends (e.g., `box2d` for Pendulum) installed.
- Seeds are handled separately for algorithm randomness (`seed`) and environment (`seed_env`) to ensure reproducibility.
- The list of benchmark functions is defined in `src/problems/function.py`. The file `functions.txt` is no longer used; all functions are available via the `Function` class.
