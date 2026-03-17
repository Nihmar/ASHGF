# ASHGF
Repository for this master degree: https://thesis.unipd.it/handle/20.500.12608/21569

---

This repository contains the implementation and comparison of several gradient-free optimization algorithms for continuous optimization and reinforcement learning problems.

## Algorithms

| Algorithm | Description |
|----------|-------------|
| **GD** | Evolution Strategy (ES) with Central Gaussian Smoothing — Algorithm 2 in thesis |
| **SGES** | Self-Guided Evolution Strategies — Algorithm 5 in thesis |
| **ASEBO** | Adaptive ES with Active Subspaces — Algorithm 3 in thesis |
| **ASGF** | Adaptive Stochastic Gradient-Free — Algorithms 6+7 in thesis |
| **ASHGF** | Adaptive Stochastic Historical Gradient-Free — Algorithms 8+9 in thesis |

## Bugfixes Applied

The implementation was corrected against the thesis specifications:

| File | Bug | Fix |
|------|-----|-----|
| `sges.py` | SGES directions overwritten by unconditional call — algorithm degenerated to plain ES | Moved to `else` branch |
| `sges.py` | `np.random.seed()` reset every iteration — identical random directions | Removed redundant seed reset |
| `asebo.py` | Gradient divided by `2σ` but NOT by `n_samples` — magnitude scales with samples | Changed to `2σ·n_samples` |
| `asgf.py` | Lipschitz estimation used only adjacent pairs, not full set I (eq 3.1) | Uses all pairs in set I |
| `ashgf.py` | Python bug: `[i,j] or [j,i] not in buffer` always True (list is truthy) | Fixed logic |
| `ashgf.py` | L∇ fallback used all directions instead of only gradient-subspace | Fixed to use `[:M]` |

## Performance Improvements

- Vectorized gradient assembly (`diffs @ directions` instead of loops)
- Batched Cholesky sampling (`z @ L.T` instead of per-direction)
- Precomputed Gauss-Hermite nodes/weights and pair index set
- Eliminated unnecessary dict storage of all steps

## Installation

```bash
pip install -r requirements.txt
```

## Running Benchmarks

Run performance profiles (from `src/` directory):

```bash
cd src
python profiles.py --n-runs 10 --workers 4
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--n-runs` | Number of runs per experiment | 10 |
| `--workers` | Number of parallel workers | 4 |
| `--dims` | Specific dimensions to run (space-separated) | 10, 100, 1000 |
| `--seed` | Random seed | 0 |
| `--overwrite` | Overwrite existing results | false |
| `--algorithms` | Specific algorithms to run (space-separated) | all |
| `--analyze` | Analyze and print summary | false |
| `--quiet` | Reduce output verbosity | false |

### Usage Examples

```bash
# Run all experiments with default settings
python profiles.py --n-runs 10 --workers 4

# Run only dimension 10
python profiles.py --dims 10 --n-runs 10 --workers 4

# Run specific dimensions
python profiles.py --dims 10 100 --n-runs 10 --workers 4

# Run only one algorithm (recalculate just ASHGF)
python profiles.py --dims 10 --algorithms ASHGF --n-runs 10

# Run only one algorithm, overwriting existing results
python profiles.py --dims 10 --algorithms ASHGF --n-runs 10 --overwrite

# Run specific algorithms
python profiles.py --dims 10 --algorithms GD ASGF --n-runs 10

# Analyze existing results (no running, just print summary)
python profiles.py --analyze

# Full example: dim=10, 10 runs, overwrite ASHGF results only
python profiles.py --dims 10 --algorithms ASHGF --n-runs 10 --overwrite --workers 4
```

Results are saved to `results/profiles/dim=<dim>/results.parquet`.

## Generating Plots

Generate convergence plots from results:

```bash
cd src
python stat_plots.py --dim 100
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dim` | Dimension to plot | 100 |
| `--functions` | Specific functions to plot (space-separated) | all from data |
| `--algorithms` | Algorithms to include (space-separated) | GD SGES ASGF ASHGF ASEBO |
| `--plot-comparison` | Generate comparison plots | false |
| `--summary` | Generate summary CSV table | false |
| `--show-plots` | Display plots interactively | false |

### Usage Examples

```bash
# Generate all plots for dimension 10
python stat_plots.py --dim 10

# Generate comparison plots (all algorithms on same chart)
python stat_plots.py --dim 10 --plot-comparison

# Generate comparison plots + summary CSV
python stat_plots.py --dim 10 --plot-comparison --summary

# Specific functions only
python stat_plots.py --dim 10 --functions sphere rastrigin levy --plot-comparison

# Specific algorithms only
python stat_plots.py --dim 10 --algorithms GD ASGF ASHGF --plot-comparison

# Full example: dim=10, specific functions, comparison + summary
python stat_plots.py --dim 10 --functions sphere rastrigin ackley levy --plot-comparison --summary
```

Plots are saved to `results/plots/dim=<dim>/<function>/`.

## Running Tests

Quick test script:

```bash
cd src
python testing_stuffs.py
```

## Running RL Experiments

```bash
cd src
python RL_problems.py
```

## Project Structure

```
src/
├── optimizers/
│   ├── base.py       # Base optimizer class
│   ├── gd.py         # ES / GD implementation
│   ├── sges.py       # SGES implementation
│   ├── asebo.py      # ASEBO implementation
│   ├── asgf.py       # ASGF implementation
│   ├── ashgf.py      # ASHGF implementation
│   └── analysis.md  # Detailed bug analysis
├── functions/
│   ├── benchmarks.py # Benchmark function definitions
│   └── __init__.py   # Function wrapper class
├── profiles.py       # Benchmark runner
├── stat_plots.py     # Plot generation
├── RL_problems.py    # RL experiments
└── testing_stuffs.py # Quick tests
```

## Benchmark Functions

The benchmark suite includes 78 functions defined in `functions.txt`. Marked functions with `**` are the core set used for comparison.
