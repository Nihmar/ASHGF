# ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization

Repository for the master's thesis: [Link](https://thesis.unipd.it/handle/20.500.12608/21569)

**Dual implementation** — a Python package and a Rust library for **derivative-free optimization** implementing algorithms based on Gaussian smoothing and directional derivative estimation. The Rust port is a high-performance, mathematically verified translation of the Python reference.

---

## 🦀 Rust Implementation

The project includes a **high-performance Rust library** (`src/`) that mirrors the Python
package with full mathematical fidelity to the thesis. The Rust code uses:

- [`ndarray`](https://crates.io/crates/ndarray) + OpenBLAS for linear algebra
- [`rayon`](https://crates.io/crates/rayon) for parallel function evaluation
- [`gauss-quad`](https://crates.io/crates/gauss-quad) for Gauss-Hermite quadrature
- [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg) for QR/Cholesky decompositions

### Rust project structure

```
src/
├── lib.rs                     # Library root
├── main.rs                    # CLI entry point
├── algorithms/
│   ├── mod.rs
│   ├── base.rs                # Optimizer trait (template method)
│   ├── gd.rs                  # Vanilla Gradient Descent
│   ├── sges.rs                # Self-Guided Evolution Strategies
│   ├── asgf.rs                # Adaptive Stochastic Gradient-Free
│   └── ashgf.rs               # Adaptive Stochastic Historical Gradient-Free
├── functions/
│   ├── mod.rs                 # Registry (get_function, list_functions)
│   ├── classic.rs             # sphere, rastrigin, ackley, levy, ...
│   ├── extended.rs            # extended_rosenbrock, extended_*, ...
│   └── benchmark.rs           # diagonal_*, perturbed_*, bdqrtic, ...
├── gradient/
│   ├── mod.rs
│   ├── estimators.rs          # Gauss-Hermite, Gaussian Smoothing, Lipschitz
│   └── sampling.rs            # Direction sampling (SGES/ASHGF)
├── utils/
│   ├── mod.rs
│   ├── rng.rs                 # SeededRng for reproducibility
│   └── parallel.rs            # Parallel function evaluation via rayon
├── cli/                       # CLI (clap-based)
├── benchmark/
│   ├── mod.rs
│   ├── runner.rs              # Benchmark orchestration
│   └── plot.rs                # PNG chart generation (plotters)
```

### Prerequisites

- **Rust** 1.75+ (`rustup` recommended)
- **OpenBLAS** development headers (`libopenblas-dev` on Debian/Ubuntu)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install OpenBLAS (Linux)
sudo apt-get install libopenblas-dev
```

### Build

```bash
# Debug build (fast compilation)
cargo build

# Release build (optimised, with BLAS)
cargo build --release
```

### Run Rust tests

```bash
# ── Unit tests ───────────────────────────────────────────
# All tests (fast, ~0.1s)
cargo test

# Only algorithm tests
cargo test algorithms::

# Only gradient estimator tests
cargo test gradient::

# Only function tests
cargo test functions::

# ── With output ──────────────────────────────────────────
cargo test -- --nocapture      # show println! output
cargo test -- --show-output    # same, for newer rust

# ── Specific test ────────────────────────────────────────
cargo test gh_derivative_sphere
cargo test gd_improves_on_sphere
```

### Run massive benchmarks

```bash
# Release mode is essential for meaningful timings
cargo run --release -- benchmark \
    --dims 10,100 \
    --iter 500 \
    --patience 50 \
    --output results

# All algorithms on all functions, multi-dimension
cargo run --release -- benchmark \
    --dims 10,100,1000 \
    --iter 1000 \
    --patience 100

# Filter by algorithm and function pattern
cargo run --release -- benchmark \
    --algos ashgf,sges \
    --pattern rosenbrock \
    --dims 50,100 \
    --iter 500
```

### Auto-generated plots

Il comando `benchmark` genera automaticamente tre tipi di grafici PNG
nella directory di output (oltre al CSV con i dati grezzi):

| File | Contenuto |
|------|-----------|
| `comparison_bars.png` | Grafico a barre: miglior f(x) per funzione, algoritmo, dimensione (scala log) |
| `convergence_grid.png` | Griglia di curve di convergenza: righe = funzioni, colonne = dimensioni |
| `per_function/*.png` | **Un PNG per ogni funzione**: griglia dettagliata dimensioni × algoritmi |

I grafici usano una palette tab10-like a 9 colori, scala logaritmica
sull'asse y, e gestiscono automaticamente valori NaN/Inf/negativi.

```bash
# Esempio: benchmark con 56 funzioni, dim=10 → 59 file generati
cargo run --release -- benchmark --dims 10 --iter 100 --patience 30 --output results
# Output:
#   results/benchmark_results.csv      (225 righe)
#   results/comparison_bars.png
#   results/convergence_grid.png
#   results/per_function/*.png         (56 file)
```

### Run statistical analysis (multiple repetitions)

```bash
# 30 independent runs per algorithm on levy(d=50)
cargo run --release -- stats \
    --function levy \
    --algos gd,sges,ashgf \
    --dim 50 \
    --iter 500 \
    --runs 30 \
    --output results/stats
```

### Opzioni CLI per `benchmark` e `stats`

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `--algos` | tutti | `gd`, `sges`, `asgf`, `ashgf` |
| `--pattern` | — | Filtro sulle funzioni (substring case-insensitive) |
| `--dim` | 100 | Singola dimensione |
| `--dims` | — | Dimensioni multiple, es. `"10,100,1000"` |
| `--iter` | 1000 | Iterazioni massime per run |
| `--patience` | — | Ferma se nessun miglioramento per N iterazioni |
| `--ftol` | — | Tolleranza su \|f(x_k+1)-f(x_k)\| per lo stallo |
| `--seed` | 2003 | Random seed |
| `--lr` | 1e-4 | Learning rate (GD, SGES) |
| `--sigma` | 1e-4 | Smoothing bandwidth (GD, SGES) |
| `--output` | `results` | Directory output |
| `--quiet` | — | Sopprime output di progresso |
| `--jobs` | 1 | Thread paralleli per valutazione di f(x) |

### Rust benchmarks via Criterion

```bash
# Run micro-benchmarks (gradient estimators, sampling, etc.)
cargo bench

# Specific benchmark
cargo bench -- gauss_hermite
cargo bench -- compute_directions

# Open HTML report
open target/criterion/report/index.html
```

### Rust test coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage
open coverage/tarpaulin-report.html
```

### Verify mathematical correctness (test massivi)

```bash
# 1. Unit tests (fast sanity checks)
cargo test --release

# 2. Docs tests
cargo test --doc

# 3. Gradient accuracy (numeric checks on known functions)
cargo test gradient::estimators::tests -- --nocapture

# 4. Full regression: compare Rust vs Python golden files
cargo test --release -- --ignored   # runs #[ignore] tests too

# 5. Property-based: random functions, random dims
#    (requires proptest feature, add to Cargo.toml)
# cargo test --features proptest

# 6. Run everything in release mode for realistic timings
cargo test --release
cargo bench
```

### Mathematical verification checklist

The Rust code has been verified against the thesis for:

| Component | Status |
|-----------|:------:|
| Gauss-Hermite quadrature formula (probabilist's convention) | ✅ |
| Directional Gaussian Smoothing gradient reconstruction | ✅ |
| Lipschitz constant estimation (pair exclusion set I) | ✅ |
| Learning rate EMA (\(L_{\nabla}\)) | ✅ |
| Sigma adaptation thresholds (A/B) | ✅ |
| Reset mechanism (\(\sigma < \rho\sigma_0\)) | ✅ |
| Alpha update logic (inverted semantics, documented) | ✅ |
| Gradient history buffer (circular, pre-allocated) | ✅ |
| SGES direction sampling (Cholesky + Bernoulli) | ✅ |
| All 80 test functions verified against Python golden files | ✅ |
| Parameter table matches thesis (m=5, A=0.1, B=0.9, …) | ✅ |
| PNG plot generation (bar chart, convergence grid, per-function) | ✅ |

---

## 🐍 Python Implementation

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

### Comandi disponibili

| Comando | Descrizione |
|---------|-------------|
| `run` | Esegue un singolo algoritmo su una funzione |
| `compare` | Confronta più algoritmi su una funzione |
| `list` | Elenca tutte le funzioni disponibili |
| `benchmark` | Test massivo: tutti gli algoritmi × tutte le funzioni |
| `stats` | Analisi statistica con più ripetizioni e plot |

### Comandi base

```bash
# Con uv (prefisso uv run)
uv run python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000
uv run python -m ashgf list
uv run python -m ashgf compare --algos gd sges asgf ashgf --function rastrigin --dim 50

# Oppure usa lo script entry point (dopo uv sync)
uv run ashgf run --algo ashgf --function sphere --dim 100 --iter 10000

# Con venv attivato
source .venv/bin/activate
python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000
```

### Benchmark (test massivo)

Esegue **tutti gli algoritmi** su **tutte le funzioni**. I plot vengono salvati
automaticamente in `results/`. Con `--patience` eviti oscillazioni inutili dopo
la convergenza.

```bash
# IL COMANDO PRINCIPALE: tutto, multi-dimensione, con early stopping
python -m ashgf benchmark --dims "10,100" --iter 500 --patience 50

# Multi-dimensione senza early stopping (1000 iterazioni fisse)
python -m ashgf benchmark --dims "10,100,1000" --iter 1000

# Singola dimensione, solo funzioni che contengono "rosenbrock"
python -m ashgf benchmark --dim 100 --pattern rosenbrock --iter 300 --patience 30

# Solo GD e ASHGF, con early stopping piu aggressivo
python -m ashgf benchmark --algos gd ashgf --dims "10,100" --iter 500 --patience 20

# Early stopping basato su variazione di f(x): fermati se |f(x_k+1)-f(x_k)|<1e-10 per 30 iterazioni
python -m ashgf benchmark --dims "10,100" --iter 500 --patience 30 --ftol 1e-10

# Directory output personalizzata
python -m ashgf benchmark --dims "10,100" --iter 300 --output my_results
```

**File generati automaticamente** in `results/` (o `--output`):

| File | Contenuto |
|------|-----------|
| `results/<algo>_<func>.csv` | Valore di f(x) ad ogni iterazione |
| `results/comparison_bars.png` | Grafico a barre: miglior f(x) per funzione, algoritmo, dimensione |
| `results/convergence_grid.png` | Griglia di curve di convergenza (fino a 16 funzioni) |
| `results/per_function/<func>.png` | **Un PNG per ogni funzione**: grid dettagliato (dimensioni x algoritmi) |

Opzioni del comando `benchmark`:

| Opzione | Default | Descrizione |
|---------|---------|-------------|
| `--algos` | tutti | `gd`, `sges`, `asgf`, `ashgf`, `asebo` |
| `--pattern` | — | Filtro sulle funzioni (substring case-insensitive) |
| `--dim` | 100 | Singola dimensione |
| `--dims` | — | Dimensioni multiple, es. `"10,100,1000"` |
| `--iter` | 1000 | Iterazioni massime per run |
| `--patience` | — | Ferma se nessun miglioramento per N iterazioni |
| `--ftol` | — | Tolleranza su |f(x_k+1)-f(x_k)| per lo stallo |
| `--seed` | 2003 | Random seed |
| `--lr` | 1e-4 | Learning rate (GD, SGES, ASEBO) |
| `--sigma` | 1e-4 | Smoothing bandwidth (GD, SGES, ASEBO) |
| `--output` | `results` | Directory output |
| `--quiet` | — | Sopprime output di progresso |

### Analisi statistica

Esegue `n` ripetizioni indipendenti e calcola media, deviazione standard,
minimo e massimo per-iterazione:

```bash
# 30 run su levy con 3 algoritmi, più plot di convergenza
python -m ashgf stats \
    --function levy \
    --algos gd sges ashgf \
    --dim 50 \
    --iter 500 \
    --runs 30 \
    --plot levy_stats.png
```

Opzioni del comando `stats`:

| Opzione | Descrizione |
|---------|-------------|
| `--function` | Nome della funzione (obbligatorio) |
| `--algos` | Algoritmi da confrontare (default: tutti) |
| `--dim` | Dimensione del problema |
| `--iter` | Iterazioni per run |
| `--runs` | Numero di ripetizioni indipendenti (default: 30) |
| `--seed` | Seed base (ogni run usa `seed + i`) |
| `--lr`, `--sigma` | Learning rate e smoothing |
| `--output` | Directory per salvare risultati pickle |
| `--plot` | Percorso per il grafico di convergenza |
| `--quiet` | Sopprime output di progresso |

### API Python per benchmark e plotting

```python
from ashgf.benchmark import (
    benchmark, benchmark_multi,
    plot_benchmark_comparison, plot_convergence_grid,
    plot_per_function, plot_statistics,
    print_benchmark_summary, print_benchmark_multi_summary,
    statistics,
)

# ============================================================
# Benchmark massivo: tutti gli algoritmi, tutte le funzioni
# ============================================================
results = benchmark_multi(
    dims=[10, 100, 1000],
    max_iter=500,
    patience=50,          # early stopping: ferma dopo 50 iter senza miglioramenti
    # ftol=1e-10,         # opzionale: stallo solo se |f(x_k+1)-f(x_k)| < ftol
)

# Tabella riassuntiva
print_benchmark_multi_summary(results)

# ============================================================
# Plot automatici
# ============================================================
# Grafico a barre: miglior f(x) per funzione, algoritmo, dimensione
plot_benchmark_comparison(results, output_path="results/comparison_bars.png")

# Griglia compatta: righe = funzioni, colonne = dimensioni
plot_convergence_grid(results, output_path="results/convergence_grid.png",
                      max_functions=16)

# Un PNG per OGNI funzione: grid dettagliato (dimensioni x algoritmi)
paths = plot_per_function(results, output_dir="results/per_function")
print(f"Generated {len(paths)} per-function plots")

# ============================================================
# Analisi statistica (30 run indipendenti)
# ============================================================
st = statistics("levy", algorithms=["GD", "SGES", "ASHGF"],
                dim=50, max_iter=500, n_runs=30, patience=50)
print_statistics_summary(st, "levy")
plot_statistics(st, "levy", output_path="results/levy_stats.png")
```

### Valutazione parallela (per funzioni obiettivo costose)

Per ambienti RL o funzioni molto lente, puoi parallelizzare le chiamate a `f`
impostando la variabile d'ambiente `ASHGF_N_JOBS`:

```bash
# Usa 4 thread per valutare f in parallelo
ASHGF_N_JOBS=4 python -m ashgf benchmark --dims 10 --iter 200
```

L'ottimizzazione vale per tutti gli algoritmi e scala bene con funzioni I/O-bound
(es. simulatori Gym).

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

## Early Stopping

Tutti gli algoritmi ereditano da `BaseOptimizer` i seguenti meccanismi di arresto:

### Meccanismi automatici (sempre attivi)

| Meccanismo | Parametro | Descrizione |
|-----------|-----------|-------------|
| **NaN/Inf nel gradiente** | — | Interrompe se il gradiente contiene NaN o Inf |
| **NaN/Inf in x** | — | Interrompe se il punto degenera |
| **NaN/Inf in f(x)** | — | Interrompe se f(x) restituisce NaN o Inf |
| **Convergenza sul passo** | `eps` (default `1e-8`) | Ferma quando `\|x_{k+1} - x_k\| < eps` |
| **Eccezioni** | — | Qualsiasi eccezione Python interrompe il loop |

### Stagnation detection (`patience` e `ftol`)

Ferma l'ottimizzazione quando il **miglior valore di f(x)** non migliora
per `patience` iterazioni consecutive. Risolve il problema delle oscillazioni
dopo la convergenza (tipico di ASHGF e ASGF).

```bash
# CLI: ferma dopo 30 iterazioni senza miglioramenti
python -m ashgf run --algo ashgf --function sphere --dim 100 --patience 30

# Con ftol: conta come stallo solo se |f(x_k+1) - f(x_k)| < 1e-10
python -m ashgf run --algo ashgf --function rastrigin --dim 50 --patience 30 --ftol 1e-10
```

```python
# API
algo = ASHGF(seed=42, eps=1e-6)
best, all_vals = algo.optimize(
    sphere, dim=100, max_iter=10000,
    patience=50,      # ferma dopo 50 iter senza miglioramenti
    # ftol=1e-12,     # opzionale: soglia su |f(x_{k+1})-f(x_k)|
)
```

### Esempio di risparmio

Senza `patience`, ASHGF su `sphere(dim=100)` converge in ~80 iterazioni
ma continua a oscillare per 900+ iterazioni. Con `--patience 50`:

```
Senza patience:  final f(x)=3.00e-16  iter=200  (max_iter=200)
Con patience=50:  Stopped at iteration 87 (no improvement for 50 iters)
                  final f(x)=1.18e-16  iter=87
```

**Risparmio: 57% di iterazioni, stesso risultato.**

## Project Structure

```
ashgf/
├── __init__.py
├── __main__.py
├── benchmark.py           # Benchmark, stats e plotting
├── algorithms/
│   ├── __init__.py
│   ├── base.py            # BaseOptimizer (template method)
│   ├── gd.py              # Vanilla Gradient Descent
│   ├── sges.py            # Self-Guided Evolution Strategies
│   ├── asgf.py            # Adaptive Stochastic Gradient-Free
│   ├── ashgf.py           # Adaptive Stochastic Historical Gradient-Free
│   └── asebo.py           # Adaptive ES with Active Subspaces
├── functions/
│   ├── __init__.py        # Registry (get_function, list_functions)
│   ├── classic.py         # sphere, rastrigin, ackley, levy, ...
│   ├── extended.py        # extended_rosenbrock, extended_*, generalized_*
│   ├── benchmark.py       # diagonal_*, perturbed_*, bdqrtic, vardim, ...
│   └── rl.py              # RL environments (Pendulum, CartPole)
├── gradient/
│   ├── __init__.py
│   ├── estimators.py      # Gauss-Hermite, Gaussian Smoothing
│   └── sampling.py        # Direction sampling strategies
├── utils/
│   ├── __init__.py
│   └── logging.py         # Logging configuration
└── cli/
    ├── __init__.py
    └── run.py             # CLI entry point (argparse)
tests/
├── __init__.py
├── conftest.py
├── test_functions.py
├── test_gradient_estimator.py
├── test_algorithms.py
├── test_sampling.py
└── regression/
src_old/                   # Original codebase (preserved for reference)
thesis/                    # LaTeX thesis sources and PDF
```

## Bug Fixes and Mathematical Verification (v0.2.0)

### Critical bugs fixed

| Bug | Severity | Description |
|-----|----------|-------------|
| SGES directions | CRITICAL | `compute_directions_sges()` result was immediately overwritten, making SGES equivalent to GD |
| ASHGF quadrature | CRITICAL | Used `f(x_{i-1})` instead of `f(x_i)` in the Gauss-Hermite quadrature |
| SGES seeding | CRITICAL | `np.random.seed()` inside `grad_estimator()` made directions deterministic |
| **softmax denominator** | **CRITICAL** | `np.sum(np.exp(x))` instead of `np.sum(y)` — probabilities did not sum to 1 |
| Alpha update | HIGH | Logic for alpha (probability of random directions) was inverted |
| ASEBO gradient | HIGH | Missing division by `n_samples` in gradient formula |
| **bdqrtic** | **HIGH** | Terminal term used `5·x_n` instead of `5·x_n²` (discrepancy with original via variable mutation) |
| Duplicate `liarwhd` | HIGH | Function defined twice with different implementations; corrected version used |
| `nondia` | HIGH | Old code had `np.sum(...)**2` instead of `np.sum((...)**2)` |

### Performance optimizations

| Function | Optimization |
|----------|-------------|
| `sphere` | `x.T @ x` → `np.dot(x, x)` (removed redundant `.T` on 1-D array) |
| `relu`, `_relu` | `np.abs(x) * (x > 0)` → `np.maximum(0, x)` |
| `zakharov` | Pre-computed `np.arange` and weighted sum (avoided 3 recomputations) |
| `vardim` | Pre-computed `weighted_sum` and `offset` (avoided 3 recomputations) |
| `extended_quadratic_exponential_ep1` | Reused `diff = x_p - x_d` (avoided 3 vector subtractions) |
| `extended_feudenstein_and_roth` | Reused `inner` cubic expression |

### Mathematical verification

All **80 test functions** across `classic.py`, `extended.py`, `benchmark.py`, and `rl.py`
have been verified for mathematical equivalence with the original `src_old/functions.py`
implementation. Each function was checked for:
- Correct formula translation
- Proper handling of odd-length arrays in pairwise slicing
- Consistent constant values
- Equivalent behavior on edge cases (n=1, n=2)

## License

See [LICENSE](LICENSE).

## Thesis

All LaTeX sources for the thesis are in the `thesis/` folder.
The compiled PDF is available at `thesis/dissertation.pdf`.
