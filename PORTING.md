# Porting Python to Rust - Plan

## Overview
This document outlines the steps to port the ASHGF optimization library from Python to Rust.

## Project Structure

### Current Python Structure (`src/`)
```
src/
├── optimizers/
│   ├── base.py          # BaseOptimizer abstract class
│   ├── gd.py            # Gradient Descent (Central Gaussian Smoothing)
│   ├── sges.py          # Stochastic Gradient Evolution Strategy
│   ├── asgf.py          # Adaptive Stochastic Gradient-Free
│   ├── asebo.py         # Adaptive Surrogate-Based Optimization
│   └── ashgf.py         # Adaptive Stochastic Historical Gradient-Free
├── functions/
│   ├── benchmarks.py    # 78 benchmark functions (Numba-optimized)
│   └── __init__.py      # Function wrapper class
├── profiles.py          # Performance profiling
├── stat_plots.py        # Plot generation
├── RL_problems.py       # Reinforcement learning experiments
└── testing_stuffs.py    # Test/demo script
```

### Target Rust Structure (`src_rust/`)
```
src_rust/
├── Cargo.toml           # Dependencies and project metadata
├── src/
│   ├── main.rs          # CLI entry point
│   ├── lib.rs           # Library root
│   ├── benchmarks/
│   │   └── mod.rs       # Criterion benchmarks
│   ├── functions/
│   │   ├── mod.rs       # Function enum and trait
│   │   └── benchmarks.rs # 78 benchmark functions
│   └── optimizers/
│       ├── mod.rs       # Optimizer trait and registry
│       ├── base.rs      # Base optimizer implementation
│       ├── gd.rs        # Gradient Descent
│       ├── sges.rs      # SGES
│       ├── asgf.rs      # ASGF
│       ├── asebo.py     # ASEBO
│       └── ashgf.rs     # ASHGF
└── tests/
    └── integration_tests.rs # Validation tests
```

## Step-by-Step Implementation Plan

### Step 1: Project Setup
1. Create Rust project: `cargo new --lib src_rust`
2. Add dependencies to `Cargo.toml`:
   - `ndarray` - N-dimensional arrays (numpy equivalent)
   - `rand` - Random number generation
   - `rand_distr` - Statistical distributions
   - ` rayon` - Parallel iteration (for batch evaluations)
   - `criterion` - Performance benchmarking
   - `approx` - Approximate comparisons for testing
   - `serde` / `serde_json` - Serialization
   - `plotters` - Plotting (optional, for results)

### Step 2: Implement Benchmark Functions
Convert all 78 functions from `functions/benchmarks.py`:

**Pure arithmetic functions (convert directly to Rust):**
- sphere, power, extended_rosenbrock, generalized_rosenbrock
- extended_white_and_holst, generalized_white_and_holst
- extended_feudenstein_and_roth, extended_baele
- extended_penalty, perturbed_quadratic, almost_perturbed_quadratic
- perturbed_quadratic_diagonal, generalized_tridiagonal_1
- extended_tridiagonal_1, diagonal_4, extended_himmelblau
- extended_psc1/sincos, extended_bd1, extended_maratos
- extended_cliff, extended_hiebert, quadratic_qf1, quadratic_qf2
- extended_quadratic_penalty_qp1/qp2
- extended_quadratic_exponential_ep1, extended_tridiagonal_2
- fletchcr, tridia, arwhead, nondia, nondquar
- dqdrtic, broyden_tridiagonal, liarwhd, engval1
- edensch, cube, nonscomp, vardim, quartc, sinquad
- extended_denschnb, extended_denschnf, dixon3dq, biggsb1
- generalized_quartic, himmelbg, himmelh, trid, zakharov
- sum_of_different_powers, cosine, sine

**Transcendental functions:**
- raydan_1/2, diagonal_1/2/3/5/7/8/9, hager
- extended_trigonometric, eg2, indef, genhumps, mccormck
- ackley, griewank, levy, rastrigin, schwefel
- fletcbv3, bdqrtic, bdexp, fh3

### Step 3: Implement Optimizer Base Trait
```rust
pub trait Optimizer {
    fn optimize<F: Fn(&[f64]) -> f64>(
        &mut self,
        function: F,
        dim: usize,
        it: usize,
        x_init: Option<&[f64]>,
        debug: bool,
    ) -> (Vec<Vec<f64>>, Vec<f64>);
}
```

### Step 4: Implement GD (Gradient Descent)
- Central Gaussian Smoothing gradient estimator
- Vectorized implementation using ndarray
- Key components: `_grad_estimator_vectorized`

### Step 5: Implement ASHGF
Complex algorithm requiring:
- Hermite quadrature (use `特殊_ortho_group` equivalent: `Qr` decomposition)
- Directional Gaussian Smoothing (DGS)
- Historical gradient subspace computation
- Adaptive sigma and learning rate

### Step 6: Implement Remaining Optimizers
- SGES: Gradient-guided direction sampling
- ASGF: Adaptive Stochastic Gradient-Free
- ASEBO: Adaptive Surrogate-Based

### Step 7: Add Test Suite
Create validation tests comparing Python and Rust outputs:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sphere_function() {
        // Test sphere function at x = [1, 2, 3] => expected = 14
    }
    
    #[test]
    fn test_optimizer_convergence() {
        // Test GD on sphere function, verify convergence
    }
}
```

### Step 8: Add Criterion.rs Benchmarks
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_sphere(c: &mut Criterion) {
    let mut group = BenchmarkGroup::new("functions");
    for dim in [10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("sphere", dim), &dim, |b, &dim| {
            let x: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1).collect();
            b.iter(|| sphere(&x));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sphere);
criterion_main!(benches);
```

## Key Differences Python -> Rust

| Python | Rust |
|--------|------|
| `numpy as np` | `ndarray::Array1` / `Array2` |
| `np.random.randn()` | `rand::random::<f64>()` |
| `np.linalg.norm()` | `ndarray_linalg::norm()` |
| `@njit(cache=True)` | `#[inline]` + pure functions |
| `Function.evaluate(x)` | `Fn(&[f64]) -> f64` closure |
| `List[T]` | `Vec<T>` |
| `Optional[X]` | `Option<X>` |
| Type hints | Type annotations + trait bounds |

## Dependencies (Cargo.toml)

```toml
[package]
name = "ashgf"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
rand = "0.8"
rand_distr = "0.4"
rayon = "1.7"
approx = "0.5"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "benchmarks"
harness = false
```

## Testing Strategy

1. **Unit Tests**: Individual benchmark functions
   - Compare against known analytical results (e.g., sphere(0) = 0)
   - Test edge cases (empty arrays, large values)

2. **Integration Tests**: Optimizer convergence
   - Run optimizer on simple functions (sphere, rosenbrock)
   - Verify final value < initial value
   - Compare convergence speed to Python version

3. **Performance Benchmarks**:
   - Benchmark each function with Criterion
   - Compare Rust vs Python (Numba) performance
   - Profile critical paths

## Notes

- Use `ndarray::azip!` for vectorized operations
- Use `rayon::par_bridge` for parallel evaluation where applicable
- Leverage Rust's iterators for functional-style code
- Consider SIMD via `std::simd` for hot loops
- Use `const` for mathematical constants (PI, E, etc.)
