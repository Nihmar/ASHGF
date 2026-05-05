//! Base trait for all optimisation algorithms.
//!
//! Implements the Template Method pattern: the [`Optimizer`] trait provides
//! a default `optimize` method that calls `grad_estimator`, `step_size`,
//! and optional hooks `setup` / `post_iteration`.

use ndarray::Array1;

use crate::utils::SeededRng;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Options shared by all optimisers.
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// If true, maximise the objective instead of minimising.
    pub maximize: bool,
    /// Stop early if no improvement for this many iterations.
    pub patience: Option<usize>,
    /// Tolerance on |f(x_{k+1}) - f(x_k)| for stagnation detection
    /// (requires `patience`).
    pub ftol: Option<f64>,
    /// Print progress every `log_interval` iterations.
    pub log_interval: usize,
    /// Number of parallel threads for function evaluation.
    /// 0 = use rayon global default (typically number of CPU cores).
    pub n_jobs: usize,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            maximize: false,
            patience: None,
            ftol: None,
            log_interval: 50,
            n_jobs: 0,
        }
    }
}

/// Result of a single optimisation run.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// Sequence of best points found: `(x, f(x))`.
    pub best_values: Vec<(Array1<f64>, f64)>,
    /// Function value at each iteration.
    pub all_values: Vec<f64>,
    /// Number of iterations actually performed.
    pub iterations: usize,
    /// Whether the algorithm converged (step size < eps).
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Optimizer trait
// ---------------------------------------------------------------------------

/// Trait for derivative-free optimisers.
///
/// Implementors must provide [`grad_estimator`](Self::grad_estimator) and
/// [`step_size`](Self::step_size).  The hooks [`setup`](Self::setup) and
/// [`post_iteration`](Self::post_iteration) have default no-op
/// implementations.
///
/// The [`optimize`](Self::optimize) method is the template method that
/// orchestrates the optimisation loop.
pub trait Optimizer {
    /// Human-readable algorithm kind (e.g. `"ASHGF"`).
    fn kind(&self) -> &'static str;

    /// Estimate the gradient `∇f(x)`.
    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64>;

    /// Return the current step size (learning rate).
    fn step_size(&self) -> f64;

    /// Hook called once before the main optimisation loop.
    fn setup(&mut self, _f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), _dim: usize, _x: &Array1<f64>) {}

    /// Hook called after each iteration (for adaptive logic).
    fn post_iteration(
        &mut self,
        _iteration: usize,
        _x: &Array1<f64>,
        _grad: &Array1<f64>,
        _f_val: f64,
    ) {
    }

    /// Compute the next point and its function value.
    ///
    /// Override to implement line-search, trust-region, or step-boost logic.
    /// Default: `x_new = x ± step_size * grad`.
    ///
    /// `f_at_x` is the cached value `f(x)` (may be `None` if not available).
    fn compute_step(
        &mut self,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        maximize: bool,
        _f_at_x: Option<f64>,
    ) -> (Array1<f64>, f64) {
        let alpha = self.step_size();
        let x_new = if maximize {
            x + &(alpha * grad)
        } else {
            x - &(alpha * grad)
        };
        let val = f(&x_new);
        (x_new, val)
    }

    /// Convergence threshold on step size ‖x_{k+1} - x_k‖.
    fn eps(&self) -> f64;

    /// Run the full optimisation loop (template method).
    fn optimize(
        &mut self,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        dim: usize,
        x_init: Option<&Array1<f64>>,
        options: &OptimizeOptions,
        rng: &mut SeededRng,
    ) -> OptimizeResult {
        let eps = self.eps();
        let mut x: Array1<f64>;

        if let Some(x0) = x_init {
            x = x0.clone();
        } else {
            // Random initial point N(0, I)
            x = Array1::from_shape_fn(dim, |_| {
                rand::Rng::sample(&mut rng.rng, rand_distr::StandardNormal)
            });
        }

        // ---- Storage ----
        let mut all_values = Vec::with_capacity(options.max_iter + 1);
        let current_val = f(&x);
        all_values.push(current_val);

        let mut x_prev = x.clone();
        let mut f_prev = current_val;

        let mut best_value = current_val;
        let mut best_values: Vec<(Array1<f64>, f64)> = vec![(x.clone(), best_value)];

        // Early-stopping state
        let mut stall_count: usize = 0;

        // ---- Hook: setup ----
        self.setup(f, dim, &x);

        tracing::info!(
            "START | algo={} | dim={} | max_iter={} | f0={:.6e}",
            self.kind(),
            dim,
            options.max_iter,
            current_val,
        );

        let mut actual_iter: usize = 0;

        for i in 1..=options.max_iter {
            if i % options.log_interval == 0 {
                let last_val = all_values.last().copied().unwrap_or(f64::NAN);
                let gap = if best_value.is_finite() && last_val.is_finite() {
                    (last_val - best_value).abs()
                } else {
                    0.0
                };
                tracing::info!(
                    "{:>5} | f={:.6e} | best={:.6e} | gap={:.2e}",
                    i,
                    last_val,
                    best_value,
                    gap,
                );
            }

            // 1. Estimate gradient
            let grad = self.grad_estimator(&x, f, rng);

            // Guard against NaN/inf in gradient
            if !grad.iter().all(|v| v.is_finite()) {
                tracing::warn!("FAIL | iter={} | gradient NaN/inf", i);
                break;
            }

            // 2. Compute next point (delegated to compute_step for subclasses)
            let f_at_x = all_values.last().copied();
            let (x_new, current_val) = self.compute_step(&x, &grad, f, options.maximize, f_at_x);

            // Guard against NaN/inf in x
            if !x_new.iter().all(|v| v.is_finite()) {
                tracing::warn!("FAIL | iter={} | x NaN/inf", i);
                break;
            }

            // Guard against NaN/inf in function value
            if !current_val.is_finite() {
                tracing::warn!("FAIL | iter={} | f(x) non-finite", i);
                all_values.push(current_val);
                break;
            }

            all_values.push(current_val);

            // 3. Track best
            let improved = if options.maximize {
                current_val > best_value
            } else {
                current_val < best_value
            };

            if improved {
                best_value = current_val;
                best_values.push((x_new.clone(), best_value));
            }

            // 4a. Stagnation detection
            if let Some(patience) = options.patience {
                if patience > 0 {
                    if improved {
                        stall_count = 0;
                    } else if let Some(ftol) = options.ftol {
                        if (current_val - f_prev).abs() < ftol {
                            stall_count += 1;
                        } else {
                            stall_count = 0;
                        }
                    } else {
                        stall_count += 1;
                    }

                    if stall_count >= patience {
                        tracing::info!(
                            "STALL | iter={} | no improvement for {} iters",
                            i,
                            patience,
                        );
                        break;
                    }
                }
            }

            // 4b. Convergence check (every 5 iterations)
            if i % 5 == 0 {
                let max_step = (&x_new - &x_prev)
                    .mapv(|v| v.abs())
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if max_step < eps {
                    tracing::info!("CONV | iter={} | step < eps", i);
                    break;
                }
            }

            x_prev = x_new.clone();
            f_prev = current_val;
            x = x_new;

            // 5. Hook: post-iteration
            self.post_iteration(i, &x, &grad, f_prev);
            actual_iter = i;
        }

        let last_val = all_values.last().copied().unwrap_or(f64::NAN);
        let total_iter = all_values.len().saturating_sub(1);
        let status = if total_iter < options.max_iter {
            "converged"
        } else {
            "max_iter"
        };
        tracing::info!(
            "DONE | iter={} | f={:.6e} | best={:.6e} | {}",
            total_iter,
            last_val,
            best_value,
            status,
        );

        let converged = actual_iter < options.max_iter;
        OptimizeResult {
            best_values,
            all_values,
            iterations: actual_iter,
            converged,
        }
    }
}
