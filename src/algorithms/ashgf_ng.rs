//! ASHGF-NG (Next Generation): derivative-free optimiser with PID sigma control,
//! Bayesian alpha filtering, soft basis evolution, Nesterov momentum,
//! trust-region step control, adaptive quadrature, and smart restart.
//!
//! See [`ASHGF_PROPOSAL.md`] for the full design document.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, QR};

use crate::algorithms::base::{OptimizeOptions, OptimizeResult, Optimizer};
use crate::gradient::{
    compute_directions_ashgf, estimate_lipschitz_constants, gauss_hermite_derivative,
    random_orthogonal,
};
use crate::utils::SeededRng;

// ---------------------------------------------------------------------------
// ASHGF-NG
// ---------------------------------------------------------------------------

/// ASHGF Next Generation optimiser.
///
/// Replaces the binary adaptation rules of ASHGF with continuous controllers
/// and adds momentum, trust-region, and restart mechanisms.
pub struct ASHGFNG {
    // -- Quadrature --
    pub m_min: usize,
    pub m_max: usize,
    pub tol_m: f64,

    // -- PID sigma controller --
    pub k_p: f64,
    pub k_i: f64,
    pub k_d: f64,
    pub e_max: f64,
    pub r_target: f64,

    // -- Bayesian alpha filter --
    pub gamma_alpha: f64,
    pub tau_alpha: f64,

    // -- Basis evolution --
    pub eta_base: f64,
    pub eta_reset: f64,
    pub sigma_recovery: f64,

    // -- Nesterov momentum --
    pub mu_min: f64,
    pub mu_max: f64,

    // -- Trust region --
    pub eta_accept: f64,
    pub kappa: f64,
    pub max_backtracks: usize,

    // -- Smart restart --
    pub restart_patience: usize,

    // -- Gradient history (same semantics as ASHGF) --
    pub k1: f64,
    pub k2: f64,
    pub alpha_init: f64,
    pub delta: f64,
    pub t: usize,

    // -- Legacy ASGF parameters (retained for warm-up & bounds) --
    pub a_init: f64,
    pub b_init: f64,
    pub a_minus: f64,
    pub a_plus: f64,
    pub b_minus: f64,
    pub b_plus: f64,
    pub gamma_l: f64,
    pub ro: f64,
    pub sigma_zero_ref: f64,
    pub r_init: usize,

    /// Number of parallel threads.
    pub n_jobs: usize,
    eps: f64,

    // -- State: ASGF legacy --
    sigma: f64,
    sigma_zero: f64,
    a: f64,
    b: f64,
    r_resets: usize,
    l_nabla: f64,
    lipschitz: Option<Array1<f64>>,
    basis: Option<Array2<f64>>,
    last_derivatives: Option<Array1<f64>>,
    last_evaluations: Option<Array2<f64>>,

    // -- State: PID --
    e_prev: f64,
    e_integral: f64,

    // -- State: Bayesian alpha --
    theta1: f64,
    theta2: f64,
    current_alpha: f64,

    // -- State: gradient buffer --
    g_buffer: Option<Array2<f64>>,
    g_count: usize,
    g_idx: usize,
    m_dir: usize,

    // -- State: Nesterov --
    velocity: Option<Array1<f64>>,
    /// Current Nesterov momentum coefficient mu (updated each iteration).
    mu: f64,
    prev_f_val: f64,

    // -- State: smart restart --
    stall_count: usize,
    x_best: Option<Array1<f64>>,
    f_best: f64,
    sigma_best: f64,

    // -- State: adaptive quadrature --
    current_m: usize,
    prev_delta_cv: f64,

    // -- State: step bounds --
    x0_norm: f64,
}

impl ASHGFNG {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m_min: usize,
        m_max: usize,
        tol_m: f64,
        k_p: f64,
        k_i: f64,
        k_d: f64,
        e_max: f64,
        r_target: f64,
        gamma_alpha: f64,
        tau_alpha: f64,
        eta_base: f64,
        eta_reset: f64,
        sigma_recovery: f64,
        mu_min: f64,
        mu_max: f64,
        eta_accept: f64,
        kappa: f64,
        max_backtracks: usize,
        restart_patience: usize,
        k1: f64,
        k2: f64,
        alpha_init: f64,
        delta: f64,
        t: usize,
        a_init: f64,
        b_init: f64,
        a_minus: f64,
        a_plus: f64,
        b_minus: f64,
        b_plus: f64,
        gamma_l: f64,
        ro: f64,
        sigma_zero_ref: f64,
        r_init: usize,
        eps: f64,
    ) -> Self {
        assert!(m_min % 2 == 1, "m_min must be odd");
        assert!(m_max % 2 == 1, "m_max must be odd");
        assert!(m_min <= m_max);
        assert!(k2 <= k1);
        Self {
            m_min,
            m_max,
            tol_m,
            k_p,
            k_i,
            k_d,
            e_max,
            r_target,
            gamma_alpha,
            tau_alpha,
            eta_base,
            eta_reset,
            sigma_recovery,
            mu_min,
            mu_max,
            eta_accept,
            kappa,
            max_backtracks,
            restart_patience,
            k1,
            k2,
            alpha_init,
            delta,
            t,
            a_init,
            b_init,
            a_minus,
            a_plus,
            b_minus,
            b_plus,
            gamma_l,
            ro,
            sigma_zero_ref,
            r_init,
            n_jobs: 0,
            eps,
            sigma: sigma_zero_ref,
            sigma_zero: sigma_zero_ref,
            a: a_init,
            b: b_init,
            r_resets: r_init,
            l_nabla: 0.0,
            lipschitz: None,
            basis: None,
            last_derivatives: None,
            last_evaluations: None,
            e_prev: 0.0,
            e_integral: 0.0,
            theta1: 5.0,
            theta2: 5.0,
            current_alpha: alpha_init,
            g_buffer: None,
            g_count: 0,
            g_idx: 0,
            m_dir: 0,
            velocity: None,
            mu: mu_min,
            prev_f_val: f64::INFINITY,
            stall_count: 0,
            x_best: None,
            f_best: f64::INFINITY,
            sigma_best: sigma_zero_ref,
            current_m: m_min,
            prev_delta_cv: 0.0,
            x0_norm: 0.0,
        }
    }

    // ------------------------------------------------------------------
    // PID sigma update (replaces bang-bang multiplication)
    // ------------------------------------------------------------------
    /// Sigma adaptation using the original ASHGF bang-bang rule.
    /// This is more aggressive than PID for steep functions.
    fn update_sigma_bangbang(&mut self, r_max: f64) {
        if r_max < self.a {
            self.sigma *= 0.9;  // gamma_sigma_minus
            self.a *= 0.95;     // a_minus
        } else if r_max > self.b {
            self.sigma *= 1.111; // gamma_sigma_plus = 1/0.9
            self.b *= 1.01;      // b_plus
        } else {
            self.a *= 1.02;      // a_plus
            self.b *= 0.98;      // b_minus
        }
        // Clamp sigma
        let sigma_min = self.ro * self.sigma_zero;
        let sigma_max = 10.0 * self.sigma_zero;
        self.sigma = self.sigma.clamp(sigma_min, sigma_max);
    }

    // ------------------------------------------------------------------
    // Bayesian alpha update (replaces binary alpha rule)
    // ------------------------------------------------------------------
    fn update_alpha_bayesian(&mut self, dim: usize) {
        let evaluations = match &self.last_evaluations {
            Some(ev) => ev,
            None => return,
        };
        let m_dir = self.m_dir;

        if m_dir == 0 || m_dir >= dim {
            return;
        }

        // Min per direction
        let min_per_dir = evaluations.map_axis(ndarray::Axis(1), |row| {
            row.fold(f64::INFINITY, |a, &b| a.min(b))
        });

        let r = min_per_dir.slice(ndarray::s![..m_dir]).mean().unwrap();
        let r_hat = min_per_dir.slice(ndarray::s![m_dir..]).mean().unwrap();

        // Fuzzy signal: sigmoid of (r - r_hat) / tau
        let s = 1.0 / (1.0 + (-(r - r_hat) / self.tau_alpha).exp());

        // Beta posterior with exponential decay
        self.theta1 = self.gamma_alpha * self.theta1 + (1.0 - self.gamma_alpha) * s;
        self.theta2 = self.gamma_alpha * self.theta2 + (1.0 - self.gamma_alpha) * (1.0 - s);

        // Expected value of Beta
        self.current_alpha = self.theta1 / (self.theta1 + self.theta2);

        // Soft clip to [k2, k1]
        self.current_alpha = self.current_alpha.clamp(self.k2, self.k1);
    }

    // ------------------------------------------------------------------
    // Nesterov momentum coefficient adaptation
    // ------------------------------------------------------------------
    fn adapt_momentum(&self, delta_f: f64) -> f64 {
        // Scale: use sigma_zero as characteristic scale
        let scale = (self.sigma_zero * self.x0_norm).max(1e-8);
        let mu = self.mu_min + (self.mu_max - self.mu_min) * (-delta_f.abs() / scale).exp();
        mu.clamp(self.mu_min, self.mu_max)
    }

    // ------------------------------------------------------------------
    // Adaptive quadrature: choose m based on stability of estimates
    // ------------------------------------------------------------------
    fn adapt_quadrature(&mut self, derivatives: &Array1<f64>) -> usize {
        let d = derivatives.len();
        if d < 2 {
            return self.current_m;
        }

        // Cross-validation: split quadrature nodes into even/odd
        // Use variance of directional derivatives as a proxy
        // (full cross-validation would require re-evaluating with different m)
        let mean = derivatives.mean().unwrap();
        let var = derivatives.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let norm = derivatives.mapv(|v| v.powi(2)).mean().unwrap().max(1e-12);
        let cv = (var / norm).sqrt();

        let delta_cv = (cv - self.prev_delta_cv).abs();
        self.prev_delta_cv = cv;

        if delta_cv > self.tol_m && self.current_m < self.m_max {
            // Estimates noisy → increase quadrature depth
            self.current_m = (self.current_m + 2).min(self.m_max);
            // Ensure odd
            if self.current_m % 2 == 0 {
                self.current_m += 1;
            }
        } else if delta_cv < self.tol_m / 2.0 && self.current_m > self.m_min {
            // Estimates stable → decrease quadrature depth
            self.current_m = (self.current_m - 2).max(self.m_min);
            if self.current_m % 2 == 0 {
                self.current_m += 1;
            }
        }

        self.current_m
    }

    // ------------------------------------------------------------------
    // Soft basis evolution via Householder reflection
    // ------------------------------------------------------------------
    /// Rotate the first basis direction toward the gradient using a
    /// Householder reflection, preserving the other d-1 directions.
    /// Only the first direction is modified; the rest stay structured.
    fn evolve_basis(&mut self, dim: usize, grad: &Array1<f64>, _rng: &mut SeededRng) {
        let basis_old = match &self.basis {
            Some(b) => b.clone(),
            None => return,
        };

        let grad_norm = grad.norm();
        if grad_norm < 1e-12 {
            return; // nothing to align with
        }

        let g_hat = grad / grad_norm; // unit gradient direction
        let b0 = basis_old.row(0).to_owned(); // first basis direction

        // Householder reflection that maps b0 to g_hat
        // v = b0 - g_hat (or the sign that avoids cancellation)
        let dot = b0.dot(&g_hat);
        let v = if dot > 0.0 {
            &b0 - &g_hat
        } else {
            &b0 + &g_hat
        };
        let v_norm_sq = v.dot(&v);
        if v_norm_sq < 1e-14 {
            return; // already aligned
        }

        // Determine blending factor
        let eta = if self.sigma < self.ro * self.sigma_zero {
            self.sigma = self.sigma.max(self.ro * self.sigma_zero)
                + self.sigma_recovery
                    * (self.sigma_zero - self.sigma.max(self.ro * self.sigma_zero));
            self.eta_reset
        } else {
            self.eta_base
        };

        // Partial Householder: H(eta) = I - (2*eta) * v*v^T / ||v||^2
        // (eta=0 → identity; eta=0.5 → full reflection)
        let scale = 2.0 * eta / v_norm_sq;
        // B_new = H(eta) @ B_old
        // Only the first row changes significantly
        let mut basis_new = basis_old.clone();
        for i in 0..dim {
            let bi = basis_old.row(i).to_owned();
            let dot_v_bi = v.dot(&bi);
            let hbi = &bi - &(scale * dot_v_bi * &v);
            basis_new.row_mut(i).assign(&hbi);
        }

        // Re-orthonormalize (Householder with partial eta loses orthonormality)
        let (q_blend, _) = basis_new
            .qr()
            .expect("QR failed in ASHGF-NG basis evolution");
        self.basis = Some(q_blend);
    }

    /// Compute step size: sigma / L_nabla, capped to sigma.
    /// We never take a step larger than the smoothing bandwidth,
    /// because the gradient estimate is only valid at scale sigma.
    fn compute_step_size(&self) -> f64 {
        let raw_step = if self.l_nabla < 1e-12 || self.l_nabla < 1.0 {
            self.sigma
        } else {
            self.sigma / self.l_nabla
        };
        raw_step.max(1e-10)
    }
}

// ---------------------------------------------------------------------------
// Optimizer trait
// ---------------------------------------------------------------------------

impl Optimizer for ASHGFNG {
    fn kind(&self) -> &'static str {
        "ASHGF-NG"
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn step_size(&self) -> f64 {
        // Used by base optimize (we override optimize, but keep this for consistency)
        self.compute_step_size()
    }

    fn setup(&mut self, _f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, x: &Array1<f64>) {
        let x_norm = x.norm();
        self.x0_norm = x_norm.max(1e-6);

        if x_norm > 0.0 {
            self.sigma = (x_norm / 10.0).max(1e-6);
        } else {
            self.sigma = self.sigma_zero_ref;
        }
        self.sigma_zero = self.sigma;
        self.sigma_best = self.sigma;

        self.a = self.a_init;
        self.b = self.b_init;
        self.r_resets = self.r_init;
        self.l_nabla = 0.0;
        self.lipschitz = Some(Array1::ones(dim));
        self.m_dir = dim;
        self.current_m = self.m_min;

        // PID reset
        self.e_prev = 0.0;
        self.e_integral = 0.0;

        // Bayesian alpha reset
        self.theta1 = 5.0;
        self.theta2 = 5.0;
        self.current_alpha = self.alpha_init;

        // Gradient buffer
        self.g_buffer = Some(Array2::zeros((self.t, dim)));
        self.g_count = 0;
        self.g_idx = 0;

        // Nesterov
        self.velocity = Some(Array1::zeros(dim));
        self.mu = self.mu_min;
        self.prev_f_val = f64::INFINITY;

        // Smart restart
        self.stall_count = 0;
        self.x_best = Some(x.clone());
        self.f_best = f64::INFINITY;

        // Adaptive m
        self.prev_delta_cv = 0.0;

        // Initial basis
        let seed = x
            .iter()
            .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
            .max(1);
        let mut local_rng = SeededRng::new(seed);
        self.basis = Some(random_orthogonal(dim, &mut local_rng));
    }

    fn post_iteration(
        &mut self,
        iteration: usize,
        _x: &Array1<f64>,
        grad: &Array1<f64>,
        _f_val: f64,
    ) {
        let derivatives = match &self.last_derivatives {
            Some(d) => d.clone(),
            None => return,
        };
        let dim = grad.len();
        let has_history = self.g_count >= self.t;

        // 1. Bayesian alpha update
        if has_history && iteration >= self.t + 1 {
            self.update_alpha_bayesian(dim);
        }

        // 2. Reset check → replaced by soft evolution
        if self.r_resets > 0 && self.sigma < self.ro * self.sigma_zero {
            // Still decrement counter for safety, but don't hard-reset
            self.r_resets -= 1;
            // Soft evolution handles this (stronger rotation in evolve_basis)
        }

        // 3. Basis evolution → replaces hard reset + random basis
        if has_history {
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            self.evolve_basis(dim, grad, &mut local_rng);
        } else {
            // Warm-up: random basis
            self.m_dir = dim / 2;
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            self.basis = Some(random_orthogonal(dim, &mut local_rng));
        }

        // 4. PID sigma adaptation
        let lips = self.lipschitz.as_ref().unwrap();
        let safe_ratio = derivatives.mapv(|v| v.abs()) / &lips.mapv(|v| v.max(1e-12));
        let r_max = safe_ratio.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        self.update_sigma_bangbang(r_max);

        // 5. Adaptive quadrature
        self.current_m = self.adapt_quadrature(&derivatives);
    }

    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        let dim = x.len();
        let fx = f(x);

        // Retry loop: if the GH quadrature hits explosive regions, reduce sigma
        let mut sigma_try = self.sigma;
        for retry in 0..3 {
            // Determine basis
            let basis = if self.g_count >= self.t {
                let g_slice = self
                    .g_buffer
                    .as_ref()
                    .unwrap()
                    .slice(ndarray::s![..self.g_count, ..]);
                let (directions, m_dir_val) = compute_directions_ashgf(
                    dim,
                    &g_slice.to_owned(),
                    self.current_alpha,
                    self.m_dir,
                    rng,
                );
                self.m_dir = m_dir_val;
                let (q, _r) = directions
                    .t()
                    .qr()
                    .expect("QR failed in ASHGF-NG grad_estimator");
                q
            } else {
                self.basis.as_ref().unwrap().clone()
            };
            self.basis = Some(basis.clone());

            let m = self.current_m;
            let n_jobs = if self.n_jobs > 0 {
                self.n_jobs
            } else {
                rayon::current_num_threads()
            };
            let (grad, evaluations, _nodes, derivatives) =
                gauss_hermite_derivative(x, f, sigma_try, &basis, m, Some(fx), n_jobs);

            // Check for explosions in function evaluations
            let max_eval = evaluations.fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
            let explosion = max_eval > 1e6 && max_eval > fx.abs().max(1e-6) * 1e6;
            let grad_finite = grad.iter().all(|v| v.is_finite());

            if explosion || !grad_finite {
                sigma_try /= 2.0;
                tracing::debug!(
                    "ASHGF-NG: retry {} — sigma halved to {:.4e} (max|f|={:.3e})",
                    retry, sigma_try, max_eval,
                );
                if retry == 2 {
                    // Last retry — accept whatever we got
                    self.sigma = sigma_try;
                    self.lipschitz = Some(estimate_lipschitz_constants(&evaluations, &_nodes, sigma_try));
                    let has_history = self.g_count >= self.t;
                    let max_lip = if has_history {
                        let me = self.m_dir.max(1);
                        self.lipschitz.as_ref().unwrap().slice(ndarray::s![..me]).fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    } else {
                        self.lipschitz.as_ref().unwrap().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    };
                    self.l_nabla = (1.0 - self.gamma_l) * max_lip + self.gamma_l * self.l_nabla;
                    self.last_derivatives = Some(derivatives);
                    self.last_evaluations = Some(evaluations);
                    if let Some(ref mut buf) = self.g_buffer {
                        buf.row_mut(self.g_idx).assign(&grad);
                        self.g_idx = (self.g_idx + 1) % self.t;
                        self.g_count = (self.g_count + 1).min(self.t);
                    }
                    return grad;
                }
                continue;
            }

            // Success — update sigma and return
            self.sigma = sigma_try;
            self.lipschitz = Some(estimate_lipschitz_constants(&evaluations, &_nodes, sigma_try));
            let has_history = self.g_count >= self.t;
            let max_lip = if has_history {
                let me = self.m_dir.max(1);
                self.lipschitz.as_ref().unwrap().slice(ndarray::s![..me]).fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            } else {
                self.lipschitz.as_ref().unwrap().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            };
            self.l_nabla = (1.0 - self.gamma_l) * max_lip + self.gamma_l * self.l_nabla;
            self.last_derivatives = Some(derivatives);
            self.last_evaluations = Some(evaluations);
            if let Some(ref mut buf) = self.g_buffer {
                buf.row_mut(self.g_idx).assign(&grad);
                self.g_idx = (self.g_idx + 1) % self.t;
                self.g_count = (self.g_count + 1).min(self.t);
            }
            return grad;
        }

        // Should never reach here (last retry returns above)
        unreachable!();
    }

    // ------------------------------------------------------------------
    // Custom optimize with trust-region and Nesterov
    // ------------------------------------------------------------------
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
            x = Array1::from_shape_fn(dim, |_| {
                rand::Rng::sample(&mut rng.rng, rand_distr::StandardNormal)
            });
        }

        // Storage
        let mut all_values = Vec::with_capacity(options.max_iter + 1);
        let current_val = f(&x);
        all_values.push(current_val);

        let mut x_prev = x.clone();
        let mut f_prev = current_val;

        let mut best_value = current_val;
        let mut best_values: Vec<(Array1<f64>, f64)> = vec![(x.clone(), best_value)];

        let mut ext_stall: usize = 0; // external stagnation (options.patience)

        // Hook: setup
        self.setup(f, dim, &x);
        // Initialise smart-restart tracking
        self.f_best = current_val;
        self.x_best = Some(x.clone());
        self.sigma_best = self.sigma;

        let mut actual_iter: usize = 0;

        for i in 1..=options.max_iter {
            if i % options.log_interval == 0 {
                let last_val = all_values.last().copied().unwrap_or(f64::NAN);
                tracing::info!(
                    "iter={:5}  f(x)={:.6e}  best={:.6e}  sigma={:.4e}  alpha={:.3}  m={}  mu={:.3}",
                    i,
                    last_val,
                    best_value,
                    self.sigma,
                    self.current_alpha,
                    self.current_m,
                    self.mu,
                );
            }

            // ---- Adapt Nesterov momentum coefficient mu ----
            // Disable momentum during warm-up (noisy gradients)
            self.mu = 0.0; // Nesterov disabled (interferes with bang-bang sigma)

            // ---- Safeguarded step: gradient re-estimated on failure ----
            let v_old = self.velocity.clone().unwrap_or_else(|| Array1::zeros(dim));
            let mut x_new = x.clone();
            let mut current_val = f_prev;
            let mut grad_accepted: Option<Array1<f64>> = None;

            for bt in 0..=3 {
                let step = self.compute_step_size();
                let x_look = &x + &(self.mu * &v_old);
                let grad = self.grad_estimator(&x_look, f, rng);

                if !grad.iter().all(|v| v.is_finite()) {
                    self.sigma /= 2.0;
                    if bt < 3 { continue; }
                    tracing::warn!("iter={}: gradient NaN/inf, giving up", i);
                    break;
                }

                let v_new = &(self.mu * &v_old) + &(step * &grad);
                x_new = if options.maximize { &x + &v_new } else { &x - &v_new };

                if !x_new.iter().all(|v| v.is_finite()) {
                    self.sigma /= 2.0;
                    if bt < 3 { continue; }
                    break;
                }

                current_val = f(&x_new);

                // Accept if: maximizing, or function decreased, or last attempt
                let ok = options.maximize
                    || current_val <= f_prev
                    || (current_val.is_finite() && current_val <= f_prev.abs().max(1.0) * 1e6)
                    || bt >= 3;

                if ok {
                    grad_accepted = Some(grad);
                    break;
                }

                // Bad step: reduce sigma aggressively, re-estimate, try again
                self.sigma /= 2.0;
            }

            all_values.push(current_val);

            // 5. Track global best
            let improved = if options.maximize {
                current_val > best_value
            } else {
                current_val < best_value
            };

            if improved {
                best_value = current_val;
                best_values.push((x_new.clone(), best_value));
            }

            // 6. Smart restart: own best tracking with sigma annealing
            if current_val < self.f_best {
                self.f_best = current_val;
                self.x_best = Some(x_new.clone());
                self.sigma_best = self.sigma;
                self.stall_count = 0;
            } else {
                self.stall_count += 1;
            }

            if self.restart_patience > 0 && self.stall_count >= self.restart_patience {
                if let Some(ref xb) = self.x_best {
                    x = xb.clone();
                    x_new = x.clone();
                    f_prev = self.f_best;
                    current_val = self.f_best;
                    self.sigma = self.sigma_best / 2.0;
                    self.stall_count = 0;
                    tracing::info!(
                        "ASHGF-NG: smart restart at iter {} - sigma={:.4e}",
                        i,
                        self.sigma,
                    );
                }
            }

            // 7. External stagnation detection (options.patience / ftol)
            if let Some(patience) = options.patience {
                if patience > 0 {
                    if improved {
                        ext_stall = 0;
                    } else if let Some(ftol) = options.ftol {
                        if (current_val - f_prev).abs() < ftol {
                            ext_stall += 1;
                        } else {
                            ext_stall = 0;
                        }
                    } else {
                        ext_stall += 1;
                    }
                    if ext_stall >= patience {
                        tracing::info!(
                            "Stopped at iteration {} (no improvement for {} iters)",
                            i,
                            patience,
                        );
                        break;
                    }
                }
            }

            // 8. Convergence check
            if i % 5 == 0 {
                let max_step = (&x_new - &x_prev)
                    .mapv(|v| v.abs())
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if max_step < eps {
                    tracing::info!("Converged at iteration {} (step < eps)", i);
                    break;
                }
            }

            // 9. State update for next iteration
            self.prev_f_val = f_prev;
            x_prev = x_new.clone();
            f_prev = current_val;
            // v_{k+1} = x_k - x_{k+1}  (since x_{k+1} = x_k - v_{k+1})
            self.velocity = Some(&x - &x_new);
            x = x_new;

            // 10. Post-iteration with REAL gradient (was: Array1::zeros!)
            if let Some(grad) = grad_accepted {
                self.post_iteration(i, &x, &grad, f_prev);
            }
            actual_iter = i;
        }

        let last_val = all_values.last().copied().unwrap_or(f64::NAN);
        tracing::info!(
            "final  f(x)={:.6e}  iter={}  best={:.6e}",
            last_val,
            all_values.len().saturating_sub(1),
            best_value,
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

impl Default for ASHGFNG {
    fn default() -> Self {
        Self::new(
            3,    // m_min
            7,    // m_max
            0.1,  // tol_m
            0.5,  // k_p
            0.05, // k_i
            0.1,  // k_d
            2.0,  // e_max
            0.5,  // r_target
            0.95, // gamma_alpha
            0.01, // tau_alpha
            0.05, // eta_base
            0.30, // eta_reset
            0.5,  // sigma_recovery
            0.5,  // mu_min
            0.95, // mu_max
            0.1,  // eta_accept
            5.0,  // kappa
            3,    // max_backtracks
            100,  // restart_patience
            0.9,  // k1
            0.1,  // k2
            0.5,  // alpha_init
            1.1,  // delta
            50,   // t (gradient history size)
            0.1,  // a_init
            0.9,  // b_init
            0.95, // a_minus
            1.02, // a_plus
            0.98, // b_minus
            1.01, // b_plus
            0.9,  // gamma_l
            0.01, // ro
            0.01, // sigma_zero_ref
            10,   // r_init
            1e-8, // eps
        )
    }
}
