//! Adaptive Stochastic Historical Gradient-Free (ASHGF) optimiser.
//!
//! ASHGF extends ASGF by incorporating a gradient history buffer, similar
//! to how SGES extends GD.  It uses Gauss-Hermite quadrature for gradient
//! estimation with directions that mix the gradient subspace (estimated
//! from past gradients) with random orthogonal directions.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, QR};

use crate::algorithms::base::{OptimizeOptions, OptimizeResult, Optimizer};
use crate::gradient::{
    compute_directions_ashgf, estimate_lipschitz_constants, gauss_hermite_derivative,
    random_orthogonal,
};
use crate::utils::SeededRng;

/// Adaptive Stochastic Historical Gradient-Free optimiser.
///
/// Combines SGES-style gradient history buffer with ASGF-style
/// Gauss-Hermite quadrature and parameter adaptation.
pub struct ASHGFS {
    // -- Quadrature & adaptation --
    pub m: usize,
    pub a_init: f64,
    pub b_init: f64,
    pub a_minus: f64,
    pub a_plus: f64,
    pub b_minus: f64,
    pub b_plus: f64,
    pub gamma_l: f64,
    pub gamma_sigma_minus: f64,
    pub gamma_sigma_plus: f64,
    pub r_init: usize,
    pub ro: f64,
    pub sigma_zero_ref: f64,

    // -- Gradient history --
    pub k1: f64,
    pub k2: f64,
    pub alpha_init: f64,
    pub delta: f64,
    pub t: usize,

    /// Number of parallel threads for function evaluation.
    pub n_jobs: usize,
    eps: f64,
    pub safeguard_active: bool,

    // -- Adaptive state --
    sigma: f64,
    sigma_zero: f64,
    a: f64,
    b: f64,
    r: usize,
    l_nabla: f64,
    lipschitz: Option<Array1<f64>>,
    basis: Option<Array2<f64>>,
    last_derivatives: Option<Array1<f64>>,

    // -- Gradient history state --
    current_alpha: f64,
    g_buffer: Option<Array2<f64>>,
    g_count: usize,
    g_idx: usize,
    m_dir: usize,
    last_evaluations: Option<Array2<f64>>,
}

impl ASHGFS {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: usize,
        a: f64,
        b: f64,
        a_minus: f64,
        a_plus: f64,
        b_minus: f64,
        b_plus: f64,
        gamma_l: f64,
        gamma_sigma_minus: f64,
        gamma_sigma_plus: f64,
        r: usize,
        ro: f64,
        sigma_zero: f64,
        k1: f64,
        k2: f64,
        alpha: f64,
        delta: f64,
        t: usize,
        eps: f64,
    ) -> Self {
        assert!(m % 2 == 1, "m must be odd");
        assert!(k2 <= k1);
        Self {
            m,
            a_init: a,
            b_init: b,
            a_minus,
            a_plus,
            b_minus,
            b_plus,
            gamma_l,
            gamma_sigma_minus,
            gamma_sigma_plus,
            r_init: r,
            ro,
            sigma_zero_ref: sigma_zero,
            k1,
            k2,
            alpha_init: alpha,
            delta,
            t,
            n_jobs: 0,
            eps,
            safeguard_active: false,
            sigma: sigma_zero,
            sigma_zero,
            a,
            b,
            r,
            l_nabla: 0.0,
            lipschitz: None,
            basis: None,
            last_derivatives: None,
            current_alpha: alpha,
            g_buffer: None,
            g_count: 0,
            g_idx: 0,
            m_dir: 0,
            last_evaluations: None,
        }
    }

    /// Update alpha based on relative performance of gradient vs random directions.
    ///
    /// `alpha` is the probability of choosing a **random** direction.
    /// * `r < r_hat` → gradient subspace better → decrease alpha (favour gradient).
    /// * `r >= r_hat` → random better → increase alpha (more exploration).
    fn update_alpha(&mut self, dim: usize) {
        let evaluations = match &self.last_evaluations {
            Some(ev) => ev,
            None => return,
        };
        let m_dir = self.m_dir;

        if m_dir == 0 || m_dir >= dim {
            return;
        }

        // Min per direction (over quadrature points)
        let min_per_dir = evaluations.map_axis(ndarray::Axis(1), |row| {
            row.fold(f64::INFINITY, |a, &b| a.min(b))
        });

        let r = min_per_dir.slice(ndarray::s![..m_dir]).mean().unwrap();
        let r_hat = min_per_dir.slice(ndarray::s![m_dir..]).mean().unwrap();

        if r < r_hat {
            self.current_alpha = (self.current_alpha / self.delta).max(self.k2);
        } else {
            self.current_alpha = (self.delta * self.current_alpha).min(self.k1);
        }
    }

    fn _run(&mut self, f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize,
            x_init: Option<&Array1<f64>>, options: &OptimizeOptions,
            rng: &mut SeededRng) -> OptimizeResult {
        let eps = self.eps();
        let mut x = if let Some(x0) = x_init { x0.clone() } else {
            Array1::from_shape_fn(dim, |_| rand::Rng::sample(&mut rng.rng, rand_distr::StandardNormal))
        };
        let mut av = Vec::with_capacity(options.max_iter + 1);
        let cv = f(&x); av.push(cv);
        let mut xp = x.clone(); let mut fp = cv;
        let mut best = cv; let mut bv = vec![(x.clone(), best)]; let mut sc: usize = 0;
        self.setup(f, dim, &x); let mut ai = 0;
        for i in 1..=options.max_iter {
            let g = self.grad_estimator(&x, f, rng);
            if !g.iter().all(|v| v.is_finite()) { break; }
            let a = self.step_size();
            let mut xn = if options.maximize { &x + &(a * &g) } else { &x - &(a * &g) };
            if !xn.iter().all(|v| v.is_finite()) { break; }
            let mut cur = f(&xn);
            if self.safeguard_active && !options.maximize && cur.is_finite() && fp.is_finite() && cur > fp {
                self.sigma /= 2.0;
                let g2 = self.grad_estimator(&x, f, rng);
                if g2.iter().all(|v| v.is_finite()) {
                    let a2 = self.step_size();
                    let xn2 = if options.maximize { &x + &(a2 * &g2) } else { &x - &(a2 * &g2) };
                    if xn2.iter().all(|v| v.is_finite()) {
                        let cv2 = f(&xn2);
                        if cv2.is_finite() && cv2 <= fp {
                            cur = cv2; xn = xn2;  // retry succeeded
                        } else {
                            cur = fp; xn = x.clone();  // reject: keep previous x
                        }
                    } else { cur = fp; xn = x.clone(); }
                } else { cur = fp; xn = x.clone(); }
            }
            if !cur.is_finite() { av.push(cur); break; }
            av.push(cur);
            let imp = if options.maximize { cur > best } else { cur < best };
            if imp { best = cur; bv.push((xn.clone(), best)); }
            if let Some(pat) = options.patience { if pat > 0 {
                if imp { sc = 0; } else if let Some(ft) = options.ftol {
                    if (cur - fp).abs() < ft { sc += 1; } else { sc = 0; }
                } else { sc += 1; }
                if sc >= pat { break; }
            }}
            if i % 5 == 0 {
                let ms = (&xn - &xp).mapv(|v| v.abs()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if ms < eps { break; }
            }
            xp = xn.clone(); fp = cur; x = xn;
            self.post_iteration(i, &x, &g, fp); ai = i;
        }
        OptimizeResult { best_values: bv, all_values: av, iterations: ai, converged: ai < options.max_iter }
    }
}
impl Optimizer for ASHGFS {
    fn kind(&self) -> &'static str {
        "ASHGF-S"
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn step_size(&self) -> f64 {
        if self.l_nabla < 1e-12 {
            self.sigma
        } else {
            self.sigma / self.l_nabla
        }
    }

    fn setup(&mut self, _f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, x: &Array1<f64>) {
        let x_norm = x.norm();
        if x_norm > 0.0 {
            self.sigma = (x_norm / 10.0).max(1e-6);
        } else {
            self.sigma = self.sigma_zero_ref;
        }
        self.sigma_zero = self.sigma;
        self.a = self.a_init;
        self.b = self.b_init;
        self.r = self.r_init;
        self.l_nabla = 0.0;
        self.lipschitz = Some(Array1::ones(dim));
        self.m_dir = dim;

        let seed = x
            .iter()
            .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
            .max(1);
        let mut local_rng = SeededRng::new(seed);
        self.basis = Some(random_orthogonal(dim, &mut local_rng));

        self.g_buffer = Some(Array2::zeros((self.t, dim)));
        self.g_count = 0;
        self.g_idx = 0;
        self.current_alpha = self.alpha_init;
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

        // 1. Alpha update
        if has_history && iteration >= self.t + 1 {
            self.update_alpha(dim);
        }

        // 2. Reset check
        if self.r > 0 && self.sigma < self.ro * self.sigma_zero {
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            self.basis = Some(random_orthogonal(dim, &mut local_rng));
            self.sigma = self.sigma_zero;
            self.a = self.a_init;
            self.b = self.b_init;
            self.r -= 1;
            self.m_dir = dim / 2;
            return;
        }

        // 3. Basis update (only during warm-up; after, grad_estimator handles it)
        if !has_history {
            self.m_dir = dim / 2;
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            self.basis = Some(random_orthogonal(dim, &mut local_rng));
        }

        // 4. Sigma adaptation
        let lips = self.lipschitz.as_ref().unwrap();
        let safe_ratio = derivatives.mapv(|v| v.abs()) / &lips.mapv(|v| v.max(1e-12));
        let value = safe_ratio.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if value < self.a {
            self.sigma *= self.gamma_sigma_minus;
            self.a *= self.a_minus;
        } else if value > self.b {
            self.sigma *= self.gamma_sigma_plus;
            self.b *= self.b_plus;
        } else {
            self.a *= self.a_plus;
            self.b *= self.b_minus;
        }
    }

    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        let dim = x.len();
        let fx = f(x);

        // Determine basis
        let basis = if self.g_count >= self.t {
            // Use gradient history
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
                .expect("QR failed in ASHGF::grad_estimator");
            q
        } else {
            self.basis.as_ref().unwrap().clone()
        };
        self.basis = Some(basis.clone());

        // Gauss-Hermite quadrature
        let n_jobs = if self.n_jobs > 0 {
            self.n_jobs
        } else {
            rayon::current_num_threads()
        };
        let (grad, evaluations, nodes, derivatives) =
            gauss_hermite_derivative(x, f, self.sigma, &basis, self.m, Some(fx), n_jobs);

        // Lipschitz constants
        self.lipschitz = Some(estimate_lipschitz_constants(
            &evaluations,
            &nodes,
            self.sigma,
        ));

        // L_nabla: during warm-up use ALL Lipschitz constants (thesis requirement);
        // after warm-up use only those from gradient-subspace directions.
        let has_history = self.g_count >= self.t;
        let max_lip = if has_history {
            let m_eff = self.m_dir.max(1);
            self.lipschitz
                .as_ref()
                .unwrap()
                .slice(ndarray::s![..m_eff])
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        } else {
            self.lipschitz
                .as_ref()
                .unwrap()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };
        self.l_nabla = (1.0 - self.gamma_l) * max_lip + self.gamma_l * self.l_nabla;

        self.last_derivatives = Some(derivatives);
        self.last_evaluations = Some(evaluations);

        // Update gradient buffer
        if let Some(ref mut buf) = self.g_buffer {
            buf.row_mut(self.g_idx).assign(&grad);
            self.g_idx = (self.g_idx + 1) % self.t;
            self.g_count = (self.g_count + 1).min(self.t);
        }

        grad
    }

    fn optimize(&mut self, f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize,
                x_init: Option<&Array1<f64>>, options: &OptimizeOptions, rng: &mut SeededRng) -> OptimizeResult {
        let seed = rng.seed;
        self.safeguard_active = false;
        let r1 = self._run(f, dim, x_init, options, &mut SeededRng::new(seed));
        let b1 = *r1.all_values.last().unwrap_or(&f64::INFINITY);
        self.safeguard_active = true;
        let r2 = self._run(f, dim, x_init, options, &mut SeededRng::new(seed));
        let b2 = *r2.all_values.last().unwrap_or(&f64::INFINITY);
        if b1.is_finite() && (!b2.is_finite() || b1 <= b2) { r1 } else if b2.is_finite() { r2 } else { r1 }
    }

}

impl Default for ASHGFS {
    fn default() -> Self {
        Self::new(
            5,
            0.1,
            0.9,
            0.95,
            1.02,
            0.98,
            1.01,
            0.9,
            0.9,
            1.0 / 0.9,
            10,
            0.01,
            0.01,
            0.9,
            0.1,
            0.5,
            1.1,
            50,
            1e-8,
        )
    }
}
