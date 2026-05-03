//! Adaptive Stochastic Gradient-Free (ASGF) optimiser.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Norm, QR};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::algorithms::base::Optimizer;
use crate::gradient::{estimate_lipschitz_constants, gauss_hermite_derivative, random_orthogonal};
use crate::utils::SeededRng;

/// Adaptive Stochastic Gradient-Free optimiser.
///
/// Uses Gauss-Hermite quadrature for gradient estimation with
/// adaptive smoothing parameter `sigma` and basis rotation.
pub struct ASGF {
    /// Number of quadrature points (odd).
    pub m: usize,
    /// Lower threshold for |D_i|/L_i ratio.
    pub a_init: f64,
    /// Upper threshold.
    pub b_init: f64,
    pub a_minus: f64,
    pub a_plus: f64,
    pub b_minus: f64,
    pub b_plus: f64,
    /// EMA factor for Lipschitz estimate.
    pub gamma_l: f64,
    /// Factor for sigma shrink.
    pub gamma_sigma: f64,
    /// Max number of resets.
    pub r_init: usize,
    /// Reset fraction threshold.
    pub ro: f64,
    /// Fallback initial sigma.
    pub sigma_zero_ref: f64,
    /// Number of parallel threads for function evaluation.
    pub n_jobs: usize,
    eps: f64,

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
}

impl ASGF {
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
        gamma_sigma: f64,
        r: usize,
        ro: f64,
        sigma_zero: f64,
        eps: f64,
    ) -> Self {
        assert!(m % 2 == 1, "m must be odd");
        Self {
            m,
            a_init: a,
            b_init: b,
            a_minus,
            a_plus,
            b_minus,
            b_plus,
            gamma_l,
            gamma_sigma,
            r_init: r,
            ro,
            sigma_zero_ref: sigma_zero,
            eps,
            n_jobs: 0,
            sigma: sigma_zero,
            sigma_zero,
            a,
            b,
            r,
            l_nabla: 0.0,
            lipschitz: None,
            basis: None,
            last_derivatives: None,
        }
    }
}

impl Optimizer for ASGF {
    fn kind(&self) -> &'static str {
        "ASGF"
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
        // We need a SeededRng here but setup doesn't receive one.
        // Use a fresh internal one seeded from the initial point hash.
        let seed = x.iter().fold(0u64, |a, &v| a.wrapping_add(v.to_bits()));
        let mut local_rng = SeededRng::new(seed.max(1));
        self.basis = Some(random_orthogonal(dim, &mut local_rng));
    }

    fn post_iteration(
        &mut self,
        _iteration: usize,
        _x: &Array1<f64>,
        grad: &Array1<f64>,
        _f_val: f64,
    ) {
        let derivatives = match &self.last_derivatives {
            Some(d) => d,
            None => return,
        };
        let dim = grad.len();

        // Reset check
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
            return;
        }

        // Basis rotation: first direction = gradient, rest random, then QR
        let grad_norm = grad.norm();
        if grad_norm > 1e-12 {
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            let mut m: Array2<f64> =
                Array2::from_shape_fn((dim, dim), |_| local_rng.rng.sample(StandardNormal));
            m.row_mut(0).assign(&(grad / grad_norm));
            let (q, _) = m.qr().expect("QR failed in ASGF::post_iteration");
            self.basis = Some(q);
        } else {
            let seed = grad
                .iter()
                .fold(0u64, |a, &v| a.wrapping_add(v.to_bits()))
                .max(1);
            let mut local_rng = SeededRng::new(seed);
            self.basis = Some(random_orthogonal(dim, &mut local_rng));
        }

        // Sigma adaptation
        let lips = self.lipschitz.as_ref().unwrap();
        let safe_lips = lips.mapv(|v| v.max(1e-12));
        let ratio = derivatives.mapv(|v| v.abs()) / &safe_lips;
        let value = ratio.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if value < self.a {
            self.sigma *= self.gamma_sigma;
            self.a *= self.a_minus;
        } else if value > self.b {
            self.sigma /= self.gamma_sigma;
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
        _rng: &mut SeededRng,
    ) -> Array1<f64> {
        let fx = f(x);
        let basis = self.basis.as_ref().expect("basis must be set in setup");
        let n_jobs = if self.n_jobs > 0 {
            self.n_jobs
        } else {
            rayon::current_num_threads()
        };

        let (grad, evaluations, nodes, derivatives) =
            gauss_hermite_derivative(x, f, self.sigma, basis, self.m, Some(fx), n_jobs);

        // Update Lipschitz constants
        self.lipschitz = Some(estimate_lipschitz_constants(
            &evaluations,
            &nodes,
            self.sigma,
        ));

        // Smoothed global Lipschitz
        let max_lip = self
            .lipschitz
            .as_ref()
            .unwrap()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        self.l_nabla = (1.0 - self.gamma_l) * max_lip + self.gamma_l * self.l_nabla;

        self.last_derivatives = Some(derivatives);

        grad
    }
}

impl Default for ASGF {
    fn default() -> Self {
        Self::new(
            5, 0.1, 0.9, 0.95, 1.02, 0.98, 1.01, 0.9, 0.9, 10, 0.01, 0.01, 1e-8,
        )
    }
}
