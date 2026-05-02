//! Self-Guided Evolution Strategies (SGES) optimiser.

use ndarray::{Array1, Array2};

use crate::algorithms::base::Optimizer;
use crate::gradient::{compute_directions, compute_directions_sges, gaussian_smoothing};
use crate::utils::SeededRng;

/// Self-Guided Evolution Strategies.
///
/// Extends Gaussian smoothing by adaptively mixing random directions
/// with directions sampled from the gradient-history subspace.
///
/// .. note::
///
///    **Alpha semantics.**  `alpha` is the probability of sampling a
///    **random** (isotropic) direction.  `1 - alpha` is the probability
///    of sampling from the gradient subspace.  This is inverted relative
///    to the thesis, where α denotes gradient-subspace probability.
///    The update rule is correspondingly inverted.
pub struct SGES {
    /// Fixed learning rate.
    pub lr: f64,
    /// Smoothing bandwidth.
    pub sigma: f64,
    /// Upper bound for alpha.
    pub k1: f64,
    /// Lower bound for alpha.
    pub k2: f64,
    /// Initial alpha (prob. of random direction).
    pub alpha_init: f64,
    /// Multiplicative factor for alpha update.
    pub delta: f64,
    /// Number of pure-random warm-up iterations.
    pub t: usize,
    eps: f64,

    // -- Adaptive state --
    /// Current alpha.
    current_alpha: f64,
    /// Gradient buffer: (t, dim).
    G_buffer: Option<Array2<f64>>,
    G_count: usize,
    G_idx: usize,
    /// Last evaluations for alpha update.
    last_evaluations: Option<Array1<f64>>,
    last_M: usize,
}

impl SGES {
    pub fn new(
        lr: f64,
        sigma: f64,
        k1: f64,
        k2: f64,
        alpha: f64,
        delta: f64,
        t: usize,
        eps: f64,
    ) -> Self {
        assert!(lr > 0.0);
        assert!(sigma > 0.0);
        assert!(k2 <= k1);
        assert!((0.0..=1.0).contains(&alpha));
        Self {
            lr,
            sigma,
            k1,
            k2,
            alpha_init: alpha,
            delta,
            t,
            eps,
            current_alpha: alpha,
            G_buffer: None,
            G_count: 0,
            G_idx: 0,
            last_evaluations: None,
            last_M: 0,
        }
    }
}

impl Optimizer for SGES {
    fn kind(&self) -> &'static str {
        "SGES"
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn step_size(&self) -> f64 {
        self.lr
    }

    fn setup(&mut self, _f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, _x: &Array1<f64>) {
        self.G_buffer = Some(Array2::zeros((self.t, dim)));
        self.G_count = 0;
        self.G_idx = 0;
        self.current_alpha = self.alpha_init;
    }

    fn post_iteration(
        &mut self,
        iteration: usize,
        _x: &Array1<f64>,
        _grad: &Array1<f64>,
        _f_val: f64,
    ) {
        if iteration < self.t {
            return;
        }

        let evaluations = match &self.last_evaluations {
            Some(ev) => ev,
            None => return,
        };
        let M = self.last_M;

        if M == 0 || M >= evaluations.len() / 2 {
            return;
        }

        // Reshape to (dim, 2) and take min per pair
        let dim = evaluations.len() / 2;
        let mut min_per_dir = Array1::zeros(dim);
        for i in 0..dim {
            let a = evaluations[2 * i];
            let b = evaluations[2 * i + 1];
            min_per_dir[i] = a.min(b);
        }

        let r = min_per_dir.slice(ndarray::s![..M]).mean().unwrap();
        let r_hat = min_per_dir.slice(ndarray::s![M..]).mean().unwrap();

        // Inverted logic: alpha = probability of RANDOM direction
        if r < r_hat {
            // Gradient subspace better → decrease alpha (less random)
            self.current_alpha = (self.current_alpha / self.delta).max(self.k2);
        } else {
            // Random better → increase alpha (more exploration)
            self.current_alpha = (self.delta * self.current_alpha).min(self.k1);
        }
    }

    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        let dim = x.len();

        let grad = if self.G_count >= self.t - 1 {
            // Use SGES directions
            let G = self.G_buffer.as_ref().unwrap();
            let G_slice = G.slice(ndarray::s![..self.G_count, ..]);
            let (directions, M) =
                compute_directions_sges(dim, &G_slice.to_owned(), self.current_alpha, rng);
            self.last_M = M;

            // Collect evaluations
            let sigma_dirs = self.sigma * &directions;
            let mut points = Vec::with_capacity(2 * dim);
            for i in 0..dim {
                let d = sigma_dirs.row(i).to_owned();
                points.push(x + &d);
                points.push(x - &d);
            }
            let results: Vec<f64> = points.iter().map(|p| f(p)).collect();
            self.last_evaluations = Some(Array1::from_vec(results.clone()));

            // Gradient assembly
            let results_arr = Array1::from_vec(results);
            let diff = results_arr.slice(ndarray::s![0..;2]).to_owned()
                - &results_arr.slice(ndarray::s![1..;2]);
            diff.dot(&directions) / (2.0 * self.sigma * dim as f64)
        } else {
            // Warm-up: pure random
            let directions = compute_directions(dim, rng);
            self.last_M = 0;
            self.last_evaluations = Some(Array1::zeros(2 * dim));
            gaussian_smoothing(x, f, self.sigma, &directions, 1)
        };

        // Update gradient buffer
        if let Some(ref mut buf) = self.G_buffer {
            buf.row_mut(self.G_idx).assign(&grad);
            self.G_idx = (self.G_idx + 1) % self.t;
            self.G_count = (self.G_count + 1).min(self.t);
        }

        grad
    }
}

impl Default for SGES {
    fn default() -> Self {
        Self::new(1e-4, 1e-4, 0.9, 0.1, 0.5, 1.1, 50, 1e-8)
    }
}
