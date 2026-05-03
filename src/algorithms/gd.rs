//! Vanilla Gradient Descent using central Gaussian smoothing.

use ndarray::Array1;

use crate::algorithms::base::Optimizer;
use crate::gradient::{compute_directions, gaussian_smoothing};
use crate::utils::SeededRng;

/// Vanilla Gradient Descent with Gaussian smoothing.
pub struct GD {
    pub lr: f64,
    pub sigma: f64,
    pub n_jobs: usize,
    eps: f64,
}

impl GD {
    pub fn new(lr: f64, sigma: f64, eps: f64) -> Self {
        assert!(lr > 0.0);
        assert!(sigma > 0.0);
        Self {
            lr,
            sigma,
            n_jobs: 0,
            eps,
        }
    }
}

impl Optimizer for GD {
    fn kind(&self) -> &'static str {
        "GD"
    }
    fn eps(&self) -> f64 {
        self.eps
    }
    fn step_size(&self) -> f64 {
        self.lr
    }
    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        let dim = x.len();
        let directions = compute_directions(dim, rng);
        let n_jobs = if self.n_jobs > 0 {
            self.n_jobs
        } else {
            rayon::current_num_threads()
        };
        gaussian_smoothing(x, f, self.sigma, &directions, n_jobs)
    }
}

impl Default for GD {
    fn default() -> Self {
        Self::new(1e-4, 1e-4, 1e-8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::base::OptimizeOptions;
    use crate::functions::get_function;

    #[test]
    fn gd_improves_on_sphere() {
        let mut gd = GD::new(1e-2, 1e-3, 1e-8);
        let f = get_function("sphere").unwrap();
        let options = OptimizeOptions {
            max_iter: 2000,
            ..Default::default()
        };
        let mut rng = SeededRng::new(42);
        let result = gd.optimize(&f, 5, None, &options, &mut rng);
        let initial = result.all_values[0];
        let final_val = *result.all_values.last().unwrap();
        assert!(
            final_val < initial * 0.5,
            "GD did not improve enough: initial={:.4e} final={:.4e}",
            initial,
            final_val,
        );
    }
}
