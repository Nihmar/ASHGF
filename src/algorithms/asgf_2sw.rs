//! ASGF-2SW: 2S with improvement-magnitude-weighted streak.
//!
//! Instead of a binary +1/−1 streak counter, each improvement or regression
//! is weighted by its *relative magnitude*.  Large improvements build
//! confidence faster (and large regressions erode it faster), giving a
//! more informative confidence signal for the 2x boost decision.

use ndarray::Array1;

use crate::algorithms::asgf::ASGF;
use crate::algorithms::base::Optimizer;
use crate::utils::SeededRng;

/// ASGF with magnitude-weighted improvement streak and 2x step boost.
///
/// Wraps an [`ASGF`] instance and overrides step computation to use
/// a weighted streak where each improvement/regression is scaled by
/// its relative magnitude.  The boost multiplier `k` ranges smoothly
/// from 1.0 to 2.0 based on the capped weighted streak.
pub struct Asgf2sw {
    /// The underlying ASGF optimiser.
    pub inner: ASGF,
    /// Streak length at which full 2x boost is reached.
    pub warmup: usize,
    /// Upper cap on the streak value.
    pub max_streak: f64,
    /// Current magnitude-weighted improvement streak.
    improve_streak: f64,
    /// Function value at the previous base point.
    prev_f_base: Option<f64>,
}

impl Asgf2sw {
    /// Create a new ASGF-2SW optimiser.
    ///
    /// All ASGF parameters are passed through to the inner [`ASGF`].
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
        warmup: usize,
        max_streak: f64,
    ) -> Self {
        Self {
            inner: ASGF::new(
                m,
                a,
                b,
                a_minus,
                a_plus,
                b_minus,
                b_plus,
                gamma_l,
                gamma_sigma,
                r,
                ro,
                sigma_zero,
                eps,
            ),
            warmup,
            max_streak,
            improve_streak: 0.0,
            prev_f_base: None,
        }
    }
}

impl Optimizer for Asgf2sw {
    fn kind(&self) -> &'static str {
        "ASGF-2SW"
    }

    fn eps(&self) -> f64 {
        self.inner.eps()
    }

    fn step_size(&self) -> f64 {
        self.inner.step_size()
    }

    fn setup(&mut self, f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, x: &Array1<f64>) {
        self.inner.setup(f, dim, x);
        self.improve_streak = 0.0;
        self.prev_f_base = None;
    }

    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        self.inner.grad_estimator(x, f, rng)
    }

    fn post_iteration(
        &mut self,
        iteration: usize,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        f_val: f64,
    ) {
        self.inner.post_iteration(iteration, x, grad, f_val);
    }

    fn compute_step(
        &mut self,
        x: &Array1<f64>,
        grad: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        maximize: bool,
        f_at_x: Option<f64>,
    ) -> (Array1<f64>, f64) {
        let step_size = self.step_size();
        let direction = if maximize {
            grad.clone()
        } else {
            -grad.clone()
        };

        let x_base = x + &(step_size * &direction);
        let f_base = f(&x_base);

        if !f_base.is_finite() {
            return (x.clone(), f(x));
        }

        // Magnitude-weighted streak update
        if let Some(prev) = self.prev_f_base {
            let denom = prev.abs().max(1e-12);
            if f_base < prev {
                let rel_imp = ((prev - f_base) / denom).min(2.0);
                self.improve_streak += 1.0 + rel_imp;
            } else if f_base > prev {
                let rel_loss = ((f_base - prev) / denom).min(2.0);
                self.improve_streak = (self.improve_streak - (1.0 + rel_loss)).max(0.0);
            }
        }
        self.prev_f_base = Some(f_base);
        self.improve_streak = self.improve_streak.min(self.max_streak);

        // Smooth confidence-based multiplier
        let confidence = (self.improve_streak / self.warmup as f64).min(1.0);
        let k = 1.0 + confidence; // ranges from 1.0 to 2.0

        if confidence > 0.0 && k > 1.01 {
            let x_big = x + &(k * step_size * &direction);
            let f_big = f(&x_big);

            // Safety gate: must beat BOTH base point AND current point
            let f_cur = f_at_x.unwrap_or_else(|| f(x));
            if f_big.is_finite() && f_big < f_base && f_big < f_cur {
                return (x_big, f_big);
            }
        }

        (x_base, f_base)
    }
}

impl Default for Asgf2sw {
    fn default() -> Self {
        Self {
            inner: ASGF::default(),
            warmup: 3,
            max_streak: 10.0,
            improve_streak: 0.0,
            prev_f_base: None,
        }
    }
}
