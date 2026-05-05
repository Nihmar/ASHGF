//! ASGF-2S: frequency-gated step boost with safety and smooth blending.
//!
//! Two fixes over ASGF-2F:
//! 1. **Safety gate**: 2x is only accepted when `f(2x) < f(x_current)`,
//!    preventing escape from already-good points.
//! 2. **Smooth multiplier**: blends from 1.0 to 2.0 proportionally to the
//!    improvement streak, avoiding the zig-zag of binary on/off.

use ndarray::Array1;

use crate::algorithms::asgf::ASGF;
use crate::algorithms::base::Optimizer;
use crate::utils::SeededRng;

/// ASGF with safety-gated, smooth-blended 2x step boost.
///
/// Wraps an [`ASGF`] instance and overrides the step computation to
/// optionally take a larger step when a streak of improvements is
/// observed.  The boost multiplier `k` ranges smoothly from 1.0 to
/// 2.0 based on the improvement streak length.
pub struct Asgf2s {
    /// The underlying ASGF optimiser.
    pub inner: ASGF,
    /// Streak length at which full 2x boost is reached.
    pub warmup: usize,
    /// Current improvement streak (0 = no streak).
    improve_streak: usize,
    /// Function value at the previous base point.
    prev_f_base: Option<f64>,
}

impl Asgf2s {
    /// Create a new ASGF-2S optimiser.
    ///
    /// All ASGF parameters are passed through to the inner [`ASGF`].
    /// `warmup` controls how many consecutive improvements are needed
    /// to reach the full 2x boost.
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
            improve_streak: 0,
            prev_f_base: None,
        }
    }
}

impl Optimizer for Asgf2s {
    fn kind(&self) -> &'static str {
        "ASGF-2S"
    }

    fn eps(&self) -> f64 {
        self.inner.eps()
    }

    fn step_size(&self) -> f64 {
        self.inner.step_size()
    }

    fn setup(&mut self, f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, x: &Array1<f64>) {
        self.inner.setup(f, dim, x);
        self.improve_streak = 0;
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

        // Update improvement streak
        if let Some(prev) = self.prev_f_base {
            if f_base < prev {
                self.improve_streak += 1;
            } else {
                self.improve_streak = self.improve_streak.saturating_sub(1);
            }
        }
        self.prev_f_base = Some(f_base);

        // Smooth confidence-based multiplier
        let confidence = (self.improve_streak as f64 / self.warmup as f64).min(1.0);
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

impl Default for Asgf2s {
    fn default() -> Self {
        Self {
            inner: ASGF::default(),
            warmup: 3,
            improve_streak: 0,
            prev_f_base: None,
        }
    }
}
