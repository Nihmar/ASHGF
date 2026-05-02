//! Parallel (and serial) evaluation of an objective function on
//! a batch of points.

use ndarray::Array1;
use rayon::prelude::*;

/// Evaluate `f` on each point in `points`, optionally in parallel.
///
/// When `n_jobs > 1` and there are at least 4 points, evaluation is
/// distributed across threads via [`rayon`].  Otherwise a plain serial
/// loop is used to avoid thread-spawning overhead.
pub fn parallel_eval<F>(f: &F, points: &[Array1<f64>], n_jobs: usize) -> Vec<f64>
where
    F: Fn(&Array1<f64>) -> f64 + Sync + ?Sized,
{
    if n_jobs <= 1 || points.len() < 4 {
        points.iter().map(|p| f(p)).collect()
    } else {
        points.par_iter().map(|p| f(p)).collect()
    }
}
