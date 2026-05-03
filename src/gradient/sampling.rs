//! Direction sampling strategies for gradient estimation.
//!
//! Provides functions for generating search directions used in
//! gradient-free optimisation:
//!
//! * Pure random (Gaussian isotropic) directions
//! * SGES-style adaptive mixing of gradient-history and random directions
//! * ASHGF-style directions (delegates to SGES)

use ndarray::Array2;
use ndarray_linalg::cholesky::UPLO;
use ndarray_linalg::{CholeskyInto, Norm, QR};
use rand::Rng;
use rand_distr::{Binomial, Distribution, StandardNormal};
use std::f64;

use crate::utils::SeededRng;

// ---------------------------------------------------------------------------
// Random orthogonal basis
// ---------------------------------------------------------------------------

/// Generate a random `(dim, dim)` orthonormal matrix via QR decomposition.
///
/// Draws i.i.d. standard-normal entries, then computes the Q factor.
/// The resulting matrix is uniformly distributed over the orthogonal
/// group O(d), which is sufficient for direction sampling.
pub fn random_orthogonal(dim: usize, rng: &mut SeededRng) -> Array2<f64> {
    let m: Array2<f64> = Array2::from_shape_fn((dim, dim), |_| rng.rng.sample(StandardNormal));
    // QR: m = Q @ R  →  Q has orthonormal columns
    let (q, _r) = m.qr().expect("QR decomposition failed");
    q
}

// ---------------------------------------------------------------------------
// Pure random directions
// ---------------------------------------------------------------------------

/// Generate a `dim × dim` matrix of i.i.d. standard-normal directions.
///
/// Rows are directions drawn from N(0, I).  They are **not** normalised.
pub fn compute_directions(dim: usize, rng: &mut SeededRng) -> Array2<f64> {
    Array2::from_shape_fn((dim, dim), |_| rng.rng.sample(StandardNormal))
}

// ---------------------------------------------------------------------------
// SGES-style directions
// ---------------------------------------------------------------------------

/// Generate directions mixing gradient-subspace and isotropic components.
///
/// Parameters
/// ----------
/// dim : usize
///     Ambient space dimension.
/// G : Array2<f64>
///     Gradient-history buffer; each row is a past gradient estimate.
/// alpha : f64
///     Probability of sampling a direction from the **random** (isotropic)
///     subspace.  `1 - alpha` is the probability of sampling from the
///     gradient subspace.  Must be in [0, 1].
/// rng : &mut SeededRng
///     Random number generator.
///
/// Returns
/// -------
/// dirs : Array2<f64>, shape (dim, dim)
///     Orthonormal matrix whose rows are unit-norm directions.
/// choices : usize
///     Number of directions actually sampled from the gradient subspace.
pub fn compute_directions_sges(
    dim: usize,
    g: &Array2<f64>,
    alpha: f64,
    rng: &mut SeededRng,
) -> (Array2<f64>, usize) {
    assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0,1]");

    // Empirical covariance of gradient buffer (column-wise)
    // g is (T, dim) → covariance is (dim, dim)
    let g_mean = g.mean_axis(ndarray::Axis(0)).unwrap();
    let g_centered = g - &g_mean.insert_axis(ndarray::Axis(0));
    let cov = g_centered.t().dot(&g_centered) / (g.nrows() as f64).max(1.0);

    // Add Tikhonov regularisation
    let eye = Array2::<f64>::eye(dim);
    let cov_reg = cov + 1e-6 * &eye;

    // How many directions from gradient subspace?
    let binom = Binomial::new(dim as u64, 1.0 - alpha).unwrap();
    let choices = binom.sample(&mut rng.rng) as usize;
    let choices = choices.min(dim);

    // --- Gradient-subspace directions ---
    let mut dirs_grad: Array2<f64> = Array2::zeros((choices, dim));
    if choices > 0 {
        // Cholesky: cov_reg = L @ L^T
        let cholesky = cov_reg
            .clone()
            .cholesky_into(UPLO::Lower)
            .unwrap_or_else(|_| {
                // Fallback: perturb diagonal more aggressively
                (cov_reg + 0.01 * &eye)
                    .cholesky_into(UPLO::Lower)
                    .unwrap_or(eye)
            });
        // Z @ L^T  →  rows are N(0, cov_reg)
        let z = Array2::from_shape_fn((choices, dim), |_| rng.rng.sample(StandardNormal));
        dirs_grad = z.dot(&cholesky.t());
    }

    // --- Random (isotropic) directions ---
    let n_random = dim - choices;
    let dirs_random = Array2::from_shape_fn((n_random, dim), |_| rng.rng.sample(StandardNormal));

    // --- Assemble ---
    let mut dirs: Array2<f64>;
    if choices > 0 {
        dirs = ndarray::concatenate(ndarray::Axis(0), &[dirs_grad.view(), dirs_random.view()])
            .unwrap();
    } else {
        dirs = dirs_random;
    }

    // Normalize every direction to unit length
    for mut row in dirs.rows_mut() {
        let nrm = row.norm();
        if nrm > 1e-12 {
            row /= nrm;
        } else {
            row.fill(1.0 / (dim as f64).sqrt());
        }
    }

    (dirs, choices)
}

// ---------------------------------------------------------------------------
// ASHGF-style directions (delegates to SGES)
// ---------------------------------------------------------------------------

/// Generate directions for ASHGF (currently delegates to SGES).
pub fn compute_directions_ashgf(
    dim: usize,
    g: &Array2<f64>,
    alpha: f64,
    _m: usize,
    rng: &mut SeededRng,
) -> (Array2<f64>, usize) {
    compute_directions_sges(dim, g, alpha, rng)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn random_orthogonal_is_orthonormal() {
        let mut rng = SeededRng::new(42);
        let q = random_orthogonal(10, &mut rng);
        let eye = q.dot(&q.t());
        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(eye[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn compute_directions_shape() {
        let mut rng = SeededRng::new(42);
        let dirs = compute_directions(10, &mut rng);
        assert_eq!(dirs.shape(), &[10, 10]);
    }

    #[test]
    fn compute_directions_sges_normalized() {
        let mut rng = SeededRng::new(42);
        let g = Array2::from_shape_fn((5, 10), |_| rng.rng.sample(StandardNormal));
        let (dirs, _) = compute_directions_sges(10, &g, 0.5, &mut rng);
        for row in dirs.rows() {
            let nrm = row.dot(&row).sqrt();
            assert_abs_diff_eq!(nrm, 1.0, epsilon = 1e-10);
        }
    }
}
