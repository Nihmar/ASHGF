//! Gradient estimators: Gaussian smoothing, Gauss-Hermite quadrature,
//! and Lipschitz-constant estimation.
//!
//! Uses the `gauss-quad` crate for quadrature nodes and weights
//! (probabilist's convention, weight `e^{-x²/2}`), avoiding the
//! `√2` factor ambiguity present in the Python reference.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};
use std::sync::Mutex;

use crate::utils::parallel::parallel_eval;
use once_cell::sync::Lazy;

// ---------------------------------------------------------------------------
// Cache for Gauss-Hermite quadrature
// ---------------------------------------------------------------------------

/// Cached (nodes, p_w, weights) for a given number of points `m`.
static GH_CACHE: Lazy<Mutex<HashMap<usize, (Array1<f64>, Array1<f64>, Array1<f64>)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Get `(p_nodes, p_w, weights)` for `m`-point Gauss-Hermite quadrature.
///
/// `gauss-quad` provides physicist's nodes/weights (weight `e^{-x²}`).
/// We convert to the probabilist's convention (weight `e^{-x²/2}`):
///   v_k = √2 · p_k,   w_k^{prob} = √2 · w_k^{phys}
///
/// Returns:
/// - `p_nodes`: probabilist's quadrature nodes `v_k`
/// - `p_w`: pre-multiplied `v_k * w_k` (for the derivative formula)
/// - `weights`: probabilist's weights `w_k`
fn get_gauss_hermite(m: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut cache = GH_CACHE.lock().unwrap();
    if let Some(entry) = cache.get(&m) {
        return entry.clone();
    }

    let quad = gauss_quad::hermite::GaussHermite::init(m);
    // Convert physicist's → probabilist's
    let p_nodes = Array1::from_vec(quad.nodes) * SQRT_2;
    let w = Array1::from_vec(quad.weights) * SQRT_2;
    let p_w = &p_nodes * &w;

    let result = (p_nodes.clone(), p_w, w);
    cache.insert(m, result.clone());
    result
}

// ---------------------------------------------------------------------------
// Gaussian smoothing (central finite difference)
// ---------------------------------------------------------------------------

/// Estimate the gradient via central Gaussian smoothing.
///
/// `g ≈ (1/(2σ·M)) Σ_j [f(x+σ·d_j) - f(x-σ·d_j)] · d_j`
///
/// Parameters
/// ----------
/// x : Array1<f64>
///     Point at which to estimate the gradient.
/// f : &dyn Fn(&Array1<f64>) -> f64 + Sync
///     Objective function.
/// sigma : f64
///     Smoothing bandwidth.
/// directions : Array2<f64>
///     Matrix of directions (rows).
/// n_jobs : usize
///     Number of parallel threads.
///
/// Returns
/// -------
/// grad : Array1<f64>
///     Estimated gradient vector.
pub fn gaussian_smoothing<F>(
    x: &Array1<f64>,
    f: &F,
    sigma: f64,
    directions: &Array2<f64>,
    n_jobs: usize,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64 + Sync + ?Sized,
{
    let m = directions.nrows();
    let _dim = x.len();

    let sigma_dirs = directions.mapv(|v| sigma * v);

    // Build all perturbed points (interleaved + and -)
    let mut points: Vec<Array1<f64>> = Vec::with_capacity(2 * m);
    for j in 0..m {
        let d = sigma_dirs.row(j).to_owned();
        points.push(x + &d);
        points.push(x - &d);
    }

    let results = parallel_eval(f, &points, n_jobs);
    let results_arr = Array1::from_vec(results);

    // diff[j] = f(x + σ·d_j) - f(x - σ·d_j)
    let diff =
        results_arr.slice(ndarray::s![0..;2]).to_owned() - &results_arr.slice(ndarray::s![1..;2]);

    // grad = Σ diff_j · d_j / (2σM)
    let grad = diff.dot(directions) / (2.0 * sigma * m as f64);

    grad
}

// ---------------------------------------------------------------------------
// Gauss-Hermite quadrature
// ---------------------------------------------------------------------------

/// Estimate directional derivatives using Gauss-Hermite quadrature.
///
/// For each basis direction `b_i` (row of `basis`):
///
/// `D_i f(x) ≈ (1/σ)·(1/√(2π)) Σ_k w_k · v_k · f(x + σ·v_k·b_i)`
///
/// where `(v_k, w_k)` are the nodes and weights of the m-point
/// Gauss-Hermite quadrature (probabilist's convention).
///
/// Parameters
/// ----------
/// x : Array1<f64>
///     Point at which to estimate.
/// f : &dyn Fn(&Array1<f64>) -> f64 + Sync
///     Objective function.
/// sigma : f64
///     Smoothing bandwidth.
/// basis : Array2<f64>
///     Orthonormal basis matrix (each row is a direction).
/// m : usize
///     Number of quadrature points (must be odd).
/// value_at_x : Option<f64>
///     `f(x)`. If `None`, it will be evaluated.
///
/// Returns
/// -------
/// grad : Array1<f64>
///     Estimated gradient `∇f(x)`.
/// evals_matrix : Array2<f64>
///     Function evaluations (d × m): row i = evaluations along direction i.
/// nodes : Array1<f64>
///     Quadrature nodes (shared across all directions).
/// derivatives : Array1<f64>
///     Estimated directional derivatives `D_i f(x)`.
pub fn gauss_hermite_derivative<F>(
    x: &Array1<f64>,
    f: &F,
    sigma: f64,
    basis: &Array2<f64>,
    m: usize,
    value_at_x: Option<f64>,
    n_jobs: usize,
) -> (Array1<f64>, Array2<f64>, Array1<f64>, Array1<f64>)
where
    F: Fn(&Array1<f64>) -> f64 + Sync + ?Sized,
{
    assert!(m % 2 == 1, "m must be odd for central node at zero");
    let dim = x.len();
    let fx = value_at_x.unwrap_or_else(|| f(x));

    let (p_nodes, p_w, _weights) = get_gauss_hermite(m);
    let mid = m / 2;

    // Scale factor: (1/σ)·(1/√(2π))
    let quad_scale = 1.0 / (sigma * (2.0 * PI).sqrt());

    // ---- Build all non-central perturbed points ----
    let sigma_nodes = sigma * &p_nodes;
    let n_perturbed = dim * (m - 1);

    let mut points: Vec<Array1<f64>> = Vec::with_capacity(n_perturbed);
    for i in 0..dim {
        let bi = basis.row(i);
        for k in 0..m {
            if k == mid {
                continue;
            }
            points.push(x + &(sigma_nodes[k] * &bi));
        }
    }

    let flat_results = parallel_eval(f, &points, n_jobs);

    // ---- Build evaluation matrix: (d, m) ----
    let mut evals_matrix = Array2::<f64>::zeros((dim, m));
    evals_matrix.column_mut(mid).fill(fx);
    for i in 0..dim {
        for (j, k) in (0..m).filter(|&k| k != mid).enumerate() {
            evals_matrix[(i, k)] = flat_results[i * (m - 1) + j];
        }
    }

    // ---- Compute directional derivatives ----
    // D_i = quad_scale · Σ_k p_w[k] · evals[i, k]
    // Since p_w[mid] == 0 (v_mid == 0), fx contributes nothing.
    let derivatives = quad_scale * evals_matrix.dot(&p_w);

    // ---- Gradient reconstruction: ∇f = B^T · D ----
    let grad = basis.t().dot(&derivatives);

    (grad, evals_matrix, p_nodes, derivatives)
}

// ---------------------------------------------------------------------------
// Lipschitz-constant estimation
// ---------------------------------------------------------------------------

/// Estimate directional Lipschitz constants from quadrature data.
///
/// Implements the thesis formula:
///
/// `L_j = max_{i,k ∈ I} |F(x+σ·p_i·ξ_j) - F(x+σ·p_k·ξ_j)| / (σ·|p_i - p_k|)`
///
/// where `I` excludes pairs symmetric about the central node:
/// `I = {{i,k} : |i - ⌊m/2⌋| ≠ |k - ⌊m/2⌋|}`.
///
/// Parameters
/// ----------
/// evaluations : Array2<f64>, shape (d, m)
///     Function evaluations along each direction.
/// nodes : Array1<f64>, shape (m,)
///     Quadrature nodes.
/// sigma : f64
///     Smoothing bandwidth.
///
/// Returns
/// -------
/// lipschitz : Array1<f64>, shape (d,)
///     Estimated Lipschitz constants per direction.
pub fn estimate_lipschitz_constants(
    evaluations: &Array2<f64>,
    nodes: &Array1<f64>,
    sigma: f64,
) -> Array1<f64> {
    let (_d, m) = evaluations.dim();
    let mid = m / 2;

    // Build all unordered pairs (i, k) with i < k, excluding symmetric pairs
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..m {
        for k in (i + 1)..m {
            let di = if i >= mid { i - mid } else { mid - i };
            let dk = if k >= mid { k - mid } else { mid - k };
            if di != dk {
                pairs.push((i, k));
            }
        }
    }

    if pairs.is_empty() {
        return Array1::zeros(_d);
    }

    // Vectorised over directions and pairs
    let mut lipschitz = Array1::<f64>::zeros(_d);
    for d_idx in 0.._d {
        let evals_row = evaluations.row(d_idx);
        let mut max_ratio: f64 = 0.0;
        for &(i, k) in &pairs {
            let diff_evals = (evals_row[i] - evals_row[k]).abs();
            let diff_pts = sigma * (nodes[i] - nodes[k]).abs();
            if diff_pts > 1e-14 {
                let ratio = diff_evals / diff_pts;
                max_ratio = max_ratio.max(ratio);
            }
        }
        lipschitz[d_idx] = max_ratio;
    }
    lipschitz
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::sampling::random_orthogonal;
    use crate::utils::SeededRng;
    use approx::assert_abs_diff_eq;

    /// Sphere: f(x) = ‖x‖², ∇f = 2x
    fn sphere(x: &Array1<f64>) -> f64 {
        x.dot(x)
    }

    #[test]
    fn gaussian_smoothing_sphere() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let n_runs = 1000;
        let mut avg_grad = Array1::zeros(3);
        for seed in 0..n_runs {
            let mut rng = SeededRng::new(seed);
            let dirs = crate::gradient::sampling::compute_directions(3, &mut rng);
            let grad = gaussian_smoothing(&x, &sphere, 1e-6, &dirs, 1);
            avg_grad = &avg_grad + &grad;
        }
        avg_grad /= n_runs as f64;
        let expected = 2.0 * &x;
        for i in 0..3 {
            assert_abs_diff_eq!(avg_grad[i], expected[i], epsilon = 3e-1);
        }
    }

    #[test]
    fn gh_derivative_sphere() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut rng = SeededRng::new(42);
        let basis = random_orthogonal(3, &mut rng);
        let (grad, _, _, _derivs) = gauss_hermite_derivative(&x, &sphere, 0.1, &basis, 5, None, 1);
        let expected = 2.0 * &x;
        // GH should be more accurate than finite diff
        for i in 0..3 {
            assert_abs_diff_eq!(grad[i], expected[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn lipschitz_sphere_is_positive() {
        let x = Array1::zeros(3);
        let mut rng = SeededRng::new(42);
        let basis = random_orthogonal(3, &mut rng);
        let (_, evals, nodes, _) = gauss_hermite_derivative(&x, &sphere, 0.1, &basis, 5, None, 1);
        let lips = estimate_lipschitz_constants(&evals, &nodes, 0.1);
        // For sphere, Lipschitz constant ≈ 2
        for l in lips.iter() {
            assert!(l.is_finite() && *l > 0.0);
        }
    }
}
