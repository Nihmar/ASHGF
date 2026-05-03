//! Classic optimisation benchmark functions.
//!
//! Each function accepts a 1-D [`ndarray::Array1`] and returns a scalar `f64`.
//! Global minima are documented in the function-level doc comments.

use ndarray::Array1;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Cached index array — avoids re-allocating [1..n] for every function call
// ---------------------------------------------------------------------------

static ARANGE_CACHE: Lazy<Mutex<HashMap<usize, Array1<f64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn cached_arange(n: usize) -> Array1<f64> {
    let mut cache = ARANGE_CACHE.lock().unwrap();
    if let Some(arr) = cache.get(&n) {
        return arr.clone();
    }
    let arr = Array1::from_iter((1..=n).map(|i| i as f64));
    cache.insert(n, arr.clone());
    arr
}

// ---------------------------------------------------------------------------
// Classical functions
// ---------------------------------------------------------------------------

/// Sphere function: `f(x) = Σ x_i²`.  Global minimum `f(0) = 0`.
pub fn sphere(x: &Array1<f64>) -> f64 {
    x.dot(x)
}

/// Rastrigin function:
/// `f(x) = 10n + Σ [x_i² - 10·cos(2π x_i)]`.  Global minimum `f(0) = 0`.
pub fn rastrigin(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x
        .iter()
        .map(|&xi| xi.powi(2) - 10.0 * (2.0 * PI * xi).cos())
        .sum();
    10.0 * n + sum
}

/// Ackley function.  Global minimum `f(0) = 0`.
pub fn ackley(x: &Array1<f64>) -> f64 {
    let a: f64 = 20.0;
    let b: f64 = 0.2;
    let c: f64 = 2.0 * PI;
    let n = x.len() as f64;

    // Single pass: accumulate squares and cosine sums
    let (sum_sq, sum_cos): (f64, f64) = x
        .iter()
        .map(|&v| (v * v, (c * v).cos()))
        .fold((0.0, 0.0), |(s1, s2), (sq, cs)| (s1 + sq, s2 + cs));

    let term1 = -a * (-b * (sum_sq / n).sqrt()).exp();
    let term2 = -(sum_cos / n).exp();
    let term3 = a + std::f64::consts::E;

    term1 + term2 + term3
}

/// Griewank function:
/// `f(x) = (1/4000) Σ x_i² - Π cos(x_i/√i) + 1`.  Global minimum `f(0) = 0`.
pub fn griewank(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let indices = cached_arange(n);

    let term1 = (1.0 / 4000.0) * x.mapv(|v| v.powi(2)).sum();
    let term2: f64 = x
        .iter()
        .zip(indices.iter())
        .map(|(&xi, &i)| (xi / i.sqrt()).cos())
        .product();
    term1 - term2 + 1.0
}

/// Levy function.  Global minimum `f(1) = 0`.
pub fn levy(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let w = x.mapv(|v| 1.0 + (v - 1.0) / 4.0);

    let term1 = (PI * w[0]).sin().powi(2);

    let term2: f64 = w
        .slice(ndarray::s![..n - 1])
        .iter()
        .map(|&wi| (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2)))
        .sum();

    let w_last = w[n - 1];
    let term3 = (w_last - 1.0).powi(2) * (1.0 + (2.0 * PI * w_last).sin().powi(2));

    term1 + term2 + term3
}

/// Schwefel function:
/// `f(x) = 418.9829·n - Σ x_i·sin(√|x_i|)`.
/// Global minimum at `x_i ≈ 420.9687`.
pub fn schwefel(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().map(|&xi| xi * (xi.abs().sqrt()).sin()).sum();
    418.9829 * n - sum
}

/// Sum of different powers: `f(x) = Σ |x_i|^{i+1}`.  Global minimum `f(0) = 0`.
pub fn sum_of_different_powers(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let exponents = cached_arange(n + 1).slice_move(ndarray::s![1..]);
    x.iter()
        .zip(exponents.iter())
        .map(|(&xi, &e)| xi.abs().powf(e))
        .sum()
}

/// Tridiagonal (Trid) function:
/// `f(x) = Σ (x_i - 1)² + Σ x_i·x_{i-1}`.
pub fn trid(x: &Array1<f64>) -> f64 {
    let term1: f64 = x.iter().map(|&xi| (xi - 1.0).powi(2)).sum();
    let term2: f64 = x
        .slice(ndarray::s![1..])
        .iter()
        .zip(x.slice(ndarray::s![..x.len() - 1]).iter())
        .map(|(&a, &b)| a * b)
        .sum();
    term1 + term2
}

/// Zakharov function:
/// `f(x) = Σ x_i² + (½ Σ i·x_i)² + (½ Σ i·x_i)⁴`.  Global minimum `f(0) = 0`.
pub fn zakharov(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let indices = cached_arange(n);
    let half_sum = 0.5 * (indices * x).sum();

    let term1 = x.mapv(|v| v.powi(2)).sum();
    let term2 = half_sum.powi(2);
    let term3 = term2.powi(2);
    term1 + term2 + term3
}

/// Cosine mixture: `f(x) = Σ_{i=1}^{n-1} cos(-0.5 x_{i+1} + x_i²)`.
pub fn cosine(x: &Array1<f64>) -> f64 {
    let n = x.len();
    x.slice(ndarray::s![..n - 1])
        .iter()
        .zip(x.slice(ndarray::s![1..]).iter())
        .map(|(&xi, &xj)| (-0.5 * xj + xi.powi(2)).cos())
        .sum()
}

/// Sine mixture: `f(x) = Σ_{i=1}^{n-1} sin(-0.5 x_{i+1} + x_i²)`.
pub fn sine(x: &Array1<f64>) -> f64 {
    let n = x.len();
    x.slice(ndarray::s![..n - 1])
        .iter()
        .zip(x.slice(ndarray::s![1..]).iter())
        .map(|(&xi, &xj)| (-0.5 * xj + xi.powi(2)).sin())
        .sum()
}

/// Sine-Cosine mixture over pairs `(x_{2k}, x_{2k+1})`.
pub fn sincos(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let m = xp.len().min(xd.len());

    let mut total = 0.0;
    for i in 0..m {
        let xp_i = xp[i];
        let xd_i = xd[i];
        let term1 = (xp_i.powi(2) + xd_i.powi(2) + xp_i * xd_i).powi(2);
        let term2 = xp_i.sin().powi(2);
        let term3 = xd_i.cos().powi(2);
        total += term1 + term2 + term3;
    }
    total
}

// ---------------------------------------------------------------------------
// Activation-style primitives (return arrays)
// ---------------------------------------------------------------------------

/// ReLU activation: `max(0, x_i)` element-wise.
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Softmax: numerically stable.
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sphere_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(sphere(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn rastrigin_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(rastrigin(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn ackley_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(ackley(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn griewank_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(griewank(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn levy_one() {
        let ones = Array1::ones(5);
        assert_abs_diff_eq!(levy(&ones), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn sum_of_different_powers_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(sum_of_different_powers(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn zakharov_zero() {
        let x = Array1::zeros(5);
        assert_abs_diff_eq!(zakharov(&x), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn sphere_known_value() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_abs_diff_eq!(sphere(&x), 14.0, epsilon = 1e-12);
    }
}
