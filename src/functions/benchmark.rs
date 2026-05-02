//! Benchmark optimisation test functions (subset from CUTEst / Andrei).
//!
//! The Python reference contains ~48 functions; this module implements
//! a curated subset covering the main structural categories (quadratic,
//! tridiagonal, perturbed, indefinite, etc.).

use ndarray::Array1;

fn cached_arange(n: usize) -> Array1<f64> {
    Array1::from_iter((1..=n).map(|i| i as f64))
}

// ---------------------------------------------------------------------------
// Quadratic / perturbed quadratic family
// ---------------------------------------------------------------------------

pub fn perturbed_quadratic(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (x * x * &i).sum()
}

pub fn almost_perturbed_quadratic(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (i + 1) as f64 * x[i].powi(2);
    }
    total + x.dot(x).powi(2) / 2.0
}

pub fn perturbed_quadratic_diagonal(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    let mut total = i.dot(&x.mapv(|v| v.powi(2)));
    total += (x.sum()).powi(2);
    total
}

// ---------------------------------------------------------------------------
// Diagonal family
// ---------------------------------------------------------------------------

pub fn diagonal_1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (x.mapv(|v| v.exp()) - &i).mapv(|v| v.powi(2)).sum()
}

pub fn diagonal_2(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (x.mapv(|v| v.sin()) - &i.mapv(|v| v / (n as f64)))
        .mapv(|v| v.powi(2))
        .sum()
}

pub fn diagonal_3(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (x.mapv(|v| v.exp() * v.sin()) + x.mapv(|v| v.exp() * v.cos()) - &i.mapv(|v| v.exp() * v.sin()))
        .mapv(|v| v.powi(2))
        .sum()
}

// ---------------------------------------------------------------------------
// Tridiagonal family
// ---------------------------------------------------------------------------

pub fn broyden_tridiagonal(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        let xi = x[i];
        let xj = x[i + 1];
        total += ((3.0 - 2.0 * xi) * xi - xj - 1.0).powi(2);
    }
    // Final term from Python: last residual uses x_n
    for i in 1..n {
        let xi = x[i];
        let xprev = x[i - 1];
        total += ((3.0 - 2.0 * xi) * xi - xprev - 1.0).powi(2);
    }
    total
}

pub fn generalized_tridiagonal_1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let i = cached_arange(n);
    let mut total = 0.0;
    for k in 0..n - 1 {
        let xk = x[k];
        let xk1 = x[k + 1];
        total += (&i * x).sum().powi(2) + (xk + xk1 - 3.0).powi(2) + (xk - xk1 + 1.0).powi(4);
    }
    total
}

// ---------------------------------------------------------------------------
// Indefinite / extended plateau family
// ---------------------------------------------------------------------------

pub fn indef(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        total += (x[i].powi(2) - x[i + 1]).powi(2);
    }
    total + (x[0] - 1.0).powi(2)
}

// ---------------------------------------------------------------------------
// Raydan family
// ---------------------------------------------------------------------------

pub fn raydan_1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (i + 1) as f64 * ((x[i].exp()) - x[i]);
    }
    total
}

pub fn raydan_2(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (i + 1) as f64 * ((x[i].exp()) - x[i].sin());
    }
    total
}

// ---------------------------------------------------------------------------
// Hager
// ---------------------------------------------------------------------------

pub fn hager(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (i * x.mapv(|v| v.exp())).sum()
}

// ---------------------------------------------------------------------------
// BDQRTIC
// ---------------------------------------------------------------------------

pub fn bdqrtic(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        total += (x[i].powi(2) + x[i + 1].powi(2) - 2.0).powi(2);
    }
    total + (x[0].powi(2) - 1.0).powi(2)
}

// ---------------------------------------------------------------------------
// Power / Engval1
// ---------------------------------------------------------------------------

pub fn power(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    (x - &i).mapv(|v| v.powi(2)).sum()
}

pub fn engval1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i].powi(2) + x[(i + 1) % n].powi(2) - 2.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// DQDRTIC / QUARTC
// ---------------------------------------------------------------------------

pub fn dqdrtic(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i] - (i + 1) as f64).powi(4);
    }
    total
}

pub fn quartc(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i] - (i + 1) as f64).powi(4);
    }
    total
}

// ---------------------------------------------------------------------------
// FLETCBV3 / FLETCHCR
// ---------------------------------------------------------------------------

pub fn fletcbv3(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    // Pairwise: xp = even indices, xd = odd indices
    for i in (0..n - 1).step_by(2) {
        let xp = x[i];
        let xd = x[i + 1];
        total += (xp - xd).powi(2) + (xp + xd - 1.0).powi(2);
    }
    total
}

pub fn fletchcr(x: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        total += (x[i + 1] - x[i] + 1.0 - x[i].powi(2)).powi(2);
    }
    total + (x[0].powi(2) - 1.0).powi(2)
}

// ---------------------------------------------------------------------------
// EG2
// ---------------------------------------------------------------------------

pub fn eg2(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i].powi(2) + (i + 1) as f64).sin();
    }
    total
}

// ---------------------------------------------------------------------------
// GENHUMPS
// ---------------------------------------------------------------------------

pub fn genhumps(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        let xi = x[i];
        total += (xi.powi(2).sin()).powi(2) - 0.5;
    }
    total / (1.0 + 0.001 * x.dot(x)).powi(2)
}

// ---------------------------------------------------------------------------
// NONDIA (non-diagonal)
// ---------------------------------------------------------------------------

pub fn nondia(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        let xi = x[i];
        let x_sum: f64 = x.iter().sum();
        total += (xi + x_sum).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// VARDIM
// ---------------------------------------------------------------------------

pub fn vardim(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let i = cached_arange(n);
    let mut total = (x[0] - 1.0).powi(2);
    for k in 1..n {
        total += (k as f64 * (x[k] - 1.0)).powi(2);
    }
    total + (x.dot(x) - 0.25).powi(2)
}

// ---------------------------------------------------------------------------
// Extended penalty variants
// ---------------------------------------------------------------------------

pub fn extended_quadratic_penalty_qp1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i] - (i + 1) as f64).powi(2);
    }
    total + (x.dot(x) - 0.25).powi(2)
}

pub fn extended_quadratic_penalty_qp2(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let mut total = 0.0;
    for i in 0..n {
        total += (x[i] - (i + 1) as f64).powi(2);
    }
    total + (x.dot(x) - 1.0).powi(2)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbed_quadratic_zero() {
        let x = Array1::zeros(5);
        assert!((perturbed_quadratic(&x) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn all_finite() {
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5]);
        assert!(perturbed_quadratic(&x).is_finite());
        assert!(diagonal_1(&x).is_finite());
        assert!(raydan_1(&x).is_finite());
        assert!(bdqrtic(&x).is_finite());
        assert!(nondia(&x).is_finite());
        assert!(vardim(&x).is_finite());
    }
}
