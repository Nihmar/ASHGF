//! Extended optimisation test functions (pairwise / chain variants).

use ndarray::Array1;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn cached_arange(n: usize) -> Array1<f64> {
    Array1::from_iter((1..=n).map(|i| i as f64))
}

// ---------------------------------------------------------------------------
// 1. extended_feudenstein_and_roth
// ---------------------------------------------------------------------------

pub fn extended_feudenstein_and_roth(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        let inner = ((5.0 - b) * b - 2.0) * b;
        let term1 = (-13.0 + a + inner).powi(2);
        let term2 = (-29.0 + a + inner).powi(2);
        total += term1 + term2;
    }
    total
}

// ---------------------------------------------------------------------------
// 2. extended_trigonometric
// ---------------------------------------------------------------------------

pub fn extended_trigonometric(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let cos_x = x.mapv(|v| v.cos());
    let indices = cached_arange(n);

    let term1 = n as f64 - cos_x.sum();
    let term2 = &indices * &(1.0 - &cos_x);
    let term3 = x.mapv(|v| v.sin());

    (&term2 + &term3 + term1).mapv(|v| v.powi(2)).sum()
}

// ---------------------------------------------------------------------------
// 3. extended_rosenbrock
// ---------------------------------------------------------------------------

pub fn extended_rosenbrock(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let c = 100.0;
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += c * (b - a.powi(2)).powi(2) + (1.0 - a).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 4. generalized_rosenbrock (chain)
// ---------------------------------------------------------------------------

pub fn generalized_rosenbrock(x: &Array1<f64>) -> f64 {
    let c = 100.0;
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        let xi = x[i];
        let xj = x[i + 1];
        total += c * (xj - xi.powi(2)).powi(2) + (1.0 - xi).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 5. extended_white_and_holst
// ---------------------------------------------------------------------------

pub fn extended_white_and_holst(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let c = 100.0;
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += c * (b - a.powi(3)).powi(2) + (1.0 - a).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 6. extended_baele
// ---------------------------------------------------------------------------

pub fn extended_baele(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (1.5 - a * (1.0 - b)).powi(2)
            + (2.25 - a * (1.0 - b.powi(2))).powi(2)
            + (2.625 - a * (1.0 - b.powi(3))).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 7. extended_penalty
// ---------------------------------------------------------------------------

pub fn extended_penalty(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let term1: f64 = x
        .slice(ndarray::s![..n - 1])
        .iter()
        .map(|&v| (v - 1.0).powi(2))
        .sum();
    let term2 = (x.mapv(|v| v.powi(2)).sum() - 0.25).powi(2);
    term1 + term2
}

// ---------------------------------------------------------------------------
// 8. extended_himmelblau
// ---------------------------------------------------------------------------

pub fn extended_himmelblau(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (a.powi(2) + b - 11.0).powi(2) + (a + b.powi(2) - 7.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 9. generalized_white_and_holst (chain)
// ---------------------------------------------------------------------------

pub fn generalized_white_and_holst(x: &Array1<f64>) -> f64 {
    let c = 100.0;
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        let xi = x[i];
        let xj = x[i + 1];
        total += c * (xj - xi.powi(3)).powi(2) + (1.0 - xi).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 10. extended_psc1
// ---------------------------------------------------------------------------

pub fn extended_psc1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (a.powi(2) + b.powi(2) + a * b).powi(2) + a.sin().powi(2) + b.cos().powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 11. extended_bd1
// ---------------------------------------------------------------------------

pub fn extended_bd1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (a.powi(2) + b - 2.0).powi(2) + ((a - 1.0).exp() - a).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 12. extended_maratos
// ---------------------------------------------------------------------------

pub fn extended_maratos(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let c = 100.0;
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += a + c * (a.powi(2) + b.powi(2) - 1.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 13. extended_cliff
// ---------------------------------------------------------------------------

pub fn extended_cliff(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        let cliff_exp = (20.0 * (a - b)).clamp(-100.0, 100.0).exp();
        total += ((a - 3.0) / 100.0).powi(2) + (a - b) + cliff_exp;
    }
    total
}

// ---------------------------------------------------------------------------
// 14. extended_hiebert
// ---------------------------------------------------------------------------

pub fn extended_hiebert(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (a - 10.0).powi(2) + (a * b - 50000.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 15. extended_tridiagonal_1
// ---------------------------------------------------------------------------

pub fn extended_tridiagonal_1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        total += (a + b - 3.0).powi(2) + (a - b + 1.0).powi(4);
    }
    total
}

// ---------------------------------------------------------------------------
// 16. extended_tridiagonal_2 (chain)
// ---------------------------------------------------------------------------

pub fn extended_tridiagonal_2(x: &Array1<f64>) -> f64 {
    let c = 0.1;
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n - 1 {
        let xi = x[i];
        let xj = x[i + 1];
        total += (xj * xi - 1.0).powi(2) + c * (xi + 1.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// 17. extended_denschnb
// ---------------------------------------------------------------------------

pub fn extended_denschnb(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        let term1 = (a - 2.0).powi(2);
        let term2 = term1 * b.powi(2);
        let term3 = (b + 1.0).powi(2);
        total += term1 + term2 + term3;
    }
    total
}

// ---------------------------------------------------------------------------
// 18. extended_denschnf
// ---------------------------------------------------------------------------

pub fn extended_denschnf(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        let term1 = 2.0 * (a + b).powi(2);
        let term2 = (a - b).powi(2) - 8.0;
        let term3 = (5.0 * a.powi(2) + (a - 3.0).powi(2) - 9.0).powi(2);
        total += (term1 + term2).powi(2) + term3;
    }
    total
}

// ---------------------------------------------------------------------------
// 19. extended_quadratic_exponential_ep1
// ---------------------------------------------------------------------------

pub fn extended_quadratic_exponential_ep1(x: &Array1<f64>) -> f64 {
    let n = x.len();
    let xp = x.slice(ndarray::s![0..n-1;2]);
    let xd = x.slice(ndarray::s![1..n;2]);
    let mut total = 0.0;
    for i in 0..xp.len().min(xd.len()) {
        let a = xp[i];
        let b = xd[i];
        let diff = a - b;
        total += (diff.exp() - 5.0).powi(2) + diff.powi(2) * (diff - 11.0).powi(2);
    }
    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock_global_min() {
        // Extended Rosenbrock minimum at xp=1, xd=1 for all pairs
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        assert!((extended_rosenbrock(&x) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn penalty_is_finite() {
        let x = Array1::zeros(10);
        assert!(extended_penalty(&x).is_finite());
    }

    #[test]
    fn himmelblau_known_min() {
        // One of the four minima: (3.0, 2.0)
        let x = Array1::from_vec(vec![3.0, 2.0]);
        assert!((extended_himmelblau(&x) - 0.0).abs() < 1e-8);
    }
}
