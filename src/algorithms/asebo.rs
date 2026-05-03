//! ASEBO: Adaptive Evolution Strategies with Active Subspaces.
//!
//! Uses PCA (via SVD) on the gradient-history buffer to identify a
//! low-dimensional *active subspace*, then samples search directions
//! from a blended covariance matrix.
//!
//! Mathematical overview (see thesis / Python docstring):
//!
//! Let G ∈ R^{T×d} be the buffer of past gradient estimates.  After
//! a warm-up of `k` iterations, we perform PCA on G and retain the
//! first `r` components such that cumulative explained variance ≥
//! threshold.  The covariance used for direction sampling is:
//!
//!   Σ = σ · [ (α/d)·I_d  +  ((1-α)/r)·U_r^T U_r ] + λ·σ·I_d
//!
//! Directions are drawn from N(0, Σ) and the gradient is estimated
//! via antithetic Gaussian smoothing (directions are NOT normalised).

use ndarray::{Array1, Array2};
use ndarray_linalg::{CholeskyInto, SVDInto, UPLO};

use crate::algorithms::base::Optimizer;
use crate::utils::{parallel_eval, SeededRng};
use rand::Rng;
use rand_distr::StandardNormal;

/// Adaptive Evolution Strategies with Active Subspaces.
pub struct ASEBO {
    /// Fixed learning rate.
    pub lr: f64,
    /// Base smoothing bandwidth.
    pub sigma: f64,
    /// Warm-up iterations before first PCA.
    pub k: usize,
    /// Tikhonov regularisation constant (added to diagonal as λ·σ·I).
    pub lambd: f64,
    /// Explained-variance-ratio threshold for active subspace dimension.
    pub thresh: f64,
    /// Maximum size of the gradient-history circular buffer.
    pub buffer_size: usize,
    /// Number of parallel threads for function evaluation.
    pub n_jobs: usize,
    eps: f64,

    // -- Internal state --
    /// Circular buffer: (buffer_size, dim).
    g_buffer: Option<Array2<f64>>,
    g_idx: usize,
    g_count: usize,
    /// Current blending parameter α ∈ [0, 1].
    current_alpha: f64,
    /// Cached active-subspace basis U_active: (r, dim).
    u_active: Option<Array2<f64>>,
    /// Number of active components r.
    n_components: usize,
    /// Counter for periodic full PCA refit.
    refit_counter: usize,
    /// Number of Monte Carlo directions M.
    m_dir: usize,
}

impl ASEBO {
    pub fn new(
        lr: f64,
        sigma: f64,
        k: usize,
        lambd: f64,
        thresh: f64,
        buffer_size: usize,
        eps: f64,
    ) -> Self {
        assert!(lr > 0.0);
        assert!(sigma > 0.0);
        assert!(k >= 2);
        assert!(buffer_size >= k);
        assert!((0.0..=1.0).contains(&lambd));
        assert!(thresh > 0.0 && thresh < 1.0);
        Self {
            lr,
            sigma,
            k,
            lambd,
            thresh,
            buffer_size,
            n_jobs: 0,
            eps,
            g_buffer: None,
            g_idx: 0,
            g_count: 0,
            current_alpha: 1.0,
            u_active: None,
            n_components: 0,
            refit_counter: 0,
            m_dir: 0,
        }
    }

    /// Perform full PCA via SVD on the gradient buffer.
    /// Returns (U_active, n_components) where U_active is (r, dim).
    fn pca_fit(&self, g_valid: &Array2<f64>, dim: usize) -> (Array2<f64>, usize) {
        let t = g_valid.nrows();
        if t < 2 {
            return (Array2::zeros((0, dim)), 0);
        }

        // Center the data (column-wise mean)
        let mean = g_valid.mean_axis(ndarray::Axis(0)).unwrap();
        let g_centered = g_valid - &mean.insert_axis(ndarray::Axis(0));

        // SVD: G_centered = U Σ V^T  (economy size)
        // V is (dim, dim) with columns = principal directions
        let (_u, sigma, vt_opt) = match g_centered.svd_into(true, true) {
            Ok((u, s, v)) => (u, s, v),
            Err(_) => {
                return (Array2::zeros((0, dim)), 0);
            }
        };

        let vt = match vt_opt {
            Some(v) => v,
            None => return (Array2::zeros((0, dim)), 0),
        };

        // Explained variance from singular values squared
        let var_total: f64 = sigma.iter().map(|&s| s * s).sum();
        if var_total < 1e-30 {
            return (Array2::zeros((0, dim)), 0);
        }

        let var_cumsum: Vec<f64> = sigma
            .iter()
            .map(|&s| s * s / var_total)
            .scan(0.0, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        // Find first index where cumulative variance >= threshold
        let r = var_cumsum
            .iter()
            .position(|&v| v >= self.thresh)
            .map(|p| p + 1)
            .unwrap_or(sigma.len());

        // Clamp: at least 10, at most dim
        let r = r.max(10).min(dim);

        // U_active = first r rows of V^T (= first r columns of V)
        let u_active = vt.slice(ndarray::s![..r, ..]).to_owned();

        (u_active, r)
    }
}

impl Optimizer for ASEBO {
    fn kind(&self) -> &'static str {
        "ASEBO"
    }

    fn eps(&self) -> f64 {
        self.eps
    }

    fn step_size(&self) -> f64 {
        self.lr
    }

    fn setup(&mut self, _f: &(dyn Fn(&Array1<f64>) -> f64 + Sync), dim: usize, _x: &Array1<f64>) {
        self.g_buffer = Some(Array2::zeros((self.buffer_size, dim)));
        self.g_idx = 0;
        self.g_count = 0;
        self.current_alpha = 1.0;
        self.u_active = None;
        self.n_components = 0;
        self.refit_counter = 0;
        self.m_dir = dim;
    }

    fn post_iteration(
        &mut self,
        _iteration: usize,
        _x: &Array1<f64>,
        _grad: &Array1<f64>,
        _f_val: f64,
    ) {
        // ASEBO adapts α inside grad_estimator; no post-iteration logic needed.
    }

    fn grad_estimator(
        &mut self,
        x: &Array1<f64>,
        f: &(dyn Fn(&Array1<f64>) -> f64 + Sync),
        rng: &mut SeededRng,
    ) -> Array1<f64> {
        let dim = x.len();
        let refit_interval = 20;
        let n_jobs = if self.n_jobs > 0 {
            self.n_jobs
        } else {
            rayon::current_num_threads()
        };

        // ==========================================================
        // 1. Determine active-subspace via PCA (after warm-up)
        // ==========================================================
        let (u_active, n_components, m_dir) = if self.g_count >= self.k {
            self.refit_counter += 1;

            let g_valid = self
                .g_buffer
                .as_ref()
                .unwrap()
                .slice(ndarray::s![..self.g_count, ..]);

            if self.refit_counter % refit_interval == 1 || self.u_active.is_none() {
                // Full PCA refit
                let (ua, nc) = self.pca_fit(&g_valid.to_owned(), dim);
                self.u_active = if nc > 0 { Some(ua.clone()) } else { None };
                self.n_components = nc;
            }

            let ua = self.u_active.clone();
            let nc = self.n_components;
            let m = if self.g_count == self.k {
                dim
            } else {
                nc.max(10)
            };

            (ua, nc, m)
        } else {
            // Warm-up: isotropic
            (None, dim, dim)
        };

        self.m_dir = m_dir;

        // ==========================================================
        // 2. Build blended covariance
        //    Σ = σ·[(α/d)·I + ((1-α)/r)·P_active] + λ·σ·I
        // ==========================================================
        // P_active = U_active^T · U_active  (dim, dim)
        let p_active: Array2<f64> = if let Some(ref ua) = u_active {
            if ua.nrows() > 0 {
                ua.t().dot(ua)
            } else {
                Array2::zeros((dim, dim))
            }
        } else {
            Array2::zeros((dim, dim))
        };

        let r_eff = if n_components > 0 { n_components } else { 1 };
        let eye = Array2::<f64>::eye(dim);
        let cov_iso = &eye * (self.current_alpha / dim as f64);
        let cov_active = &p_active * ((1.0 - self.current_alpha) / r_eff as f64);
        let mut cov = (cov_iso + cov_active) * self.sigma;

        // Tikhonov regularisation
        if self.lambd > 0.0 {
            cov = cov + &eye * (self.lambd * self.sigma);
        }

        // ==========================================================
        // 3. Sample directions from N(0, Σ) via Cholesky
        //    A = L @ Z^T  →  rows of A^T = directions (M, dim)
        // ==========================================================
        let a: Array2<f64> = match cov.cholesky_into(UPLO::Lower) {
            Ok(l) => {
                let z = Array2::from_shape_fn((dim, m_dir), |_| rng.rng.sample(StandardNormal));
                let az = l.dot(&z);
                az.t().to_owned() // (M, dim)
            }
            Err(_) => {
                tracing::warn!("ASEBO: Cholesky failed — falling back to isotropic directions.");
                Array2::from_shape_fn((m_dir, dim), |_| rng.rng.sample(StandardNormal))
            }
        };

        // ==========================================================
        // 4. Antithetic gradient estimation
        // ==========================================================
        let sigma_a = &a * self.sigma; // (M, dim) pre-scaled
        let mut points: Vec<Array1<f64>> = Vec::with_capacity(2 * m_dir);
        for j in 0..m_dir {
            let d = sigma_a.row(j).to_owned();
            points.push(x + &d);
            points.push(x - &d);
        }

        let results: Vec<f64> = parallel_eval(f, &points, n_jobs);
        let results_arr = Array1::from_vec(results);

        // diff[j] = f(x + σ·d_j) - f(x - σ·d_j)
        let diff = results_arr.slice(ndarray::s![0..;2]).to_owned()
            - &results_arr.slice(ndarray::s![1..;2]);

        // grad = Σ diff_j · a_j / (2σ·M)
        let grad = diff.dot(&a) / (2.0 * self.sigma * m_dir as f64);

        // ==========================================================
        // 5. Update circular gradient-history buffer (with decay)
        // ==========================================================
        if let Some(ref mut buf) = self.g_buffer {
            // Exponential decay on existing entries
            if self.g_count > 0 {
                for i in 0..self.g_count.min(self.buffer_size) {
                    buf.row_mut(i).mapv_inplace(|v| v * 0.99);
                }
            }
            buf.row_mut(self.g_idx).assign(&grad);
            self.g_idx = (self.g_idx + 1) % self.buffer_size;
            self.g_count = (self.g_count + 1).min(self.buffer_size);
        }

        // ==========================================================
        // 6. Adapt blending parameter α
        //    α = ‖P_ort·g‖ / ‖P_active·g‖  (clipped to [0, 1])
        //    Use: ‖P_ort·g‖² = ‖g‖² - ‖P_active·g‖²
        // ==========================================================
        if self.g_count >= self.k {
            if let Some(ref ua) = u_active {
                if ua.nrows() > 0 {
                    let g_active = ua.dot(&grad); // (r,)
                    let norm_active_sq = g_active.dot(&g_active);
                    let norm_total_sq = grad.dot(&grad);
                    let norm_ort_sq = (norm_total_sq - norm_active_sq).max(0.0);
                    let norm_active = norm_active_sq.sqrt();
                    let norm_ort = norm_ort_sq.sqrt();

                    if norm_active > 1e-12 {
                        self.current_alpha = (norm_ort / norm_active).clamp(0.0, 1.0);
                    }
                }
            }
        }

        grad
    }
}

impl Default for ASEBO {
    fn default() -> Self {
        Self::new(1e-4, 1e-4, 50, 0.1, 1e-4, 200, 1e-8)
    }
}
