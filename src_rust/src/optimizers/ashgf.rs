use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, StandardNormal};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct ASHGF {
    pub k1: f64,
    pub k2: f64,
    pub alpha: f64,
    pub delta: f64,
    pub t: usize,
    seed: u64,
    pub eps: f64,
    // Parametri dell'algoritmo (corrispondono a ASHGF.data in Python)
    m: usize,
    a_init: f64,
    b_init: f64,
    a_minus: f64,
    a_plus: f64,
    b_minus: f64,
    b_plus: f64,
    gamma_l: f64,
    gamma_sigma_plus: f64,
    gamma_sigma_minus: f64,
    r_init: usize,
    ro: f64,
    // sigma_zero calcolato a runtime da ‖x₀‖
}

// ---------------------------------------------------------------------------
// FIX 1: Quadratura di Gauss-Hermite via algoritmo di Golub-Welsch
//
// Restituisce (nodes_std, weights_std) per la misura N(0,1):
//   nodes_std  = v_m * √2
//   weights_std = w_m / √π
// dove (v_m, w_m) sono i nodi/pesi per e^{-x²} (forma "physicist").
// ---------------------------------------------------------------------------
fn gauss_hermite(m: usize) -> (Vec<f64>, Vec<f64>) {
    // Matrice di Jacobi tridiagonale: J[i,i±1] = √((i+1)/2)
    let mut j = DMatrix::zeros(m, m);
    for i in 0..m - 1 {
        let off = ((i + 1) as f64 / 2.0).sqrt();
        j[(i, i + 1)] = off;
        j[(i + 1, i)] = off;
    }

    let eigen = j.symmetric_eigen();

    // Ordina per autovalore crescente (nodi in ordine crescente)
    let mut pairs: Vec<(f64, usize)> = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &ev)| (ev, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let sqrt_pi = std::f64::consts::PI.sqrt();
    let sqrt_2 = 2.0_f64.sqrt();

    // nodes_std = v * √2,  weights_std = q0² (i pesi sommano a 1)
    let nodes_std: Vec<f64> = pairs.iter().map(|&(v, _)| v * sqrt_2).collect();
    let weights_std: Vec<f64> = pairs
        .iter()
        .map(|&(_, orig)| {
            let q0 = eigen.eigenvectors[(0, orig)];
            sqrt_pi * q0 * q0 / sqrt_pi
        })
        .collect();

    (nodes_std, weights_std)
}

// ---------------------------------------------------------------------------
// FIX 2: Matrice ortogonale casuale secondo la distribuzione di Haar
//
// Usa gaussiane (non uniformi) + QR + correzione del segno su diag(R).
// ---------------------------------------------------------------------------
fn haar_orthogonal(dim: usize, rng: &mut StdRng) -> DMatrix<f64> {
    let data: Vec<f64> = (0..dim * dim).map(|_| StandardNormal.sample(rng)).collect();
    let a = DMatrix::from_column_slice(dim, dim, &data);

    let qr = a.qr();
    let mut q = qr.q();
    let r = qr.r();

    // Correggi i segni: sign(R[i,i])
    for i in 0..dim {
        let s = if r[(i, i)] >= 0.0 { 1.0 } else { -1.0 };
        for row in 0..dim {
            q[(row, i)] *= s;
        }
    }

    // Garantisce det = +1 (SO(dim))
    if q.determinant() < 0.0 {
        for row in 0..dim {
            q[(row, 0)] *= -1.0;
        }
    }

    q
}

// ---------------------------------------------------------------------------
// FIX 3: Campionamento da gaussiana multivariata N(0, C)
//
// Usa la decomposizione spettrale di C per gestire covarinaze
// semi-definite positive (con autovalori negativi clampati a 0).
//   1. C = V * diag(λ) * Vᵀ  (eigendecomposition)
//   2. λ' = max(λ, 0)         (garantisce PSD)
//   3. L = V * diag(√λ')      (radice quadrata di C)
//   4. x = L * z, z ~ N(0, I)
// ---------------------------------------------------------------------------
fn multivariate_normal_sample(cov: &DMatrix<f64>, dim: usize, rng: &mut StdRng) -> Vec<f64> {
    let eigen = cov.clone().symmetric_eigen();

    // L = V * diag(sqrt(max(λ,0)))
    let mut l = eigen.eigenvectors.clone(); // dim × dim, colonne = autovettori
    for (i, &ev) in eigen.eigenvalues.iter().enumerate() {
        let scale = ev.max(0.0).sqrt();
        for row in 0..dim {
            l[(row, i)] *= scale;
        }
    }

    // z ~ N(0, I_dim)
    let z: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
    let z_vec = DVector::from_column_slice(&z);

    // x = L * z
    let x_vec = l * z_vec;
    x_vec.as_slice().to_vec()
}

// ---------------------------------------------------------------------------
// FIX 3: Direzioni da covarianza dei gradienti storici
//
// Identico a Python _compute_directions_sges:
//   cov_G = cov(G_clean)  (non-biased)
//   M ~ Binomiale(dim, alpha)
//   dirs_G[i] ~ N(0, cov_G)      per i = 0..M
//   dirs_rand[i] ~ N(0, I_dim)   per i = M..dim
//   Normalizza ogni riga a norma unitaria
//
// Restituisce (flat row-major Vec<f64> di lunghezza dim*dim, M).
// ---------------------------------------------------------------------------
fn compute_directions_sges(
    dim: usize,
    g_history: &[Vec<f64>],
    alpha: f64,
    rng: &mut StdRng,
) -> (Vec<f64>, usize) {
    // Filtra gradienti con valori non finiti
    let g_clean: Vec<&Vec<f64>> = g_history
        .iter()
        .filter(|g| g.iter().all(|v| v.is_finite()))
        .collect();

    // Costruisce matrice di covarianza (o identità se non ci sono abbastanza dati)
    let cov_g: DMatrix<f64> = if g_clean.len() >= 2 {
        let n = g_clean.len();

        // Centra i dati
        let means: Vec<f64> = (0..dim)
            .map(|j| g_clean.iter().map(|g| g[j]).sum::<f64>() / n as f64)
            .collect();
        let flat: Vec<f64> = g_clean
            .iter()
            .flat_map(|g| g.iter().zip(&means).map(|(gj, mj)| gj - mj))
            .collect();
        let data = DMatrix::from_row_slice(n, dim, &flat);

        // cov = (data^T * data) / (n-1) — come numpy.cov
        let mut cov = (data.transpose() * data) / (n - 1) as f64;

        // Simmetrizza
        let cov_sym = (&cov + cov.transpose()) * 0.5;
        cov = cov_sym;

        // Correggi autovalori negativi: cov -= lambda_min * I  se lambda_min < 0
        let min_ev = cov
            .clone()
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_ev < 0.0 {
            for i in 0..dim {
                cov[(i, i)] -= min_ev;
            }
        }

        cov
    } else {
        DMatrix::identity(dim, dim)
    };

    // M ~ Binomiale(dim, alpha)  [FIX: distribuzione corretta]
    let binom =
        Binomial::new(dim as u64, alpha.clamp(0.0, 1.0)).expect("alpha deve essere in [0,1]");
    let m_dirs = (binom.sample(rng) as usize).clamp(0, dim);

    // Campiona le direzioni
    let mut dirs_flat = vec![0.0f64; dim * dim];

    // Prime m_dirs righe: da N(0, cov_G)
    for i in 0..m_dirs {
        let sample = multivariate_normal_sample(&cov_g, dim, rng);
        let norm: f64 = sample.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        let base = i * dim;
        for (j, &s) in sample.iter().enumerate() {
            dirs_flat[base + j] = s * scale;
        }
    }

    // Righe rimanenti: da N(0, I_dim)
    for i in m_dirs..dim {
        let base = i * dim;
        let mut norm_sq = 0.0f64;
        for j in 0..dim {
            let v: f64 = StandardNormal.sample(rng);
            dirs_flat[base + j] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for j in 0..dim {
                dirs_flat[base + j] /= norm;
            }
        }
    }

    (dirs_flat, m_dirs)
}

// ---------------------------------------------------------------------------
// FIX 2: Ortogonalizza le direzioni via QR su dirs^T
//
// Equivale a Python: orth(dirs.T).T + completamento se rango < dim
// Input: dirs_flat = Vec<f64> row-major (dim righe = direzioni normalizzate)
// Output: DMatrix<f64> dim×dim con righe ortonormali (la base)
// ---------------------------------------------------------------------------
fn orthogonalize_directions(dirs_flat: &[f64], dim: usize) -> DMatrix<f64> {
    let dirs = DMatrix::from_row_slice(dim, dim, dirs_flat);

    // QR su dirs^T: le colonne di Q sono la base ortonormale delle direzioni
    let qr = dirs.transpose().qr();
    let mut q = qr.q(); // dim × dim
    let r = qr.r();

    // Correzione del segno per consistenza con Haar
    for i in 0..dim {
        let s = if r[(i, i)] >= 0.0 { 1.0 } else { -1.0 };
        for row in 0..dim {
            q[(row, i)] *= s;
        }
    }

    // basis[i, :] = i-esima direzione → trasponiamo Q
    let mut basis = q.transpose();

    // Garantisce det = +1
    if basis.determinant() < 0.0 {
        for col in 0..dim {
            basis[(0, col)] *= -1.0;
        }
    }

    basis
}

// ---------------------------------------------------------------------------
// Impl ASHGF
// ---------------------------------------------------------------------------

impl ASHGF {
    pub fn new() -> Self {
        Self {
            k1: 0.9,
            k2: 0.1,
            alpha: 0.5,
            delta: 1.1,
            t: 50,
            seed: 2003,
            eps: 1e-8,
            m: 5,
            a_init: 0.1,
            b_init: 0.9,
            a_minus: 0.95,
            a_plus: 1.02,
            b_minus: 0.98,
            b_plus: 1.01,
            gamma_l: 0.9,
            gamma_sigma_plus: 1.0 / 0.9,
            gamma_sigma_minus: 0.9,
            r_init: 10,
            ro: 0.01,
        }
    }

    // -----------------------------------------------------------------------
    // Stimatore del gradiente DGS con Gauss-Hermite (eq. 2.22)
    //
    // FIX 1: usa gauss_hermite() pre-calcolato, non valori hardcoded.
    // FIX 4: rimosso il clamp lip.max(1.0) → usa lip senza floor artificiale.
    // FIX 4: rimosso il cap lr.min(0.4).
    // -----------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn grad_estimator<F>(
        &self,
        x: &[f64],
        sigma: f64,
        dim: usize,
        lipschitz_coefficients: &[f64],
        basis: &DMatrix<f64>,
        f: F,
        l_nabla: f64,
        m_dirs: usize,
        value: f64,
        nodes_std: &[f64],
        weights_std: &[f64],
        pair_indices: &[(usize, usize)],
    ) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, f64, Vec<Vec<f64>>)
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        let m = nodes_std.len();
        let mid = m / 2;
        let pert: Vec<f64> = nodes_std.iter().map(|n| sigma * n).collect();

        let mut evaluations: Vec<Vec<f64>> = Vec::with_capacity(dim);
        let mut derivatives = vec![0.0f64; dim];

        for j in 0..dim {
            let mut evals_j = vec![0.0f64; m];
            for k in 0..m {
                if k == mid {
                    evals_j[k] = value;
                } else {
                    let point: Vec<f64> = x
                        .iter()
                        .enumerate()
                        .map(|(idx, xi)| xi + pert[k] * basis[(j, idx)])
                        .collect();
                    evals_j[k] = f(&point);
                }
            }

            // Derivata direzionale: (1/σ) Σ_k w_k * n_k * F_k
            let deriv: f64 = evals_j
                .iter()
                .zip(weights_std.iter().zip(nodes_std.iter()))
                .map(|(fk, (wk, nk))| wk * nk * fk)
                .sum::<f64>()
                / sigma;
            derivatives[j] = deriv;
            evaluations.push(evals_j);
        }

        // Gradiente: g = Σⱼ derivatives[j] * basis[j,:]
        let mut grad = vec![0.0f64; dim];
        for j in 0..dim {
            let d = derivatives[j];
            for k in 0..dim {
                grad[k] += d * basis[(j, k)];
            }
        }

        // Costanti di Lipschitz locali (eq. 3.1)
        let mut new_lipschitz = lipschitz_coefficients.to_vec();
        for j in 0..dim {
            let mut lip = 0.0f64;
            let evals_j = &evaluations[j];
            for &(a, b) in pair_indices {
                let denom = sigma * (nodes_std[a] - nodes_std[b]);
                if denom.abs() > 1e-12 {
                    let val = (evals_j[a] - evals_j[b]).abs() / denom.abs();
                    if val > lip {
                        lip = val;
                    }
                }
            }
            new_lipschitz[j] = lip;
        }

        // L_G = max delle costanti di Lipschitz nelle prime m_dirs direzioni
        let m_eff = m_dirs.max(1).min(dim);
        let l_g = new_lipschitz[..m_eff]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        // FIX 4: nessun clamp artificiale su l_g né su lr
        let l_nabla_new = (1.0 - self.gamma_l) * l_g + self.gamma_l * l_nabla;
        let lr = sigma / l_nabla_new.max(1e-12);

        (
            grad,
            new_lipschitz,
            lr,
            derivatives,
            l_nabla_new,
            evaluations,
        )
    }

    // -----------------------------------------------------------------------
    // Subroutine (Algorithm 9): adatta sigma e ricostruisce la base
    //
    // FIX 2: reset usa haar_orthogonal() (gaussiane, non uniformi).
    // FIX 3: fase historical usa compute_directions_sges() con covarianza.
    // FIX 4: lip.max(1e-10) come Python, non lip.max(1.0).
    // -----------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn subroutine(
        &self,
        sigma: f64,
        sigma_zero: f64,
        grad: &[f64],
        derivatives: &[f64],
        lipschitz_coefficients: &[f64],
        mut a: f64,
        mut b: f64,
        mut r: usize,
        g_history: &[Vec<f64>],
        alpha: f64,
        historical: bool,
        rng: &mut StdRng,
    ) -> (f64, DMatrix<f64>, f64, f64, usize, usize) {
        let dim = grad.len();

        // Reset se sigma è scesa troppo e ci sono ancora reset disponibili
        if r > 0 && sigma < self.ro * sigma_zero {
            let basis = haar_orthogonal(dim, rng);
            r -= 1;
            return (sigma_zero, basis, self.a_init, self.b_init, r, dim);
        }

        // Costruisce la base
        let (basis, m_dirs) = if historical {
            // Campiona direzioni dalla covarianza dei gradienti storici
            let (dirs_flat, m) = compute_directions_sges(dim, g_history, alpha, rng);
            // Ortogonalizza via QR (equivalente a orth(dirs.T).T in Python)
            let b = orthogonalize_directions(&dirs_flat, dim);
            (b, m)
        } else {
            // Warmup: base casuale secondo Haar
            (haar_orthogonal(dim, rng), dim)
        };

        // Adatta sigma in base al rapporto max(|derivata| / Lipschitz)  [FIX 4]
        let ratio: f64 = derivatives
            .iter()
            .zip(lipschitz_coefficients.iter())
            .map(|(d, l)| d.abs() / l.max(1e-10))
            .fold(0.0f64, f64::max);

        let sigma_new = if ratio < a {
            a *= self.a_minus;
            sigma * self.gamma_sigma_minus
        } else if ratio > b {
            b *= self.b_plus;
            sigma * self.gamma_sigma_plus
        } else {
            a *= self.a_plus;
            b *= self.b_minus;
            sigma
        };

        (sigma_new, basis, a, b, r, m_dirs)
    }
}

impl Default for ASHGF {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trait Optimizer
// ---------------------------------------------------------------------------

impl Optimizer for ASHGF {
    fn name(&self) -> &'static str {
        "Adaptive Stochastic Historical Gradient-Free"
    }

    fn optimize<F>(
        &mut self,
        function: F,
        dim: usize,
        it: usize,
        x_init: Option<&[f64]>,
        debug: bool,
        itprint: usize,
    ) -> Result<OptimizerResult, OptimizerError>
    where
        F: Fn(&[f64]) -> f64 + Copy + Sync,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut x: Vec<f64> = match x_init {
            Some(init) => {
                if init.len() != dim {
                    return Err(OptimizerError::DimensionMismatch {
                        expected: dim,
                        got: init.len(),
                    });
                }
                init.to_vec()
            }
            None => (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect(),
        };

        let mut current_val = function(&x);
        let mut best_value = current_val;
        let mut best_points = vec![OptimizerPoint {
            x: x.clone(),
            value: best_value,
        }];
        let mut all_values = vec![current_val];

        // sigma_zero = ‖x₀‖ / 10  (identico a Python)
        let norm_x: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        let sigma_zero = norm_x / 10.0;
        let mut sigma = sigma_zero;
        let mut a = self.a_init;
        let mut b = self.b_init;
        let mut r = self.r_init;
        let mut l_nabla = 0.0f64;
        let mut m_dirs = dim;

        // FIX 5: inizializzazione identica a Python: np.ones(dim)
        let mut lipschitz_coefficients = vec![1.0f64; dim];

        // FIX 5: base casuale iniziale secondo Haar (non identità)
        let mut basis = haar_orthogonal(dim, &mut rng);

        // Precalcola nodi e pesi GH  [FIX 1]
        let (nodes_std, weights_std) = gauss_hermite(self.m);
        let mid = self.m / 2;

        // Coppie per la stima di Lipschitz (eq. 3.1)
        let pair_indices: Vec<(usize, usize)> = (0..self.m)
            .flat_map(|aa| (aa + 1..self.m).map(move |bb| (aa, bb)))
            .filter(|&(aa, bb)| (aa as i64 - mid as i64).abs() != (bb as i64 - mid as i64).abs())
            .collect();

        let mut g_history: Vec<Vec<f64>> = Vec::new();
        let mut alpha = self.alpha;

        if debug {
            println!(
                "algorithm: ashgf  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        let mut x_prev = x.clone(); // [FIX 4] criterio di arresto corretto

        for i in 1..=it {
            if debug && i % itprint == 0 {
                println!(
                    "{}th iteration - value: {}  last best value: {}",
                    i, current_val, best_value
                );
            }

            // ----------------------------------------------------------------
            // Stima del gradiente
            // ----------------------------------------------------------------
            let (grad, new_lipschitz, lr, derivatives, new_l_nabla, evaluations) = self
                .grad_estimator(
                    &x,
                    sigma,
                    dim,
                    &lipschitz_coefficients,
                    &basis,
                    function,
                    l_nabla,
                    m_dirs,
                    current_val,
                    &nodes_std,
                    &weights_std,
                    &pair_indices,
                );

            if !grad.iter().all(|g| g.is_finite()) || !lr.is_finite() {
                if debug {
                    println!("Warning: non-finite gradient or lr at iteration {}", i);
                }
                break;
            }

            // Aggiorna la storia dei gradienti (FIFO, max t)
            g_history.push(grad.clone());
            if g_history.len() > self.t {
                g_history.remove(0);
            }

            // ----------------------------------------------------------------
            // Passo di discesa: x ← x - lr * grad
            // ----------------------------------------------------------------
            x_prev.copy_from_slice(&x);
            x.iter_mut()
                .zip(grad.iter())
                .for_each(|(xi, gi)| *xi -= lr * gi);

            current_val = function(&x);
            all_values.push(current_val);

            if current_val < best_value {
                best_value = current_val;
                best_points.push(OptimizerPoint {
                    x: x.clone(),
                    value: best_value,
                });
            }

            // ----------------------------------------------------------------
            // Criterio di arresto: ‖x − x_prev‖₂ < eps  [FIX 4]
            // ----------------------------------------------------------------
            let norm_diff: f64 = x
                .iter()
                .zip(x_prev.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
                if debug {
                    println!(
                        "Converged at iteration {} (norm_diff = {:.2e})",
                        i, norm_diff
                    );
                }
                break;
            }

            // ----------------------------------------------------------------
            // Decide se usare i gradienti storici [FIX 6/7]
            // ----------------------------------------------------------------
            let historical = i >= self.t;

            // Adatta alpha (solo per i >= t+1, identico a Python)  [FIX 6]
            if i > self.t && m_dirs < dim {
                let vals_g: Vec<f64> = (0..m_dirs)
                    .map(|j| evaluations[j].iter().cloned().fold(f64::INFINITY, f64::min))
                    .collect();
                let vals_ort: Vec<f64> = (m_dirs..dim)
                    .map(|j| evaluations[j].iter().cloned().fold(f64::INFINITY, f64::min))
                    .collect();

                let r_g = if !vals_g.is_empty() {
                    Some(vals_g.iter().sum::<f64>() / vals_g.len() as f64)
                } else {
                    None
                };
                let r_g_ort = if !vals_ort.is_empty() {
                    Some(vals_ort.iter().sum::<f64>() / vals_ort.len() as f64)
                } else {
                    None
                };

                // Se entrambe le regioni hanno dati, aggiorna alpha
                if let (Some(rg), Some(rg_ort)) = (r_g, r_g_ort) {
                    alpha = if rg < rg_ort {
                        (self.delta * alpha).min(self.k1)
                    } else {
                        (alpha / self.delta).max(self.k2)
                    };
                }
                // Se una regione è vuota, alpha resta invariato (come Python)
            }

            // ----------------------------------------------------------------
            // Subroutine: adatta sigma e ricostruisce la base
            // ----------------------------------------------------------------
            let (new_sigma, new_basis, new_a, new_b, new_r, new_m) = self.subroutine(
                sigma,
                sigma_zero,
                &grad,
                &derivatives,
                &new_lipschitz,
                a,
                b,
                r,
                &g_history,
                alpha,
                historical,
                &mut rng,
            );
            sigma = new_sigma;
            basis = new_basis;
            a = new_a;
            b = new_b;
            r = new_r;
            m_dirs = new_m;

            l_nabla = new_l_nabla;
            lipschitz_coefficients = new_lipschitz;
        }

        if debug {
            println!(
                "\nlast evaluation: {}  last_iterate: {}  best evaluation: {}\n",
                all_values.last().unwrap(),
                all_values.len() - 1,
                best_value
            );
        }

        Ok(OptimizerResult {
            best_points,
            all_values,
        })
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&v| v * v).sum()
    }

    /// Nodi e pesi GH per m=5: ∑w = 1 e ∑w*n² ≈ 1 (varianza N(0,1))
    #[test]
    fn test_gauss_hermite_m5() {
        let (nodes, weights) = gauss_hermite(5);
        let sum_w: f64 = weights.iter().sum();
        assert!(
            (sum_w - 1.0).abs() < 1e-10,
            "weights should sum to 1, got {sum_w}"
        );
        let var: f64 = weights
            .iter()
            .zip(nodes.iter())
            .map(|(w, n)| w * n * n)
            .sum();
        assert!((var - 1.0).abs() < 1e-6, "∑ w*n² should ≈ 1, got {var}");
        assert!(nodes[2].abs() < 1e-10, "middle node should be 0");
        assert!(
            (nodes[0] + nodes[4]).abs() < 1e-10,
            "nodes should be symmetric"
        );
    }

    /// haar_orthogonal: QᵀQ ≈ I e det ≈ +1
    #[test]
    fn test_haar_orthogonal() {
        let mut rng = StdRng::seed_from_u64(42);
        for dim in [2, 3, 5, 10] {
            let q = haar_orthogonal(dim, &mut rng);
            let qtq = q.transpose() * &q;
            let identity = DMatrix::<f64>::identity(dim, dim);
            let err: f64 = (&qtq - &identity).norm();
            assert!(err < 1e-10, "QᵀQ should be I for dim={dim}, err={err}");
            let d = q.determinant();
            assert!(
                (d - 1.0).abs() < 1e-8,
                "det should be +1 for dim={dim}, got {d}"
            );
        }
    }

    /// orthogonalize_directions: la base risultante deve avere righe ortonormali
    #[test]
    fn test_orthogonalize_directions() {
        let mut rng = StdRng::seed_from_u64(7);
        let dim = 6;
        // Genera direzioni random normalizzate (non ortogonali)
        let flat: Vec<f64> = (0..dim * dim)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let basis = orthogonalize_directions(&flat, dim);
        let identity = DMatrix::<f64>::identity(dim, dim);
        let bbt = &basis * basis.transpose();
        let err = (&bbt - &identity).norm();
        assert!(err < 1e-9, "rows should be orthonormal, err={err}");
    }

    #[test]
    fn test_ashgf_convergence() {
        let mut ashgf = ASHGF::new();
        let result = ashgf
            .optimize(sphere, 10, 200, None, false, 25)
            .expect("optimization should succeed");
        let initial = result.all_values[0];
        let final_best = result.best_value();
        assert!(
            final_best < initial,
            "ASHGF should improve (initial={initial}, best={final_best})"
        );
    }

    #[test]
    fn test_reproducibility() {
        let mut a1 = ASHGF::new();
        let mut a2 = ASHGF::new();
        let r1 = a1.optimize(sphere, 5, 30, None, false, 25).unwrap();
        let r2 = a2.optimize(sphere, 5, 30, None, false, 25).unwrap();
        assert_eq!(r1.all_values, r2.all_values);
        assert_eq!(r1.best_value(), r2.best_value());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut ashgf = ASHGF::new();
        let result = ashgf.optimize(sphere, 10, 10, Some(&[1.0, 2.0, 3.0]), false, 1);
        assert!(matches!(
            result,
            Err(OptimizerError::DimensionMismatch {
                expected: 10,
                got: 3
            })
        ));
    }

    #[test]
    fn test_historical_phase_activates() {
        // t=3 → dal quarto step usa i gradienti storici
        let mut ashgf = ASHGF {
            t: 3,
            ..ASHGF::new()
        };
        let result = ashgf
            .optimize(sphere, 6, 20, None, false, 1)
            .expect("should not panic");
        assert!(!result.all_values.is_empty());
    }
}
