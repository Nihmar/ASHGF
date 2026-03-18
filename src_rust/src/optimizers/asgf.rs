use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::DMatrix;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct ASGF {
    seed: u64,
    pub eps: f64,
    // Parametri dell'algoritmo (corrispondono a ASGF.data in Python)
    m: usize,
    a_init: f64,
    b_init: f64,
    a_minus: f64,
    a_plus: f64,
    b_minus: f64,
    b_plus: f64,
    gamma_l: f64,
    gamma_sigma: f64,
    r_init: usize,
    ro: f64,
    // sigma_zero viene calcolato a runtime in base a ‖x₀‖
}

// ---------------------------------------------------------------------------
// Helper: quadratura di Gauss-Hermite
//
// FIX 1: implementazione corretta tramite algoritmo di Golub-Welsch.
//
// Restituisce (nodes_std, weights_std) per la misura N(0,1):
//   nodes_std  = v_m * √2
//   weights_std = w_m / √π
// dove (v_m, w_m) sono i nodi/pesi della quadratura di Gauss-Hermite
// rispetto alla misura peso e^{-x²} (forma "physicist").
//
// La matrice di Jacobi tridiagonale per la forma physicist è:
//   J[i,i]   = 0
//   J[i,i±1] = √((i+1)/2)   per i = 0..m-2
// I nodi sono gli autovalori di J; i pesi sono w_i = √π * q_i[0]²
// dove q_i è l'autovettore normalizzato corrispondente.
// ---------------------------------------------------------------------------
fn gauss_hermite(m: usize) -> (Vec<f64>, Vec<f64>) {
    // Costruisci la matrice di Jacobi (m × m, simmetrica tridiagonale)
    let mut j = DMatrix::zeros(m, m);
    for i in 0..m - 1 {
        let off = ((i + 1) as f64 / 2.0).sqrt();
        j[(i, i + 1)] = off;
        j[(i + 1, i)] = off;
    }

    // Decomposizione spettrale: J = Q Λ Qᵀ
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

    // nodes_std = v_m * √2,  weights_std = w_m / √π
    let nodes_std: Vec<f64> = pairs.iter().map(|&(v, _)| v * sqrt_2).collect();
    let weights_std: Vec<f64> = pairs
        .iter()
        .map(|&(_, orig)| {
            // w_i = √π * q_i[0]²  dove q_i è la prima componente dell'autovettore
            let q0 = eigen.eigenvectors[(0, orig)];
            sqrt_pi * q0 * q0 / sqrt_pi // = q0²  (i pesi sommano a 1 dopo la divisione)
        })
        .collect();

    (nodes_std, weights_std)
}

// ---------------------------------------------------------------------------
// Helper: matrice ortogonale casuale secondo la distribuzione di Haar
//
// FIX 2: usa gaussiane (non uniformi) per ottenere la distribuzione di Haar.
// ---------------------------------------------------------------------------
//
// Procedura standard (metodo QR):
// 1. Genera A: dim×dim con entrate i.i.d. ~ N(0,1)
// 2. Calcola A = QR via decomposizione QR
// 3. Correggi il segno: moltiplica la colonna i di Q per sign(R[i,i])
//    → Q è uniformemente distribuita su O(dim)
// 4. Se det(Q) < 0, nega la prima colonna → Q ∈ SO(dim)
fn haar_orthogonal(dim: usize, rng: &mut StdRng) -> DMatrix<f64> {
    // Passo 1: matrice dim×dim di gaussiane
    let data: Vec<f64> = (0..dim * dim).map(|_| StandardNormal.sample(rng)).collect();
    let a = DMatrix::from_column_slice(dim, dim, &data); // col-major

    // Passo 2: decomposizione QR
    let qr = a.qr();
    let mut q = qr.q(); // dim × dim
    let r = qr.r(); // dim × dim upper-triangular

    // Passo 3: correggi i segni delle colonne (sign(R[i,i]))
    for i in 0..dim {
        let s = if r[(i, i)] >= 0.0 { 1.0 } else { -1.0 };
        for row in 0..dim {
            q[(row, i)] *= s;
        }
    }

    // Passo 4: garantisce det = +1 (SO(dim))
    if q.determinant() < 0.0 {
        for row in 0..dim {
            q[(row, 0)] *= -1.0;
        }
    }

    q
}

// ---------------------------------------------------------------------------
// Impl ASGF
// ---------------------------------------------------------------------------

impl ASGF {
    pub fn new() -> Self {
        Self {
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
            gamma_sigma: 0.9,
            r_init: 2,
            ro: 0.01,
        }
    }

    // -----------------------------------------------------------------------
    // Stimatore del gradiente DGS con quadratura di Gauss-Hermite (eq. 2.22)
    //
    // FIX 1: usa gauss_hermite() invece di valori hardcoded sbagliati.
    // FIX 4: rimosso il cap arbitrario su lr (era .min(0.4) nella versione errata).
    // FIX 5: rimosso il floor artificiale lip.max(1.0) (era artificioso).
    // -----------------------------------------------------------------------

    /// Per ogni direzione di base ξⱼ, valuta F in m punti di quadratura lungo
    /// quella direzione e assembla la stima della derivata direzionale.
    /// Stima anche le costanti di Lipschitz locali per ogni direzione (eq. 3.1).
    ///
    /// # Argomenti
    /// - `basis`: matrice dim×dim le cui RIGHE sono le direzioni ortonormali
    /// - `value`: F(x) già noto (viene usato come nodo centrale k = m/2)
    ///
    /// # Restituzione
    /// `(grad, new_lipschitz, lr, derivatives, l_nabla_new)`
    #[allow(clippy::too_many_arguments)]
    fn grad_estimator<F>(
        &self,
        x: &[f64],
        m: usize,
        sigma: f64,
        dim: usize,
        lipschitz_coefficients: &[f64],
        basis: &DMatrix<f64>,
        f: F,
        l_nabla: f64,
        value: f64,
        nodes_std: &[f64],
        weights_std: &[f64],
        pair_indices: &[(usize, usize)],
    ) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, f64)
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        // Perturbazioni: σ * nodes_std
        let pert: Vec<f64> = nodes_std.iter().map(|n| sigma * n).collect();
        let mid = m / 2;

        // evaluations[j][k] = F(x + pert[k] * basis_row[j])
        let mut evaluations: Vec<Vec<f64>> = Vec::with_capacity(dim);
        let mut derivatives = vec![0.0f64; dim];

        for j in 0..dim {
            let mut evals_j = vec![0.0f64; m];
            for k in 0..m {
                if k == mid {
                    evals_j[k] = value;
                } else {
                    // point = x + pert[k] * basis[j, :]
                    let point: Vec<f64> = x
                        .iter()
                        .enumerate()
                        .map(|(idx, xi)| xi + pert[k] * basis[(j, idx)])
                        .collect();
                    evals_j[k] = f(&point);
                }
            }

            // Derivata direzionale: (1/σ) Σ_k weights_std[k] * nodes_std[k] * F_k
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

        // Costanti di Lipschitz locali (eq. 3.1) su tutte le coppie in I
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

        // Aggiorna L_nabla e calcola il learning rate (eq. 3.2)
        // FIX 4: nessun cap artificiale su lr
        let l_nabla_new = (1.0 - self.gamma_l) * new_lipschitz[0] + self.gamma_l * l_nabla;
        let lr = sigma / l_nabla_new.max(1e-12);

        (grad, new_lipschitz, lr, derivatives, l_nabla_new)
    }

    // -----------------------------------------------------------------------
    // Subroutine (Algorithm 7): adatta sigma e ricostruisce la base
    //
    // FIX 2: usa haar_orthogonal() per il reset della base.
    // FIX 2: costruisce la nuova base via QR sul transposto (identico a Python).
    // FIX 5: lip.max(1e-10) come in Python, non lip.max(1.0).
    // -----------------------------------------------------------------------

    /// Adatta sigma e costruisce una nuova base ortonormale con il gradiente
    /// come prima riga.
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
        rng: &mut StdRng,
    ) -> (f64, DMatrix<f64>, f64, f64, usize) {
        let dim = grad.len();

        // Reset se sigma è scesa troppo in basso e ci sono ancora reset disponibili
        if r > 0 && sigma < self.ro * sigma_zero {
            let basis = haar_orthogonal(dim, rng); // FIX 2
            r -= 1;
            return (sigma_zero, basis, self.a_init, self.b_init, r);
        }

        // Costruisce base con grad come prima riga, poi QR per ortonormalizzare
        // (identico a Python: basis[0] = grad/||grad||; Q,_ = qr(basis.T); basis = Q.T)
        let mut mat = haar_orthogonal(dim, rng); // base casuale iniziale (Haar)
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm > 1e-10 {
            // Sostituisce la prima riga con il gradiente normalizzato
            for j in 0..dim {
                mat[(0, j)] = grad[j] / grad_norm;
            }
        }

        // QR su mat^T: le colonne di Q corrispondono alle righe ortonormalizzate di mat
        // Python: Q, _ = la.qr(basis.T); basis = Q.T
        let qr = mat.transpose().qr();
        let mut q = qr.q(); // dim × dim, colonne ortonormali
        let r_mat = qr.r();
        // Correggi i segni (come in haar_orthogonal, per consistenza)
        for i in 0..dim {
            let s = if r_mat[(i, i)] >= 0.0 { 1.0 } else { -1.0 };
            for row in 0..dim {
                q[(row, i)] *= s;
            }
        }
        let basis = q.transpose(); // basis[i,:] = i-esima direzione

        // Adatta sigma basandosi su max(|derivata| / Lipschitz)  (eq. 3.3)
        // FIX 5: usa max(lip, 1e-10) come in Python
        let mut lip_clamped = lipschitz_coefficients.to_vec();
        for l in lip_clamped.iter_mut() {
            *l = l.max(1e-10);
        }

        let ratio: f64 = derivatives
            .iter()
            .zip(lip_clamped.iter())
            .map(|(d, l)| d.abs() / l)
            .fold(0.0f64, f64::max);

        let sigma_new = if ratio < a {
            a *= self.a_minus;
            sigma * self.gamma_sigma
        } else if ratio > b {
            b *= self.b_plus;
            sigma / self.gamma_sigma
        } else {
            a *= self.a_plus;
            b *= self.b_minus;
            sigma
        };

        (sigma_new, basis, a, b, r)
    }
}

impl Default for ASGF {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trait Optimizer
// ---------------------------------------------------------------------------

impl Optimizer for ASGF {
    fn name(&self) -> &'static str {
        "Adaptive Stochastic Gradient-Free"
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
        let mut l_nabla = 0.0_f64;

        // FIX 5: inizializzazione identica a Python: np.ones(dim)
        let mut lipschitz_coefficients = vec![1.0f64; dim];

        // Base ortogonale casuale iniziale (distribuzione di Haar)
        let mut basis = haar_orthogonal(dim, &mut rng);

        // Precalcola nodi e pesi di Gauss-Hermite  [FIX 1]
        let (nodes_std, weights_std) = gauss_hermite(self.m);
        let mid = self.m / 2;

        // Insieme delle coppie I per la stima di Lipschitz (eq. 3.1)
        let pair_indices: Vec<(usize, usize)> = (0..self.m)
            .flat_map(|aa| (aa + 1..self.m).map(move |bb| (aa, bb)))
            .filter(|&(aa, bb)| (aa as i64 - mid as i64).abs() != (bb as i64 - mid as i64).abs())
            .collect();

        if debug {
            println!(
                "algorithm: asgf  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        let mut x_prev = x.clone(); // [FIX 3] per il criterio di arresto

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
            let (grad, new_lipschitz, lr, derivatives, new_l_nabla) = self.grad_estimator(
                &x,
                self.m,
                sigma,
                dim,
                &lipschitz_coefficients,
                &basis,
                function,
                l_nabla,
                current_val,
                &nodes_std,
                &weights_std,
                &pair_indices,
            );

            // Controllo valori finiti
            if !grad.iter().all(|g| g.is_finite()) || !lr.is_finite() {
                if debug {
                    println!("Warning: non-finite gradient or lr at iteration {}", i);
                }
                break;
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
            // Criterio di arresto: ‖x - x_prev‖₂ < eps  [FIX 3]
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
            // Subroutine: adatta sigma e ricostruisce la base
            // ----------------------------------------------------------------
            let (new_sigma, new_basis, new_a, new_b, new_r) = self.subroutine(
                sigma,
                sigma_zero,
                &grad,
                &derivatives,
                &new_lipschitz,
                a,
                b,
                r,
                &mut rng,
            );
            sigma = new_sigma;
            basis = new_basis;
            a = new_a;
            b = new_b;
            r = new_r;

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

    /// Verifica che i nodi e pesi GH per m=5 abbiano la proprietà fondamentale:
    /// ∑ weights_std = 1  e  ∑ weights_std * nodes_std² ≈ 1  (varianza N(0,1))
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
        assert!(
            (var - 1.0).abs() < 1e-6,
            "∑ w*n² should ≈ 1 (variance of N(0,1)), got {var}"
        );
        // Simmetria dei nodi
        assert!(
            (nodes[0] + nodes[4]).abs() < 1e-10,
            "nodes should be symmetric"
        );
        assert!(
            (nodes[1] + nodes[3]).abs() < 1e-10,
            "nodes should be symmetric"
        );
        assert!(nodes[2].abs() < 1e-10, "middle node should be 0");
    }

    /// Verifica che haar_orthogonal produca una matrice ortogonale con det ≈ +1
    #[test]
    fn test_haar_orthogonal() {
        let mut rng = StdRng::seed_from_u64(42);
        for dim in [2, 3, 5, 10] {
            let q = haar_orthogonal(dim, &mut rng);
            // Q^T Q ≈ I
            let qtq = q.transpose() * &q;
            let identity = DMatrix::<f64>::identity(dim, dim);
            let err: f64 = (&qtq - &identity).norm();
            assert!(err < 1e-10, "QᵀQ should be I for dim={dim}, err={err}");
            // det ≈ +1
            let d = q.determinant();
            assert!(
                (d - 1.0).abs() < 1e-8,
                "det should be +1 for dim={dim}, got {d}"
            );
        }
    }

    #[test]
    fn test_asgf_convergence() {
        let mut asgf = ASGF::new();
        let result = asgf
            .optimize(sphere, 10, 200, None, false, 25)
            .expect("optimization should succeed");

        let initial = result.all_values[0];
        let final_best = result.best_value();
        assert!(
            final_best < initial,
            "ASGF should improve from initial value (initial={initial}, best={final_best})"
        );
    }

    #[test]
    fn test_reproducibility() {
        let mut a1 = ASGF::new();
        let mut a2 = ASGF::new();

        let r1 = a1.optimize(sphere, 5, 30, None, false, 25).unwrap();
        let r2 = a2.optimize(sphere, 5, 30, None, false, 25).unwrap();

        assert_eq!(r1.all_values, r2.all_values);
        assert_eq!(r1.best_value(), r2.best_value());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut asgf = ASGF::new();
        let result = asgf.optimize(sphere, 10, 10, Some(&[1.0, 2.0, 3.0]), false, 1);
        assert!(matches!(
            result,
            Err(OptimizerError::DimensionMismatch {
                expected: 10,
                got: 3
            })
        ));
    }
}
