use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Tipi pubblici
// ---------------------------------------------------------------------------

/// Un punto trovato durante l'ottimizzazione.
#[derive(Debug, Clone)]
pub struct OptimizerPoint {
    /// Coordinate del punto.
    pub x: Vec<f64>,
    /// Valore della funzione obiettivo in quel punto.
    pub value: f64,
}

/// Risultato di una sessione di ottimizzazione.
///
/// - `best_points`: sequenza dei miglioramenti (ogni volta che si trova un nuovo minimo).
/// - `all_values`:  valore della funzione ad ogni iterazione.
pub struct OptimizerResult {
    pub best_points: Vec<OptimizerPoint>,
    pub all_values: Vec<f64>,
}

impl OptimizerResult {
    /// Valore del miglior punto trovato.
    pub fn best_value(&self) -> f64 {
        self.best_points
            .last()
            .map(|p| p.value)
            .unwrap_or(f64::INFINITY)
    }

    /// Riferimento al miglior punto trovato.
    pub fn best_point(&self) -> Option<&OptimizerPoint> {
        self.best_points.last()
    }
}

// ---------------------------------------------------------------------------
// Errori
// ---------------------------------------------------------------------------

/// Errori che possono occorrere durante la costruzione o l'esecuzione
/// di un ottimizzatore.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    InvalidParameter(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for OptimizerError {}

// ---------------------------------------------------------------------------
// Helper condiviso (equivalente a BaseOptimizer._validate_x_init)
// ---------------------------------------------------------------------------

/// Valida (o genera casualmente) il punto iniziale.
///
/// - Se `x_init` è `None`, campiona da N(0, I).
/// - Se `x_init` è `Some(slice)`, controlla che la dimensione coincida con `dim`.
///
/// Restituisce `Err(OptimizerError::DimensionMismatch)` in caso di mismatch.
pub fn validate_x_init(
    x_init: Option<&[f64]>,
    dim: usize,
    rng: &mut StdRng,
) -> Result<Vec<f64>, OptimizerError> {
    match x_init {
        None => Ok((0..dim).map(|_| StandardNormal.sample(rng)).collect()),
        Some(init) => {
            if init.len() != dim {
                Err(OptimizerError::DimensionMismatch {
                    expected: dim,
                    got: init.len(),
                })
            } else {
                Ok(init.to_vec())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

pub trait Optimizer {
    /// Nome dell'algoritmo (per log e debug).
    fn name(&self) -> &'static str;

    /// Esegue l'ottimizzazione.
    ///
    /// # Errori
    /// Restituisce `Err` se i parametri sono invalidi (es. `x_init` con
    /// dimensione errata).
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
        F: Fn(&[f64]) -> f64 + Copy;
}
