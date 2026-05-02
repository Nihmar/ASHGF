//! Seeded random number generator wrapper for reproducible optimisation.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// A reproducible random number generator.
///
/// Wraps [`StdRng`] (ChaCha12) initialised from a `u64` seed.
/// Passed explicitly to every function that requires randomness,
/// avoiding any implicit global state.
#[derive(Debug, Clone)]
pub struct SeededRng {
    pub rng: StdRng,
    pub seed: u64,
}

impl SeededRng {
    /// Create a new generator from the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a generator with a randomly chosen seed (non-reproducible).
    pub fn from_entropy() -> Self {
        use rand::RngCore;
        let seed = rand::rngs::OsRng.next_u64();
        Self::new(seed)
    }
}
