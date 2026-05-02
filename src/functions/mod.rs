//! Function registry for optimisation test functions.
//!
//! Provides a central registry mapping function names to callable
//! function pointers, supporting both analytical test functions
//! and (optionally) RL environments.

pub mod benchmark;
pub mod classic;
pub mod extended;

use ndarray::Array1;
use once_cell::sync::Lazy;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

/// A scalar test function `f: R^d → R`.
pub type TestFunction = fn(&Array1<f64>) -> f64;

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

static REGISTRY: Lazy<HashMap<&'static str, TestFunction>> = Lazy::new(|| {
    let mut m: HashMap<&'static str, TestFunction> = HashMap::new();

    // -- classic --
    m.insert("sphere", classic::sphere);
    m.insert("rastrigin", classic::rastrigin);
    m.insert("ackley", classic::ackley);
    m.insert("griewank", classic::griewank);
    m.insert("levy", classic::levy);
    m.insert("schwefel", classic::schwefel);
    m.insert("sum_of_different_powers", classic::sum_of_different_powers);
    m.insert("trid", classic::trid);
    m.insert("zakharov", classic::zakharov);
    m.insert("cosine", classic::cosine);
    m.insert("sine", classic::sine);
    m.insert("sincos", classic::sincos);

    // -- extended --
    m.insert(
        "extended_feudenstein_and_roth",
        extended::extended_feudenstein_and_roth,
    );
    m.insert("extended_trigonometric", extended::extended_trigonometric);
    m.insert("extended_rosenbrock", extended::extended_rosenbrock);
    m.insert("generalized_rosenbrock", extended::generalized_rosenbrock);
    m.insert(
        "extended_white_and_holst",
        extended::extended_white_and_holst,
    );
    m.insert("extended_baele", extended::extended_baele);
    m.insert("extended_penalty", extended::extended_penalty);
    m.insert("extended_himmelblau", extended::extended_himmelblau);
    m.insert(
        "generalized_white_and_holst",
        extended::generalized_white_and_holst,
    );
    m.insert("extended_psc1", extended::extended_psc1);
    m.insert("extended_bd1", extended::extended_bd1);
    m.insert("extended_maratos", extended::extended_maratos);
    m.insert("extended_cliff", extended::extended_cliff);
    m.insert("extended_hiebert", extended::extended_hiebert);
    m.insert("extended_tridiagonal_1", extended::extended_tridiagonal_1);
    m.insert("extended_tridiagonal_2", extended::extended_tridiagonal_2);
    m.insert("extended_denschnb", extended::extended_denschnb);
    m.insert("extended_denschnf", extended::extended_denschnf);
    m.insert(
        "extended_quadratic_exponential_ep1",
        extended::extended_quadratic_exponential_ep1,
    );

    // -- benchmark --
    m.insert("perturbed_quadratic", benchmark::perturbed_quadratic);
    m.insert(
        "almost_perturbed_quadratic",
        benchmark::almost_perturbed_quadratic,
    );
    m.insert(
        "perturbed_quadratic_diagonal",
        benchmark::perturbed_quadratic_diagonal,
    );
    m.insert("diagonal_1", benchmark::diagonal_1);
    m.insert("diagonal_2", benchmark::diagonal_2);
    m.insert("diagonal_3", benchmark::diagonal_3);
    m.insert("broyden_tridiagonal", benchmark::broyden_tridiagonal);
    m.insert(
        "generalized_tridiagonal_1",
        benchmark::generalized_tridiagonal_1,
    );
    m.insert("indef", benchmark::indef);
    m.insert("raydan_1", benchmark::raydan_1);
    m.insert("raydan_2", benchmark::raydan_2);
    m.insert("hager", benchmark::hager);
    m.insert("bdqrtic", benchmark::bdqrtic);
    m.insert("power", benchmark::power);
    m.insert("engval1", benchmark::engval1);
    m.insert("dqdrtic", benchmark::dqdrtic);
    m.insert("quartc", benchmark::quartc);
    m.insert("fletcbv3", benchmark::fletcbv3);
    m.insert("fletchcr", benchmark::fletchcr);
    m.insert("eg2", benchmark::eg2);
    m.insert("genhumps", benchmark::genhumps);
    m.insert("nondia", benchmark::nondia);
    m.insert("vardim", benchmark::vardim);
    m.insert(
        "extended_quadratic_penalty_qp1",
        benchmark::extended_quadratic_penalty_qp1,
    );
    m.insert(
        "extended_quadratic_penalty_qp2",
        benchmark::extended_quadratic_penalty_qp2,
    );

    m
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Retrieve a test function by name.
///
/// Returns `None` if the name is not registered.
pub fn get_function(name: &str) -> Option<TestFunction> {
    REGISTRY.get(name).copied()
}

/// Return a sorted list of all registered function names.
pub fn list_functions() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = REGISTRY.keys().copied().collect();
    v.sort_unstable();
    v
}

/// Return the number of registered functions.
pub fn num_functions() -> usize {
    REGISTRY.len()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_contains_sphere() {
        assert!(get_function("sphere").is_some());
    }

    #[test]
    fn registry_contains_rastrigin() {
        let f = get_function("rastrigin").unwrap();
        let x = Array1::zeros(3);
        assert!((f(&x) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn list_returns_sorted() {
        let names = list_functions();
        assert!(names.len() >= 40); // at least 40
                                    // verify sorted
        for w in names.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn unknown_function_returns_none() {
        assert!(get_function("nonexistent_func_xyz").is_none());
    }
}
