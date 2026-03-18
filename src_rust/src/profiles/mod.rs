pub mod experiment;
pub mod io;

pub use experiment::{analyze_results, get_default_functions, Algorithm, ExperimentResult, Runner};
pub use io::{load_results, save_results};
