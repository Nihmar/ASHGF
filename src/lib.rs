//! ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization.
//!
//! A library for derivative-free optimisation implementing algorithms
//! based on Gaussian smoothing and directional derivative estimation.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use ashgf::algorithms::{GD, Optimizer, OptimizeOptions};
//! use ashgf::functions::get_function;
//! use ashgf::utils::SeededRng;
//!
//! let mut gd = GD::default();
//! let f = get_function("sphere").unwrap();
//! let options = OptimizeOptions::default();
//! let mut rng = SeededRng::new(42);
//! let result = gd.optimize(&f, 10, None, &options, &mut rng);
//! println!("Best value: {:e}", result.best_values.last().unwrap().1);
//! ```

pub mod algorithms;
pub mod benchmark;
pub mod cli;
pub mod functions;
pub mod gradient;
pub mod utils;
