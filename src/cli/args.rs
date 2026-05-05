//! CLI argument parsing via `clap` derive macros.

use clap::{Parser, Subcommand, ValueEnum};

/// Adaptive Stochastic Historical Gradient-Free optimisation.
#[derive(Parser)]
#[command(name = "ashgf", version = env!("CARGO_PKG_VERSION"), about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run a single algorithm on a test function.
    Run(RunArgs),
    /// Compare multiple algorithms on a test function.
    Compare(CompareArgs),
    /// List all available test functions.
    List,
    /// Run all algorithms on all (or matching) test functions.
    Benchmark(BenchmarkArgs),
    /// Run multiple trials and compute convergence statistics.
    Stats(StatsArgs),
}

/// Algorithms available via the CLI.
#[derive(Clone, Debug, PartialEq, ValueEnum)]
#[clap(rename_all = "lower")]
pub enum AlgoName {
    Gd,
    Sges,
    Asgf,
    #[clap(name = "asgf-2s")]
    Asgf2s,
    #[clap(name = "asgf-2sw")]
    Asgf2sw,
    Ashgf,
    AshgfNg,
    AshgfS,
    Asebo,
}

impl std::fmt::Display for AlgoName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgoName::Gd => write!(f, "gd"),
            AlgoName::Sges => write!(f, "sges"),
            AlgoName::Asgf => write!(f, "asgf"),
            AlgoName::Asgf2s => write!(f, "asgf-2s"),
            AlgoName::Asgf2sw => write!(f, "asgf-2sw"),
            AlgoName::Ashgf => write!(f, "ashgf"),
            AlgoName::AshgfNg => write!(f, "ashgf-ng"),
            AlgoName::AshgfS => write!(f, "ashgf-s"),
            AlgoName::Asebo => write!(f, "asebo"),
        }
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct RunArgs {
    /// Algorithm to use.
    #[arg(long, default_value = "gd")]
    pub algo: AlgoName,

    /// Test function name (use 'list' to list all).
    #[arg(long)]
    pub function: String,

    /// Problem dimension.
    #[arg(long, default_value = "100")]
    pub dim: usize,

    /// Number of iterations.
    #[arg(long, default_value = "1000")]
    pub iter: usize,

    /// Random seed.
    #[arg(long, default_value = "2003")]
    pub seed: u64,

    /// Learning rate (for GD, SGES).
    #[arg(long, default_value = "1e-4")]
    pub lr: f64,

    /// Smoothing bandwidth (for GD, SGES).
    #[arg(long, default_value = "1e-4")]
    pub sigma: f64,

    /// Stop if no improvement for N iterations.
    #[arg(long)]
    pub patience: Option<usize>,

    /// Tolerance on f(x) change for stagnation.
    #[arg(long)]
    pub ftol: Option<f64>,

    /// Suppress progress output.
    #[arg(long)]
    pub quiet: bool,
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct CompareArgs {
    /// Algorithms to compare.
    #[arg(long, num_args = 1.., default_values_t = vec![AlgoName::Gd, AlgoName::Sges])]
    pub algos: Vec<AlgoName>,

    /// Test function name.
    #[arg(long)]
    pub function: String,

    /// Problem dimension.
    #[arg(long, default_value = "100")]
    pub dim: usize,

    /// Number of iterations.
    #[arg(long, default_value = "1000")]
    pub iter: usize,

    /// Random seed.
    #[arg(long, default_value = "2003")]
    pub seed: u64,

    /// Stop if no improvement for N iterations.
    #[arg(long)]
    pub patience: Option<usize>,

    /// Tolerance on f(x) change.
    #[arg(long)]
    pub ftol: Option<f64>,

    /// Suppress output.
    #[arg(long)]
    pub quiet: bool,
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Algorithms to include (default: all).
    #[arg(long, num_args = 1..)]
    pub algos: Option<Vec<AlgoName>>,

    /// Only include functions matching this substring.
    #[arg(long)]
    pub pattern: Option<String>,

    /// Single dimension.
    #[arg(long)]
    pub dim: Option<usize>,

    /// Comma-separated dimensions, e.g. '10,100,1000'.
    #[arg(long)]
    pub dims: Option<String>,

    /// Number of iterations per run.
    #[arg(long, default_value = "1000")]
    pub iter: usize,

    /// Random seed.
    #[arg(long, default_value = "2003")]
    pub seed: u64,

    /// Learning rate.
    #[arg(long, default_value = "1e-4")]
    pub lr: f64,

    /// Smoothing bandwidth.
    #[arg(long, default_value = "1e-4")]
    pub sigma: f64,

    /// Output directory for CSV results.
    #[arg(long, default_value = "results")]
    pub output: String,

    /// Stop if no improvement for N iterations.
    #[arg(long)]
    pub patience: Option<usize>,

    /// Tolerance on f(x) change.
    #[arg(long)]
    pub ftol: Option<f64>,

    /// Number of parallel workers.
    #[arg(long, default_value = "1")]
    pub jobs: usize,

    /// Suppress per-run output.
    #[arg(long)]
    pub quiet: bool,
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[derive(Parser)]
pub struct StatsArgs {
    /// Test function name.
    #[arg(long)]
    pub function: String,

    /// Algorithms to include (default: all).
    #[arg(long, num_args = 1..)]
    pub algos: Option<Vec<AlgoName>>,

    /// Problem dimension.
    #[arg(long, default_value = "100")]
    pub dim: usize,

    /// Number of iterations per run.
    #[arg(long, default_value = "1000")]
    pub iter: usize,

    /// Number of independent repetitions.
    #[arg(long, default_value = "30")]
    pub runs: usize,

    /// Base random seed.
    #[arg(long, default_value = "2003")]
    pub seed: u64,

    /// Learning rate.
    #[arg(long, default_value = "1e-4")]
    pub lr: f64,

    /// Smoothing bandwidth.
    #[arg(long, default_value = "1e-4")]
    pub sigma: f64,

    /// Output directory for pickled results.
    #[arg(long)]
    pub output: Option<String>,

    /// Number of parallel workers.
    #[arg(long, default_value = "1")]
    pub jobs: usize,

    /// Suppress per-run output.
    #[arg(long)]
    pub quiet: bool,
}
