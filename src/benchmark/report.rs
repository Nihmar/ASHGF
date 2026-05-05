//! Automatic Markdown report generation after benchmark runs.
//!
//! Produces a `REPORT.md` in the output directory summarising algorithm
//! performance across functions and dimensions.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::benchmark::runner::RunResult;

/// Generate a Markdown report and write it to `output_dir/REPORT.md`.
pub fn generate_report(
    results: &[RunResult],
    dimensions: &[usize],
    seed: u64,
    max_iter: usize,
    output_dir: &Path,
) {
    let mut md = String::new();
    let now = chrono_now();

    // --- Header ---
    writeln!(md, "# Benchmark Report — ASHGF").unwrap();
    writeln!(md).unwrap();
    writeln!(md, "**Generated**: {now}").unwrap();
    writeln!(md, "**Seed**: {seed}").unwrap();
    writeln!(md, "**Max iterations**: {max_iter}").unwrap();
    writeln!(md, "**Dimensions**: {}", dimensions.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")).unwrap();
    writeln!(md, "**Functions**: {}  ", func_count(results)).unwrap();
    writeln!(md, "**Algorithms**: {}", algo_list(results)).unwrap();
    writeln!(md).unwrap();

    // --- 1. Overall leaderboard ---
    writeln!(md, "## Overall Leaderboard (wins per function)").unwrap();
    writeln!(md).unwrap();

    let leaderboard = compute_wins(results);
    writeln!(md, "| Algorithm | Wins | Win % |").unwrap();
    writeln!(md, "|-----------|------|-------|").unwrap();
    let total_funcs = func_count(results) * dimensions.len();
    for (algo, wins) in &leaderboard {
        let pct = if total_funcs > 0 {
            *wins as f64 / total_funcs as f64 * 100.0
        } else {
            0.0
        };
        writeln!(md, "| {algo} | {wins} | {pct:.1}% |").unwrap();
    }
    writeln!(md).unwrap();

    // --- 2. Convergence summary ---
    writeln!(md, "## Convergence Rate").unwrap();
    writeln!(md).unwrap();
    writeln!(md, "| Algorithm | Converged | Total | Rate |").unwrap();
    writeln!(md, "|-----------|-----------|-------|------|").unwrap();
    for (algo, (conv, total)) in &convergence_stats(results) {
        let rate = if *total > 0 {
            *conv as f64 / *total as f64 * 100.0
        } else {
            0.0
        };
        writeln!(md, "| {algo} | {conv} | {total} | {rate:.1}% |").unwrap();
    }
    writeln!(md).unwrap();

    // --- 3. Per-dimension tables ---
    for &dim in dimensions {
        let dim_results: Vec<_> = results.iter().filter(|r| r.dim == dim).cloned().collect();
        if dim_results.is_empty() {
            continue;
        }
        writeln!(md, "---").unwrap();
        writeln!(md).unwrap();
        writeln!(md, "## Dimension d = {dim}").unwrap();
        writeln!(md).unwrap();

        // Collect unique function names and algorithm names
        let mut funcs = BTreeSet::new();
        let mut algos = BTreeSet::new();
        for r in &dim_results {
            funcs.insert(r.function.clone());
            algos.insert(r.algorithm.clone());
        }
        let funcs: Vec<_> = funcs.into_iter().collect();
        let algos: Vec<_> = algos.into_iter().collect();

        // Best value table: rows = functions, cols = algorithms
        let best_map = build_best_map(&dim_results);

        // Header row
        let mut header = String::from("| Function ");
        for algo in &algos {
            write!(header, "| {algo} ").unwrap();
        }
        header.push('|');
        writeln!(md, "{header}").unwrap();

        // Separator
        let mut sep = String::from("|----------");
        for _ in &algos {
            write!(sep, "|----------").unwrap();
        }
        sep.push('|');
        writeln!(md, "{sep}").unwrap();

        // Data rows
        for func in &funcs {
            let mut row = format!("| {func} ");
            let mut func_bests: Vec<f64> = Vec::new();
            for algo in &algos {
                let key = (algo.clone(), func.clone());
                if let Some(val) = best_map.get(&key) {
                    row.push_str(&format!("| {:.4e} ", val));
                    func_bests.push(*val);
                } else {
                    row.push_str("| — ");
                }
            }
            // Mark winner(s) with bold
            let best = func_bests
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min);
            if best.is_finite() {
                for algo in &algos {
                    let key = (algo.clone(), func.clone());
                    if let Some(&val) = best_map.get(&key) {
                        if val.is_finite() && (val - best).abs() < 1e-12 {
                            let old = format!("| {:.4e} ", val);
                            let new = format!("| **{:.4e}** ", val);
                            row = row.replace(&old, &new);
                        }
                    }
                }
            }
            row.push('|');
            writeln!(md, "{row}").unwrap();
        }
        writeln!(md).unwrap();

        // Per-dimension wins
        let dim_wins = compute_wins_for_dim(&dim_results);
        writeln!(md, "### Wins (d={dim})").unwrap();
        writeln!(md).unwrap();
        writeln!(md, "| Algorithm | Wins |").unwrap();
        writeln!(md, "|-----------|------|").unwrap();
        for (algo, wins) in &dim_wins {
            writeln!(md, "| {algo} | {wins} |").unwrap();
        }
        writeln!(md).unwrap();
    }

    // Write to file
    let report_path = output_dir.join("REPORT.md");
    if let Err(e) = fs::write(&report_path, md) {
        tracing::error!("Cannot write REPORT.md: {e}");
    } else {
        tracing::info!("REPORT.md written to {}", report_path.display());
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a map `(algo, func) -> best_value` from results.
fn build_best_map(results: &[RunResult]) -> BTreeMap<(String, String), f64> {
    let mut map: BTreeMap<(String, String), f64> = BTreeMap::new();
    for r in results {
        let key = (r.algorithm.clone(), r.function.clone());
        let val = r.best_value;
        if val.is_finite() {
            map.entry(key)
                .and_modify(|v: &mut f64| {
                    if val < *v {
                        *v = val;
                    }
                })
                .or_insert(val);
        }
    }
    map
}

/// Count how many unique functions are in the results.
fn func_count(results: &[RunResult]) -> usize {
    results.iter().map(|r| &r.function).collect::<BTreeSet<_>>().len()
}

/// Comma-separated list of unique algorithms.
fn algo_list(results: &[RunResult]) -> String {
    let algos: BTreeSet<_> = results.iter().map(|r| r.algorithm.as_str()).collect();
    algos.into_iter().collect::<Vec<_>>().join(", ")
}

/// Count per-algorithm wins (lowest best_value on each function).
/// A win is awarded to every algorithm that achieves the minimum value.
fn compute_wins(results: &[RunResult]) -> Vec<(String, usize)> {
    let funcs: BTreeSet<_> = results
        .iter()
        .map(|r| (r.dim, r.function.clone()))
        .collect();
    let algos: BTreeSet<String> = results.iter().map(|r| r.algorithm.clone()).collect();
    let best_map = build_best_map(results);

    let mut win_counts: BTreeMap<String, usize> = BTreeMap::new();
    for algo in &algos {
        win_counts.insert(algo.clone(), 0);
    }

    for (_dim, func) in &funcs {
        let best = algos
            .iter()
            .filter_map(|a| best_map.get(&(a.clone(), func.clone())))
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, |a: f64, &b: &f64| a.min(b));

        if best.is_finite() {
            for algo in &algos {
                if let Some(&val) = best_map.get(&(algo.clone(), func.clone())) {
                    if val.is_finite() && (val - best).abs() < 1e-12 {
                        *win_counts.get_mut(algo).unwrap() += 1;
                    }
                }
            }
        }
    }

    let mut result: Vec<_> = win_counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    result
}

/// Per-dimension win count.
fn compute_wins_for_dim(results: &[RunResult]) -> Vec<(String, usize)> {
    compute_wins(results)
}

/// Convergence statistics: (algo -> (converged, total)).
fn convergence_stats(results: &[RunResult]) -> BTreeMap<String, (usize, usize)> {
    let mut map: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in results {
        let entry = map.entry(r.algorithm.clone()).or_default();
        entry.1 += 1;
        if r.converged {
            entry.0 += 1;
        }
    }
    map
}

/// Current UTC timestamp as a human-readable string.
fn chrono_now() -> String {
    // Avoid adding chrono dependency: use a minimal inline implementation.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Naive UTC date conversion (good enough for a report header).
    let secs_per_day: u64 = 86400;
    let days_since_epoch = ts / secs_per_day;
    let time_of_day = ts % secs_per_day;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;

    // Algorithm from Howard Hinnant: https://howardhinnant.github.io/date_algorithms.html
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02} {hours:02}:{minutes:02} UTC")
}
