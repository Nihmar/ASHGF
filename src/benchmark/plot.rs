//! Plotting functions for benchmark results using the `plotters` crate.
//!
//! Generates PNG charts:
//! - Bar chart comparing best values per function / algorithm / dimension
//! - Convergence grid (rows = functions, columns = dimensions)
//! - Per-function detailed grids

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use plotters::prelude::*;

use crate::benchmark::runner::RunResult;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Colour palette (tab10-like).
const PALETTE: &[RGBColor] = &[
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
    RGBColor(188, 189, 34),
    RGBColor(23, 190, 207),
];

fn algo_colour(idx: usize) -> RGBColor {
    PALETTE[idx % PALETTE.len()]
}

/// Clamp a value to a safe range for log-scale plotting (avoid 0, Inf, NaN).
fn safe_log_val(v: f64) -> f64 {
    if v.is_nan() || v.is_infinite() || v <= 0.0 {
        1e-30
    } else {
        v
    }
}

/// Collect unique sorted algorithms, functions, and dimensions from results.
fn collect_keys(results: &[RunResult]) -> (Vec<String>, Vec<String>, Vec<usize>) {
    let mut algos = BTreeSet::new();
    let mut funcs = BTreeSet::new();
    let mut dims = BTreeSet::new();
    for r in results {
        algos.insert(r.algorithm.clone());
        funcs.insert(r.function.clone());
        dims.insert(r.dim);
    }
    (
        algos.into_iter().collect(),
        funcs.into_iter().collect(),
        dims.into_iter().collect(),
    )
}

/// Build a lookup: (function, dim, algo) -> best_value.
fn build_lookup(results: &[RunResult]) -> BTreeMap<(String, usize, String), f64> {
    let mut map = BTreeMap::new();
    for r in results {
        map.insert(
            (r.function.clone(), r.dim, r.algorithm.clone()),
            r.best_value,
        );
    }
    map
}

// ---------------------------------------------------------------------------
// Bar chart: best value per function, algorithm, dimension
// ---------------------------------------------------------------------------

/// Generate a grouped bar chart comparing best f(x) across algorithms,
/// functions, and dimensions.  One subplot per dimension.
pub fn plot_comparison_bars(results: &[RunResult], output_path: &Path) {
    let (algos, funcs, dims) = collect_keys(results);
    if algos.is_empty() || funcs.is_empty() || dims.is_empty() {
        tracing::warn!("No data to plot.");
        return;
    }
    let lookup = build_lookup(results);
    let n_algos = algos.len();
    let n_dims = dims.len();

    let root = BitMapBackend::new(output_path, (1800 * n_dims as u32, 1200)).into_drawing_area();
    root.fill(&WHITE).ok();

    let panels = root.split_evenly((1, n_dims));

    for (col, &dim) in dims.iter().enumerate() {
        let panel = &panels[col];
        let n_funcs = funcs.len();

        // Determine y range — iterate all (func, algo) pairs
        let mut y_min = f64::INFINITY;
        let mut y_max = 0.0f64;
        for f in funcs.iter() {
            for a in algos.iter() {
                if let Some(&v) = lookup.get(&(f.clone(), dim, a.clone())) {
                    let sv = safe_log_val(v);
                    y_min = y_min.min(sv);
                    y_max = y_max.max(sv);
                }
            }
        }
        y_min = y_min.max(1e-30_f64);
        y_max = y_max.max(y_min * 10.0);

        let mut chart = ChartBuilder::on(panel)
            .caption(format!("dim = {dim}"), ("sans-serif", 22))
            .margin(15)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0f64..funcs.len() as f64,
                (y_min.log10()..y_max.log10()).step(1.0),
            )
            .unwrap();

        chart
            .configure_mesh()
            .x_labels(n_funcs)
            .x_label_formatter(&|i: &f64| {
                let idx = *i as usize;
                if idx < funcs.len() {
                    let s = &funcs[idx];
                    if s.len() > 12 {
                        format!("{}..", &s[..10])
                    } else {
                        s.clone()
                    }
                } else {
                    String::new()
                }
            })
            .x_label_style(("sans-serif", 9, &BLACK))
            .y_label_formatter(&|v| format!("1e{:.0}", v))
            .y_desc("Best f(x)")
            .draw()
            .ok();

        let bar_width = 0.8 / n_algos as f64;

        for (i, algo) in algos.iter().enumerate() {
            let colour = algo_colour(i);
            let offset = i as f64 * bar_width;
            let bars: Vec<(f64, f64)> = funcs
                .iter()
                .enumerate()
                .filter_map(|(j, f)| {
                    lookup
                        .get(&(f.clone(), dim, algo.clone()))
                        .map(|&v| (j as f64 + offset, safe_log_val(v).log10()))
                })
                .collect();

            chart
                .draw_series(
                    bars.iter().map(|&(x, y)| {
                        Rectangle::new([(x, 0.0), (x + bar_width, y)], colour.filled())
                    }),
                )
                .ok()
                .map(|s| {
                    s.label(algo.clone()).legend(move |(x, y)| {
                        Rectangle::new([(x, y), (x + 20, y + 12)], colour.filled())
                    })
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .ok();
    }

    root.present().ok();
    tracing::info!("Comparison bar chart saved to {}", output_path.display());
}

// ---------------------------------------------------------------------------
// Convergence grid (requires per-iteration data)
// ---------------------------------------------------------------------------

/// Full run result with per-iteration history for plotting.
#[derive(Debug, Clone)]
pub struct RunResultWithHistory {
    pub algorithm: String,
    pub function: String,
    pub dim: usize,
    pub values: Vec<f64>,
    pub best_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl From<RunResult> for RunResultWithHistory {
    fn from(r: RunResult) -> Self {
        Self {
            algorithm: r.algorithm,
            function: r.function,
            dim: r.dim,
            values: vec![r.final_value],
            best_value: r.best_value,
            iterations: r.iterations,
            converged: r.converged,
        }
    }
}

/// Generate a convergence grid: rows = functions, columns = dimensions.
/// Each cell shows f(x) vs iterations for all algorithms.
pub fn plot_convergence_grid(results: &[RunResultWithHistory], output_path: &Path) {
    let (algos, funcs, dims) = collect_keys_from_history(results);
    if algos.is_empty() || funcs.is_empty() || dims.is_empty() {
        tracing::warn!("No data for convergence plot.");
        return;
    }

    let n_funcs = funcs.len();
    let n_dims = dims.len();

    // High-resolution cells + legend strip at bottom
    let cell_w = 1200u32;
    let cell_h = 900u32;
    let legend_h = 60u32;
    let total_h = cell_h * n_funcs as u32 + legend_h;
    let total_w = cell_w * n_dims as u32;

    let root = BitMapBackend::new(output_path, (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE).ok();

    // Split: grid area + legend area
    let (grid_area, legend_area) = root.split_vertically(cell_h * n_funcs as u32);
    let panels = grid_area.split_evenly((n_funcs, n_dims));

    for row in 0..n_funcs {
        for col in 0..n_dims {
            let panel = &panels[row * n_dims + col];
            let fn_name = &funcs[row];
            let dim = dims[col];

            let mut max_iter = 0usize;
            let mut all_series: Vec<(&str, Vec<(usize, f64)>)> = Vec::new();

            for algo in &algos {
                let points: Vec<(usize, f64)> = results
                    .iter()
                    .filter(|r| r.function == *fn_name && r.dim == dim && r.algorithm == *algo)
                    .flat_map(|r| {
                        r.values
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (i, safe_log_val(v)))
                    })
                    .collect();
                max_iter = max_iter.max(points.len());
                if !points.is_empty() {
                    all_series.push((algo, points));
                }
            }

            let x_max = if max_iter > 1 { max_iter - 1 } else { 1 };
            let y_vals: Vec<f64> = all_series
                .iter()
                .flat_map(|(_, p)| p.iter().map(|&(_, v)| v))
                .collect();
            let y_min = y_vals
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b))
                .max(1e-30_f64);
            let y_max = y_vals
                .iter()
                .fold(1e-30_f64, |a, &b| a.max(b))
                .max(y_min * 10.0);

            let mut chart = ChartBuilder::on(panel)
                .caption(
                    if row == 0 {
                        format!("dim={dim}")
                    } else {
                        String::new()
                    },
                    ("sans-serif", 20),
                )
                .margin(12)
                .x_label_area_size(55)
                .y_label_area_size(85)
                .build_cartesian_2d(0..x_max, (y_min.log10()..y_max.log10()).step(0.5))
                .unwrap();

            chart.configure_mesh().disable_mesh().draw().ok();

            chart
                .configure_mesh()
                .light_line_style(&RGBColor(230, 230, 230))
                .x_labels(5)
                .y_labels(5)
                .y_label_formatter(&|v| format!("1e{:.0}", v))
                .draw()
                .ok();

            if col == 0 {
                chart
                    .configure_mesh()
                    .y_desc(if fn_name.len() > 28 {
                        format!("{}..", &fn_name[..26])
                    } else {
                        fn_name.clone()
                    })
                    .draw()
                    .ok();
            }

            for (i, (_algo, points)) in all_series.iter().enumerate() {
                let colour = algo_colour(i);
                let line_points: Vec<(usize, f64)> = points
                    .iter()
                    .map(|&(x, v)| (x, safe_log_val(v).log10()))
                    .collect();
                if line_points.len() >= 2 {
                    chart
                        .draw_series(LineSeries::new(line_points, &colour))
                        .ok();
                }
            }
        }
    }

    // ---- Global legend at the bottom (manual drawing, more reliable) ----
    let legend_bg = legend_area.margin(5, 5, 5, 5);
    legend_bg.fill(&RGBColor(245, 245, 245)).ok();

    // Title
    legend_bg
        .draw_text(
            "Legend:",
            &TextStyle::from(("sans-serif", 14)).color(&BLACK),
            (5, 2),
        )
        .ok();

    // Colour swatches + algorithm names in a horizontal row
    let swatch_w = 22i32;
    let swatch_h = 14i32;
    let text_offset = swatch_w + 4;
    let y_center = 24i32;

    for (i, algo) in algos.iter().enumerate() {
        let colour = algo_colour(i);
        let x0 = 10 + i as i32 * 160;

        // Colour swatch
        legend_bg
            .draw(&Rectangle::new(
                [
                    (x0, y_center - swatch_h / 2),
                    (x0 + swatch_w, y_center + swatch_h / 2),
                ],
                colour.filled(),
            ))
            .ok();

        // Algorithm name
        legend_bg
            .draw_text(
                algo,
                &TextStyle::from(("sans-serif", 12)).color(&BLACK),
                (x0 + text_offset, y_center - 7),
            )
            .ok();
    }

    root.present().ok();
    tracing::info!("Convergence grid saved to {}", output_path.display());
}

// ---------------------------------------------------------------------------
// Per-function plots: one PNG per function, grid (dims × algos)
// ---------------------------------------------------------------------------

/// Generate one PNG per function.  Each PNG contains a grid:
/// rows = dimensions, columns = algorithms, showing f(x) vs iterations.
pub fn plot_per_function(results: &[RunResultWithHistory], output_dir: &Path) -> Vec<String> {
    std::fs::create_dir_all(output_dir).ok();

    let (algos, funcs, dims) = collect_keys_from_history(results);
    if algos.is_empty() || funcs.is_empty() || dims.is_empty() {
        tracing::warn!("No data for per-function plots.");
        return vec![];
    }

    let n_dims = dims.len();
    let n_algos = algos.len();
    let cell_w = 800u32;
    let cell_h = 560u32;

    let mut saved = Vec::new();

    for fn_name in &funcs {
        let path = output_dir.join(format!("{fn_name}.png"));
        let root = BitMapBackend::new(&path, (cell_w * n_algos as u32, cell_h * n_dims as u32))
            .into_drawing_area();
        root.fill(&WHITE).ok();

        let panels = root.split_evenly((n_dims, n_algos));

        for row in 0..n_dims {
            for col in 0..n_algos {
                let dim = dims[row];
                let algo = &algos[col];
                let panel = &panels[row * n_algos + col];

                let points: Vec<(usize, f64)> = results
                    .iter()
                    .filter(|r| r.function == *fn_name && r.dim == dim && r.algorithm == *algo)
                    .flat_map(|r| {
                        r.values
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (i, safe_log_val(v)))
                    })
                    .collect();

                let x_max = if points.len() > 1 {
                    points.len() - 1
                } else {
                    1
                };
                let y_vals: Vec<f64> = points.iter().map(|&(_, v)| v).collect();
                let y_min = y_vals
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b))
                    .max(1e-30_f64);
                let y_max = y_vals
                    .iter()
                    .fold(1e-30_f64, |a, &b| a.max(b))
                    .max(y_min * 100.0);

                let mut chart = ChartBuilder::on(panel)
                    .caption(
                        if row == 0 { algo.as_str() } else { "" },
                        ("sans-serif", 14),
                    )
                    .margin(5)
                    .x_label_area_size(35)
                    .y_label_area_size(55)
                    .build_cartesian_2d(0..x_max, (y_min.log10()..y_max.log10()).step(0.5))
                    .unwrap();

                chart.configure_mesh().disable_mesh().draw().ok();
                chart
                    .configure_mesh()
                    .light_line_style(&RGBColor(230, 230, 230))
                    .x_labels(3)
                    .y_labels(3)
                    .y_label_formatter(&|v| format!("1e{:.0}", v))
                    .draw()
                    .ok();

                if col == 0 {
                    chart
                        .configure_mesh()
                        .y_desc(format!("d={dim}"))
                        .draw()
                        .ok();
                }

                if !points.is_empty() {
                    let colour = algo_colour(col);
                    let line_points: Vec<(usize, f64)> = points
                        .iter()
                        .map(|&(x, v)| (x, safe_log_val(v).log10()))
                        .collect();
                    if line_points.len() >= 2 {
                        chart.draw_series(LineSeries::new(line_points, colour)).ok();
                    }
                }
            }
        }

        root.present().ok();
        saved.push(path.to_string_lossy().to_string());
    }

    tracing::info!(
        "Generated {} per-function plot(s) in {}",
        saved.len(),
        output_dir.display()
    );
    saved
}

// ---------------------------------------------------------------------------
// Helpers for history-based results
// ---------------------------------------------------------------------------

fn collect_keys_from_history(
    results: &[RunResultWithHistory],
) -> (Vec<String>, Vec<String>, Vec<usize>) {
    let mut algos = BTreeSet::new();
    let mut funcs = BTreeSet::new();
    let mut dims = BTreeSet::new();
    for r in results {
        algos.insert(r.algorithm.clone());
        funcs.insert(r.function.clone());
        dims.insert(r.dim);
    }
    (
        algos.into_iter().collect(),
        funcs.into_iter().collect(),
        dims.into_iter().collect(),
    )
}
