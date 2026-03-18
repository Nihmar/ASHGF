use crate::plots::{ResultsData, Statistics};
use anyhow::Result;
use plotters::prelude::*;
use std::path::Path;

const FIGURE_SIZE: (u32, u32) = (800, 600);
const BACKGROUND_COLOR: RGBColor = RGBColor(255, 255, 255);
const TEXT_COLOR: RGBColor = RGBColor(0, 0, 0);

const ALGORITHM_COLORS: &[(&str, RGBColor)] = &[
    ("GD", RGBColor(0, 0, 255)),
    ("SGES", RGBColor(0, 128, 0)),
    ("ASGF", RGBColor(255, 165, 0)),
    ("ASHGF", RGBColor(255, 0, 0)),
    ("ASEBO", RGBColor(128, 0, 128)),
];

fn get_algorithm_color(algorithm: &str) -> RGBColor {
    ALGORITHM_COLORS
        .iter()
        .find(|(name, _)| *name == algorithm)
        .map(|(_, color)| *color)
        .unwrap_or_else(|| RGBColor(100, 100, 100))
}

pub fn plot_convergence_with_stats(
    results: &ResultsData,
    function: &str,
    algorithm: &str,
    output_path: &Path,
) -> Result<()> {
    let filtered = results.filter(
        Some(&[function.to_string()]),
        Some(&[algorithm.to_string()]),
    );

    if filtered.results.is_empty() {
        println!("No data for {} - {}", function, algorithm);
        return Ok(());
    }

    let values_list: Vec<_> = filtered
        .results
        .iter()
        .map(|r| r.values.clone())
        .filter(|v| !v.is_empty())
        .collect();

    if values_list.is_empty() {
        println!("No valid data for {} - {}", function, algorithm);
        return Ok(());
    }

    let stats = Statistics::compute(&values_list);
    let n_points = stats.mean.len();

    if n_points == 0 {
        return Ok(());
    }

    let min_val = stats
        .min
        .iter()
        .filter(|&&v| v.is_finite())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = stats
        .max
        .iter()
        .filter(|&&v| v.is_finite())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = if min_val.is_finite() {
        min_val * 0.1
    } else {
        1e-10
    };
    let y_max = if max_val.is_finite() {
        max_val * 10.0
    } else {
        1e10
    };

    let root = BitMapBackend::new(output_path, FIGURE_SIZE).into_drawing_area();
    root.fill(&BACKGROUND_COLOR)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - {}", function, algorithm),
            ("sans-serif", 24).into_font().color(&TEXT_COLOR),
        )
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..(n_points as u32 - 1), (y_min)..(y_max))?;

    chart
        .configure_mesh()
        .x_desc("Iterations t")
        .y_desc("f(x_t)")
        .y_labels(10)
        .label_style(("sans-serif", 14).into_font().color(&TEXT_COLOR))
        .draw()?;

    let color = get_algorithm_color(algorithm);
    let color_for_legend = color.clone();

    let min_points: Vec<(u32, f64)> = (0..n_points)
        .filter(|&i| stats.min[i].is_finite())
        .map(|i| (i as u32, stats.min[i]))
        .collect();
    let max_points: Vec<(u32, f64)> = (0..n_points)
        .filter(|&i| stats.max[i].is_finite())
        .map(|i| (i as u32, stats.max[i]))
        .collect();
    let mean_points: Vec<(u32, f64)> = (0..n_points)
        .filter(|&i| stats.mean[i].is_finite())
        .map(|i| (i as u32, stats.mean[i]))
        .collect();

    chart
        .draw_series(LineSeries::new(min_points, color.clone()))?
        .label("min")
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], color_for_legend.stroke_width(1))
        });

    chart
        .draw_series(LineSeries::new(max_points, color.clone()))?
        .label("max")
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], color_for_legend.stroke_width(1))
        });

    chart
        .draw_series(LineSeries::new(mean_points, color.stroke_width(2)))?
        .label("mean")
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], color_for_legend.stroke_width(2))
        });

    for i in 1..n_points {
        let y1 = (stats.mean[i - 1] - stats.std[i - 1]).max(stats.min[i - 1] * 0.5);
        let y2 = (stats.mean[i] + stats.std[i]).min(stats.max[i] * 2.0);
        if y1.is_finite() && y2.is_finite() && y1 > 0.0 && y2 > 0.0 {
            let y_range = y1.min(stats.min[i.min(n_points - 1)] * 0.5);
            let _ = chart.plotting_area().draw(&Rectangle::new(
                [(i as u32 - 1, y_range), (i as u32, y2)],
                ShapeStyle::from(color.mix(0.3)).filled(),
            ));
        }
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.7))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    println!("Saved {:?}", output_path);
    Ok(())
}

pub fn plot_all_algorithms(results: &ResultsData, function: &str, output_dir: &Path) -> Result<()> {
    let algorithms = results.get_algorithms();

    std::fs::create_dir_all(output_dir)?;

    let output_path = output_dir.join(format!("{}_comparison.png", function));
    let root = BitMapBackend::new(&output_path, FIGURE_SIZE).into_drawing_area();
    root.fill(&BACKGROUND_COLOR)?;

    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut n_points = 0usize;

    for algorithm in &algorithms {
        let filtered = results.filter(
            Some(&[function.to_string()]),
            Some(&[algorithm.to_string()]),
        );
        let values_list: Vec<_> = filtered
            .results
            .iter()
            .map(|r| r.values.clone())
            .filter(|v| !v.is_empty())
            .collect();

        if !values_list.is_empty() {
            let stats = Statistics::compute(&values_list);
            n_points = n_points.max(stats.mean.len());

            for &v in &stats.min {
                if v.is_finite() {
                    min_val = min_val.min(v);
                }
            }
            for &v in &stats.max {
                if v.is_finite() {
                    max_val = max_val.max(v);
                }
            }
        }
    }

    let y_min = if min_val.is_finite() {
        min_val * 0.1
    } else {
        1e-10
    };
    let y_max = if max_val.is_finite() {
        max_val * 10.0
    } else {
        1e10
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} - All Algorithms", function),
            ("sans-serif", 24).into_font().color(&TEXT_COLOR),
        )
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0u32..(n_points as u32 - 1), (y_min)..(y_max))?;

    chart
        .configure_mesh()
        .x_desc("Iterations t")
        .y_desc("f(x_t)")
        .y_labels(10)
        .label_style(("sans-serif", 14).into_font().color(&TEXT_COLOR))
        .draw()?;

    for algorithm in &algorithms {
        let filtered = results.filter(
            Some(&[function.to_string()]),
            Some(&[algorithm.to_string()]),
        );
        let values_list: Vec<_> = filtered
            .results
            .iter()
            .map(|r| r.values.clone())
            .filter(|v| !v.is_empty())
            .collect();

        if values_list.is_empty() {
            continue;
        }

        let stats = Statistics::compute(&values_list);
        let color = get_algorithm_color(algorithm);
        let mean_points: Vec<(u32, f64)> = (0..stats.mean.len())
            .filter(|&i| stats.mean[i].is_finite())
            .map(|i| (i as u32, stats.mean[i]))
            .collect();

        let color_for_legend = color.clone();
        let alg_label = algorithm.clone();
        chart
            .draw_series(LineSeries::new(mean_points, color.stroke_width(2)))?
            .label(alg_label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color_for_legend.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.7))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    println!("Saved {:?}", output_path);
    Ok(())
}

pub fn generate_summary_csv(results: &ResultsData, output_path: &Path) -> Result<()> {
    let mut csv_content =
        String::from("function,algorithm,n_runs,mean_final,std_final,min_final,mean_best\n");

    let grouped = results.group_by_function_algorithm();
    let mut keys: Vec<_> = grouped.keys().collect();
    keys.sort();

    for (func, alg) in keys {
        let values_list: Vec<_> = grouped
            .get(&(func.clone(), alg.clone()))
            .unwrap()
            .iter()
            .map(|r| r.values.clone())
            .collect();

        if values_list.is_empty() {
            continue;
        }

        let final_values: Vec<f64> = values_list
            .iter()
            .filter(|v| !v.is_empty())
            .map(|v| *v.last().unwrap())
            .collect();

        let best_values: Vec<f64> = values_list
            .iter()
            .filter(|v| !v.is_empty())
            .map(|v| v.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
            .collect();

        let n = final_values.len() as f64;
        let mean_final = final_values.iter().sum::<f64>() / n;
        let std_final = if n > 1.0 {
            let var: f64 = final_values
                .iter()
                .map(|&v| (v - mean_final).powi(2))
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        let min_final = final_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_best = best_values.iter().sum::<f64>() / n;

        csv_content.push_str(&format!(
            "{},{},{},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            func, alg, n, mean_final, std_final, min_final, mean_best
        ));
    }

    std::fs::write(output_path, csv_content)?;
    println!("Saved summary to {:?}", output_path);
    Ok(())
}
