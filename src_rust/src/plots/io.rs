use anyhow::Context;
use arrow::array::{Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RunResult {
    pub function: String,
    pub algorithm: String,
    pub run: i32,
    pub values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<String>,
}

#[derive(Debug)]
pub struct ResultsData {
    pub results: Vec<RunResult>,
}

impl ResultsData {
    pub fn from_parquet(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path).context("Failed to open parquet file")?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create parquet reader")?
            .build()
            .context("Failed to build record batch reader")?;

        let mut results = Vec::new();

        for batch_result in reader {
            let batch = batch_result.context("Failed to read batch")?;
            let num_rows = batch.num_rows();

            let function_col = batch
                .column_by_name("function")
                .context("Missing 'function' column")?;
            let algorithm_col = batch
                .column_by_name("algorithm")
                .context("Missing 'algorithm' column")?;
            let run_col = batch
                .column_by_name("run")
                .context("Missing 'run' column")?;
            let values_col = batch
                .column_by_name("values")
                .context("Missing 'values' column")?;
            let warnings_col = batch.column_by_name("warnings");

            for i in 0..num_rows {
                let function = extract_string(function_col.as_ref(), i)?;
                let algorithm = extract_string(algorithm_col.as_ref(), i)?;
                let run = extract_int(run_col.as_ref(), i)?;
                let values = extract_values(values_col.as_ref(), i)?;
                let warnings = warnings_col
                    .as_ref()
                    .and_then(|col| extract_optional_string(col.as_ref(), i));

                results.push(RunResult {
                    function,
                    algorithm,
                    run,
                    values,
                    warnings,
                });
            }
        }

        Ok(Self { results })
    }

    pub fn filter(&self, functions: Option<&[String]>, algorithms: Option<&[String]>) -> Self {
        let results: Vec<RunResult> = self
            .results
            .iter()
            .filter(|r| {
                let func_ok = functions.is_none_or(|f| f.contains(&r.function));
                let alg_ok = algorithms.is_none_or(|a| a.contains(&r.algorithm));
                func_ok && alg_ok
            })
            .cloned()
            .collect();
        Self { results }
    }

    pub fn get_functions(&self) -> Vec<String> {
        let mut functions: Vec<String> = self.results.iter().map(|r| r.function.clone()).collect();
        functions.sort();
        functions.dedup();
        functions
    }

    pub fn get_algorithms(&self) -> Vec<String> {
        let mut algorithms: Vec<String> =
            self.results.iter().map(|r| r.algorithm.clone()).collect();
        algorithms.sort();
        algorithms.dedup();
        algorithms
    }

    pub fn group_by_function_algorithm(&self) -> HashMap<(String, String), Vec<&RunResult>> {
        let mut grouped: HashMap<(String, String), Vec<&RunResult>> = HashMap::new();
        for result in &self.results {
            grouped
                .entry((result.function.clone(), result.algorithm.clone()))
                .or_default()
                .push(result);
        }
        grouped
    }
}

pub fn get_results_path(dim: u32) -> std::path::PathBuf {
    let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(std::path::Path::new("."));
    project_root
        .join("results")
        .join("profiles")
        .join(format!("dim={}", dim))
        .join("results.parquet")
}

fn extract_string(col: &dyn arrow::array::Array, idx: usize) -> anyhow::Result<String> {
    let string_arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .context("Expected StringArray")?;
    Ok(string_arr.value(idx).to_string())
}

fn extract_int(col: &dyn arrow::array::Array, idx: usize) -> anyhow::Result<i32> {
    if let Some(int_arr) = col.as_any().downcast_ref::<Int32Array>() {
        Ok(int_arr.value(idx))
    } else {
        anyhow::bail!("Expected Int32Array")
    }
}

fn extract_values(col: &dyn arrow::array::Array, idx: usize) -> anyhow::Result<Vec<f64>> {
    let string_arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .context("Expected StringArray for values")?;

    let json_str = string_arr.value(idx);
    let values: Vec<Option<f64>> =
        serde_json::from_str(json_str).context("Failed to parse values JSON")?;

    Ok(values
        .into_iter()
        .map(|v| v.unwrap_or(f64::INFINITY))
        .collect())
}

fn extract_optional_string(col: &dyn arrow::array::Array, idx: usize) -> Option<String> {
    if col.is_null(idx) {
        return None;
    }
    let string_arr = col.as_any().downcast_ref::<StringArray>()?;
    Some(string_arr.value(idx).to_string())
}
