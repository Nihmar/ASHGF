use std::fs::File;
use std::sync::Arc;

use anyhow::Context;
use arrow::array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub function: String,
    pub algorithm: String,
    pub run: i32,
    pub values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<String>,
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

pub fn save_results(results: &[RunResult], dim: u32) -> anyhow::Result<()> {
    let path = get_results_path(dim);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(&path)?;
    let schema = schema();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(WriterProperties::new()))
        .context("Failed to create ArrowWriter")?;
    let batch = results_to_batch(results, &schema)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

pub fn load_results(dim: u32) -> anyhow::Result<Vec<RunResult>> {
    let path = get_results_path(dim);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(&path).context("Failed to open parquet file")?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("Failed to create parquet reader")?
        .build()
        .context("Failed to build record batch reader")?;

    let mut all_results = Vec::new();

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

            all_results.push(RunResult {
                function,
                algorithm,
                run,
                values,
                warnings,
            });
        }
    }

    Ok(all_results)
}

fn schema() -> SchemaRef {
    Schema::new(vec![
        Field::new("function", arrow::datatypes::DataType::Utf8, false),
        Field::new("algorithm", arrow::datatypes::DataType::Utf8, false),
        Field::new("run", arrow::datatypes::DataType::Int32, false),
        Field::new("values", arrow::datatypes::DataType::Utf8, false),
        Field::new("warnings", arrow::datatypes::DataType::Utf8, true),
    ])
    .into()
}

fn results_to_batch(results: &[RunResult], schema: &SchemaRef) -> anyhow::Result<RecordBatch> {
    let functions: Vec<&str> = results.iter().map(|r| r.function.as_str()).collect();
    let algorithms: Vec<&str> = results.iter().map(|r| r.algorithm.as_str()).collect();
    let runs: Vec<i32> = results.iter().map(|r| r.run).collect();
    let values: Vec<String> = results
        .iter()
        .map(|r| serde_json::to_string(&r.values).unwrap_or_default())
        .collect();
    let warnings: Vec<Option<&str>> = results.iter().map(|r| r.warnings.as_deref()).collect();

    let function_array = Arc::new(StringArray::from(functions)) as ArrayRef;
    let algorithm_array = Arc::new(StringArray::from(algorithms)) as ArrayRef;
    let run_array = Arc::new(Int32Array::from(runs)) as ArrayRef;
    let values_array = Arc::new(StringArray::from(values)) as ArrayRef;
    let warnings_array = Arc::new(StringArray::from(warnings)) as ArrayRef;

    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            function_array,
            algorithm_array,
            run_array,
            values_array,
            warnings_array,
        ],
    )
    .context("Failed to create record batch")
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
