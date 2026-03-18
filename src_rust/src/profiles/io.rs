use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};

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
        .join("results.json")
}

pub fn save_results(results: &[RunResult], dim: u32) -> anyhow::Result<()> {
    let path = get_results_path(dim);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);

    serde_json::to_writer_pretty(&mut writer, results)?;
    writer.flush()?;

    Ok(())
}

pub fn load_results(dim: u32) -> anyhow::Result<Vec<RunResult>> {
    let path = get_results_path(dim);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let results: Vec<RunResult> = serde_json::from_reader(reader)?;

    Ok(results)
}
