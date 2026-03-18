use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
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
    pub fn from_json(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let results: Vec<RunResult> = serde_json::from_reader(reader)?;
        Ok(Self { results })
    }

    pub fn filter(&self, functions: Option<&[String]>, algorithms: Option<&[String]>) -> Self {
        let results: Vec<RunResult> = self
            .results
            .iter()
            .filter(|r| {
                let func_ok = functions.map_or(true, |f| f.contains(&r.function));
                let alg_ok = algorithms.map_or(true, |a| a.contains(&r.algorithm));
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
        .join("results.json")
}
