use ndarray::{arr1, Array1};

#[derive(Debug, Clone)]
pub struct Statistics {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Statistics {
    pub fn compute(values_list: &[Vec<f64>]) -> Self {
        if values_list.is_empty() {
            return Self {
                min: vec![],
                max: vec![],
                mean: vec![],
                std: vec![],
            };
        }

        let max_len = values_list.iter().map(|v| v.len()).max().unwrap_or(0);

        if max_len == 0 {
            return Self {
                min: vec![],
                max: vec![],
                mean: vec![],
                std: vec![],
            };
        }

        let padded: Vec<Array1<f64>> = values_list
            .iter()
            .map(|v| {
                let arr = arr1(v);
                if arr.len() < max_len {
                    let mut extended = arr.to_vec();
                    extended.resize(max_len, f64::NAN);
                    arr1(&extended)
                } else {
                    arr
                }
            })
            .collect();

        let _n = padded.len() as f64;

        let mut min = vec![f64::INFINITY; max_len];
        let mut max = vec![f64::NEG_INFINITY; max_len];
        let mut mean = vec![0.0; max_len];
        let mut std = vec![0.0; max_len];

        for arr in &padded {
            for (i, &val) in arr.iter().enumerate() {
                if !val.is_nan() {
                    min[i] = min[i].min(val);
                    max[i] = max[i].max(val);
                }
            }
        }

        for i in 0..max_len {
            let sum: f64 = padded
                .iter()
                .filter(|arr| !arr[i].is_nan())
                .map(|arr| arr[i])
                .sum();
            let count: usize = padded.iter().filter(|arr| !arr[i].is_nan()).count();
            mean[i] = if count > 0 {
                sum / count as f64
            } else {
                f64::NAN
            };
        }

        for i in 0..max_len {
            let sum_sq: f64 = padded
                .iter()
                .filter(|arr| !arr[i].is_nan())
                .map(|arr| {
                    let diff = arr[i] - mean[i];
                    diff * diff
                })
                .sum();
            let count: usize = padded.iter().filter(|arr| !arr[i].is_nan()).count();
            std[i] = if count > 1 {
                (sum_sq / (count - 1) as f64).sqrt()
            } else {
                0.0
            };
        }

        Self {
            min,
            max,
            mean,
            std,
        }
    }
}

pub fn compute_final_values(values_list: &[Vec<f64>]) -> Vec<f64> {
    values_list
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.last().copied().unwrap_or(f64::NAN))
        .collect()
}

pub fn compute_best_values(values_list: &[Vec<f64>]) -> Vec<f64> {
    values_list
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| v.iter().cloned().fold(f64::INFINITY, f64::min))
        .collect()
}
