pub type OptimizerResult = (Vec<Vec<f64>>, Vec<f64>);

pub trait Optimizer {
    fn optimize<F>(
        &mut self,
        function: F,
        dim: usize,
        it: usize,
        x_init: Option<&[f64]>,
        debug: bool,
        itprint: usize,
    ) -> OptimizerResult
    where
        F: Fn(&[f64]) -> f64 + Copy;

    fn name(&self) -> &'static str;
}
