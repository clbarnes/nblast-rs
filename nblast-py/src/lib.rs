use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;

use nblast::{table_to_fn, DistDot, DotPropIdx, NblastArena, Precision};

#[pyclass]
pub struct ArenaWrapper {
    // TODO: can this box be avoided?
    arena: NblastArena<Box<dyn Fn(&DistDot) -> Precision>>,
}

fn vec_to_array3<T: Sized + Copy>(v: &Vec<T>) -> [T; 3] {
    [v[0], v[1], v[2]]
}

#[pymethods]
impl ArenaWrapper {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        dist_thresholds: Vec<f64>,
        dot_thresholds: Vec<f64>,
        cells: Vec<f64>,
    ) -> PyResult<()> {
        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);
        Ok(obj.init(Self {
            arena: NblastArena::new(Box::new(score_fn)),
        }))
    }

    fn add_points(&mut self, _py: Python, points: Vec<Vec<f64>>) -> PyResult<usize> {
        // TODO: avoid this copy?
        let v: Vec<[f64; 3]> = points.iter().map(vec_to_array3).collect();
        self.arena
            .add_points(&v)
            .map_err(|s| PyErr::new::<exceptions::RuntimeError, _>(s))
    }

    pub fn query_target(
        &self,
        _py: Python,
        query_idx: DotPropIdx,
        target_idx: DotPropIdx,
        normalise: bool,
        symmetric: bool,
    ) -> Option<f64> {
        self.arena
            .query_target(query_idx, target_idx, normalise, symmetric)
    }

    pub fn queries_targets(
        &self,
        _py: Python,
        query_idxs: Vec<DotPropIdx>,
        target_idxs: Vec<DotPropIdx>,
        normalise: bool,
        symmetric: bool,
    ) -> Option<HashMap<(DotPropIdx, DotPropIdx), f64>> {
        self.arena
            .queries_targets(&query_idxs, &target_idxs, normalise, symmetric)
    }

    pub fn all_v_all(
        &self,
        _py: Python,
        normalise: bool,
        symmetric: bool,
    ) -> Option<HashMap<(DotPropIdx, DotPropIdx), Precision>> {
        self.arena.all_v_all(normalise, symmetric)
    }

    pub fn len(&self, _py: Python) -> usize {
        self.arena.len()
    }
}

#[pymodule]
fn pynblast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ArenaWrapper>()?;

    Ok(())
}
