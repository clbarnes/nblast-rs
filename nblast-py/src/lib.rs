use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;

use nblast::{
    table_to_fn, DistDot, NblastArena, NeuronIdx, Precision, RStarPointTangents,
};

#[pyclass]
pub struct ArenaWrapper {
    // TODO: can this box be avoided?
    arena: NblastArena<RStarPointTangents, Box<dyn Fn(&DistDot) -> Precision>>,
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
        let neuron = RStarPointTangents::new(points.iter().map(vec_to_array3).collect())
            .map_err(|s| PyErr::new::<exceptions::RuntimeError, _>(s))?;
        Ok(self.arena.add_neuron(neuron))
    }

    pub fn query_target(
        &self,
        _py: Python,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetric: bool,
    ) -> Option<f64> {
        self.arena
            .query_target(query_idx, target_idx, normalize, symmetric)
    }

    pub fn queries_targets(
        &self,
        _py: Python,
        query_idxs: Vec<NeuronIdx>,
        target_idxs: Vec<NeuronIdx>,
        normalize: bool,
        symmetric: bool,
    ) -> HashMap<(NeuronIdx, NeuronIdx), f64> {
        self.arena
            .queries_targets(&query_idxs, &target_idxs, normalize, symmetric)
    }

    pub fn all_v_all(
        &self,
        _py: Python,
        normalize: bool,
        symmetric: bool,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        self.arena.all_v_all(normalize, symmetric)
    }

    pub fn len(&self, _py: Python) -> usize {
        self.arena.len()
    }

    pub fn self_hit(&self, _py: Python, idx: NeuronIdx) -> Option<Precision> {
        self.arena.self_hit(idx)
    }

    pub fn points(&self, _py: Python, idx: NeuronIdx) -> Option<Vec<[Precision; 3]>> {
        self.arena.points(idx)
    }

    pub fn tangents(&self, _py: Python, idx: NeuronIdx) -> Option<Vec<[Precision; 3]>> {
        self.arena.tangents(idx).map(|vectors| vectors.map(|v| [*v.x, *v.y, *v.z]))  // TODO: fix this
    }
}

#[pymodule]
fn pynblast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ArenaWrapper>()?;

    Ok(())
}
