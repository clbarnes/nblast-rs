use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use core::fmt::Debug;

use nblast::nalgebra::base::{Vector3, Unit};
use nblast::{table_to_fn, DistDot, NblastArena, NeuronIdx, Precision, RStarPointTangents, Symmetry};

#[pyclass]
pub struct ArenaWrapper {
    // TODO: can this box be avoided?
    arena: NblastArena<RStarPointTangents, Box<dyn Fn(&DistDot) -> Precision>>,
}

fn vec_to_array3<T: Sized + Copy>(v: &Vec<T>) -> [T; 3] {
    [v[0], v[1], v[2]]
}

fn vec_to_unitvector3<T: 'static + Sized + Copy + PartialEq + Debug>(v: &Vec<T>) -> Unit<Vector3<T>> {
    Unit::new_unchecked(Vector3::new(v[0], v[1], v[2]))
}

fn str_to_sym(s: &str) -> Result<Symmetry, ()> {
    match s {
        "arithmetic_mean" => Ok(Symmetry::ArithmeticMean),
        "geometric_mean" => Ok(Symmetry::GeometricMean),
        "harmonic_mean" => Ok(Symmetry::HarmonicMean),
        "min" => Ok(Symmetry::Min),
        "max" => Ok(Symmetry::Max),
        _ => Err(()),
    }
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

    fn add_points_tangents(&mut self, _py: Python, points: Vec<Vec<f64>>, tangents: Vec<Vec<f64>>) -> PyResult<usize> {
        let neuron = RStarPointTangents::new_with_tangents(
            points.iter().map(vec_to_array3).collect(),
            tangents.iter().map(vec_to_unitvector3).collect(),
        ).map_err(|s| PyErr::new::<exceptions::RuntimeError, _>(s))?;
        Ok(self.arena.add_neuron(neuron))
    }

    pub fn query_target(
        &self,
        _py: Python,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetry: Option<&str>,
    ) -> PyResult<Option<f64>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised"))?),
            _ => None,
        };
        Ok(self.arena
            .query_target(query_idx, target_idx, normalize, &sym)
        )
    }

    pub fn queries_targets(
        &self,
        _py: Python,
        query_idxs: Vec<NeuronIdx>,
        target_idxs: Vec<NeuronIdx>,
        normalize: bool,
        symmetry: Option<&str>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), f64>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised"))?),
            _ => None,
        };
        Ok(self.arena
            .queries_targets(&query_idxs, &target_idxs, normalize, &sym))
    }

    pub fn all_v_all(
        &self,
        _py: Python,
        normalize: bool,
        symmetry: Option<&str>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), Precision>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised"))?),
            _ => None,
        };
        Ok(self.arena.all_v_all(normalize, &sym))
    }

    pub fn len(&self, _py: Python) -> usize {
        self.arena.len()
    }

    pub fn self_hit(&self, _py: Python, idx: NeuronIdx) -> Option<Precision> {
        self.arena.self_hit(idx)
    }

    pub fn points(&self, _py: Python, idx: NeuronIdx) -> Option<Vec<Vec<Precision>>> {
        self.arena
            .points(idx)
            .map(|points| points.into_iter().map(|p| p.to_vec()).collect())
    }

    pub fn tangents(&self, _py: Python, idx: NeuronIdx) -> Option<Vec<Vec<Precision>>> {
        // TODO: better way to do this?
        self.arena.tangents(idx).map(|vectors| {
            vectors
                .into_iter()
                .map(|v| v.into_iter().map(|t| *t).collect())
                .collect()
        })
    }
}

#[pymodule]
fn pynblast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ArenaWrapper>()?;

    Ok(())
}
