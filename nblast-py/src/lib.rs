use core::fmt::Debug;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;

use neurarbor::slab_tree::{NodeId, Tree};
use neurarbor::{edges_to_tree_with_data, resample_tree_points, Location, TopoArbor};

use nblast::nalgebra::base::{Unit, Vector3};
use nblast::{
    table_to_fn, DistDot, NblastArena, NeuronIdx, Precision, RStarTangentsAlphas, Symmetry,
    TangentAlpha,
};

fn vec_to_array3<T: Sized + Copy>(v: &Vec<T>) -> [T; 3] {
    [v[0], v[1], v[2]]
}

fn vec_to_unitvector3<T: 'static + Sized + Copy + PartialEq + Debug>(
    v: &Vec<T>,
) -> Unit<Vector3<T>> {
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

#[cfg(not(test))]
#[pyclass]
pub struct ArenaWrapper {
    // TODO: can this box be avoided?
    arena: NblastArena<RStarTangentsAlphas, Box<dyn Fn(&DistDot) -> Precision + Sync + Send>>,
    k: usize,
}

#[cfg(not(test))]
#[pymethods]
impl ArenaWrapper {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        dist_thresholds: Vec<f64>,
        dot_thresholds: Vec<f64>,
        cells: Vec<f64>,
        k: usize,
    ) -> PyResult<()> {
        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);
        obj.init(Self {
            arena: NblastArena::new(Box::new(score_fn)),
            k,
        });
        Ok(())
    }

    pub fn add_points(&mut self, py: Python, points: Vec<Vec<f64>>) -> PyResult<usize> {
        py.allow_threads(|| {
            let neuron = RStarTangentsAlphas::new(points.iter().map(vec_to_array3), self.k)
                .map_err(PyErr::new::<exceptions::RuntimeError, _>)?;
            Ok(self.arena.add_neuron(neuron))
        })
    }

    pub fn add_points_tangents_alphas(
        &mut self,
        py: Python,
        points: Vec<Vec<f64>>,
        tangents: Vec<Vec<f64>>,
        alphas: Vec<f64>,
    ) -> PyResult<usize> {
        py.allow_threads(|| {
            let tangents_alphas = tangents
                .iter()
                .zip(alphas.iter())
                .map(|(t, a)| TangentAlpha {
                    tangent: vec_to_unitvector3(t),
                    alpha: *a,
                })
                .collect();
            let neuron = RStarTangentsAlphas::new_with_tangents_alphas(
                points.iter().map(vec_to_array3),
                tangents_alphas,
            )
            .map_err(PyErr::new::<exceptions::RuntimeError, _>)?;
            Ok(self.arena.add_neuron(neuron))
        })
    }

    pub fn query_target(
        &self,
        py: Python,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetry: Option<&str>,
        use_alpha: bool,
    ) -> PyResult<Option<f64>> {
        py.allow_threads(|| {
            let sym = match symmetry {
                Some(s) => Some(str_to_sym(s).map_err(|_| {
                    PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised")
                })?),
                _ => None,
            };
            Ok(self
                .arena
                .query_target(query_idx, target_idx, normalize, &sym, use_alpha))
        })
    }

    pub fn queries_targets(
        &self,
        query_idxs: Vec<NeuronIdx>,
        target_idxs: Vec<NeuronIdx>,
        normalize: bool,
        symmetry: Option<&str>,
        use_alpha: bool,
        threads: Option<usize>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), f64>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| {
                PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised")
            })?),
            _ => None,
        };
        Ok(self.arena.queries_targets(
            &query_idxs,
            &target_idxs,
            normalize,
            &sym,
            use_alpha,
            threads,
        ))
    }

    pub fn all_v_all(
        &self,
        _py: Python,
        normalize: bool,
        symmetry: Option<&str>,
        use_alpha: bool,
        threads: Option<usize>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), Precision>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| {
                PyErr::new::<exceptions::ValueError, _>("Symmetry type not recognised")
            })?),
            _ => None,
        };
        Ok(self.arena.all_v_all(normalize, &sym, use_alpha, threads))
    }

    pub fn len(&self, _py: Python) -> usize {
        self.arena.len()
    }

    pub fn self_hit(&self, _py: Python, idx: NeuronIdx, use_alpha: bool) -> Option<Precision> {
        self.arena.self_hit(idx, use_alpha)
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
                .map(|v| v.into_iter().cloned().collect())
                .collect()
        })
    }

    pub fn alphas(&self, _py: Python, idx: NeuronIdx) -> Option<Vec<Precision>> {
        self.arena.alphas(idx)
    }
}

#[cfg(not(test))]
#[pyclass]
struct ResamplingArbor {
    tree: Tree<(usize, [Precision; 3])>,
    tnid_to_id: HashMap<usize, NodeId>,
}

#[cfg(not(test))]
#[pymethods]
impl ResamplingArbor {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        table: Vec<(usize, Option<usize>, Precision, Precision, Precision)>,
    ) -> PyResult<()> {
        let edges_with_data: Vec<_> = table
            .iter()
            .map(|(child, parent, x, y, z)| (*child, *parent, [*x, *y, *z]))
            .collect();
        let (tree, tnid_to_id) = edges_to_tree_with_data(&edges_with_data)
            .map_err(|_| PyErr::new::<exceptions::ValueError, _>("Could not construct tree"))?;
        obj.init(Self { tree, tnid_to_id });
        Ok(())
    }

    pub fn prune_at(&mut self, py: Python, ids: Vec<usize>) -> usize {
        py.allow_threads(|| {
            let inner_ids: Vec<_> = ids
                .iter()
                .filter_map(|k| self.tnid_to_id.get(k).cloned())
                .collect();
            self.tree.prune_at(&inner_ids).len()
        })
    }

    pub fn prune_branches_containing(&mut self, py: Python, ids: Vec<usize>) -> usize {
        py.allow_threads(|| {
            let inner_ids: Vec<_> = ids
                .iter()
                .filter_map(|k| self.tnid_to_id.get(k).cloned())
                .collect();
            self.tree.prune_branches_containing(&inner_ids).len()
        })
    }

    pub fn prune_below_strahler(&mut self, py: Python, threshold: usize) -> usize {
        py.allow_threads(|| self.tree.prune_below_strahler(threshold).len())
    }

    pub fn prune_beyond_branches(&mut self, py: Python, threshold: usize) -> usize {
        py.allow_threads(|| self.tree.prune_beyond_branches(threshold).len())
    }

    pub fn prune_beyond_steps(&mut self, py: Python, threshold: usize) -> usize {
        py.allow_threads(|| self.tree.prune_beyond_steps(threshold).len())
    }

    pub fn points(&self, py: Python, resample: Option<Precision>) -> Vec<Vec<Precision>> {
        py.allow_threads(|| {
            if let Some(len) = resample {
                resample_tree_points(&self.tree, len)
                    .into_iter()
                    .map(|a| a.to_vec())
                    .collect()
            } else {
                self.tree
                    .root()
                    .unwrap()
                    .traverse_pre_order()
                    .map(|n| n.data().location().to_vec())
                    .collect()
            }
        })
    }

    pub fn skeleton(
        &self,
        py: Python,
    ) -> Vec<(usize, Option<usize>, Precision, Precision, Precision)> {
        py.allow_threads(|| {
            self.tree
                .root()
                .unwrap()
                .traverse_pre_order()
                .map(|n| {
                    let (tnid, loc) = n.data();
                    (
                        *tnid,
                        n.parent().map(|p| p.data().0),
                        loc[0],
                        loc[1],
                        loc[2],
                    )
                })
                .collect()
        })
    }

    pub fn root(&self, _py: Python) -> usize {
        self.tree.root().unwrap().data().0
    }
}

#[cfg(not(test))]
#[pymodule]
fn pynblast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ArenaWrapper>()?;
    m.add_class::<ResamplingArbor>()?;

    #[pyfn(m, "get_version")]
    fn get_version(_py: Python) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    Ok(())
}
