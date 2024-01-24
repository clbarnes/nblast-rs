use numpy::ndarray::Array;
use pyo3::exceptions::PyValueError;
use pyo3::exceptions::{self, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::f64::{INFINITY, NEG_INFINITY};
use std::fmt::Display;
use std::io::{Cursor, Read, Write};
use std::str::FromStr;

use neurarbor::slab_tree::{NodeId, Tree};
use neurarbor::{edges_to_tree_with_data, resample_tree_points, Location, SpatialArbor, TopoArbor};

use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use nblast::nalgebra::base::{Unit, Vector3};
use nblast::{
    BinLookup, NblastArena, NblastNeuron, Neuron, NeuronIdx, Precision, RangeTable, ScoreCalc,
    ScoreMatrixBuilder, Symmetry, TangentAlpha, DEFAULT_BACKEND,
};

use nblast::rayon;
use rayon::prelude::*;

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

#[pyclass]
#[derive(Debug, Clone)]
pub struct ArenaWrapper {
    arena: NblastArena<Neuron>,
    k: usize,
}

#[pymethods]
impl ArenaWrapper {
    #[new]
    pub fn __new__(
        dist_thresholds: Vec<f64>,
        dot_thresholds: Vec<f64>,
        cells: Vec<f64>,
        k: usize,
        use_alpha: bool,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let rtable = RangeTable::new_from_bins(vec![dist_thresholds, dot_thresholds], cells)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("{}", e)))?;
        let score_calc = ScoreCalc::Table(rtable);
        let mut arena = NblastArena::new(score_calc, use_alpha);
        if let Some(t) = threads {
            arena = arena.with_threads(t);
        }
        Ok(Self { arena, k })
    }

    pub fn add_points(&mut self, _py: Python, points: PyReadonlyArray2<f64>) -> PyResult<usize> {
        let pshape = points.shape();
        if pshape[1] != 3 {
            return Err(PyErr::new::<PyValueError, _>("Points were not 3D"));
        }
        let neuron = np_to_neuron(points, self.k).map_err(PyErr::new::<PyValueError, _>)?;
        Ok(self.arena.add_neuron(neuron))
    }

    pub fn add_points_tangents_alphas(
        &mut self,
        _py: Python,
        points: PyReadonlyArray2<f64>,
        tangents: PyReadonlyArray2<f64>,
        alphas: PyReadonlyArray1<f64>,
    ) -> PyResult<usize> {
        let pshape = points.shape();
        if pshape[1] != 3 {
            return Err(PyErr::new::<PyValueError, _>("Points were not 3D"));
        }
        let tshape = points.shape();
        if tshape[1] != 3 {
            return Err(PyErr::new::<PyValueError, _>("Tangents were not 3D"));
        }
        if pshape[0] != tshape[0] || pshape[0] != alphas.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "Points, tangents, and alphas have inconsistent lengths",
            ));
        }
        let tangents_alphas = tangents
            .as_array()
            .rows()
            .into_iter()
            .zip(alphas.as_array())
            .map(|(t, a)| TangentAlpha {
                tangent: Unit::new_unchecked(Vector3::new(t[0], t[1], t[2])),
                alpha: *a,
            })
            .collect();
        let neuron = Neuron::new_with_tangents_alphas(
            points
                .as_array()
                .rows()
                .into_iter()
                .map(|r| [r[0], r[1], r[2]])
                .collect(),
            tangents_alphas,
        )
        .map_err(PyErr::new::<PyValueError, _>)?;
        Ok(self.arena.add_neuron(neuron))
    }

    pub fn query_target(
        &self,
        py: Python,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetry: Option<&str>,
    ) -> PyResult<Option<f64>> {
        py.allow_threads(|| {
            let sym = match symmetry {
                Some(s) => Some(str_to_sym(s).map_err(|_| {
                    PyErr::new::<exceptions::PyValueError, _>("Symmetry type not recognised")
                })?),
                _ => None,
            };
            Ok(self
                .arena
                .query_target(query_idx, target_idx, normalize, &sym))
        })
    }

    pub fn queries_targets(
        &self,
        query_idxs: Vec<NeuronIdx>,
        target_idxs: Vec<NeuronIdx>,
        normalize: bool,
        symmetry: Option<&str>,
        max_centroid_dist: Option<Precision>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), f64>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| {
                PyErr::new::<exceptions::PyValueError, _>(format!(
                    "Symmetry type '{}' not recognised",
                    s
                ))
            })?),
            _ => None,
        };
        Ok(self.arena.queries_targets(
            &query_idxs,
            &target_idxs,
            normalize,
            &sym,
            max_centroid_dist,
        ))
    }

    pub fn all_v_all(
        &self,
        _py: Python,
        normalize: bool,
        symmetry: Option<&str>,
        max_centroid_dist: Option<Precision>,
    ) -> PyResult<HashMap<(NeuronIdx, NeuronIdx), Precision>> {
        let sym = match symmetry {
            Some(s) => Some(str_to_sym(s).map_err(|_| {
                PyErr::new::<exceptions::PyValueError, _>("Symmetry type not recognised")
            })?),
            _ => None,
        };
        Ok(self.arena.all_v_all(normalize, &sym, max_centroid_dist))
    }

    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn self_hit(&self, _py: Python, idx: NeuronIdx) -> Option<Precision> {
        self.arena.self_hit(idx)
    }

    pub fn points<'py>(&self, py: Python<'py>, idx: NeuronIdx) -> Option<&'py PyArray2<Precision>> {
        let nrn = self.arena.neuron(idx)?;
        let points = nrn.points();
        let v = points.flatten().collect::<Vec<_>>();
        Some(
            Array::from_shape_vec((v.len() / 3, 3), v)
                .unwrap()
                .into_pyarray(py),
        )
    }

    pub fn tangents<'py>(
        &self,
        py: Python<'py>,
        idx: NeuronIdx,
    ) -> Option<&'py PyArray2<Precision>> {
        let nrn = self.arena.neuron(idx)?;
        let tangents = nrn.tangents();

        let v = tangents
            .into_iter()
            .fold(Vec::with_capacity(nrn.len() * 3), |mut v, t| {
                v.extend(t.into_inner().iter());
                v
            });
        Some(
            Array::from_shape_vec((nrn.len(), 3), v)
                .unwrap()
                .into_pyarray(py),
        )
    }

    pub fn alphas<'py>(&self, py: Python<'py>, idx: NeuronIdx) -> Option<&'py PyArray1<Precision>> {
        let v = self.arena.neuron(idx)?.alphas().collect();
        Some(PyArray1::from_vec(py, v))
    }

    /// Get the neuron as an array with N rows and 7 columns.
    /// The columns are the point locations x, y, z;
    /// the normalized tangent vectors x, y, z;
    /// and the alpha value, for each of the N points.
    pub fn neuron_array<'py>(
        &self,
        py: Python<'py>,
        id: NeuronIdx,
    ) -> Option<&'py PyArray2<Precision>> {
        let nrn = self.arena.neuron(id)?;
        let v = nrn.points().zip(nrn.tangents()).zip(nrn.alphas()).fold(
            Vec::with_capacity(nrn.len() * 7),
            |mut v, ((p, t), a)| {
                v.extend(p);
                v.extend(t.as_slice());
                v.push(a);
                v
            },
        );

        Some(
            Array::from_shape_vec((nrn.len(), 7), v)
                .unwrap()
                .into_pyarray(py),
        )
    }

    /// Add a neuron serialized by [serialize_neuron](#method.serialize_neuron).
    ///
    /// Note that a serialized neuron is only valid for one particular backend
    /// and neighborhood size.
    /// Unlike adding a neuron by its points or points/tangents/alphas,
    /// this saves having to rebuild the spatial index.
    pub fn add_serialized_neuron<'py>(
        &mut self,
        _py: Python<'py>,
        bytes: &[u8],
        format: &str,
    ) -> PyResult<NeuronIdx> {
        let fmt = NeuronFormat::from_str(format).map_err(|msg| PyValueError::new_err(msg))?;
        let n = fmt.read_neuron(bytes)?;
        Ok(self.arena.add_neuron(n))
    }

    /// Serialize a neuron and its spatial index.
    ///
    /// Note that a serialized neuron is only valid for one particular backend
    /// and neighborhood size; the format may not be stable.
    pub fn serialize_neuron<'py>(
        &self,
        py: Python<'py>,
        idx: NeuronIdx,
        format: &str,
    ) -> PyResult<Option<&'py PyBytes>> {
        let fmt = NeuronFormat::from_str(format).map_err(|msg| PyValueError::new_err(msg))?;
        let Some(nrn) = self.arena.neuron(idx) else {
            return Ok(None);
        };
        let mut buf = Cursor::new(Vec::default());
        fmt.write_neuron(nrn, &mut buf)?;
        Ok(Some(PyBytes::new(py, buf.into_inner().as_slice())))
    }

    pub fn deepcopy<'py>(&self, _py: Python<'py>) -> Self {
        self.clone()
    }
}

enum NeuronFormat {
    Json,
    Cbor,
    // consider bincode, postcard, flexbuffers
}

impl NeuronFormat {
    fn write_neuron<W: Write>(&self, nrn: &Neuron, w: &mut W) -> PyResult<()> {
        match self {
            Self::Json => {
                serde_json::to_writer(w, nrn).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
            Self::Cbor => {
                ciborium::into_writer(nrn, w).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
        }
    }

    fn read_neuron<R: Read>(&self, r: R) -> PyResult<Neuron> {
        match self {
            Self::Json => {
                serde_json::from_reader(r).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
            Self::Cbor => {
                ciborium::from_reader(r).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
        }
    }
}

impl Display for NeuronFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeuronFormat::Json => f.write_str("json"),
            NeuronFormat::Cbor => f.write_str("cbor"),
        }
    }
}

impl FromStr for NeuronFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "cbor" => Ok(Self::Cbor),
            _ => Err(format!("Unknown format '{s}'")),
        }
    }
}

#[pyclass]
struct ResamplingArbor {
    tree: Tree<(usize, [Precision; 3])>,
    tnid_to_id: HashMap<usize, NodeId>,
}

#[pymethods]
impl ResamplingArbor {
    #[new]
    fn __new__(
        table: Vec<(usize, Option<usize>, Precision, Precision, Precision)>,
    ) -> PyResult<Self> {
        let edges_with_data: Vec<_> = table
            .iter()
            .map(|(child, parent, x, y, z)| (*child, *parent, [*x, *y, *z]))
            .collect();
        let (tree, tnid_to_id) = edges_to_tree_with_data(&edges_with_data)
            .map_err(|_| PyErr::new::<exceptions::PyValueError, _>("Could not construct tree"))?;
        Ok(Self { tree, tnid_to_id })
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

    pub fn prune_twigs(&mut self, py: Python, threshold: Precision) -> usize {
        py.allow_threads(|| self.tree.prune_twigs(threshold).len())
    }

    pub fn prune_beyond_distance(&mut self, py: Python, threshold: Precision) -> usize {
        py.allow_threads(|| self.tree.prune_beyond_distance(threshold).len())
    }

    pub fn cable_length(&mut self, py: Python) -> Precision {
        py.allow_threads(|| self.tree.cable_length())
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

    pub fn copy(&self, _py: Python) -> Self {
        // todo: this is probably inefficient. Implement clone for slab_tree::Tree?
        let edges_with_data: Vec<_> = self
            .skeleton(_py)
            .into_iter()
            .map(|(node, parent, x, y, z)| (node, parent, [x, y, z]))
            .collect();
        let (tree, tnid_to_id) = edges_to_tree_with_data(&edges_with_data).unwrap();
        Self { tree, tnid_to_id }
    }
}

#[allow(non_snake_case)]
fn is_Nx3<T: Element>(p: &PyReadonlyArray2<T>) -> Result<(), &'static str> {
    if p.shape()[1] != 3 {
        return Err("Array must be Nx3");
    }
    Ok(())
}

fn np_to_neuron(points: PyReadonlyArray2<Precision>, k: usize) -> Result<Neuron, &'static str> {
    is_Nx3(&points)?;
    Neuron::new(
        points
            .as_array()
            .rows()
            .into_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect(),
        k,
    )
}

fn make_neurons_many(
    points_list: Vec<PyReadonlyArray2<Precision>>,
    k: usize,
    threads: Option<usize>,
) -> Result<Vec<Neuron>, &'static str> {
    if let Some(t) = threads {
        let vp = points_list
            .into_iter()
            .map(|np| {
                is_Nx3(&np)?;
                Ok(np
                    .as_array()
                    .rows()
                    .into_iter()
                    .map(|r| [r[0], r[1], r[2]])
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, &str>>()?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();

        Ok(pool.install(|| {
            vp.into_par_iter()
                .map(|ps| Neuron::new(ps, k).expect("Invalid neuron"))
                .collect()
        }))
    } else {
        points_list
            .into_iter()
            .map(|ps| np_to_neuron(ps, k))
            .collect()
    }
}

fn inner_bounds_to_binlookup(mut v: Vec<Precision>) -> BinLookup<Precision> {
    v.insert(0, NEG_INFINITY);
    v.push(INFINITY);
    BinLookup::new(v, (true, true)).expect("Failed to build BinLookup")
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_score_matrix(
    _py: Python,
    points: Vec<PyReadonlyArray2<Precision>>,
    k: usize,
    seed: u64,
    use_alpha: bool,
    matching_sets: Vec<Vec<usize>>,
    nonmatching_sets: Option<Vec<Vec<usize>>>,
    dist_n_bins: Option<usize>,
    dist_inner_bounds: Option<Vec<Precision>>,
    dot_n_bins: Option<usize>,
    dot_inner_bounds: Option<Vec<Precision>>,
    max_matching_pairs: Option<usize>,
    max_nonmatching_pairs: Option<usize>,
    threads: Option<usize>,
) -> PyResult<(Vec<Precision>, Vec<Precision>, Vec<Precision>)> {
    let neurons = make_neurons_many(points, k, threads).map_err(PyRuntimeError::new_err)?;
    let mut smatb = ScoreMatrixBuilder::new(neurons, seed);
    smatb.set_threads(threads).set_use_alpha(use_alpha);

    if let Some(mmp) = max_matching_pairs {
        smatb.set_max_matching_pairs(mmp);
    }

    if let Some(mnmp) = max_nonmatching_pairs {
        smatb.set_max_nonmatching_pairs(mnmp);
    }

    for m in matching_sets.into_iter() {
        smatb.add_matching_set(&m);
    }
    if let Some(ns) = nonmatching_sets {
        for n in ns.into_iter() {
            smatb.add_nonmatching_set(&n);
        }
    }

    if let Some(inner) = dist_inner_bounds {
        smatb.set_dist_lookup(inner_bounds_to_binlookup(inner));
    } else if let Some(n) = dist_n_bins {
        smatb.set_n_dist_bins(n);
    } else {
        unimplemented!("should supply dist_inner_bounds or dist_n_bins");
    }

    if let Some(inner) = dot_inner_bounds {
        smatb.set_dot_lookup(inner_bounds_to_binlookup(inner));
    } else if let Some(n) = dot_n_bins {
        smatb.set_n_dot_bins(n);
    } else {
        unimplemented!("should supply dot_inner_bounds or dot_n_bins");
    }

    let mut table = smatb.build().expect("Failed to build score matrix");
    let dot_bounds = table.bins_lookup.lookups.pop().unwrap().bin_boundaries;
    let dist_bounds = table.bins_lookup.lookups.pop().unwrap().bin_boundaries;
    Ok((dist_bounds, dot_bounds, table.cells))
}

#[pyfunction]
fn get_version(_py: Python) -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[pyfunction]
fn backend(_py: Python) -> String {
    DEFAULT_BACKEND.to_string()
}

#[pymodule]
fn pynblast(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ArenaWrapper>()?;
    m.add_class::<ResamplingArbor>()?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(build_score_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(backend, m)?)?;

    Ok(())
}
