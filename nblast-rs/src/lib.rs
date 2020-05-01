//! Implementation of the NBLAST algorithm for quantifying neurons' morphological similarity.
//! Originally published in
//! [Costa et al. (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4961245/)
//! and implemented as part of the
//! [NeuroAnatomy Toolbox](http://natverse.org/).
//!
//! # Algorithm
//!
//! Each neuron is passed in as a point cloud sample (the links between the points are not required).
//! A tangent vector is calculated for each point, based on its location and that of its 4 nearest neighbors.
//! To query the similarity of neuron `Q` to neuron `T`:
//!
//! - Take a point and its associated tangent in `Q`
//!   - Find the nearest point in `T`, and its associated tangent
//!   - Compute the distance between the two points
//!   - Compute the absolute dot product of the two tangents
//!   - Apply some empirically-derived function to the (distance, dot_product) tuple
//!     - As published, this is the log probabity ratio of any pair belonging to closely related or unrelated neurons
//! - Repeat for all points, summing the results
//!
//! The result is not easily comparable:
//! it is highly dependent on the size of the point cloud
//! and is not commutative, i.e. `f(Q, T) != f(T, Q)`.
//!
//! To make queries between two pairs of neurons comparable,
//! the result can be normalized by the "self-hit" score of the query, i.e. `f(Q, Q)`.
//!
//! To make the result commutative, the forward `f(Q, T)` and backward `f(T, Q)` scores can be combined in some way.
//! This library supports several means (arithmetic, harmonic, and geometric), the minimum, and the maximum.
//! The choice will depend on the application.
//! This can be applied after the scores are normalized.
//!
//! More information on the algorithm can be found
//! [here](http://jefferislab.org/si/nblast).
//!
//! # Usage
//!
//! The [QueryNeuron](trait.QueryNeuron.html) and [TargetNeuron](trait.TargetNeuron.html) traits
//! define types which can be compared with NBLAST.
//! All `TargetNeuron`s are also `QueryNeuron`s.
//!
//! [QueryPointTangents](struct.QueryPointTangents.html) and
//! [RStarPointTangents](struct.RStarPointTangents.html) implement these, respectively.
//! Both can be created with pre-calculated tangents, or calculate them on instantiation.
//!
//! The [NblastArena](struct.NblastArena.html) contains a collection of `TargetNeuron`s
//! and a function to apply to pointwise (distance, absolute dot product) pairs to generate
//! a score for that point match, for convenient many-to-many comparisons.
//! A pre-calculated table of point match scores can be converted into a function with [table_to_fn](fn.table_to_fn.html).
use nalgebra::base::{Matrix3, Unit, Vector3};
use rstar::primitives::PointWithData;
use rstar::RTree;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use nalgebra;

// NOTE: will panic if this is changed due to use of Matrix3x5
// const N_NEIGHBORS: usize = 5;

/// Floating point precision type used internally
pub type Precision = f64;
pub type Point3 = [Precision; 3];
pub type Normal3 = Unit<Vector3<Precision>>;

type PointWithIndex = PointWithData<usize, Point3>;

fn geometric_mean(a: Precision, b: Precision) -> Precision {
    (a.max(0.0) * b.max(0.0)).sqrt()
}

#[derive(Copy, Clone, Debug)]
pub struct TangentAlpha {
    pub tangent: Normal3,
    pub alpha: Precision,
}

impl TangentAlpha {
    fn new_from_points<'a>(points: impl Iterator<Item = &'a Point3>) -> Self {
        let inertia = calc_inertia(points);
        let eig = inertia.symmetric_eigen();
        let mut sum = 0.0;
        let mut vals: Vec<_> = eig
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(idx, v)| {
                sum += v;
                (idx, v)
            })
            .collect();
        vals.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let alpha = (vals[0].1 - vals[1].1) / sum;

        // ? new_unchecked
        let tangent = Unit::new_normalize(eig.eigenvectors.column(vals[0].0).into());

        Self { tangent, alpha }
    }
}

/// Enumeration of methods to ensure that queries are symmetric/ commutative
/// (i.e. f(q, t) = f(t, q)).
/// Specific applications will require different methods.
/// Geometric and harmonic means bound the output to be >= 0.0.
/// Geometric mean may work best with non-normalized queries.
/// Min may work if an unknown one of the query and target is incomplete.
pub enum Symmetry {
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    Min,
    Max,
}

fn apply_symmetry(
    symmetry: &Symmetry,
    query_score: Precision,
    target_score: Precision,
) -> Precision {
    match symmetry {
        Symmetry::ArithmeticMean => (query_score + target_score) / 2.0,
        Symmetry::GeometricMean => geometric_mean(query_score, target_score),
        Symmetry::HarmonicMean => {
            if query_score.max(0.0) * target_score.max(0.0) == 0.0 {
                0.0
            } else {
                2.0 / (1.0 / query_score + 1.0 / target_score)
            }
        }
        Symmetry::Min => query_score.min(target_score),
        Symmetry::Max => query_score.max(target_score),
    }
}

/// The result of comparing two (point, tangent) tuples.
/// Contains the Euclidean distance between the points,
/// and the absolute dot product of the (unit) tangents,
/// i.e. the absolute cosine of the angle between them.
#[derive(Debug, Clone, Copy)]
pub struct DistDot {
    pub dist: Precision,
    pub dot: Precision,
}

impl Default for DistDot {
    fn default() -> Self {
        Self {
            dist: 0.0,
            dot: 1.0,
        }
    }
}

pub trait Neuron {
    /// Number of points in the neuron.
    fn len(&self) -> usize;

    /// Whether the number of points is 0.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return an owned copy of the points present in the neuron.
    /// The order is not guaranteed, but is consistent with
    /// [tangents](#method.tangents).
    fn points(&self) -> Vec<Point3>;

    /// Return an owned copy of the unit tangents present in the neuron.
    /// The order is not guaranteed, but is consistent with
    /// [points](#method.points).
    fn tangents(&self) -> Vec<Normal3>;

    /// Return an owned copy of the alpha values for points in the neuron.
    /// The order is consistent with [points](#method.points)
    /// and [tangents](#method.tangents).
    fn alphas(&self) -> Vec<Precision>;
}

/// Trait for objects which can be used as queries
/// (not necessarily as targets) with NBLAST.
/// See [TargetNeuron](trait.TargetNeuron.html).
pub trait QueryNeuron: Neuron {
    /// Calculate the raw NBLAST score by comparing this neuron to
    /// the given target neuron, using the given score function.
    /// The score function is applied to each point match distance and summed.
    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
        use_alpha: bool,
    ) -> Precision;

    /// The raw NBLAST score if this neuron was compared with itself using the given score function.
    /// Used for normalisation.
    fn self_hit(&self, score_fn: &impl Fn(&DistDot) -> Precision, use_alpha: bool) -> Precision;
}

fn subtract_points(p1: &Point3, p2: &Point3) -> Point3 {
    let mut result = [0.0; 3];
    for ((rref, v1), v2) in result.iter_mut().zip(p1).zip(p2) {
        *rref = v1 - v2;
    }
    result
}

fn center_points<'a>(points: impl Iterator<Item = &'a Point3>) -> impl Iterator<Item = Point3> {
    let mut points_vec = Vec::default();
    let mut means: Point3 = [0.0, 0.0, 0.0];
    for pt in points {
        points_vec.push(*pt);
        for (sum, v) in means.iter_mut().zip(pt.iter()) {
            *sum += v;
        }
    }

    for val in means.iter_mut() {
        *val /= points_vec.len() as Precision;
    }
    let subtract = move |p| subtract_points(&p, &means);
    points_vec.into_iter().map(subtract)
}

fn dot(a: &[Precision], b: &[Precision]) -> Precision {
    a.iter()
        .zip(b.iter())
        .fold(0.0, |sum, (ax, bx)| sum + ax * bx)
}

/// Calculate inertia from iterator of points.
/// This is an implementation of matrix * matrix.transpose(),
/// to sidestep the fixed-size constraints of linalg's built-in classes.
/// Only calculates the lower triangle and diagonal.
fn calc_inertia<'a>(points: impl Iterator<Item = &'a Point3>) -> Matrix3<Precision> {
    let mut xs = Vec::default();
    let mut ys = Vec::default();
    let mut zs = Vec::default();
    for point in center_points(points) {
        xs.push(point[0]);
        ys.push(point[1]);
        zs.push(point[2]);
    }
    Matrix3::new(
        dot(&xs, &xs),
        0.0,
        0.0,
        dot(&ys, &xs),
        dot(&ys, &ys),
        0.0,
        dot(&zs, &xs),
        dot(&zs, &ys),
        dot(&zs, &zs),
    )
}

fn points_to_rtree(
    points: impl Iterator<Item = impl std::borrow::Borrow<Point3>>,
) -> Result<RTree<PointWithIndex>, &'static str> {
    Ok(RTree::bulk_load(
        points
            .enumerate()
            .map(|(idx, point)| PointWithIndex::new(idx, *point.borrow()))
            .collect(),
    ))
}

fn points_to_rtree_tangents_alphas(
    points: impl Iterator<Item = impl std::borrow::Borrow<Point3>> + ExactSizeIterator + Clone,
    k: usize,
) -> Result<(RTree<PointWithIndex>, Vec<TangentAlpha>), &'static str> {
    if points.len() < k {
        return Err("Too few points to generate tangents");
    }
    let rtree = points_to_rtree(points.clone())?;
    let tangents_alphas = points
        .map(|p| {
            TangentAlpha::new_from_points(
                rtree
                    .nearest_neighbor_iter(p.borrow())
                    .take(k)
                    .map(|pwd| pwd.position()),
            )
        })
        .collect();

    Ok((rtree, tangents_alphas))
}

/// Minimal struct to use as the query (not the target) of an NBLAST
/// comparison.
/// Equivalent to "dotprops" in the reference implementation.
#[derive(Clone)]
pub struct PointsTangentsAlphas {
    /// Locations of points in point cloud.
    points: Vec<Point3>,
    /// For each point in the cloud, a unit-length vector and a colinearity metric.
    tangents_alphas: Vec<TangentAlpha>,
}

impl PointsTangentsAlphas {
    /// Calculates tangents from the given points.
    /// Note that this constructs a spatial index in order to calculate the tangents,
    /// and then throws it away: you may as well use a [TargetNeuron](trait.TargetNeuron.html)
    /// type, with regards to performance.
    /// `k` is the number of points tangents will be calculated with,
    /// and includes the point itself.
    pub fn new(points: Vec<Point3>, k: usize) -> Result<Self, &'static str> {
        points_to_rtree_tangents_alphas(points.iter(), k).map(|(_, tangents_alphas)| Self {
            points,
            tangents_alphas,
        })
    }
}

impl Neuron for PointsTangentsAlphas {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn points(&self) -> Vec<Point3> {
        self.points.clone()
    }

    fn tangents(&self) -> Vec<Normal3> {
        self.tangents_alphas.iter().map(|ta| ta.tangent).collect()
    }

    fn alphas(&self) -> Vec<Precision> {
        self.tangents_alphas.iter().map(|ta| ta.alpha).collect()
    }
}

impl QueryNeuron for PointsTangentsAlphas {
    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
        use_alpha: bool,
    ) -> Precision {
        let mut score_total: Precision = 0.0;

        for (q_pt, q_ta) in self.points.iter().zip(self.tangents_alphas.iter()) {
            let alpha = if use_alpha { Some(q_ta.alpha) } else { None };
            score_total += score_fn(&target.nearest_match_dist_dot(q_pt, &q_ta.tangent, alpha));
        }
        score_total
    }

    fn self_hit(&self, score_fn: &impl Fn(&DistDot) -> Precision, use_alpha: bool) -> Precision {
        if use_alpha {
            self.tangents_alphas
                .iter()
                .map(|ta| {
                    score_fn(&DistDot {
                        dist: 0.0,
                        dot: ta.alpha,
                    })
                })
                .fold(0.0, |total, s| total + s)
        } else {
            score_fn(&DistDot {
                dist: 0.0,
                dot: 1.0,
            }) * self.len() as Precision
        }
    }
}

/// Trait describing a neuron which can be the target (or the query)
/// of an NBLAST match.
pub trait TargetNeuron: QueryNeuron {
    /// For a given point and tangent vector,
    /// get the distance to its nearest point in the target, and the absolute dot product
    /// with that neighbor's tangent (i.e. absolute cosine of the angle, as they are both unit-length).
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot;
}

/// Target neuron using an [R*-tree](https://en.wikipedia.org/wiki/R*_tree) for spatial queries.
#[derive(Clone)]
pub struct RStarTangentsAlphas {
    rtree: RTree<PointWithIndex>,
    tangents_alphas: Vec<TangentAlpha>,
}

impl RStarTangentsAlphas {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new<T: std::borrow::Borrow<Point3>>(
        points: impl IntoIterator<
            Item = T,
            IntoIter = impl Iterator<Item = T> + ExactSizeIterator + Clone,
        >,
        k: usize,
    ) -> Result<Self, &'static str> {
        points_to_rtree_tangents_alphas(points.into_iter(), k).map(|(rtree, tangents_alphas)| {
            RStarTangentsAlphas {
                rtree,
                tangents_alphas,
            }
        })
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas<T: std::borrow::Borrow<Point3>>(
        points: impl IntoIterator<
            Item = T,
            IntoIter = impl Iterator<Item = T> + ExactSizeIterator + Clone,
        >,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Result<Self, &'static str> {
        points_to_rtree(points.into_iter()).map(|rtree| RStarTangentsAlphas {
            rtree,
            tangents_alphas,
        })
    }
}

impl Neuron for RStarTangentsAlphas {
    fn len(&self) -> usize {
        self.tangents_alphas.len()
    }

    fn points(&self) -> Vec<Point3> {
        let mut unsorted: Vec<&PointWithIndex> = self.rtree.iter().collect();
        unsorted.sort_by_key(|pwd| pwd.data);
        unsorted.into_iter().map(|pwd| *pwd.position()).collect()
    }

    fn tangents(&self) -> Vec<Normal3> {
        self.tangents_alphas.iter().map(|ta| ta.tangent).collect()
    }

    fn alphas(&self) -> Vec<Precision> {
        self.tangents_alphas.iter().map(|ta| ta.alpha).collect()
    }
}

impl QueryNeuron for RStarTangentsAlphas {
    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
        use_alpha: bool,
    ) -> Precision {
        let mut score_total: Precision = 0.0;
        for q_pt_idx in self.rtree.iter() {
            let tangent_alpha = self.tangents_alphas[q_pt_idx.data];
            let alpha = if use_alpha {
                Some(tangent_alpha.alpha)
            } else {
                None
            };
            let dd =
                target.nearest_match_dist_dot(q_pt_idx.position(), &tangent_alpha.tangent, alpha);
            let score = score_fn(&dd);
            score_total += score;
        }
        score_total
    }

    fn self_hit(&self, score_fn: &impl Fn(&DistDot) -> Precision, use_alpha: bool) -> Precision {
        if use_alpha {
            self.tangents_alphas
                .iter()
                .map(|ta| {
                    score_fn(&DistDot {
                        dist: 0.0,
                        dot: ta.alpha,
                    })
                })
                .fold(0.0, |total, s| total + s)
        } else {
            score_fn(&DistDot {
                dist: 0.0,
                dot: 1.0,
            }) * self.len() as Precision
        }
    }
}

impl TargetNeuron for RStarTangentsAlphas {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.rtree
            .nearest_neighbor_iter_with_distance(point)
            .next()
            .map(|(element, dist2)| {
                let ta = self.tangents_alphas[element.data];
                let raw_dot = ta.tangent.dot(tangent).abs();
                let dot = match alpha {
                    Some(a) => raw_dot * geometric_mean(a, ta.alpha),
                    None => raw_dot,
                };
                DistDot {
                    dist: dist2.sqrt(),
                    dot,
                }
            })
            .expect("impossible")
    }
}

// ? consider using nalgebra's Point3 in PointWithIndex, for consistency
// ^ can't implement rstar::Point for nalgebra::geometry::Point3 because of orphan rules
// TODO: replace Precision with float generic

/// Given the upper bounds of a number of bins, find which bin the value falls into.
/// Values outside of the range fall into the bottom or top bin.
fn find_bin_binary(value: Precision, upper_bounds: &[Precision]) -> usize {
    let raw = match upper_bounds.binary_search_by(|bound| bound.partial_cmp(&value).unwrap()) {
        Ok(v) => v + 1,
        Err(v) => v,
    };
    let highest = upper_bounds.len() - 1;
    if raw > highest {
        highest
    } else {
        raw
    }
}

/// Convert an empirically-derived table mapping pointwise distance and tangent absolute dot products
/// to pointwise scores into a function which can be passed to neuron queries.
/// These scores are then summed across all points in the query to give the raw NBLAST score.
///
/// Cells are passed in dist-major order
/// i.e. if the original table had distance bins in the left margin
/// and dot product bins on the top margin,
/// the cells should be given in row-major order.
///
/// Each bin is identified by its upper bound:
/// the lower bound is implicitly the previous bin's upper bound, or zero.
/// The output is constrained to the limits of the table.
pub fn table_to_fn(
    dist_thresholds: Vec<Precision>,
    dot_thresholds: Vec<Precision>,
    cells: Vec<Precision>,
) -> impl Fn(&DistDot) -> Precision {
    if dist_thresholds.len() * dot_thresholds.len() != cells.len() {
        panic!("Number of cells in table do not match number of columns/rows");
    }

    move |dd: &DistDot| -> Precision {
        let col_idx = find_bin_binary(dd.dot, &dot_thresholds);
        let row_idx = find_bin_binary(dd.dist, &dist_thresholds);

        let lin_idx = row_idx * dot_thresholds.len() + col_idx;
        cells[lin_idx]
    }
}

#[derive(Clone)]
struct NeuronSelfHit<N: QueryNeuron> {
    neuron: N,
    self_hit: Precision,
    self_hit_alpha: Precision,
}

impl<N: QueryNeuron> NeuronSelfHit<N> {
    fn score(&self, use_alpha: bool) -> Precision {
        if use_alpha {
            self.self_hit_alpha
        } else {
            self.self_hit
        }
    }
}

/// Struct for caching a number of neurons for multiple comparable NBLAST queries.
#[derive(Clone)]
pub struct NblastArena<N, F>
where
    N: TargetNeuron,
    F: Fn(&DistDot) -> Precision,
{
    neurons_scores: Vec<NeuronSelfHit<N>>,
    score_fn: F,
}

pub type NeuronIdx = usize;

// TODO: caching strategy
impl<N, F> NblastArena<N, F>
where
    N: TargetNeuron + Sync,
    F: Fn(&DistDot) -> Precision + Sync,
{
    pub fn new(score_fn: F) -> Self {
        Self {
            neurons_scores: Vec::default(),
            score_fn,
        }
    }

    fn next_id(&self) -> NeuronIdx {
        self.neurons_scores.len()
    }

    /// Returns an index which is then used to make queries.
    pub fn add_neuron(&mut self, neuron: N) -> NeuronIdx {
        let idx = self.next_id();
        let self_hit = neuron.self_hit(&self.score_fn, false);
        let self_hit_alpha = neuron.self_hit(&self.score_fn, true);
        self.neurons_scores.push(NeuronSelfHit {
            neuron,
            self_hit,
            self_hit_alpha,
        });
        idx
    }

    /// Make a single query using the given indexes.
    /// `normalize` divides the result by the self-hit score of the query neuron.
    /// `symmetry`, if `Some`, also calculates the reverse score
    /// (normalizing it if necessary), and then applies a function to ensure
    /// that the query is symmetric/ commutative.
    pub fn query_target(
        &self,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetry: &Option<Symmetry>,
        use_alpha: bool,
    ) -> Option<Precision> {
        // ? consider separate methods
        let q = self.neurons_scores.get(query_idx)?;
        let t = self.neurons_scores.get(target_idx)?;
        let mut score = q.neuron.query(&t.neuron, &self.score_fn, use_alpha);
        if normalize {
            score /= q.score(use_alpha)
        }
        match symmetry {
            Some(s) => {
                let mut score2 = t.neuron.query(&q.neuron, &self.score_fn, use_alpha);
                if normalize {
                    score2 /= t.score(use_alpha);
                }
                Some(apply_symmetry(s, score, score2))
            }
            _ => Some(score),
        }
    }

    /// Make many queries using the Cartesian product of the query and target indices.
    /// `threads` configures parallelisation if the `parallel` feature is enabled:
    /// `None` is done in serial, and for `Some(n)`, `n` is passed to
    /// [rayon::ThreadPoolBuilder::num_threads](https://docs.rs/rayon/1.3.0/rayon/struct.ThreadPoolBuilder.html#method.num_threads).
    ///
    /// See [query_target](#method.query_target) for details on `normalize` and `symmetry`.
    pub fn queries_targets(
        &self,
        query_idxs: &[NeuronIdx],
        target_idxs: &[NeuronIdx],
        normalize: bool,
        symmetry: &Option<Symmetry>,
        use_alpha: bool,
        threads: Option<usize>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let mut out = HashMap::with_capacity(query_idxs.len() * target_idxs.len());
        let mut out_keys: HashSet<(NeuronIdx, NeuronIdx)> = HashSet::default();
        let mut jobs: HashSet<(NeuronIdx, NeuronIdx)> = HashSet::default();
        for q_idx in query_idxs {
            for t_idx in target_idxs {
                let key = (*q_idx, *t_idx);
                if q_idx == t_idx {
                    if let Some(ns) = self.neurons_scores.get(*q_idx) {
                        out.insert(key, if normalize { 1.0 } else { ns.score(use_alpha) });
                    };
                    continue;
                }
                out_keys.insert(key);
                jobs.insert(key);
                if symmetry.is_some() {
                    jobs.insert((*t_idx, *q_idx));
                }
            }
        }

        let jobs_vec: Vec<_> = jobs.into_iter().collect();
        let raw = pairs_to_raw(self, &jobs_vec, normalize, use_alpha, threads);

        for key in out_keys.into_iter() {
            if let Some(forward) = raw.get(&key) {
                if let Some(s) = symmetry {
                    // ! this applies symmetry twice if idx is in both input and output,
                    // but it's a cheap function
                    if let Some(backward) = raw.get(&(key.1, key.0)) {
                        out.insert(key, apply_symmetry(s, *forward, *backward));
                    }
                } else {
                    out.insert(key, *forward);
                }
            }
        }
        out
    }

    pub fn self_hit(&self, idx: NeuronIdx, use_alpha: bool) -> Option<Precision> {
        self.neurons_scores.get(idx).map(|n| n.score(use_alpha))
    }

    /// Query every neuron against every other neuron.
    /// See [queries_targets](#method.queries_targets) for more details.
    pub fn all_v_all(
        &self,
        normalize: bool,
        symmetry: &Option<Symmetry>,
        use_alpha: bool,
        threads: Option<usize>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let idxs: Vec<NeuronIdx> = (0..self.len()).collect();
        self.queries_targets(&idxs, &idxs, normalize, symmetry, use_alpha, threads)
    }

    pub fn is_empty(&self) -> bool {
        self.neurons_scores.is_empty()
    }

    /// Number of neurons in the arena.
    pub fn len(&self) -> usize {
        self.neurons_scores.len()
    }

    pub fn points(&self, idx: NeuronIdx) -> Option<Vec<Point3>> {
        self.neurons_scores.get(idx).map(|n| n.neuron.points())
    }

    pub fn tangents(&self, idx: NeuronIdx) -> Option<Vec<Normal3>> {
        self.neurons_scores.get(idx).map(|n| n.neuron.tangents())
    }

    pub fn alphas(&self, idx: NeuronIdx) -> Option<Vec<Precision>> {
        self.neurons_scores.get(idx).map(|n| n.neuron.alphas())
    }
}

fn pairs_to_raw_serial<N, F>(
    arena: &NblastArena<N, F>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision>
where
    N: TargetNeuron + Sync,
    F: Fn(&DistDot) -> Precision + Sync,
{
    pairs
        .iter()
        .filter_map(|(q_idx, t_idx)| {
            arena
                .query_target(*q_idx, *t_idx, normalize, &None, use_alpha)
                .map(|s| ((*q_idx, *t_idx), s))
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn pairs_to_raw<N, F>(
    arena: &NblastArena<N, F>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
    threads: Option<usize>,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision>
where
    N: TargetNeuron + Sync,
    F: Fn(&DistDot) -> Precision + Sync,
{
    pairs_to_raw_serial(arena, pairs, normalize, use_alpha)
}

#[cfg(feature = "parallel")]
fn pairs_to_raw<N, F>(
    arena: &NblastArena<N, F>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
    threads: Option<usize>,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision>
where
    N: TargetNeuron + Sync,
    F: Fn(&DistDot) -> Precision + Sync,
{
    if let Some(t) = threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            pairs
                .par_iter()
                .filter_map(|(q_idx, t_idx)| {
                    arena
                        .query_target(*q_idx, *t_idx, normalize, &None, use_alpha)
                        .map(|s| ((*q_idx, *t_idx), s))
                })
                .collect()
        })
    } else {
        pairs_to_raw_serial(arena, pairs, normalize, use_alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Precision = 0.001;
    const N_NEIGHBORS: usize = 5;

    fn add_points(a: &Point3, b: &Point3) -> Point3 {
        let mut out = [0., 0., 0.];
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            out[idx] = x + y;
        }
        out
    }

    fn make_points(offset: &Point3, step: &Point3, count: usize) -> Vec<Point3> {
        let mut out = Vec::default();
        out.push(*offset);

        for _ in 0..count - 1 {
            let to_push = add_points(out.last().unwrap(), step);
            out.push(to_push);
        }

        out
    }

    #[test]
    fn construct() {
        let points = make_points(&[0., 0., 0.], &[1., 0., 0.], 10);
        PointsTangentsAlphas::new(points.clone(), N_NEIGHBORS).expect("Query construction failed");
        RStarTangentsAlphas::new(&points, N_NEIGHBORS).expect("Target construction failed");
    }

    fn is_close(val1: Precision, val2: Precision) -> bool {
        (val1 - val2).abs() < EPSILON
    }

    fn assert_close(val1: Precision, val2: Precision) {
        if !is_close(val1, val2) {
            panic!("Not close:\n\t{:?}\n\t{:?}", val1, val2);
        }
    }

    #[test]
    fn unit_tangents_eig() {
        let (points, _) = tangent_data();
        let tangent = TangentAlpha::new_from_points(points.iter()).tangent;
        assert_close(tangent.dot(&tangent), 1.0)
    }

    fn equivalent_tangents(tan1: &Normal3, tan2: &Normal3) -> bool {
        is_close(tan1.dot(tan2).abs(), 1.0)
    }

    fn tangent_data() -> (Vec<Point3>, Normal3) {
        // calculated from implementation known to be correct
        let expected = Unit::new_normalize(Vector3::from_column_slice(&[
            -0.939_392_2,
            0.313_061_82,
            0.139_766_18,
        ]));

        // points in first row of data/dotprops/ChaMARCM-F000586_seg002.csv
        let points = vec![
            [
                329.679_962_158_203,
                72.718_803_405_761_7,
                31.028_469_085_693_4,
            ],
            [
                328.647_399_902_344,
                73.046_119_689_941_4,
                31.537_061_691_284_2,
            ],
            [
                335.219_879_150_391,
                70.710_479_736_328_1,
                30.398_145_675_659_2,
            ],
            [
                332.611_389_160_156,
                72.322_929_382_324_2,
                30.887_334_823_608_4,
            ],
            [
                331.770_782_470_703,
                72.434_440_612_793,
                31.169_372_558_593_8,
            ],
        ];

        (points, expected)
    }

    // #[test]
    // #[ignore]
    // fn test_tangent_svd() {
    //     let (points, expected) = tangent_data();
    //     let tangent = points_to_tangent_svd(points.iter()).expect("Failed to create tangent");
    //     println!("tangent is {:?}", tangent);
    //     println!("  expected {:?}", expected);
    //     assert!(equivalent_tangents(&tangent, &expected))
    // }

    #[test]
    fn test_tangent_eig() {
        let (points, expected) = tangent_data();
        let tangent = TangentAlpha::new_from_points(points.iter()).tangent;
        if !equivalent_tangents(&tangent, &expected) {
            panic!(
                "Non-equivalent tangents:\n\t{:?}\n\t{:?}",
                tangent, expected
            )
        }
    }

    /// dist_thresholds, dot_thresholds, values
    fn score_mat() -> (Vec<Precision>, Vec<Precision>, Vec<Precision>) {
        let dists = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let dots = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mut values = vec![];
        let n_values = dots.len() * dists.len();
        for v in 0..n_values {
            values.push(v as Precision);
        }
        (dists, dots, values)
    }

    #[test]
    fn test_score_fn() {
        let (dists, dots, values) = score_mat();
        let func = table_to_fn(dists, dots, values);
        assert_close(
            func(&DistDot {
                dist: 0.0,
                dot: 0.0,
            }),
            0.0,
        );
        assert_close(
            func(&DistDot {
                dist: 0.0,
                dot: 0.1,
            }),
            1.0,
        );
        assert_close(
            func(&DistDot {
                dist: 11.0,
                dot: 0.0,
            }),
            10.0,
        );
        assert_close(
            func(&DistDot {
                dist: 55.0,
                dot: 0.0,
            }),
            40.0,
        );
        assert_close(
            func(&DistDot {
                dist: 55.0,
                dot: 10.0,
            }),
            49.0,
        );
        assert_close(
            func(&DistDot {
                dist: 15.0,
                dot: 0.15,
            }),
            11.0,
        );
    }

    // #[test]
    // fn test_find_bin_linear() {
    //     let dots = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    //     assert_eq!(find_bin_linear(0.0, &dots), 0);
    //     assert_eq!(find_bin_linear(0.15, &dots), 1);
    //     assert_eq!(find_bin_linear(0.95, &dots), 9);
    //     assert_eq!(find_bin_linear(-10.0, &dots), 0);
    //     assert_eq!(find_bin_linear(10.0, &dots), 9);
    //     assert_eq!(find_bin_linear(0.1, &dots), 1);
    // }

    #[test]
    fn test_find_bin_binary() {
        let dots = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        assert_eq!(find_bin_binary(0.0, &dots), 0);
        assert_eq!(find_bin_binary(0.15, &dots), 1);
        assert_eq!(find_bin_binary(0.95, &dots), 9);
        assert_eq!(find_bin_binary(-10.0, &dots), 0);
        assert_eq!(find_bin_binary(10.0, &dots), 9);
        assert_eq!(find_bin_binary(0.1, &dots), 1);
    }

    #[test]
    fn score_function() {
        let dist_thresholds = vec![1.0, 2.0];
        let dot_thresholds = vec![0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);

        let q_points = make_points(&[0., 0., 0.], &[1.0, 0.0, 0.0], 10);
        let query = PointsTangentsAlphas::new(q_points.clone(), N_NEIGHBORS)
            .expect("Query construction failed");
        let query2 = RStarTangentsAlphas::new(&q_points, N_NEIGHBORS).expect("Construction failed");
        let target = RStarTangentsAlphas::new(
            &make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10),
            N_NEIGHBORS,
        )
        .expect("Construction failed");

        assert_close(
            query.query(&target, &score_fn, false),
            query2.query(&target, &score_fn, false),
        );
        assert_close(
            query.self_hit(&score_fn, false),
            query2.self_hit(&score_fn, false),
        );
        let score = query.query(&query2, &score_fn, false);
        let self_hit = query.self_hit(&score_fn, false);
        println!("score: {:?}, self-hit {:?}", score, self_hit);
        assert_close(
            query.query(&query2, &score_fn, false),
            query.self_hit(&score_fn, false),
        );
    }

    #[test]
    fn arena() {
        let dist_thresholds = vec![1.0, 2.0];
        let dot_thresholds = vec![0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);

        let query =
            RStarTangentsAlphas::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10), N_NEIGHBORS)
                .expect("Construction failed");
        let target = RStarTangentsAlphas::new(
            &make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10),
            N_NEIGHBORS,
        )
        .expect("Construction failed");

        let mut arena = NblastArena::new(score_fn);
        let q_idx = arena.add_neuron(query);
        let t_idx = arena.add_neuron(target);

        let no_norm = arena
            .query_target(q_idx, t_idx, false, &None, false)
            .expect("should exist");
        let self_hit = arena
            .query_target(q_idx, q_idx, false, &None, false)
            .expect("should exist");

        assert!(
            arena
                .query_target(q_idx, t_idx, true, &None, false)
                .expect("should exist")
                - no_norm / self_hit
                < EPSILON
        );
        assert_eq!(
            arena.query_target(q_idx, t_idx, false, &Some(Symmetry::ArithmeticMean), false),
            arena.query_target(t_idx, q_idx, false, &Some(Symmetry::ArithmeticMean), false),
        );

        let out =
            arena.queries_targets(&[q_idx, t_idx], &[t_idx, q_idx], false, &None, false, None);
        assert_eq!(out.len(), 4);
    }

    fn test_symmetry(symmetry: &Symmetry, a: Precision, b: Precision) {
        assert_close(
            apply_symmetry(symmetry, a, b),
            apply_symmetry(symmetry, b, a),
        )
    }

    fn test_symmetry_multiple(symmetry: &Symmetry) {
        for (a, b) in vec![(0.3, 0.7), (0.0, 0.7), (-1.0, 0.7), (100.0, 1000.0)].into_iter() {
            test_symmetry(symmetry, a, b);
        }
    }

    #[test]
    fn symmetry_arithmetic() {
        test_symmetry_multiple(&Symmetry::ArithmeticMean)
    }

    #[test]
    fn symmetry_harmonic() {
        test_symmetry_multiple(&Symmetry::HarmonicMean)
    }

    #[test]
    fn symmetry_geometric() {
        test_symmetry_multiple(&Symmetry::GeometricMean)
    }

    #[test]
    fn symmetry_min() {
        test_symmetry_multiple(&Symmetry::Min)
    }

    #[test]
    fn symmetry_max() {
        test_symmetry_multiple(&Symmetry::Max)
    }
}
