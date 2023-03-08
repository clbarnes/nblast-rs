//! Implementation of the NBLAST algorithm for quantifying neurons' morphological similarity.
//! Originally published in
//! [Costa et al. (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4961245/)
//! and implemented as part of the
//! [NeuroAnatomy Toolbox](http://natverse.org/).
//!
//! # Algorithm
//!
//! Each neuron is passed in as a point cloud sample (the links between the points are not required).
//! A tangent vector is calculated for each point, based on its location and that of its nearest neighbors.
//! Additionally, an `alpha` value is calculated, which describes how colinear the neighbors are,
//! between 0 and 1.
//!
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
//! The backbone of the neuron is the most easily sampled and most stereotyped part of its morphology,
//! and therefore should be focused on for comparisons.
//! However, a lot of cable is in dendrites, which can cause problems when reconstructed in high resolution.
//! Queries can be weighted towards straighter, less branched regions by multiplying the absolute dot product
//! for each point match by the geometric mean of the two alpha values.
//!
//! More information on the algorithm can be found
//! [here](http://jefferislab.org/si/nblast).
//!
//! # Usage
//!
//! The [QueryNeuron](trait.QueryNeuron.html) and [TargetNeuron](trait.TargetNeuron.html) traits
//! define types which can be compared with NBLAST.
//! All `TargetNeuron`s are also `QueryNeuron`s.
//! Both are [Neuron](trait.Neuron.html)s.
//!
//! [PointsTangentsAlphas](struct.PointsTangentsAlphas.html) and
//! [RStarTangentsAlphas](struct.RStarTangentsAlphas.html) implement these, respectively.
//! Both can be created with pre-calculated tangents and alphas, or calculate them on instantiation.
//!
//! The [NblastArena](struct.NblastArena.html) contains a collection of `TargetNeuron`s
//! and a function to apply to pointwise [DistDot](struct.DistDot.html)s to generate
//! a score for that point match, for convenient many-to-many comparisons.
//! A pre-calculated table of point match scores can be converted into a function with [table_to_fn](fn.table_to_fn.html).
use nalgebra::base::{Matrix3, Unit, Vector3};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use nalgebra;

mod smat;
pub use smat::ScoreMatrixBuilder;

mod table_lookup;
pub use table_lookup::{BinLookup, NdBinLookup, RangeTable};

mod neurons;
use neurons::rstar::points_to_rtree_tangents_alphas;
pub use neurons::rstar::RStarTangentsAlphas;
pub use neurons::{Neuron, QueryNeuron, TargetNeuron};

#[cfg(feature = "nabo")]
pub use neurons::nabo::NaboTangentsAlphas;

/// Floating point precision type used internally
pub type Precision = f64;
/// 3D point type used internally
pub type Point3 = [Precision; 3];
/// 3D unit-length vector type used internally
pub type Normal3 = Unit<Vector3<Precision>>;

fn centroid<'a, T: IntoIterator<Item = &'a Point3>>(points: T) -> Point3 {
    let mut len: f64 = 0.0;
    let mut out = [0.0; 3];
    for p in points {
        len += 1.0;
        for idx in 0..3 {
            out[idx] += p[idx];
        }
    }
    for el in &mut out {
        *el /= len;
    }
    out
}

fn geometric_mean(a: Precision, b: Precision) -> Precision {
    (a.max(0.0) * b.max(0.0)).sqrt()
}

fn harmonic_mean(a: Precision, b: Precision) -> Precision {
    if a <= 0.0 || b <= 0.0 {
        0.0
    } else {
        2.0 / (1.0 / a + 1.0 / b)
    }
}

/// A tangent, alpha pair associated with a point.
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
/// (i.e. `f(q, t) = f(t, q)`).
/// Specific applications will require different methods.
/// Geometric and harmonic means bound the output to be >= 0.0.
/// Geometric mean may work best with non-normalized queries.
/// Max may work if an unknown one of the query and target is incomplete.
pub enum Symmetry {
    ArithmeticMean,
    GeometricMean,
    HarmonicMean,
    Min,
    Max,
}

impl Symmetry {
    pub fn apply(&self, query_score: Precision, target_score: Precision) -> Precision {
        match self {
            Symmetry::ArithmeticMean => (query_score + target_score) / 2.0,
            Symmetry::GeometricMean => geometric_mean(query_score, target_score),
            Symmetry::HarmonicMean => harmonic_mean(query_score, target_score),
            Symmetry::Min => query_score.min(target_score),
            Symmetry::Max => query_score.max(target_score),
        }
    }
}

impl Default for Symmetry {
    fn default() -> Self {
        Symmetry::GeometricMean
    }
}

/// The result of comparing two (point, tangent) tuples.
/// Contains the Euclidean distance between the points,
/// and the absolute dot product of the (unit) tangents,
/// i.e. the absolute cosine of the angle between them
/// (possibly scaled by the geometric mean of the alphas).
#[derive(Debug, Clone, Copy)]
pub struct DistDot {
    pub dist: Precision,
    pub dot: Precision,
}

impl DistDot {
    fn to_idxs(
        self,
        dist_thresholds: &[Precision],
        dot_thresholds: &[Precision],
    ) -> (usize, usize) {
        let dist_bin = find_bin_binary(self.dist, dist_thresholds);
        let dot_bin = find_bin_binary(self.dot, dot_thresholds);
        (dist_bin, dot_bin)
    }

    fn to_linear_idx(self, dist_thresholds: &[Precision], dot_thresholds: &[Precision]) -> usize {
        let (row_idx, col_idx) = self.to_idxs(dist_thresholds, dot_thresholds);
        row_idx * dot_thresholds.len() + col_idx
    }
}

impl Default for DistDot {
    fn default() -> Self {
        Self {
            dist: 0.0,
            dot: 1.0,
        }
    }
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

    fn centroid(&self) -> Point3 {
        centroid(self.points.iter())
    }

    fn tangents(&self) -> Vec<Normal3> {
        self.tangents_alphas.iter().map(|ta| ta.tangent).collect()
    }

    fn alphas(&self) -> Vec<Precision> {
        self.tangents_alphas.iter().map(|ta| ta.alpha).collect()
    }
}

impl QueryNeuron for PointsTangentsAlphas {
    fn query_dist_dots(&self, target: &impl TargetNeuron, use_alpha: bool) -> Vec<DistDot> {
        self.points
            .iter()
            .zip(self.tangents_alphas.iter())
            .map(|(q_pt, q_ta)| {
                let alpha = if use_alpha { Some(q_ta.alpha) } else { None };
                target.nearest_match_dist_dot(q_pt, &q_ta.tangent, alpha)
            })
            .collect()
    }

    fn query(
        &self,
        target: &impl TargetNeuron,
        use_alpha: bool,
        score_calc: &ScoreCalc,
    ) -> Precision {
        let mut score_total: Precision = 0.0;

        for (q_pt, q_ta) in self.points.iter().zip(self.tangents_alphas.iter()) {
            let alpha = if use_alpha { Some(q_ta.alpha) } else { None };
            score_total +=
                score_calc.calc(&target.nearest_match_dist_dot(q_pt, &q_ta.tangent, alpha));
        }
        score_total
    }

    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision {
        if use_alpha {
            self.tangents_alphas
                .iter()
                .map(|ta| {
                    score_calc.calc(&DistDot {
                        dist: 0.0,
                        dot: ta.alpha,
                    })
                })
                .fold(0.0, |total, s| total + s)
        } else {
            score_calc.calc(&DistDot {
                dist: 0.0,
                dot: 1.0,
            }) * self.len() as Precision
        }
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

    move |dd: &DistDot| -> Precision { cells[dd.to_linear_idx(&dist_thresholds, &dot_thresholds)] }
}

pub fn range_table_to_fn(
    range_table: RangeTable<Precision, Precision>,
) -> impl Fn(&DistDot) -> Precision {
    move |dd: &DistDot| -> Precision { *range_table.lookup(&[dd.dist, dd.dot]) }
}

trait Location {
    fn location(&self) -> &Point3;

    fn distance2_to<T: Location>(&self, other: T) -> Precision {
        self.location()
            .iter()
            .zip(other.location().iter())
            .map(|(a, b)| a * a + b * b)
            .sum()
    }

    fn distance_to<T: Location>(&self, other: T) -> Precision {
        self.distance2_to(other).sqrt()
    }
}

impl Location for Point3 {
    fn location(&self) -> &Point3 {
        self
    }
}

impl Location for &Point3 {
    fn location(&self) -> &Point3 {
        self
    }
}

#[derive(Clone)]
struct NeuronSelfHit<N: QueryNeuron> {
    neuron: N,
    self_hit: Precision,
    self_hit_alpha: Precision,
    centroid: [Precision; 3],
}

impl<N: QueryNeuron> NeuronSelfHit<N> {
    fn new(neuron: N, self_hit: Precision, self_hit_alpha: Precision) -> Self {
        let centroid = neuron.centroid();
        Self {
            neuron,
            self_hit,
            self_hit_alpha,
            centroid,
        }
    }

    fn score(&self, use_alpha: bool) -> Precision {
        if use_alpha {
            self.self_hit_alpha
        } else {
            self.self_hit
        }
    }
}

#[derive(Debug, Clone)]
pub enum ScoreCalc {
    // Func(Box<dyn Fn(&DistDot) -> Precision + Sync>),
    Table(RangeTable<Precision, Precision>),
}

impl ScoreCalc {
    pub fn calc(&self, dist_dot: &DistDot) -> Precision {
        match self {
            // Self::Func(func) => func(dist_dot),
            Self::Table(tab) => *tab.lookup(&[dist_dot.dist, dist_dot.dot]),
        }
    }
}

/// Struct for caching a number of neurons for multiple comparable NBLAST queries.
pub struct NblastArena<N>
where
    N: TargetNeuron,
{
    neurons_scores: Vec<NeuronSelfHit<N>>,
    score_calc: ScoreCalc,
}

pub type NeuronIdx = usize;

impl<N> NblastArena<N>
where
    N: TargetNeuron + Sync,
{
    pub fn new(score_calc: ScoreCalc) -> Self {
        Self {
            neurons_scores: Vec::default(),
            score_calc,
        }
    }

    fn next_id(&self) -> NeuronIdx {
        self.neurons_scores.len()
    }

    /// Returns an index which is then used to make queries.
    pub fn add_neuron(&mut self, neuron: N) -> NeuronIdx {
        let idx = self.next_id();
        let self_hit = neuron.self_hit(&self.score_calc, false);
        let self_hit_alpha = neuron.self_hit(&self.score_calc, true);
        self.neurons_scores
            .push(NeuronSelfHit::new(neuron, self_hit, self_hit_alpha));
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
        let mut score = q.neuron.query(&t.neuron, use_alpha, &self.score_calc);
        if normalize {
            score /= q.score(use_alpha)
        }
        match symmetry {
            Some(s) => {
                let mut score2 = t.neuron.query(&q.neuron, use_alpha, &self.score_calc);
                if normalize {
                    score2 /= t.score(use_alpha);
                }
                Some(s.apply(score, score2))
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
    #[allow(clippy::too_many_arguments)] // todo: refactor normalize/symmetry/use_alpha into separate struct
    pub fn queries_targets(
        &self,
        query_idxs: &[NeuronIdx],
        target_idxs: &[NeuronIdx],
        normalize: bool,
        symmetry: &Option<Symmetry>,
        use_alpha: bool,
        threads: Option<usize>,
        max_centroid_dist: Option<Precision>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let mut out = HashMap::with_capacity(query_idxs.len() * target_idxs.len());
        let mut out_keys: HashSet<(NeuronIdx, NeuronIdx)> = HashSet::default();
        let mut jobs: HashSet<(NeuronIdx, NeuronIdx)> = HashSet::default();
        for q_idx in query_idxs {
            if q_idx >= &self.len() {
                continue;
            }
            for t_idx in target_idxs {
                if t_idx >= &self.len() {
                    continue;
                }
                let key = (*q_idx, *t_idx);
                if q_idx == t_idx {
                    if let Some(ns) = self.neurons_scores.get(*q_idx) {
                        out.insert(key, if normalize { 1.0 } else { ns.score(use_alpha) });
                    };
                    continue;
                }

                if let Some(d) = max_centroid_dist {
                    if !self
                        .centroids_within_distance(*q_idx, *t_idx, d)
                        .expect("Already checked indices")
                    {
                        continue;
                    }
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
                        out.insert(key, s.apply(*forward, *backward));
                    }
                } else {
                    out.insert(key, *forward);
                }
            }
        }
        out
    }

    pub fn centroids_within_distance(
        &self,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        max_centroid_dist: Precision,
    ) -> Option<bool> {
        self.neurons_scores.get(query_idx).and_then(|q| {
            self.neurons_scores
                .get(target_idx)
                .map(|t| q.centroid.distance_to(t.centroid) < max_centroid_dist)
        })
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
        max_centroid_dist: Option<Precision>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let idxs: Vec<NeuronIdx> = (0..self.len()).collect();
        self.queries_targets(
            &idxs,
            &idxs,
            normalize,
            symmetry,
            use_alpha,
            threads,
            max_centroid_dist,
        )
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

fn pairs_to_raw_serial<N: TargetNeuron + Sync>(
    arena: &NblastArena<N>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
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
fn pairs_to_raw<N>(
    arena: &NblastArena<N>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
    _threads: Option<usize>,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision>
where
    N: TargetNeuron + Sync,
{
    pairs_to_raw_serial(arena, pairs, normalize, use_alpha)
}

#[cfg(feature = "parallel")]
fn pairs_to_raw<N: TargetNeuron + Sync>(
    arena: &NblastArena<N>,
    pairs: &[(NeuronIdx, NeuronIdx)],
    normalize: bool,
    use_alpha: bool,
    threads: Option<usize>,
) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
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
mod test {
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
        println!("Comparing values:\n\tval1: {:?}\n\tval2: {:?}", val1, val2);
        (val1 - val2).abs() < EPSILON
    }

    fn assert_close(val1: Precision, val2: Precision) {
        if !is_close(val1, val2) {
            panic!("Not close:\n\t{:?}\n\t{:?}", val1, val2);
        }
    }

    #[test]
    fn unit_tangents_eig() {
        let (points, _, _) = tangent_data();
        let tangent = TangentAlpha::new_from_points(points.iter()).tangent;
        assert_close(tangent.dot(&tangent), 1.0)
    }

    fn equivalent_tangents(tan1: &Normal3, tan2: &Normal3) -> bool {
        is_close(tan1.dot(tan2).abs(), 1.0)
    }

    fn tangent_data() -> (Vec<Point3>, Normal3, Precision) {
        // calculated from implementation known to be correct
        let tangent = Unit::new_normalize(Vector3::from_column_slice(&[
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

        let alpha = 0.844_842_871_450_449;

        (points, tangent, alpha)
    }

    #[test]
    fn test_tangent_eig() {
        let (points, exp_tan, _exp_alpha) = tangent_data();
        let ta = TangentAlpha::new_from_points(points.iter());
        if !equivalent_tangents(&ta.tangent, &exp_tan) {
            panic!(
                "Non-equivalent tangents:\n\t{:?}\n\t{:?}",
                ta.tangent, exp_tan
            )
        }
        // tested from the python side
        // assert_close(ta.alpha, exp_alpha);
    }

    #[test]
    fn test_neuron() {
        let (points, exp_tan, _exp_alpha) = tangent_data();
        let tgt = RStarTangentsAlphas::new(points, N_NEIGHBORS).unwrap();
        assert!(equivalent_tangents(&tgt.tangents()[0], &exp_tan));
        // tested from the python side
        // assert_close(tgt.alphas()[0], exp_alpha);
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
    fn test_score_calc() {
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

    // #[test]
    // fn score_function() {
    //     let dist_thresholds = vec![1.0, 2.0];
    //     let dot_thresholds = vec![0.5, 1.0];
    //     let cells = vec![1.0, 2.0, 4.0, 8.0];

    //     let score_calc = ScoreCalc::Func(Box::new(table_to_fn(dist_thresholds, dot_thresholds, cells)));

    //     let q_points = make_points(&[0., 0., 0.], &[1.0, 0.0, 0.0], 10);
    //     let query = PointsTangentsAlphas::new(q_points.clone(), N_NEIGHBORS)
    //         .expect("Query construction failed");
    //     let query2 = RStarTangentsAlphas::new(&q_points, N_NEIGHBORS).expect("Construction failed");
    //     let target = RStarTangentsAlphas::new(
    //         &make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10),
    //         N_NEIGHBORS,
    //     )
    //     .expect("Construction failed");

    //     assert_close(
    //         query.query(&target, false, &score_calc),
    //         query2.query(&target, false, &score_calc),
    //     );
    //     assert_close(
    //         query.self_hit(&score_calc, false),
    //         query2.self_hit(&score_calc, false),
    //     );
    //     let score = query.query(&query2, false, &score_calc);
    //     let self_hit = query.self_hit(&score_calc, false);
    //     println!("score: {:?}, self-hit {:?}", score, self_hit);
    //     assert_close(
    //         query.query(&query2, false, &score_calc),
    //         query.self_hit(&score_calc, false),
    //     );
    // }

    #[test]
    fn arena() {
        let dist_thresholds = vec![0.0, 1.0, 2.0];
        let dot_thresholds = vec![0.0, 0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        // let score_calc = ScoreCalc::Func(Box::new(table_to_fn(dist_thresholds, dot_thresholds, cells)));
        let score_calc = ScoreCalc::Table(
            RangeTable::new_from_bins(vec![dist_thresholds, dot_thresholds], cells).unwrap(),
        );

        let query =
            RStarTangentsAlphas::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10), N_NEIGHBORS)
                .expect("Construction failed");
        let target = RStarTangentsAlphas::new(
            &make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10),
            N_NEIGHBORS,
        )
        .expect("Construction failed");

        let mut arena = NblastArena::new(score_calc);
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

        let out = arena.queries_targets(
            &[q_idx, t_idx],
            &[t_idx, q_idx],
            false,
            &None,
            false,
            None,
            None,
        );
        assert_eq!(out.len(), 4);
    }

    fn test_symmetry(symmetry: &Symmetry, a: Precision, b: Precision) {
        assert_close(symmetry.apply(a, b), symmetry.apply(b, a))
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

    // #[test]
    // fn alpha_changes_results() {
    //     let (points, _, _) = tangent_data();
    //     let neuron = RStarTangentsAlphas::new(points, N_NEIGHBORS).unwrap();
    //     let score_calc = ScoreCalc::Func(Box::new(|dd: &DistDot| dd.dot));

    //     let sh = neuron.self_hit(&score_calc, false);
    //     let sh_a = neuron.self_hit(&score_calc, true);
    //     assert!(!is_close(sh, sh_a));

    //     let q = neuron.query(&neuron, false, &score_calc);
    //     let q_a = neuron.query(&neuron, true, &score_calc);

    //     assert!(!is_close(q, q_a));
    // }
}
