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
use std::collections::HashMap;

pub use nalgebra;

// NOTE: will panic if this is changed due to use of Matrix3x5
// const N_NEIGHBORS: usize = 5;

/// Floating point precision type used internally
pub type Precision = f64;

type PointWithIndex = PointWithData<usize, [Precision; 3]>;

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
        Symmetry::GeometricMean => (query_score.max(0.0) * target_score.max(0.0)).sqrt(),
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

/// Trait for objects which can be used as queries
/// (not necessarily as targets) with NBLAST.
/// See [TargetNeuron](trait.TargetNeuron.html).
pub trait QueryNeuron {
    /// Number of points in the neuron.
    fn len(&self) -> usize;

    /// Whether the number of points is 0.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calculate the raw NBLAST score by comparing this neuron to
    /// the given target neuron, using the given score function.
    /// The score function is applied to each point match distance and summed.
    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
    ) -> Precision;

    /// The raw NBLAST score if this neuron was compared with itself using the given score function.
    /// Used for normalisation.
    fn self_hit(&self, score_fn: &impl Fn(&DistDot) -> Precision) -> Precision {
        score_fn(&DistDot::default()) * self.len() as Precision
    }

    /// Return an owned copy of the points present in the neuron.
    /// The order is not guaranteed, but is consistent with
    /// [tangents](#method.tangents).
    fn points(&self) -> Vec<[Precision; 3]>;

    /// Return an owned copy of the unit tangents present in the neuron.
    /// The order is not guaranteed, but is consistent with
    /// [points](#method.points).
    fn tangents(&self) -> Vec<Unit<Vector3<Precision>>>;
}

/// Minimal struct to use as the query (not the target) of an NBLAST
/// comparison.
#[derive(Clone)]
pub struct QueryPointTangents {
    /// Locations of points in point cloud.
    points: Vec<[Precision; 3]>,
    /// Unit-length tangent vectors for each point in the cloud.
    tangents: Vec<Unit<Vector3<Precision>>>,
}

fn subtract_points(p1: &[Precision; 3], p2: &[Precision; 3]) -> [Precision; 3] {
    let mut result = [0.0; 3];
    for ((rref, v1), v2) in result.iter_mut().zip(p1).zip(p2) {
        *rref = v1 - v2;
    }
    result
}

fn center_points<'a>(
    points: impl Iterator<Item = &'a [Precision; 3]>,
) -> impl Iterator<Item = [Precision; 3]> {
    let mut points_vec = Vec::default();
    let mut means: [Precision; 3] = [0.0, 0.0, 0.0];
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
    a.iter().zip(b.iter()).fold(0.0, |sum, (ax, bx)| sum + ax * bx)
}

/// Calculate inertia from iterator of points.
/// This is an implementation of matrix * matrix.transpose(),
/// to sidestep the fixed-size constraints of linalg's built-in classes.
/// Only calculates the lower triangle and diagonal.
fn calc_inertia<'a>(points: impl Iterator<Item = &'a [Precision; 3]>) -> Matrix3<Precision> {
    let mut xs = Vec::default();
    let mut ys = Vec::default();
    let mut zs = Vec::default();
    for point in center_points(points) {
        xs.push(point[0]);
        ys.push(point[1]);
        zs.push(point[2]);
    }
    Matrix3::new(
        dot(&xs, &xs), 0.0, 0.0,
        dot(&ys, &xs), dot(&ys, &ys), 0.0,
        dot(&zs, &xs), dot(&zs, &ys), dot(&zs, &zs),
    )
}

fn points_to_tangent_eig<'a>(
    points: impl Iterator<Item = &'a [Precision; 3]>,
) -> Option<Unit<Vector3<Precision>>> {
    let points_vec: Vec<_> = points.collect();
    let inertia = calc_inertia(points_vec.iter().cloned());
    let eig = inertia.symmetric_eigen();
    // TODO: new_unchecked
    // TODO: better copying in general
    Some(Unit::new_normalize(Vector3::from_iterator(
        eig.eigenvectors
            .column(eig.eigenvalues.argmax().0)
            .iter()
            .cloned(),
    )))
}

// ! doesn't work
// fn points_to_tangent_svd<'a>(
//     points: impl Iterator<Item = &'a [Precision; 3]>,
// ) -> Option<Unit<Vector3<Precision>>> {
//     let cols_vec: Vec<Vector3<Precision>> = center_points(points)
//         .map(|p| Vector3::from_column_slice(&p))
//         .collect();
//     let neighbor_mat = Matrix3x5::from_columns(&cols_vec);
//     let svd = neighbor_mat.svd(false, true);

//     let (idx, _val) = svd.singular_values.argmax();

//     svd.v_t.map(|v_t| {
//         Unit::new_normalize(Vector3::from_iterator(v_t.column(idx).iter().cloned()))
//     })
// }

fn points_to_rtree(points: &[[Precision; 3]]) -> Result<RTree<PointWithIndex>, &'static str> {
    Ok(RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .map(|(idx, point)| PointWithIndex::new(idx, *point))
            .collect(),
    ))
}

fn points_to_rtree_tangents(
    points: &[[Precision; 3]], k: usize,
) -> Result<(RTree<PointWithIndex>, Vec<Unit<Vector3<Precision>>>), &'static str> {
    if points.len() < k {
        return Err("Too few points to generate tangents");
    }
    let rtree = points_to_rtree(points)?;

    let mut tangents: Vec<Unit<Vector3<Precision>>> = Vec::with_capacity(rtree.size());

    for point in points.iter() {
        match points_to_tangent_eig(
            rtree
                .nearest_neighbor_iter(&point)
                .take(k)
                .map(|pwd| pwd.position()),
        ) {
            Some(t) => tangents.push(t),
            None => return Err("Failed to SVD"),
        }
    }

    Ok((rtree, tangents))
}

impl QueryPointTangents {
    /// Calculates tangents from the given points.
    /// Note that this constructs a spatial index in order to calculate the tangents,
    /// and then throws it away: you may as well use a [TargetNeuron](trait.TargetNeuron.html)
    /// type, with regards to performance.
    /// `k` is the number of points tangents will be calculated with,
    /// and includes the point itself.
    pub fn new(points: Vec<[Precision; 3]>, k: usize) -> Result<Self, &'static str> {
        points_to_rtree_tangents(&points, k).map(|(_, tangents)| Self { points, tangents })
    }
}

impl QueryNeuron for QueryPointTangents {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
    ) -> Precision {
        let mut score_total: Precision = 0.0;
        for (q_pt, q_tan) in self.points.iter().zip(self.tangents.iter()) {
            score_total += score_fn(&target.nearest_match_dist_dot(q_pt, q_tan));
        }
        score_total
    }

    fn points(&self) -> Vec<[Precision; 3]> {
        self.points.clone()
    }

    fn tangents(&self) -> Vec<Unit<Vector3<Precision>>> {
        self.tangents.clone()
    }
}

pub trait TargetNeuron: QueryNeuron {
    /// For a given point and tangent vector,
    /// get the distance to its nearest point in the target, and the absolute dot product
    /// with that neighbor's tangent (i.e. absolute cosine of the angle, as they are both unit-length).
    fn nearest_match_dist_dot(
        &self,
        point: &[Precision; 3],
        tangent: &Unit<Vector3<Precision>>,
    ) -> DistDot;
}

/// Target neuron using an [R*-tree](https://en.wikipedia.org/wiki/R*_tree) for spatial queries.
#[derive(Clone)]
pub struct RStarPointTangents {
    rtree: RTree<PointWithIndex>,
    tangents: Vec<Unit<Vector3<Precision>>>,
}

impl RStarPointTangents {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<[Precision; 3]>, k: usize) -> Result<Self, &'static str> {
        points_to_rtree_tangents(&points, k).map(|(rtree, tangents)| Self { rtree, tangents })
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents(
        points: Vec<[Precision; 3]>,
        tangents: Vec<Unit<Vector3<Precision>>>,
    ) -> Result<Self, &'static str> {
        points_to_rtree(&points).map(|rtree| Self { rtree, tangents })
    }
}

impl QueryNeuron for RStarPointTangents {
    fn len(&self) -> usize {
        self.tangents.len()
    }

    fn query(
        &self,
        target: &impl TargetNeuron,
        score_fn: &impl Fn(&DistDot) -> Precision,
    ) -> Precision {
        let mut score_total: Precision = 0.0;
        for q_pt_idx in self.rtree.iter() {
            let dd =
                target.nearest_match_dist_dot(q_pt_idx.position(), &self.tangents[q_pt_idx.data]);
            let score = score_fn(&dd);
            score_total += score;
        }
        score_total
    }

    fn points(&self) -> Vec<[Precision; 3]> {
        let mut unsorted: Vec<&PointWithIndex> = self.rtree.iter().collect();
        unsorted.sort_by_key(|pwd| pwd.data);
        unsorted.into_iter().map(|pwd| *pwd.position()).collect()
    }

    fn tangents(&self) -> Vec<Unit<Vector3<Precision>>> {
        self.tangents.clone()
    }
}

impl TargetNeuron for RStarPointTangents {
    fn nearest_match_dist_dot(
        &self,
        point: &[Precision; 3],
        tangent: &Unit<Vector3<Precision>>,
    ) -> DistDot {
        self.rtree
            .nearest_neighbor_iter_with_distance(point)
            .next()
            .map(|(element, dist2)| {
                let this_tangent = self.tangents[element.data];
                let dot = this_tangent.dot(tangent).abs();
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

// fn find_bin_linear(value: Precision, upper_bounds: &[Precision]) -> usize {
//     let mut out = 0;
//     for bound in upper_bounds.iter() {
//         if &value < bound {
//             return out;
//         }
//         out += 1;
//     }
//     out - 1
// }

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

/// Struct for caching a number of neurons for multiple comparable NBLAST queries.
#[derive(Clone)]
pub struct NblastArena<N, F>
where
    N: TargetNeuron,
    F: Fn(&DistDot) -> Precision,
{
    neurons_scores: Vec<(N, Precision)>,
    score_fn: F,
}

pub type NeuronIdx = usize;

// TODO: caching strategy
impl<N, F> NblastArena<N, F>
where
    N: TargetNeuron,
    F: Fn(&DistDot) -> Precision,
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
        let score = neuron.self_hit(&self.score_fn);
        self.neurons_scores.push((neuron, score));
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
    ) -> Option<Precision> {
        // ? consider separate methods
        let q = self.neurons_scores.get(query_idx)?;
        let t = self.neurons_scores.get(target_idx)?;
        let mut score = q.0.query(&t.0, &self.score_fn);
        if normalize {
            score /= q.1;
        }
        match symmetry {
            Some(s) => {
                let mut score2 = t.0.query(&q.0, &self.score_fn);
                if normalize {
                    score2 /= t.1;
                }
                Some(apply_symmetry(s, score, score2))
            }
            _ => Some(score),
        }
    }

    /// Make many queries using the Cartesian product of the query and target indices.
    /// See [query_target](#method.query_target) for more details.
    pub fn queries_targets(
        &self,
        query_idxs: &[NeuronIdx],
        target_idxs: &[NeuronIdx],
        normalize: bool,
        symmetry: &Option<Symmetry>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let mut out = HashMap::with_capacity(query_idxs.len() * target_idxs.len());

        // ? lots of unnecessary index operations
        for q_idx in query_idxs.iter() {
            for t_idx in target_idxs.iter() {
                let key = (*q_idx, *t_idx);
                if q_idx == t_idx {
                    // if neurons are present and identical, 1.0 or self-hit (always symmetric)
                    if let Some(ns) = self.neurons_scores.get(*q_idx) {
                        out.insert(key, if normalize { 1.0 } else { ns.1 });
                    };
                } else if symmetry.is_some() {
                    // otherwise, if symmetric, use reverse query score if it's in the result set
                    // or generate the result if not
                    match out.get(&(*t_idx, *q_idx)).map_or_else(
                        || self.query_target(*q_idx, *t_idx, normalize, symmetry),
                        |s| Some(*s),
                    ) {
                        Some(s) => out.insert(key, s),
                        _ => None,
                    };
                } else {
                    // otherwise, generate (asymmetric) result
                    match self.query_target(*q_idx, *t_idx, normalize, &None) {
                        Some(s) => out.insert(key, s),
                        _ => None,
                    };
                }
            }
        }
        out
    }

    pub fn self_hit(&self, idx: NeuronIdx) -> Option<Precision> {
        self.neurons_scores.get(idx).map(|(_, s)| *s)
    }

    /// Query every neuron against every other neuron.
    /// See [queries_targets](#method.queries_targets) for more details.
    pub fn all_v_all(
        &self,
        normalize: bool,
        symmetry: &Option<Symmetry>,
    ) -> HashMap<(NeuronIdx, NeuronIdx), Precision> {
        let idxs: Vec<NeuronIdx> = (0..self.len()).collect();
        self.queries_targets(&idxs, &idxs, normalize, symmetry)
    }

    pub fn is_empty(&self) -> bool {
        self.neurons_scores.is_empty()
    }

    /// Number of neurons in the arena.
    pub fn len(&self) -> usize {
        self.neurons_scores.len()
    }

    pub fn points(&self, idx: NeuronIdx) -> Option<Vec<[Precision; 3]>> {
        self.neurons_scores.get(idx).map(|(n, _)| n.points())
    }

    pub fn tangents(&self, idx: NeuronIdx) -> Option<Vec<Unit<Vector3<Precision>>>> {
        self.neurons_scores.get(idx).map(|(n, _)| n.tangents())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Precision = 0.001;
    const N_NEIGHBORS: usize = 5;

    fn add_points(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        let mut out = [0., 0., 0.];
        for (idx, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            out[idx] = x + y;
        }
        out
    }

    fn make_points(offset: &[f64; 3], step: &[f64; 3], count: usize) -> Vec<[f64; 3]> {
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
        QueryPointTangents::new(points.clone(), N_NEIGHBORS).expect("Query construction failed");
        RStarPointTangents::new(points, N_NEIGHBORS).expect("Target construction failed");
    }

    fn is_close(val1: Precision, val2: Precision) -> bool {
        (val1 - val2).abs() < EPSILON
    }

    fn assert_close(val1: Precision, val2: Precision) {
        if !is_close(val1, val2) {
            panic!("Not close:\n\t{:?}\n\t{:?}", val1, val2);
        }
    }

    // #[test]
    // fn unit_tangents_svd() {
    //     let (points, _) = tangent_data();
    //     let tangent = points_to_tangent_svd(points.iter()).expect("SVD failed");
    //     assert_close(tangent.dot(&tangent), 1.0)
    // }

    #[test]
    fn unit_tangents_eig() {
        let (points, _) = tangent_data();
        let tangent = points_to_tangent_eig(points.iter()).expect("eig failed");
        assert_close(tangent.dot(&tangent), 1.0)
    }

    fn equivalent_tangents(
        tan1: &Unit<Vector3<Precision>>,
        tan2: &Unit<Vector3<Precision>>,
    ) -> bool {
        is_close(tan1.dot(tan2).abs(), 1.0)
    }

    fn tangent_data() -> (Vec<[Precision; 3]>, Unit<Vector3<Precision>>) {
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
        let tangent = points_to_tangent_eig(points.iter()).expect("Failed to create tangent");
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
        let query = QueryPointTangents::new(q_points.clone(), N_NEIGHBORS).expect("Query construction failed");
        let query2 = RStarPointTangents::new(q_points, N_NEIGHBORS).expect("Construction failed");
        let target = RStarPointTangents::new(make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10), N_NEIGHBORS)
            .expect("Construction failed");

        assert_close(
            query.query(&target, &score_fn),
            query2.query(&target, &score_fn),
        );
        assert_close(query.self_hit(&score_fn), query2.self_hit(&score_fn));
        let score = query.query(&query2, &score_fn);
        let self_hit = query.self_hit(&score_fn);
        println!("score: {:?}, self-hit {:?}", score, self_hit);
        assert_close(query.query(&query2, &score_fn), query.self_hit(&score_fn));
    }

    #[test]
    fn arena() {
        let dist_thresholds = vec![1.0, 2.0];
        let dot_thresholds = vec![0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);

        let query = RStarPointTangents::new(make_points(&[0., 0., 0.], &[1., 0., 0.], 10), N_NEIGHBORS)
            .expect("Construction failed");
        let target = RStarPointTangents::new(make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10), N_NEIGHBORS)
            .expect("Construction failed");

        let mut arena = NblastArena::new(score_fn);
        let q_idx = arena.add_neuron(query);
        let t_idx = arena.add_neuron(target);

        let no_norm = arena
            .query_target(q_idx, t_idx, false, &None)
            .expect("should exist");
        let self_hit = arena
            .query_target(q_idx, q_idx, false, &None)
            .expect("should exist");

        assert!(
            arena
                .query_target(q_idx, t_idx, true, &None)
                .expect("should exist")
                - no_norm / self_hit
                < EPSILON
        );
        assert_eq!(
            arena.query_target(q_idx, t_idx, false, &Some(Symmetry::ArithmeticMean)),
            arena.query_target(t_idx, q_idx, false, &Some(Symmetry::ArithmeticMean)),
        );

        let out = arena.queries_targets(&[q_idx, t_idx], &[t_idx, q_idx], false, &None);
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
