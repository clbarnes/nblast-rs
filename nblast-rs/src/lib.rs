use nalgebra::base::{Matrix3x6, Vector3, Unit};
use rstar::primitives::PointWithData;
use rstar::RTree;
use std::collections::HashMap;

const N_NEIGHBORS: usize = 5;
pub type Precision = f64;
type PointWithIndex = PointWithData<usize, [Precision; 3]>;

#[derive(Debug, Clone, Copy)]
pub struct DistDot {
    pub dist: Precision,
    pub dot: Precision,
}

#[derive(Clone)]
pub struct PointTangents {
    rtree: RTree<PointWithIndex>,
    tangents: Vec<Unit<Vector3<Precision>>>,
}

// TODO: check orientation of matrices, may need to transpose everything
// ? consider using nalgebra's Point3 in PointWithIndex, for consistency
// ^ can't implement rstar::Point for nalgebra::geometry::Point3 because of orphan rules
// TODO: replace Precision with float generic

fn get_tangent(points: Vec<Vector3<Precision>>) -> Option<Unit<Vector3<Precision>>> {
    // TODO: make more generic (e.g. take an iterator of slices)

    let neighbor_mat: Matrix3x6<Precision> = Matrix3x6::from_columns(&points);
    let svd = neighbor_mat.svd(false, true);
    svd.v_t.map(|v_t| Unit::new_normalize(Vector3::from_iterator(v_t.column(0).iter().cloned())))
}

impl PointTangents {
    pub fn new(points: &[[Precision; 3]]) -> Result<Self, &'static str> {
        if points.len() < N_NEIGHBORS + 1 {
            return Err("Not enough points");
        }

        let rtree = RTree::bulk_load(
            points
                .iter()
                .enumerate()
                .map(|(idx, point)| PointWithIndex::new(idx, *point))
                .collect(),
        );

        let mut tangents: Vec<Unit<Vector3<Precision>>> = Vec::with_capacity(rtree.size());

        for point in points.iter() {
            let nearest_it = rtree.nearest_neighbor_iter(&point);

            // TODO: would probably be faster to read the matrix as single elements
            // from a flat_map, but this intermediate should help prevent row/column order bugs
            let columns: Vec<Vector3<Precision>> = nearest_it
                .take(N_NEIGHBORS + 1)
                .map(|p| Vector3::from_column_slice(p.position()))
                .collect();

            match get_tangent(columns) {
                Some(t) => tangents.push(t),
                None => return Err("Failed to SVD"),
            }
        }

        Ok(Self { rtree, tangents })
    }

    pub fn from_points_tangents(
        points: &[[Precision; 3]],
        tangents: Vec<Unit<Vector3<Precision>>>,
    ) -> Result<Self, &'static str> {
        if points.len() != tangents.len() {
            return Err("Tangents do not match points");
        }

        let rtree = RTree::bulk_load(
            points
                .iter()
                .enumerate()
                .map(|(idx, point)| PointWithIndex::new(idx, *point))
                .collect(),
        );

        Ok(Self { rtree, tangents })
    }

    /// For a given point and tangent vector,
    /// get the distance to its nearest neighbor and dot product with that neighbor's tangent
    pub fn nearest_match_dist_dot(
        &self,
        point: &[Precision; 3],
        tangent: &Unit<Vector3<Precision>>,
    ) -> DistDot {
        self.rtree
            .nearest_neighbor_iter_with_distance(point)
            .next()
            .map(|(element, dist)| {
                let this_tangent = self.tangents[element.data];
                let dot = this_tangent.dot(tangent).abs();
                DistDot { dist, dot }
            })
            .expect("impossible")
    }

    /// For every segment in self, find the nearest segment in target,
    /// and return the distance and dot product of the points and tangents respectively.
    ///
    /// Return order is arbitrary
    pub fn query_target_dist_dots(&self, target: &Self) -> Vec<DistDot> {
        let mut out: Vec<DistDot> = Vec::with_capacity(self.rtree.size());

        // TODO: parallelise
        for q_point in self.rtree.iter() {
            out.push(
                target.nearest_match_dist_dot(q_point.position(), &self.tangents[q_point.data]),
            );
        }

        out
    }

    /// Get the raw NBLAST score for this pair of neuron pointtangents.
    pub fn query_target<F>(&self, target: &Self, score_fn: &F) -> Precision
    where
        F: Fn(&DistDot) -> Precision,
    {
        let mut out: f64 = 0.0;
        for q_point in self.rtree.iter() {
            out += score_fn(
                &target.nearest_match_dist_dot(q_point.position(), &self.tangents[q_point.data]),
            );
        }
        out
    }

    /// Get the raw NBLAST scores for this neuron with every one of the given targets.
    pub fn query_targets<F>(&self, targets: &[&Self], score_fn: &F) -> Vec<Precision>
    where
        F: Fn(&DistDot) -> Precision,
    {
        // TODO: parallelise
        targets
            .iter()
            .map(|t| self.query_target(&t, score_fn))
            .collect()
    }

    /// Get the raw NBLAST score of this neuron with itself, for normalisation.
    pub fn self_hit<F>(&self, score_fn: &F) -> Precision
    where
        F: Fn(&DistDot) -> Precision,
    {
        // Every point will match with itself, so distance will be 0.
        // Dot product = cos(angle=0) * (norm^2=1) = 1 * 1 = 1
        score_fn(&DistDot{dist: 0.0, dot: 1.0}) * self.tangents.len() as Precision
    }
}

/// For slice of (query, targets[]) pairs, find all the raw NBLAST scores.
pub fn queries_targets<F>(
    qs_ts: &[(&PointTangents, Vec<&PointTangents>)],
    score_fn: &F,
) -> Vec<Vec<Precision>>
where
    F: Fn(&DistDot) -> Precision,
{
    qs_ts
        .iter()
        .map(|(q, ts)| q.query_targets(ts, score_fn))
        .collect()
}

/// Given the upper bounds of a number of bins, find which bin the value falls into.
/// Values outside of the range fall into the bottom and top bin.
fn find_bin(value: Precision, upper_bounds: &[Precision]) -> usize {
    let mut out = 0;

    for bound in upper_bounds.iter() {
        if &value < bound {
            return out;
        }
        out += 1;
    }

    out - 1
}

/// Convert an empirically-derived table of NBLAST scores to a function
/// which can be passed to dotprop queries.
///
/// Cells are passed in dot-major order
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
        let col_idx = find_bin(dd.dot, &dot_thresholds);
        let row_idx = find_bin(dd.dist, &dist_thresholds);

        let lin_idx = row_idx * dot_thresholds.len() + col_idx;
        cells[lin_idx]
    }
}

pub struct NblastArena<F>
where
    F: Fn(&DistDot) -> Precision,
{
    pointtangents_scores: Vec<(PointTangents, Precision)>,
    score_fn: F,
}

pub type DotPropIdx = usize;

// TODO: caching strategy
impl<F> NblastArena<F>
where
    F: Fn(&DistDot) -> Precision,
{
    pub fn new(score_fn: F) -> Self {
        Self {
            pointtangents_scores: Vec::default(),
            score_fn,
        }
    }

    fn next_id(&self) -> DotPropIdx {
        self.pointtangents_scores.len()
    }

    pub fn add_pointtangents(&mut self, pointtangents: PointTangents) -> DotPropIdx {
        let idx = self.next_id();
        let score = pointtangents.self_hit(&self.score_fn);
        self.pointtangents_scores.push((pointtangents, score));
        idx
    }

    // TODO: Results instead of Options?

    fn get_pointtangents_norm_score(&self, idx: DotPropIdx) -> Option<&(PointTangents, f64)> {
        self.pointtangents_scores.get(idx)
    }

    fn get_many_pointtangents_norm_score(&self, idxs: &[DotPropIdx]) -> Option<Vec<&(PointTangents, f64)>> {
        let mut out = Vec::with_capacity(idxs.len());
        for idx in idxs.iter() {
            if let Some(dp_s) = self.get_pointtangents_norm_score(*idx) {
                out.push(dp_s)
            } else {
                return None;
            }
        }
        Some(out)
    }

    pub fn add_points(&mut self, points: &[[Precision; 3]]) -> Result<usize, &'static str> {
        let pointtangents = PointTangents::new(points)?;
        Ok(self.add_pointtangents(pointtangents))
    }

    fn _query_target(
        &self,
        query: &(PointTangents, Precision),
        target: &(PointTangents, Precision),
        normalise: bool,
        symmetric: bool,
    ) -> Precision {
        let mut score = query.0.query_target(&target.0, &self.score_fn);
        if normalise {
            score /= query.1;
        }
        if symmetric {
            let mut score2 = target.0.query_target(&query.0, &self.score_fn);
            if normalise {
                score2 /= target.1;
            }
            score += score2;
            score /= 2.0;
        }
        score
    }

    pub fn query_target(
        &self,
        query_idx: DotPropIdx,
        target_idx: DotPropIdx,
        normalise: bool,
        symmetric: bool,
    ) -> Option<Precision> {
        if query_idx == target_idx {
            if normalise {
                Some(1.0)
            } else {
                self.pointtangents_scores.get(query_idx).map(|pair| pair.1)
            }
        } else {
            self.get_many_pointtangents_norm_score(&[query_idx, target_idx])
                .map(|a| self._query_target(a[0], a[1], normalise, symmetric))
        }
    }

    pub fn queries_targets(
        &self,
        query_idxs: &[DotPropIdx],
        target_idxs: &[DotPropIdx],
        normalise: bool,
        symmetric: bool,
    ) -> Option<HashMap<(DotPropIdx, DotPropIdx), Precision>> {
        let query_doubles = self.get_many_pointtangents_norm_score(query_idxs)?;
        let target_doubles = self.get_many_pointtangents_norm_score(target_idxs)?;
        let mut out = HashMap::with_capacity(query_idxs.len() * target_idxs.len());

        for (q_idx, q) in query_idxs.iter().zip(query_doubles.iter()) {
            for (t_idx, t) in target_idxs.iter().zip(target_doubles.iter()) {
                let key = (*q_idx, *t_idx);
                if q_idx == t_idx {
                    let mut val = 1.0;
                    if !normalise {
                        val *= q.1;
                    }
                    out.insert(key, val);
                } else if symmetric {
                    out.entry((*t_idx, *q_idx))
                        .or_insert_with(|| self._query_target(q, t, normalise, symmetric));
                } else {
                    let score = self._query_target(q, t, normalise, symmetric);
                    out.insert(key, score);
                }
            }
        }
        Some(out)
    }

    pub fn all_v_all(
        &self,
        normalise: bool,
        symmetric: bool,
    ) -> Option<HashMap<(DotPropIdx, DotPropIdx), Precision>> {
        let idxs: Vec<DotPropIdx> = (0..self.len()).collect();
        self.queries_targets(&idxs, &idxs, normalise, symmetric)
    }

    pub fn is_empty(&self) -> bool {
        self.pointtangents_scores.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pointtangents_scores.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: Precision = 0.001;

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
        PointTangents::new(&points).expect("Construction failed");
    }

    fn assert_close(val1: Precision, val2: Precision) {
        assert!((val1 - val2).abs() < EPSILON);
    }

    #[test]
    fn tangents_are_unit() {
        let points = make_points(&[0., 0., 0.], &[1_000_000., 0., 0.,], 6);
        let vectors: Vec<Vector3<Precision>> = points.iter()
            .map(|p| Vector3::from_column_slice(p))
            .collect();
        let tangent = get_tangent(vectors).expect("SVD failed");
        assert_close(tangent.dot(&tangent), 1.0)
    }

    #[test]
    fn query() {
        let query = PointTangents::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10))
            .expect("Construction failed");
        let target = PointTangents::new(&make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10))
            .expect("Construction failed");

        let dist_dots = query.query_target_dist_dots(&target);
        assert_eq!(dist_dots.len(), 10);
    }

    #[test]
    fn score_function() {
        let dist_thresholds = vec![1.0, 2.0];
        let dot_thresholds = vec![0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);

        let query = PointTangents::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10))
            .expect("Construction failed");
        let target = PointTangents::new(&make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10))
            .expect("Construction failed");

        let _score = query.query_target(&target, &score_fn);
    }

    #[test]
    fn arena() {
        let dist_thresholds = vec![1.0, 2.0];
        let dot_thresholds = vec![0.5, 1.0];
        let cells = vec![1.0, 2.0, 4.0, 8.0];

        let score_fn = table_to_fn(dist_thresholds, dot_thresholds, cells);

        let query = PointTangents::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10))
            .expect("Construction failed");
        let target = PointTangents::new(&make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10))
            .expect("Construction failed");

        let mut arena = NblastArena::new(score_fn);
        let q_idx = arena.add_pointtangents(query);
        let t_idx = arena.add_pointtangents(target);

        let no_norm = arena
            .query_target(q_idx, t_idx, false, false)
            .expect("should exist");
        let self_hit = arena
            .query_target(q_idx, q_idx, false, false)
            .expect("should exist");
        assert!(
            arena
                .query_target(q_idx, t_idx, true, false)
                .expect("should exist")
                - no_norm / self_hit
                < EPSILON
        );
        assert_eq!(
            arena.query_target(q_idx, t_idx, false, true),
            arena.query_target(t_idx, q_idx, false, true),
        );
    }
}
