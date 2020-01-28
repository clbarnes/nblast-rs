use nalgebra::base::{Matrix3x6, Vector3};
use rstar::primitives::PointWithData;
use rstar::RTree;

const N_NEIGHBORS: usize = 5;
type Precision = f64;
type PointWithIndex = PointWithData<usize, [Precision; 3]>;

pub struct DistDot {
    pub dist: Precision,
    pub dot: Precision,
}

pub struct DotProps {
    rtree: RTree<PointWithIndex>,
    tangents: Vec<Vector3<Precision>>,
}

// TODO: check orientation of matrices, may need to transpose everything

impl DotProps {
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

        let mut tangents: Vec<Vector3<Precision>> = Vec::with_capacity(rtree.size());

        for point in points.iter() {
            let nearest_it = rtree.nearest_neighbor_iter(&point);

            // TODO: would probably be faster to read the matrix as single elements
            // from a flat_map, but this intermediate should help prevent row/column order bugs
            let columns: Vec<Vector3<Precision>> = nearest_it
                .take(N_NEIGHBORS + 1)
                .map(|p| Vector3::from_column_slice(p.position()))
                .collect();

            let neighbor_mat: Matrix3x6<Precision> = Matrix3x6::from_columns(&columns);

            let svd = neighbor_mat.svd(false, true);
            if let Some(v_t) = svd.v_t {
                tangents.push(Vector3::from_iterator(v_t.column(0).iter().cloned()));
            } else {
                return Err("Failed to SVD");
            }
        }

        Ok(Self { rtree, tangents })
    }

    pub fn nearest_match_dist_dot(
        &self,
        point: &[Precision; 3],
        tangent: &Vector3<Precision>,
    ) -> DistDot {
        if let Some((element, dist)) = self.rtree.nearest_neighbor_iter_with_distance(point).next()
        {
            let this_tangent = self.tangents[element.data];
            let dot = this_tangent.dot(tangent).abs();
            DistDot { dist, dot }
        } else {
            panic!("Impossible")
        }
    }

    /// Arbitrary order
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
}

pub fn queries_targets<F>(
    qs_ts: &[(&DotProps, Vec<&DotProps>)],
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

/// Cells in dot-major order
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

#[cfg(test)]
mod tests {
    use super::*;

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
        DotProps::new(&points).expect("Construction failed");
    }

    #[test]
    fn query() {
        let query = DotProps::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10))
            .expect("Construction failed");
        let target = DotProps::new(&make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10))
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

        let query = DotProps::new(&make_points(&[0., 0., 0.], &[1., 0., 0.], 10))
            .expect("Construction failed");
        let target = DotProps::new(&make_points(&[0.5, 0., 0.], &[1.1, 0., 0.], 10))
            .expect("Construction failed");

        let score = query.query_target(&target, &score_fn);
    }
}
