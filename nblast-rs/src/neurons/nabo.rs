//! Neuron types using the [nabo](https://crates.io/crates/nabo) crate as a backend.
use super::{Neuron, QueryNeuron, TargetNeuron};
use crate::{
    centroid, geometric_mean, DistDot, Normal3, Point3, Precision, ScoreCalc, TangentAlpha,
};
use nabo::{KDTree, NotNan, Point};
use std::borrow::Borrow;

#[derive(Clone, Copy, Debug)]
struct NaboPointWithIndex {
    pub point: [NotNan<Precision>; 3],
    pub index: usize,
}

impl NaboPointWithIndex {
    pub fn new(point: &Point3, index: usize) -> Self {
        Self {
            point: [
                NotNan::new(point[0]).unwrap(),
                NotNan::new(point[1]).unwrap(),
                NotNan::new(point[2]).unwrap(),
            ],
            index,
        }
    }
}

impl From<Point3> for NaboPointWithIndex {
    fn from(p: Point3) -> Self {
        Self::new(&p, 0)
    }
}

impl From<NaboPointWithIndex> for Point3 {
    fn from(p: NaboPointWithIndex) -> Self {
        [*p.point[0], *p.point[1], *p.point[2]]
    }
}

impl Default for NaboPointWithIndex {
    fn default() -> Self {
        Self {
            point: [
                NotNan::new(0.0).unwrap(),
                NotNan::new(0.0).unwrap(),
                NotNan::new(0.0).unwrap(),
            ],
            index: 0,
        }
    }
}

impl Point<Precision> for NaboPointWithIndex {
    fn get(&self, i: u32) -> nabo::NotNan<Precision> {
        self.point[i as usize]
    }

    fn set(&mut self, i: u32, value: nabo::NotNan<Precision>) {
        self.point[i as usize] = value;
    }

    const DIM: u32 = 3;
}

fn points_to_nabo(
    points: impl Iterator<Item = impl Borrow<Point3>>,
) -> KDTree<Precision, NaboPointWithIndex> {
    let all_points: Vec<_> = points
        .enumerate()
        .map(|(idx, p)| NaboPointWithIndex::new(p.borrow(), idx))
        .collect();
    KDTree::new(&all_points)
}

fn points_to_nabo_tangents_alphas(
    points: impl Iterator<Item = impl Borrow<Point3>>,
    k: usize,
) -> (KDTree<Precision, NaboPointWithIndex>, Vec<TangentAlpha>) {
    let all_points: Vec<_> = points
        .enumerate()
        .map(|(idx, p)| NaboPointWithIndex::new(p.borrow(), idx))
        .collect();
    let tree = KDTree::new(&all_points);
    let k_u32 = k as u32;
    let tas = all_points
        .iter()
        .map(|p| {
            let ns: Vec<Point3> = tree.knn(k_u32, p).iter().map(|n| n.point.into()).collect();
            TangentAlpha::new_from_points(ns.iter())
        })
        .collect();
    (tree, tas)
}

/// Target neuron using a KDTree from the nabo crate
pub struct NaboTangentsAlphas {
    tree: KDTree<Precision, NaboPointWithIndex>,
    points_tangents_alphas: Vec<(Point3, TangentAlpha)>,
}

impl NaboTangentsAlphas {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<Point3>, k: usize) -> Self {
        let (tree, tangents_alphas) = points_to_nabo_tangents_alphas(points.iter(), k);
        Self {
            tree,
            points_tangents_alphas: points.into_iter().zip(tangents_alphas).collect(),
        }
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas<T: std::borrow::Borrow<Point3>>(
        points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Self {
        let tree = points_to_nabo(points.iter());
        Self {
            tree,
            points_tangents_alphas: points.into_iter().zip(tangents_alphas).collect(),
        }
    }
}

impl Neuron for NaboTangentsAlphas {
    fn len(&self) -> usize {
        self.points_tangents_alphas.len()
    }

    fn points(&self) -> Vec<Point3> {
        self.points_tangents_alphas
            .iter()
            .map(|pta| pta.0)
            .collect()
    }

    fn centroid(&self) -> Point3 {
        centroid(self.points().iter())
    }

    fn tangents(&self) -> Vec<Normal3> {
        self.points_tangents_alphas
            .iter()
            .map(|pta| pta.1.tangent)
            .collect()
    }

    fn alphas(&self) -> Vec<Precision> {
        self.points_tangents_alphas
            .iter()
            .map(|pta| pta.1.alpha)
            .collect()
    }
}

impl QueryNeuron for NaboTangentsAlphas {
    fn query_dist_dots(&self, target: &impl TargetNeuron, use_alpha: bool) -> Vec<DistDot> {
        self.points_tangents_alphas
            .iter()
            .map(|(p, tangent_alpha)| {
                let alpha = if use_alpha {
                    Some(tangent_alpha.alpha)
                } else {
                    None
                };
                target.nearest_match_dist_dot(p, &tangent_alpha.tangent, alpha)
            })
            .collect()
    }

    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision {
        if use_alpha {
            self.points_tangents_alphas
                .iter()
                .map(|(_, ta)| {
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

impl TargetNeuron for NaboTangentsAlphas {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        let nn = self
            .tree
            .knn(1, &(*point).into())
            .pop()
            .expect("No points in other tree");
        let idx = nn.point.index;
        let (_, ta) = self.points_tangents_alphas[idx];

        let raw_dot = ta.tangent.dot(tangent).abs();
        let dot = match alpha {
            Some(a) => raw_dot * geometric_mean(a, ta.alpha),
            None => raw_dot,
        };
        DistDot {
            dist: nn.dist2.sqrt(),
            dot,
        }
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     const EPSILON: Precision = 0.001;
//     const N_NEIGHBORS: usize = 5;
// }
