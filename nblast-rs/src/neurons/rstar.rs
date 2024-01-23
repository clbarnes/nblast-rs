//! Neuron types using the [rstar](https://crates.io/crates/rstar) crate as a backend.
use super::{NblastNeuron, QueryNeuron, TargetNeuron};
use crate::{geometric_mean, DistDot, Normal3, Point3, Precision, ScoreCalc, TangentAlpha};
use rstar::{primitives::GeomWithData, PointDistance, RTree};
use std::borrow::Borrow;

type PointWithIndex = GeomWithData<Point3, usize>;

fn points_to_rtree(
    points: impl Iterator<Item = impl Borrow<Point3>>,
) -> Result<RTree<PointWithIndex>, &'static str> {
    Ok(RTree::bulk_load(
        points
            .enumerate()
            .map(|(idx, point)| PointWithIndex::new(*point.borrow(), idx))
            .collect(),
    ))
}

pub(crate) fn points_to_rtree_tangents_alphas(
    points: impl Iterator<Item = impl std::borrow::Borrow<Point3>> + ExactSizeIterator + Clone,
    k: usize,
) -> Result<(RTree<PointWithIndex>, Vec<TangentAlpha>), &'static str> {
    if points.len() < k {
        return Err("Too few points to generate tangents");
    }
    let rtree = points_to_rtree(points.clone())?;
    let tangents_alphas = rtree
        .iter()
        .map(|geom| {
            let p = geom.geom();
            TangentAlpha::new_from_points(
                rtree.nearest_neighbor_iter(p).take(k).map(|pwd| pwd.geom()),
            )
        })
        .collect();

    Ok((rtree, tangents_alphas))
}

/// Target neuron using an [R*-tree](https://en.wikipedia.org/wiki/R*_tree) for spatial queries.
#[derive(Clone)]
pub struct RstarNeuron {
    rtree: RTree<PointWithIndex>,
    tangents_alphas: Vec<TangentAlpha>,
}

impl RstarNeuron {
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
            RstarNeuron {
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
        let rtree = points_to_rtree(points.into_iter())?;
        let ta = rtree
            .iter()
            .map(|gwd| {
                let idx = gwd.data;
                tangents_alphas[idx]
            })
            .collect();
        Ok(Self {
            rtree,
            tangents_alphas: ta,
        })
    }
}

impl NblastNeuron for RstarNeuron {
    fn len(&self) -> usize {
        self.tangents_alphas.len()
    }

    fn points(&self) -> impl Iterator<Item = Point3> + '_ {
        self.rtree.iter().map(|gwd| *gwd.geom())
    }

    fn tangents(&self) -> impl Iterator<Item = Normal3> + '_ {
        self.tangents_alphas.iter().map(|ta| ta.tangent)
    }

    fn alphas(&self) -> impl Iterator<Item = Precision> + '_ {
        self.tangents_alphas.iter().map(|ta| ta.alpha)
    }
}

impl QueryNeuron for RstarNeuron {
    fn query_dist_dots<'a>(
        &'a self,
        target: &'a impl TargetNeuron,
        use_alpha: bool,
    ) -> impl Iterator<Item = DistDot> + 'a {
        self.rtree.iter().map(move |q_pt_idx| {
            let tangent_alpha = self.tangents_alphas[q_pt_idx.data];
            let alpha = if use_alpha {
                Some(tangent_alpha.alpha)
            } else {
                None
            };
            target.nearest_match_dist_dot(q_pt_idx.geom(), &tangent_alpha.tangent, alpha)
        })
    }

    fn query(
        &self,
        target: &impl TargetNeuron,
        use_alpha: bool,
        score_calc: &ScoreCalc,
    ) -> Precision {
        let mut score_total: Precision = 0.0;
        for q_pt_idx in self.rtree.iter() {
            let tangent_alpha = self.tangents_alphas[q_pt_idx.data];
            let alpha = if use_alpha {
                Some(tangent_alpha.alpha)
            } else {
                None
            };
            let dd = target.nearest_match_dist_dot(q_pt_idx.geom(), &tangent_alpha.tangent, alpha);
            let score = score_calc.calc(&dd);
            score_total += score;
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

impl TargetNeuron for RstarNeuron {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.rtree
            .nearest_neighbor(point)
            .map(|element| {
                let ta = self.tangents_alphas[element.data];
                let raw_dot = ta.tangent.dot(tangent).abs();
                let dot = match alpha {
                    Some(a) => raw_dot * geometric_mean(a, ta.alpha),
                    None => raw_dot,
                };
                DistDot {
                    dist: element.distance_2(point).sqrt(),
                    dot,
                }
            })
            .expect("impossible")
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;

//     const EPSILON: Precision = 0.001;
//     const N_NEIGHBORS: usize = 5;
// }
