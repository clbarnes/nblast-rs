//! Neuron types using the [kiddo](https://crates.io/crates/kiddo) crate as a backend.
use super::{NblastNeuron, QueryNeuron, TargetNeuron};
use crate::{
    centroid, geometric_mean, DistDot, Normal3, Point3, Precision, ScoreCalc, TangentAlpha,
};
use kiddo::{ImmutableKdTree, SquaredEuclidean};

type KdTree = ImmutableKdTree<Precision, 3>;

/// Target neuron using a KDTree from the kiddo crate.
///
/// By default, this uses approximate nearest neighbour for one-off lookups (as used in NBLAST scoring).
/// However, in tests it is *very* approximate.
/// See the [ExactKiddoTangentsAlphas] for exact 1NN.
pub struct KiddoTangentsAlphas {
    tree: KdTree,
    points_tangents_alphas: Vec<(Point3, TangentAlpha)>,
}

impl KiddoTangentsAlphas {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<Point3>, k: usize) -> Self {
        let tree: KdTree = points.as_slice().into();
        let points_tangents_alphas = points
            .iter()
            .map(|p| {
                let neighbors = tree.nearest_n::<SquaredEuclidean>(p, k);

                let pts = neighbors.iter().map(|nn| &points[nn.item as usize]);
                (*p, TangentAlpha::new_from_points(pts))
            })
            .collect();

        Self {
            tree,
            points_tangents_alphas,
        }
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas(
        points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Self {
        let tree: KdTree = points.as_slice().into();
        Self {
            tree,
            points_tangents_alphas: points.into_iter().zip(tangents_alphas).collect(),
        }
    }

    fn nearest_match_dist_dot_inner(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
        exact: bool,
    ) -> DistDot {
        let nn = if exact {
            self.tree.nearest_one::<SquaredEuclidean>(point)
        } else {
            self.tree.approx_nearest_one::<SquaredEuclidean>(point)
        };

        let (_, ta) = self.points_tangents_alphas[nn.item as usize];

        let raw_dot = ta.tangent.dot(tangent).abs();
        let dot = match alpha {
            Some(a) => raw_dot * geometric_mean(a, ta.alpha),
            None => raw_dot,
        };
        DistDot {
            dist: nn.distance.sqrt(),
            dot,
        }
    }
}

impl NblastNeuron for KiddoTangentsAlphas {
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

impl QueryNeuron for KiddoTangentsAlphas {
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

impl TargetNeuron for KiddoTangentsAlphas {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.nearest_match_dist_dot_inner(point, tangent, alpha, false)
    }
}

pub struct ExactKiddoTangentsAlphas(KiddoTangentsAlphas);

impl ExactKiddoTangentsAlphas {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<Point3>, k: usize) -> Self {
        Self(KiddoTangentsAlphas::new(points, k))
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas(
        points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Self {
        Self(KiddoTangentsAlphas::new_with_tangents_alphas(
            points,
            tangents_alphas,
        ))
    }
}

impl NblastNeuron for ExactKiddoTangentsAlphas {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn points(&self) -> Vec<Point3> {
        self.0.points()
    }

    fn tangents(&self) -> Vec<Normal3> {
        self.0.tangents()
    }

    fn alphas(&self) -> Vec<Precision> {
        self.0.alphas()
    }
}

impl QueryNeuron for ExactKiddoTangentsAlphas {
    fn query_dist_dots(&self, target: &impl TargetNeuron, use_alpha: bool) -> Vec<DistDot> {
        self.0.query_dist_dots(target, use_alpha)
    }

    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision {
        self.0.self_hit(score_calc, use_alpha)
    }
}

impl TargetNeuron for ExactKiddoTangentsAlphas {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.0
            .nearest_match_dist_dot_inner(point, tangent, alpha, true)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use fastrand::Rng;

    fn random_points(n: usize, rng: &mut Rng) -> Vec<Point3> {
        std::iter::repeat_with(|| [rng.f64(), rng.f64(), rng.f64()])
            .take(n)
            .collect()
    }

    #[test]
    fn stable_index() {
        let mut rng = Rng::with_seed(1991);
        let pts = random_points(1000, &mut rng);
        let tree = KdTree::new_from_slice(pts.as_slice());
        for (exp_idx, pt) in pts.iter().enumerate() {
            let stored_idx = tree.nearest_one::<SquaredEuclidean>(pt).item as usize;
            assert_eq!(exp_idx, stored_idx)
        }
    }
}
