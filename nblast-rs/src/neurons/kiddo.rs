//! Neuron types using the [kiddo](https://crates.io/crates/kiddo) crate as a backend.
use super::{NblastNeuron, QueryNeuron, TargetNeuron};
use crate::{geometric_mean, DistDot, Normal3, Point3, Precision, ScoreCalc, TangentAlpha};
use kiddo::{ImmutableKdTree, SquaredEuclidean};

type KdTree = ImmutableKdTree<Precision, 3>;

/// Target neuron using a KDTree from the kiddo crate.
///
/// By default, this uses approximate nearest neighbour for one-off lookups (as used in NBLAST scoring).
/// However, in tests it is *very* approximate.
/// See the [KiddoNeuron] for exact 1NN.
#[derive(Clone, Debug)]
pub struct KiddoNeuron {
    tree: KdTree,
    points_tangents_alphas: Vec<(Point3, TangentAlpha)>,
}

impl KiddoNeuron {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<Point3>, k: usize) -> Result<Self, &'static str> {
        let tree: KdTree = points.as_slice().into();
        if points.len() < k {
            return Err("Not enough points to calculate neighborhood");
        }
        let points_tangents_alphas = points
            .iter()
            .map(|p| {
                let neighbors = tree.nearest_n::<SquaredEuclidean>(p, k);

                let pts = neighbors.iter().map(|nn| &points[nn.item as usize]);
                (*p, TangentAlpha::new_from_points(pts))
            })
            .collect();

        Ok(Self {
            tree,
            points_tangents_alphas,
        })
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas(
        points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Result<Self, &'static str> {
        if points.len() != tangents_alphas.len() {
            return Err("Mismatch in points and tangents_alphas length");
        }
        let tree: KdTree = points.as_slice().into();
        Ok(Self {
            tree,
            points_tangents_alphas: points.into_iter().zip(tangents_alphas).collect(),
        })
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

impl NblastNeuron for KiddoNeuron {
    fn len(&self) -> usize {
        self.points_tangents_alphas.len()
    }

    fn points(&self) -> impl Iterator<Item = Point3> + '_ {
        self.points_tangents_alphas.iter().map(|pta| pta.0)
    }

    fn tangents(&self) -> impl Iterator<Item = Normal3> + '_ {
        self.points_tangents_alphas.iter().map(|pta| pta.1.tangent)
    }

    fn alphas(&self) -> impl Iterator<Item = Precision> + '_ {
        self.points_tangents_alphas.iter().map(|pta| pta.1.alpha)
    }
}

impl QueryNeuron for KiddoNeuron {
    fn query_dist_dots<'a>(
        &'a self,
        target: &'a impl TargetNeuron,
        use_alpha: bool,
    ) -> impl Iterator<Item = DistDot> + 'a {
        self.points_tangents_alphas
            .iter()
            .map(move |(p, tangent_alpha)| {
                let alpha = if use_alpha {
                    Some(tangent_alpha.alpha)
                } else {
                    None
                };
                target.nearest_match_dist_dot(p, &tangent_alpha.tangent, alpha)
            })
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

impl TargetNeuron for KiddoNeuron {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.nearest_match_dist_dot_inner(point, tangent, alpha, true)
    }
}

#[derive(Debug, Clone)]
pub struct ApproxKiddoNeuron(KiddoNeuron);

impl ApproxKiddoNeuron {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new(points: Vec<Point3>, k: usize) -> Result<Self, &'static str> {
        KiddoNeuron::new(points, k).map(Self)
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas(
        points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Result<Self, &'static str> {
        KiddoNeuron::new_with_tangents_alphas(points, tangents_alphas).map(Self)
    }
}

impl NblastNeuron for ApproxKiddoNeuron {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn points(&self) -> impl Iterator<Item = Point3> + '_ {
        self.0.points()
    }

    fn tangents(&self) -> impl Iterator<Item = Normal3> + '_ {
        self.0.tangents()
    }

    fn alphas(&self) -> impl Iterator<Item = Precision> + '_ {
        self.0.alphas()
    }
}

impl QueryNeuron for ApproxKiddoNeuron {
    fn query_dist_dots<'a>(
        &'a self,
        target: &'a impl TargetNeuron,
        use_alpha: bool,
    ) -> impl Iterator<Item = DistDot> + '_ {
        self.0.query_dist_dots(target, use_alpha)
    }

    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision {
        self.0.self_hit(score_calc, use_alpha)
    }
}

impl TargetNeuron for ApproxKiddoNeuron {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.0
            .nearest_match_dist_dot_inner(point, tangent, alpha, false)
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
