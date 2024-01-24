use crate::{
    geometric_mean, DistDot, NblastNeuron, Normal3, Point3, Precision, QueryNeuron, ScoreCalc,
    TangentAlpha, TargetNeuron,
};
use bosque::tree::{build_tree, build_tree_with_indices, nearest_k, nearest_one};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BosqueNeuron {
    points: Vec<Point3>,
    tangents_alphas: Vec<TangentAlpha>,
}

impl BosqueNeuron {
    pub fn new(mut points: Vec<Point3>, k: usize) -> Result<Self, &'static str> {
        if points.len() < k {
            return Err("Not enough points to calculate neighborhood");
        }
        build_tree(points.as_mut());
        let tangents_alphas = points
            .iter()
            .map(|p| {
                TangentAlpha::new_from_points(
                    nearest_k(points.as_slice(), p, k)
                        .into_iter()
                        // todo: indirect through indices?
                        .map(|(_d, idx)| points.get(idx).unwrap()),
                )
            })
            .collect();
        Ok(Self {
            points,
            tangents_alphas,
        })
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas(
        mut points: Vec<Point3>,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Result<Self, &'static str> {
        if points.len() != tangents_alphas.len() {
            return Err("Mismatch in points and tangents_alphas length");
        }
        let mut indices: Vec<_> = (0..(points.len() as u32)).collect();
        build_tree_with_indices(points.as_mut(), indices.as_mut());
        let tas = indices
            .into_iter()
            .map(|idx32| tangents_alphas[idx32 as usize])
            .collect();
        Ok(Self {
            points,
            tangents_alphas: tas,
        })
    }
}

impl NblastNeuron for BosqueNeuron {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn points(&self) -> impl Iterator<Item = Point3> + '_ {
        self.points.iter().cloned()
    }

    fn tangents(&self) -> impl Iterator<Item = Normal3> + '_ {
        self.tangents_alphas.iter().map(|ta| ta.tangent)
    }

    fn alphas(&self) -> impl Iterator<Item = Precision> + '_ {
        self.tangents_alphas.iter().map(|ta| ta.alpha)
    }
}

impl QueryNeuron for BosqueNeuron {
    fn query_dist_dots<'a>(
        &'a self,
        target: &'a impl crate::TargetNeuron,
        use_alpha: bool,
    ) -> impl Iterator<Item = crate::DistDot> + 'a {
        self.points
            .iter()
            .zip(self.tangents_alphas.iter())
            .map(move |(p, ta)| {
                let alpha = if use_alpha { Some(ta.alpha) } else { None };
                target.nearest_match_dist_dot(p, &ta.tangent, alpha)
            })
    }

    fn query(
        &self,
        target: &impl TargetNeuron,
        use_alpha: bool,
        score_calc: &ScoreCalc,
    ) -> Precision {
        self.query_dist_dots(target, use_alpha)
            .map(|dd| score_calc.calc(&dd))
            .sum()
    }

    fn self_hit(&self, score_calc: &crate::ScoreCalc, use_alpha: bool) -> crate::Precision {
        if use_alpha {
            self.tangents_alphas
                .iter()
                .map(|ta| {
                    score_calc.calc(&DistDot {
                        dist: 0.0,
                        dot: ta.alpha,
                    })
                })
                .sum()
        } else {
            score_calc.calc(&DistDot {
                dist: 0.0,
                dot: 1.0,
            }) * self.len() as Precision
        }
    }
}

impl TargetNeuron for BosqueNeuron {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &crate::Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        let (dist, idx) = nearest_one(self.points.as_slice(), point);
        // todo: redirect through indices?
        let ta = self.tangents_alphas[idx];
        let raw_dot = ta.tangent.dot(tangent).abs();
        let dot = match alpha {
            Some(a) => raw_dot * geometric_mean(a, ta.alpha),
            None => raw_dot,
        };
        DistDot { dist, dot }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use fastrand::Rng;

    fn random_points(n: usize, rng: &mut Rng) -> Vec<Point3> {
        iter::repeat_with(|| [rng.f64(), rng.f64(), rng.f64()])
            .take(n)
            .collect()
    }

    #[cfg(feature = "kiddo")]
    #[test]
    fn expected_idxs() {
        use crate::neurons::kiddo::KiddoNeuron;

        let mut rng = Rng::with_seed(1991);
        let pts = random_points(100, &mut rng);
        let t_kid = KiddoNeuron::new(pts.clone(), 5).unwrap();
        let b_kid = BosqueNeuron::new(pts, 5).unwrap();

        let pts2 = random_points(5, &mut rng);
        let query = KiddoNeuron::new(pts2, 5).unwrap();

        let t_res: Vec<_> = query.query_dist_dots(&t_kid, false).collect();
        let b_res: Vec<_> = query.query_dist_dots(&b_kid, false).collect();

        assert_eq!(t_res, b_res);
    }
}
