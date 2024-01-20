//! Neurites which can be queried against each other.
use crate::{centroid, DistDot, Normal3, Point3, Precision, ScoreCalc, TangentAlpha};

use self::rstar::RStarTangentsAlphas;

// pub mod fnntw;
#[cfg(feature = "nabo")]
pub mod nabo;
pub mod rstar;

/// Trait describing a point cloud representing a neuron.
pub trait NblastNeuron {
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

    fn centroid(&self) -> Point3 {
        centroid(&self.points())
    }

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
pub trait QueryNeuron: NblastNeuron {
    /// Calculate the distance and (alpha-scaled) absolute dot products for point matches
    /// between this and a target neuron.
    fn query_dist_dots(&self, target: &impl TargetNeuron, use_alpha: bool) -> Vec<DistDot>;

    /// Calculate the raw NBLAST score by comparing this neuron to
    /// the given target neuron, using the given score function.
    /// The score function is applied to each point match distance and summed.
    fn query(
        &self,
        target: &impl TargetNeuron,
        use_alpha: bool,
        score_calc: &ScoreCalc,
    ) -> Precision {
        self.query_dist_dots(target, use_alpha)
            .iter()
            .map(|dd| score_calc.calc(dd))
            .sum()
    }

    /// The raw NBLAST score if this neuron was compared with itself using the given score function.
    /// Used for normalisation.
    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision;
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

#[derive(Clone)]
pub struct Neuron(rstar::RStarTangentsAlphas);

impl Neuron {
    /// Calculate tangents from constructed R*-tree.
    /// `k` is the number of points to calculate each tangent with.
    pub fn new<T: std::borrow::Borrow<Point3>>(
        points: impl IntoIterator<
            Item = T,
            IntoIter = impl Iterator<Item = T> + ExactSizeIterator + Clone,
        >,
        k: usize,
    ) -> Result<Self, &'static str> {
        Ok(Neuron(RStarTangentsAlphas::new(points, k)?))
    }

    /// Use pre-calculated tangents.
    pub fn new_with_tangents_alphas<T: std::borrow::Borrow<Point3>>(
        points: impl IntoIterator<
            Item = T,
            IntoIter = impl Iterator<Item = T> + ExactSizeIterator + Clone,
        >,
        tangents_alphas: Vec<TangentAlpha>,
    ) -> Result<Self, &'static str> {
        Ok(Neuron(RStarTangentsAlphas::new_with_tangents_alphas(
            points,
            tangents_alphas,
        )?))
    }
}

impl NblastNeuron for Neuron {
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

impl QueryNeuron for Neuron {
    fn query_dist_dots(&self, target: &impl TargetNeuron, use_alpha: bool) -> Vec<DistDot> {
        self.0.query_dist_dots(target, use_alpha)
    }

    fn self_hit(&self, score_calc: &ScoreCalc, use_alpha: bool) -> Precision {
        self.0.self_hit(score_calc, use_alpha)
    }
}

impl TargetNeuron for Neuron {
    fn nearest_match_dist_dot(
        &self,
        point: &Point3,
        tangent: &Normal3,
        alpha: Option<Precision>,
    ) -> DistDot {
        self.0.nearest_match_dist_dot(point, tangent, alpha)
    }
}
