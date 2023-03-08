use crate::{centroid, DistDot, Normal3, Point3, Precision, ScoreCalc};

// pub mod fnntw;
#[cfg(feature = "nabo")]
pub mod nabo;
pub mod rstar;

/// Trait describing a point cloud representing a neuron.
pub trait Neuron {
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
pub trait QueryNeuron: Neuron {
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
