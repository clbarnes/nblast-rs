//! Neurites which can be queried against each other.
//!
//! Spatial lookups are the slowest part of NBLAST.
//! As such, there are [NblastNeuron] implementations here using a number of different backends selected at compile time using cargo features.
//! [Neuron] is a convenient type alias for our recommended backend given those available.
use crate::{centroid, DistDot, Normal3, Point3, Precision, ScoreCalc};

#[cfg(feature = "bosque")]
pub mod bosque;
#[cfg(feature = "kiddo")]
pub mod kiddo;
#[cfg(feature = "nabo")]
pub mod nabo;
#[cfg(feature = "rstar")]
pub mod rstar;

#[cfg(feature = "serde")]
pub mod any;

cfg_if::cfg_if! {
    if #[cfg(feature = "kiddo")] {
        pub type Neuron = self::kiddo::KiddoNeuron;
        pub const DEFAULT_BACKEND: &'static str = "kiddo";
    } else if #[cfg(feature = "bosque")] {
        pub type Neuron = self::bosque::BosqueNeuron;
        pub const DEFAULT_BACKEND: &'static str = "bosque";
    } else if #[cfg(feature = "rstar")] {
        pub type Neuron = self::rstar::RstarNeuron;
        pub const DEFAULT_BACKEND: &'static str = "rstar";
    } else if #[cfg(feature = "nabo")] {
        pub type Neuron = self::nabo::NaboNeuron;
        pub const DEFAULT_BACKEND: &'static str = "nabo";
    } else {
        compile_error!("No spatial query backend selected");
    }
}

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
    fn points(&self) -> impl Iterator<Item = Point3> + '_;

    fn centroid(&self) -> Point3 {
        centroid(self.points())
    }

    /// Return an owned copy of the unit tangents present in the neuron.
    /// The order is not guaranteed, but is consistent with
    /// [points](#method.points).
    fn tangents(&self) -> impl Iterator<Item = Normal3> + '_;

    /// Return an owned copy of the alpha values for points in the neuron.
    /// The order is consistent with [points](#method.points)
    /// and [tangents](#method.tangents).
    fn alphas(&self) -> impl Iterator<Item = Precision> + '_;
}

/// Trait for objects which can be used as queries
/// (not necessarily as targets) with NBLAST.
/// See [TargetNeuron].
pub trait QueryNeuron: NblastNeuron {
    /// Calculate the distance and (alpha-scaled) absolute dot products for point matches
    /// between this and a target neuron.
    fn query_dist_dots<'a>(
        &'a self,
        target: &'a impl TargetNeuron,
        use_alpha: bool,
    ) -> impl Iterator<Item = DistDot> + 'a;

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
            .map(|dd| score_calc.calc(&dd))
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
