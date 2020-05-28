//! The crux of the NBLAST algorithm is converting `(distance, abs_dot)` point matches
//! into a meaningful score.
//!
//! The strategy here generates a lookup table whose indices are dist and dot bins.
//! A neuron match score is the sum of the values of the lookup table over every point match.
//!
//! The values in the table cells are the log2 odds ratio of a dist-dot in that cell
//! being from a matching over a non-matching neuron pair.
//! This is calculated by adding a large pool of neurons
//! (most of which would not match), with small groups of neurons which should match each other.
//! Calculate all the point matches between all pairs of neurons in those sets,
//! then draw random pairs of neurons from either the whole neuron pool,
//! or an explicitly non-matching subset.
//! Normalise the counts falling into each cell by the total number of matching
//! and non-matching dist-dots, and calculate the log2 odds ratios.
//!
//! This is handled by the ScoreMatrixBuilder.
//! The dist and dot bins for the table are can be set manually or determined
//! based on how many bins there should be, distributed linearly
//! and logarithmically (with a given base) respectively.
use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand_pcg::Pcg32;
use rayon::prelude::*;
use std::collections::HashSet;
use std::error;
use std::fmt;
use std::iter;

use crate::{DistDot, Precision, TargetNeuron};

const EPSILON: Precision = 1e-6;

/// If both indexes are in the neuron slice, get the DistDots between them.
fn idx_to_distdots(
    neurons: &[impl TargetNeuron],
    q_idx: usize,
    t_idx: usize,
    use_alpha: bool,
) -> Option<Vec<DistDot>> {
    let q = neurons.get(q_idx)?;
    let t = neurons.get(t_idx)?;
    Some(q.query_dist_dots(t, use_alpha))
}

/// For all given `(query, target)` index pairs, find DistDots and flatten
/// into a single array.
fn pairs_to_distdots_ser(
    neurons: &[impl TargetNeuron],
    jobs: impl IntoIterator<Item = (usize, usize)>,
    use_alpha: bool,
) -> Vec<DistDot> {
    jobs.into_iter()
        .filter_map(|(q_idx, t_idx)| idx_to_distdots(neurons, q_idx, t_idx, use_alpha))
        .flatten()
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn pairs_to_distdots<T: TargetNeuron + Sync>(
    neurons: &[T],
    jobs: Vec<(usize, usize)>,
    use_alpha: bool,
    threads: Option<usize>,
) -> Vec<DistDot> {
    pairs_to_distdots_ser(neurons, jobs, use_alpha)
}

#[cfg(feature = "parallel")]
fn pairs_to_distdots<T: TargetNeuron + Sync>(
    neurons: &[T],
    jobs: Vec<(usize, usize)>,
    use_alpha: bool,
    threads: Option<usize>,
) -> Vec<DistDot> {
    if let Some(t) = threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        pool.install(|| {
            jobs.into_par_iter()
                .filter_map(|(q_idx, t_idx)| idx_to_distdots(neurons, q_idx, t_idx, use_alpha))
                .flatten()
                .collect()
        })
    } else {
        pairs_to_distdots_ser(neurons, jobs, use_alpha)
    }
}

/// Calculate the DistDots for matching and non-matching neurons.
///
/// Every non-repeating 2-length permutation of neurons within each matching set is used.
/// Then, pairs of neurons are taken randomly from the non-matching set
/// (if `None`, use all neurons) until there are at least as many non-matching DistDots
/// as there are matching.
///
/// Return a tuple of vecs of matching and non-matching DistDots.
fn match_nonmatch_distdots<T: TargetNeuron + Sync>(
    neurons: &[T],
    matching_sets: &[HashSet<usize>],
    non_matching_set: Option<Vec<usize>>,
    use_alpha: bool,
    seed: u64,
    threads: Option<usize>,
) -> (Vec<DistDot>, Vec<DistDot>) {
    let mut matching_len = 0;
    let mut matching_jobs: HashSet<(usize, usize)> = HashSet::default();

    // for every matching set
    for matching_set in matching_sets {
        // for every possible query idx
        for q_idx in matching_set.iter() {
            // if the idx is in the given neurons
            let q_len = match neurons.get(*q_idx) {
                Some(n) => n.len(),
                None => continue,
            };
            // for every possible target idx
            for t_idx in matching_set.iter() {
                // if t!=q, t exists, and the pair is not already addressed
                if t_idx != q_idx
                    && t_idx < &neurons.len()
                    && matching_jobs.insert((*q_idx, *t_idx))
                {
                    // keep track of how many distdots we're producing
                    matching_len += q_len
                }
            }
        }
    }

    // if nonmatching not given, use all neurons
    let nonmatching_idxs = non_matching_set
        .or_else(|| Some((0..neurons.len()).collect()))
        .unwrap();

    if matching_jobs.len() > nonmatching_idxs.len() * (nonmatching_idxs.len() - 1) {
        panic!("Not enough non-matching neurons")
    }

    let range = Uniform::new(0, nonmatching_idxs.len());

    let mut rng = Pcg32::seed_from_u64(seed);
    let mut nonmatching_jobs: HashSet<(usize, usize)> = HashSet::default();

    // randomly pick nonmatching pairs until we have requested as many distdots
    // as we did for matching
    while matching_len > 0 {
        let q_idx = nonmatching_idxs[range.sample(&mut rng)];
        let t_idx = nonmatching_idxs[range.sample(&mut rng)];

        let key = (q_idx, t_idx);
        if q_idx != t_idx && !matching_jobs.contains(&key) && nonmatching_jobs.insert(key) {
            matching_len -= neurons[q_idx].len()
        }
    }

    // ! passing round vecs is easier to parallelise but unnecessarily RAM-intensive
    // can we use iterators instead?

    let matching_dd = pairs_to_distdots(
        &neurons,
        matching_jobs.drain().collect(),
        use_alpha,
        threads,
    );

    let nonmatching_dd = pairs_to_distdots(
        &neurons,
        nonmatching_jobs.drain().collect(),
        use_alpha,
        threads,
    );

    (matching_dd, nonmatching_dd)
}

/// Calculate the scores associated with cells of a table defined by dist and dot thresholds.
///
/// This is the log2 odds ratio of a DistDot in that cell
/// having been drawn from a matching pair of neurons over a non-matching pair.
fn calculate_cells(
    dist_thresholds: &[Precision],
    dot_thresholds: &[Precision],
    matching_distdots: Vec<DistDot>,
    nonmatching_distdots: Vec<DistDot>,
) -> Vec<Precision> {
    let n_cells = dist_thresholds.len() * dot_thresholds.len();
    let mut matching_count: Vec<Precision> = iter::repeat(0.0).take(n_cells).collect();
    let mut nonmatching_count = matching_count.clone();

    // There may be more nonmatching distdots than matching ones.
    // Therefore, the nonmatching count in each cell should be scaled up.
    let matching_factor =
        nonmatching_distdots.len() as Precision / matching_distdots.len() as Precision;

    for dd in matching_distdots {
        matching_count[dd.to_linear_idx(dist_thresholds, dot_thresholds)] += matching_factor;
    }
    for dd in nonmatching_distdots {
        nonmatching_count[dd.to_linear_idx(dist_thresholds, dot_thresholds)] += 1.0;
    }

    matching_count
        .into_iter()
        .zip(nonmatching_count.into_iter())
        .map(|(match_count, nonmatch_count)| {
            // add epsilon to prevent division by 0
            ((match_count + EPSILON) / (nonmatch_count + EPSILON)).log2()
        })
        .collect()
}

#[derive(Debug)]
pub struct ScoreMatBuildErr {}

impl fmt::Display for ScoreMatBuildErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bins not set or no matching neurons given")
    }
}

impl error::Error for ScoreMatBuildErr {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

/// Calculate a score matrix (lookup table for converting point matches
/// into NBLAST scores) from real data using some sets of matching and non-matching neurons.
///
/// At a minimum, one matching set must be added (`.add_matching_set`),
/// and the dist and dot bins must either be set automatically or calculated
/// (`.set_(n_)?(dist|dot)_bins`).
pub struct ScoreMatrixBuilder<T: TargetNeuron> {
    // Data set of neuron point clouds
    neurons: Vec<T>,
    seed: u64,
    // Sets of neurons, as indexes into `self.neurons`, which should match
    matching_sets: Vec<HashSet<usize>>,
    nonmatching: Option<Vec<usize>>,
    use_alpha: bool,
    threads: Option<usize>,
    dist_bins_upper: Option<Vec<Precision>>,
    dot_bins_upper: Option<Vec<Precision>>,
}

impl<T: TargetNeuron + Sync> ScoreMatrixBuilder<T> {
    /// `seed` is used when randomly selecting non-matching neurons to use as controls.
    pub fn new(neurons: Vec<T>, seed: u64) -> Self {
        Self {
            neurons,
            seed,
            matching_sets: Vec::default(),
            nonmatching: None,
            use_alpha: false,
            threads: Some(0),
            dist_bins_upper: None,
            dot_bins_upper: None,
        }
    }

    /// Add a set of neurons which are considered to be mutually matching,
    /// as indices into the `ScoreMatrixBuilder.neurons`.
    /// This can be done several times with different sets.
    /// Matching sets should be small subsets of the total neuron count.
    pub fn add_matching_set(&mut self, matching: HashSet<usize>) -> &mut Self {
        self.matching_sets.push(matching);
        self
    }

    /// Optionally list neurons which are mutually non-matching,
    /// as indices into `ScoreMatrixBuilder.neurons`,
    /// from which non-matching pairs will be randomly drawn.
    /// By default, all neurons are included,
    /// as matching sets are expected to be only a small portion of the total neurons.
    ///
    /// Candidate non-matching pairs are made sure not to be in the matching pairs,
    /// so this can safely be ignored.
    pub fn set_nonmatching(&mut self, nonmatching: Vec<usize>) -> &mut Self {
        self.nonmatching = Some(nonmatching);
        self
    }

    /// Alpha values are a measure of how colinear a neuron's local neighborhood is.
    /// Score matrices calculated using alpha can only be used for queries using alpha,
    /// and vice versa.
    ///
    /// False by default.
    pub fn set_use_alpha(&mut self, use_alpha: bool) -> &mut Self {
        self.use_alpha = use_alpha;
        self
    }

    /// How many threads to use for calculating the pair matches.
    ///
    /// `None` means the calculation is done in serial;
    /// for `Some(n)`, `n = 0` means use all available CPUs (default),
    /// and any other value uses that many cores.
    pub fn set_threads(&mut self, threads: Option<usize>) -> &mut Self {
        self.threads = threads;
        self
    }

    /// Manually set the upper bounds of the distance bins.
    /// The lowest bound is implicitly 0.
    /// The upper bound is effectively ignored, as values above that value
    /// snap into the highest bin anyway.
    pub fn set_dist_bins<'a>(&'a mut self, dist_bins_upper: Vec<Precision>) -> &mut Self {
        self.dist_bins_upper = Some(dist_bins_upper);
        self
    }

    /// Automatically generate distance bins by logarithmically interpolating
    /// between 0 and `greatest_lower_bound`, using a logarithm of the given `base`.
    pub fn set_n_dist_bins(
        &mut self,
        n_dist_bins: usize,
        greatest_lower_bound: Precision,
        base: Precision,
    ) -> &mut Self {
        let log_greatest = greatest_lower_bound.log(base);
        let step = log_greatest / (n_dist_bins - 1) as Precision;
        let mut v: Vec<_> = (1..n_dist_bins)
            .map(|n| (step * n as Precision).powf(base))
            .collect();
        // ? better way to get infinity
        v.push(1.0 / 0.0);
        self.dist_bins_upper = Some(v);
        self
    }

    /// Manually set the upper bounds of the abs dot product bins.
    /// The lowest bound is implicitly 0.
    /// The upper bound is effectively ignored, as values above that value
    /// snap into the highest bin anyway.
    ///
    /// Because the tangent vectors are unit-length,
    /// absolute dot products between them are between 0 and 1
    /// (although float precision means values slightly above 1 are possible).
    pub fn set_dot_bins<'a>(&'a mut self, dot_bins_upper: Vec<Precision>) -> &'a mut Self {
        self.dot_bins_upper = Some(dot_bins_upper);
        self
    }

    /// Automatically generate abs dot product bins by linearly interpolating between
    /// 0 and 1.
    pub fn set_n_dot_bins(&mut self, n_dot_bins: usize) -> &mut Self {
        let step = 1.0 / n_dot_bins as Precision;
        self.set_dot_bins(
            (1..(n_dot_bins + 1))
                .map(|n| step * n as Precision)
                .collect(),
        )
    }

    /// Return a tuple of `(dist_bins, dot_bins, cells)` in the forms accepted by
    /// `table_to_fn`.
    pub fn build(
        self,
    ) -> Result<(Vec<Precision>, Vec<Precision>, Vec<Precision>), ScoreMatBuildErr> {
        let dist_bins = self.dist_bins_upper.ok_or(ScoreMatBuildErr {})?;
        let dot_bins = self.dot_bins_upper.ok_or(ScoreMatBuildErr {})?;
        if self.matching_sets.is_empty() {
            return Err(ScoreMatBuildErr {});
        }

        let (matching, nonmatching) = match_nonmatch_distdots(
            &self.neurons,
            &self.matching_sets,
            self.nonmatching,
            self.use_alpha,
            self.seed,
            self.threads,
        );
        let cells = calculate_cells(&dist_bins, &dot_bins, matching, nonmatching);

        Ok((dist_bins, dot_bins, cells))
    }
}
