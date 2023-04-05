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
//! based on how many bins there should be, distributed logarithmically
//! (with a given base) and linearly respectively.
// use rand::distributions::{Distribution, Uniform};
// use rand::SeedableRng;
// use rand_pcg::Pcg32;
use fastrand;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::HashSet;
use std::error;
use std::fmt;
use std::iter;
use std::ops::Range;
#[cfg(feature = "rayon")]
use std::sync::mpsc::channel;

use crate::Neuron;
use crate::{BinLookup, NdBinLookup, RangeTable};
use crate::{DistDot, Precision, TargetNeuron};

const EPSILON: Precision = 1e-6;

type JobSet = HashSet<(usize, usize)>;

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

/// Generate `count` logarithmically-spaced values from `base^min_exp` up to (and including) `base^max_exp`
fn logspace(
    base: Precision,
    min_exp: Precision,
    max_exp: Precision,
    count: usize,
) -> Vec<Precision> {
    // TODO: do this better
    assert!(count > 2);
    let step = (max_exp - min_exp) / (count - 1) as Precision;

    (0..count)
        .map(|idx| base.powf(min_exp + idx as Precision * step))
        .collect()
}

struct PairSampler {
    sets: Vec<Vec<usize>>,
    cumu_weights: Vec<f64>,
    pub rng: fastrand::Rng,
}

impl PairSampler {
    fn new(sets: Vec<Vec<usize>>, seed: Option<u64>) -> Self {
        let rng = fastrand::Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }
        let mut weights: Vec<_> = sets.iter().map(|v| v.len().pow(2) as f64).collect();
        let total: f64 = weights.iter().sum();
        let mut acc = 0.0;
        for x in &mut weights {
            acc += *x / total;
            *x = acc;
        }
        Self {
            sets,
            cumu_weights: weights,
            rng,
        }
    }

    pub fn from_sets(sets: &[HashSet<usize>], seed: Option<u64>) -> Self {
        let vecs = sets
            .iter()
            .map(|s| {
                if s.len() < 2 {
                    panic!("Gave set with <2 components");
                }
                let mut v: Vec<_> = s.iter().cloned().collect();
                v.sort();
                v
            })
            .collect();
        Self::new(vecs, seed)
    }

    fn outer_idx(&mut self) -> usize {
        if self.cumu_weights.len() == 1 {
            return 0;
        }
        let rand = self.rng.f64();
        match self
            .cumu_weights
            .binary_search_by(|w| w.partial_cmp(&rand).expect("NaN weight"))
        {
            Ok(idx) => idx,
            Err(idx) => idx,
        }
    }

    pub fn sample(&mut self) -> (usize, usize) {
        let set_idx = self.outer_idx();
        let set = &self.sets[set_idx];
        let first = self.rng.usize(0..set.len());
        let mut second = first;
        while second == first {
            second = self.rng.usize(0..set.len());
        }
        (first, second)
    }

    /// If not enough unique pairs can be found, the Err response contains all pairs, but fewer than requested.
    pub fn sample_n(
        &mut self,
        n: usize,
    ) -> Result<HashSet<(usize, usize)>, HashSet<(usize, usize)>> {
        let mut out = HashSet::default();

        // If n is not much less than the total number of pairs,
        // easier to enumerate all pairs and shuffle from them.
        if n * 10 < self.n_pairs() {
            let mut count = 0;
            while out.len() < n {
                out.insert(self.sample());
                count += 1;
                if count >= n * 10 {
                    return Err(out);
                }
            }
        } else {
            let mut v = self.exhaust();
            self.rng.shuffle(&mut v);

            while out.len() < n {
                if let Some(s) = v.pop() {
                    out.insert(s);
                } else {
                    return Err(out);
                }
            }
        }
        Ok(out)
    }

    /// Upper bound on the number of pairs.
    /// If some pair sets intersect, there could be fewer unique pairs.
    pub fn n_pairs(&self) -> usize {
        self.sets
            .iter()
            .map(|v| {
                let len = v.len();
                if len < 2 {
                    0
                } else {
                    len * (len - 1)
                }
            })
            .sum()
    }

    pub fn n_sets(&self) -> usize {
        self.sets.len()
    }

    pub fn exhaust(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::default();

        for set in self.sets.iter() {
            for q_idx in set.iter() {
                for t_idx in set.iter() {
                    if t_idx != q_idx {
                        out.push((*q_idx, *t_idx));
                    }
                }
            }
        }

        out
    }
}

pub struct TrainingSampler {
    rng: fastrand::Rng,
    matching_sets: Vec<HashSet<usize>>,
    nonmatching_sets: Option<Vec<HashSet<usize>>>,
    n_neurons: usize,
}

impl TrainingSampler {
    pub fn new(n_neurons: usize, seed: Option<u64>) -> Self {
        let rng = fastrand::Rng::new();
        if let Some(s) = seed {
            rng.seed(s);
        }
        Self {
            rng,
            matching_sets: Vec::default(),
            nonmatching_sets: None,
            n_neurons,
        }
    }

    /// Add a set of neurons which are considered to be mutually matching,
    /// as indices into a vec of neurons of length `self.n_neurons`.
    /// This can be done several times with different sets.
    /// Matching sets should be small subsets of the total neuron count.
    ///
    /// Indices outside of the neuron vec will be silently filtered out.
    /// Filtered sets with length <2 will also be silently ignored.
    pub fn add_matching_set(&mut self, matching: &[usize]) -> &mut Self {
        // ensure all neurons exist
        let set: HashSet<usize> = matching
            .iter()
            .filter_map(|idx| {
                if idx < &self.n_neurons {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();
        if set.len() >= 2 {
            self.matching_sets.push(set);
        }
        self
    }

    /// Optionally list neuron sets which are mutually non-matching,
    /// as indices into a vec of neurons of length `self.n_neurons`,
    /// from which non-matching pairs will be randomly drawn.
    /// By default, all neurons are included,
    /// as matching sets are expected to be only a small portion of the total neurons.
    ///
    /// Candidate non-matching pairs are made sure not to be in the matching pairs,
    /// so this can safely be ignored.
    pub fn add_nonmatching_set(&mut self, nonmatching: &[usize]) -> &mut Self {
        let set: HashSet<usize> = nonmatching
            .iter()
            .filter_map(|idx| {
                if idx < &self.n_neurons {
                    Some(*idx)
                } else {
                    None
                }
            })
            .collect();

        if set.len() < 2 {
            return self;
        }

        if let Some(ref mut vector) = self.nonmatching_sets {
            vector.push(set);
        } else {
            let mut v = Vec::default();
            v.push(set);
            self.nonmatching_sets = Some(v);
        }
        self
    }

    /// Generate the "jobs" for matching and nonmatching neuron queries.
    ///
    /// If n_matching is `None`, uses all unique ordered matching pairs.
    /// If n_nonmatching is `None`, uses the same number as there are matching pairs.
    ///
    /// The first element of the tuple is the matching jobs; the second set is non-matching.
    pub fn make_jobs(
        &self,
        n_matching: Option<usize>,
        n_nonmatching: Option<usize>,
    ) -> (JobSet, JobSet) {
        let mut match_sampler =
            PairSampler::from_sets(&self.matching_sets, Some(self.rng.u64(0..u64::MAX)));

        let matching_jobs: JobSet = match n_matching {
            Some(n) => match match_sampler.sample_n(n) {
                Ok(jobs) => jobs,
                Err(jobs) => jobs,
            },
            None => match_sampler.exhaust().into_iter().collect(),
        };

        // if nonmatching not given, use all neurons
        let nonmatch_seed = Some(self.rng.u64(0..u64::MAX));
        let mut nonmatch_sampler = self.nonmatching_sets.as_ref().map_or_else(
            || {
                let v = vec![(0..self.n_neurons).collect()];
                PairSampler::from_sets(&v, nonmatch_seed)
            },
            |s| PairSampler::from_sets(&s, nonmatch_seed),
        );

        let n_nm = n_nonmatching.unwrap_or(matching_jobs.len());

        if n_nm > nonmatch_sampler.n_pairs() {
            panic!("Not enough non-matching neurons")
        }

        let nonmatching_jobs = nonmatch_sampler.sample_n(n_nm).unwrap_or_else(|s| s);

        (matching_jobs, nonmatching_jobs)
    }
}

fn all_distdots<T: TargetNeuron>(
    neurons: &[T],
    jobs: &[(usize, usize)],
    use_alpha: bool,
) -> Vec<DistDot> {
    jobs.iter()
        .map(|(q, t)| neurons[*q].query_dist_dots(&neurons[*t], use_alpha))
        .flatten()
        .collect()
}

/// Uses global thread pool by default. To use your own threadpool, use
///
/// ```rust
/// let pool = rayon::ThreadPoolBuilder::new()
///     .num_threads(n_threads)
///     .build()
///     .unwrap();
/// let results = pool.install(|| all_distdots_par(&neurons, &jobs, use_alpha));
/// ```
#[cfg(feature = "parallel")]
pub fn all_distdots_par<T: TargetNeuron + Sync>(
    neurons: &[T],
    jobs: &[(usize, usize)],
    use_alpha: bool,
) -> Vec<DistDot> {
    jobs.par_iter()
        .map(|(q, t)| neurons[*q].query_dist_dots(&neurons[*t], use_alpha))
        .flatten()
        .collect()
}

/// Find indices into a flattened array of `DistDot`s (as returned by `all_distdots` et al.)
/// which correspond to particular jobs.
/// i.e. element `i` of the output will contain a range which corresponds to the job at `jobs[i]`.
/// If the same neurons and jobs were given to `all_distdots`, that range could index the output
/// to find the distdots corresponding to that job.
pub fn job_idxs<T: Neuron>(neurons: &[T], jobs: &[(usize, usize)]) -> Vec<Range<usize>> {
    let mut start = 0;
    let mut out = Vec::with_capacity(jobs.len());
    for (q, _) in jobs.iter() {
        let len = neurons[*q].len();
        out.push(start..start + len);
        start += len;
    }
    out
}

enum LookupArgs {
    Lookup(BinLookup<Precision>),
    NBins(usize),
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
    nonmatching_sets: Option<Vec<HashSet<usize>>>,
    use_alpha: bool,
    threads: Option<usize>,
    dist_bin_lookup: Option<BinLookup<Precision>>,
    dot_bin_lookup: Option<BinLookup<Precision>>,
}

impl<T: TargetNeuron + Sync> ScoreMatrixBuilder<T> {
    /// `seed` is used when randomly selecting non-matching neurons to use as controls.
    pub fn new(neurons: Vec<T>, seed: u64) -> Self {
        Self {
            neurons,
            seed,
            matching_sets: Vec::default(),
            nonmatching_sets: None,
            use_alpha: false,
            threads: Some(0),
            dist_bin_lookup: None,
            dot_bin_lookup: None,
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

    /// Optionally list neuron sets which are mutually non-matching,
    /// as indices into `ScoreMatrixBuilder.neurons`,
    /// from which non-matching pairs will be randomly drawn.
    /// By default, all neurons are included,
    /// as matching sets are expected to be only a small portion of the total neurons.
    ///
    /// Candidate non-matching pairs are made sure not to be in the matching pairs,
    /// so this can safely be ignored.
    pub fn add_nonmatching_set(&mut self, nonmatching: HashSet<usize>) -> &mut Self {
        if let Some(ref mut vector) = self.nonmatching_sets {
            vector.push(nonmatching);
        } else {
            let mut v = Vec::default();
            v.push(nonmatching);
            self.nonmatching_sets = Some(v);
        }
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

    /// Manually set the bounds of the distance bins.
    /// The first and last bounds are effectively ignored,
    /// as values outside that range are snapped to the bottom and top bins.
    pub fn set_dist_bins(&mut self, bins: Vec<Precision>) -> &mut Self {
        self.dist_bin_lookup = Some(BinLookup::new(bins, (true, true)).expect("Illegal bins"));
        self
    }

    /// Automatically generate distance bins by logarithmically interpolating
    /// between `base^min_exp` (which should be small) and `base^max_exp`.
    pub fn set_n_dist_bins(
        &mut self,
        n_bins: usize,
        base: Precision,
        min_exp: Precision,
        max_exp: Precision,
    ) -> &mut Self {
        let mut v = logspace(base, min_exp, max_exp, n_bins);
        v.push(1.0 / 0.0);
        self.set_dist_bins(v)
    }

    /// Manually set the upper bounds of the abs dot product bins.
    /// The first and last bounds are effectively ignored, as values outside those values
    /// snap into their closest bins.
    ///
    /// Because the tangent vectors are unit-length,
    /// absolute dot products between them are between 0 and 1
    /// (although float precision means values slightly above 1 are possible).
    pub fn set_dot_bins(&mut self, dot_bin_boundaries: Vec<Precision>) -> &mut Self {
        self.dot_bin_lookup =
            Some(BinLookup::new(dot_bin_boundaries, (true, true)).expect("Illegal bins"));
        self
    }

    /// Automatically generate abs dot product bins by linearly interpolating between
    /// 0 and 1.
    pub fn set_n_dot_bins(&mut self, n_bins: usize) -> &mut Self {
        let step = 1.0 / (n_bins + 1) as Precision;
        self.set_dot_bins((0..(n_bins + 1)).map(|n| step * n as Precision).collect())
    }

    /// Return a tuple of `(dist_bins, dot_bins, cells)` in the forms accepted by
    /// `table_to_fn`.
    pub fn build(&self) -> Result<RangeTable<Precision, Precision>, ScoreMatBuildErr> {
        if self.matching_sets.is_empty() {
            return Err(ScoreMatBuildErr {});
        }
        let dist_bin_lookup = match &self.dist_bin_lookup {
            Some(lookup) => lookup.clone(),
            None => return Err(ScoreMatBuildErr {}),
        };
        let dot_bin_lookup = match &self.dot_bin_lookup {
            Some(lookup) => lookup.clone(),
            None => return Err(ScoreMatBuildErr {}),
        };

        let dist_dot_lookup = NdBinLookup::new(vec![dist_bin_lookup, dot_bin_lookup]);

        let (match_jobs, nonmatch_jobs) = self._match_nonmatch_jobs();
        let matching_factor = nonmatch_jobs.len() as Precision / match_jobs.len() as Precision;

        let match_counts = self._pairs_to_counts(match_jobs, &dist_dot_lookup);
        let nonmatch_counts = self._pairs_to_counts(nonmatch_jobs, &dist_dot_lookup);

        let cells = match_counts
            .into_iter()
            .zip(nonmatch_counts.into_iter())
            .map(|(match_count, nonmatch_count)| {
                // add epsilon to prevent division by 0
                ((match_count as Precision * matching_factor + EPSILON)
                    / (nonmatch_count as Precision + EPSILON))
                    .log2()
            })
            .collect();

        Ok(RangeTable {
            bins_lookup: dist_dot_lookup,
            cells,
        })
    }

    /// Generate the "jobs" for matching and nonmatching neuron queries.
    ///
    /// Every non-repeating 2-length permutation of neurons within each matching set is used.
    /// Then, pairs of neurons are taken randomly from the non-matching set
    /// (if `None`, use all neurons) until there are at least as many non-matching DistDots
    /// as there are matching.
    fn _match_nonmatch_jobs(&self) -> (JobSet, JobSet) {
        let rng = fastrand::Rng::with_seed(self.seed);

        let match_sampler = PairSampler::from_sets(&self.matching_sets, Some(rng.u64(0..u64::MAX)));
        let matching_jobs: JobSet = match_sampler.exhaust().into_iter().collect();

        // if nonmatching not given, use all neurons
        let nonmatch_seed = Some(rng.u64(0..u64::MAX));
        let mut nonmatch_sampler = self.nonmatching_sets.as_ref().map_or_else(
            || {
                let v = vec![(0..self.neurons.len()).collect()];
                PairSampler::from_sets(&v, nonmatch_seed)
            },
            |s| PairSampler::from_sets(&s, nonmatch_seed),
        );

        if matching_jobs.len() > nonmatch_sampler.n_pairs() {
            panic!("Not enough non-matching neurons")
        }

        let nonmatching_jobs = nonmatch_sampler
            .sample_n(matching_jobs.len())
            .unwrap_or_else(|s| s);

        (matching_jobs, nonmatching_jobs)
    }

    /// If both indexes are in the neuron slice, get the DistDots between them.
    fn _idx_to_distdots(&self, q_idx: usize, t_idx: usize) -> Option<Vec<DistDot>> {
        let q = self.neurons.get(q_idx)?;
        let t = self.neurons.get(t_idx)?;
        Some(q.query_dist_dots(t, self.use_alpha))
    }

    /// For all given `(query, target)` index pairs, find DistDots and flatten
    /// into a single array.
    fn _pairs_to_counts_ser(
        &self,
        jobs: impl IntoIterator<Item = (usize, usize)>,
        dist_dot_lookup: &NdBinLookup<Precision>,
    ) -> Vec<usize> {
        let mut counts: Vec<usize> = iter::repeat(0).take(dist_dot_lookup.n_cells).collect();

        let distdots = jobs
            .into_iter()
            .filter_map(|(q_idx, t_idx)| self._idx_to_distdots(q_idx, t_idx))
            .flatten();

        for dd in distdots {
            let idx = dist_dot_lookup.to_linear_idx(&[dd.dist, dd.dot]).unwrap();
            counts[idx] += 1;
        }
        counts
    }

    #[cfg(not(feature = "parallel"))]
    fn _pairs_to_counts(
        &self,
        jobs: HashSet<(usize, usize)>,
        dist_dot_lookup: &NdBinLookup<Precision>,
    ) -> Vec<usize> {
        self._pairs_to_counts_ser(jobs, dist_dot_lookup)
    }

    #[cfg(feature = "parallel")]
    fn _pairs_to_counts(
        &self,
        jobs: HashSet<(usize, usize)>,
        dist_dot_lookup: &NdBinLookup<Precision>,
    ) -> Vec<usize> {
        if let Some(t) = self.threads {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();

            pool.install(|| {
                let (sender, receiver) = channel();

                jobs.into_par_iter()
                    .filter_map(|(q_idx, t_idx)| self._idx_to_distdots(q_idx, t_idx))
                    .flatten()
                    .map(|dd| dist_dot_lookup.to_linear_idx(&[dd.dist, dd.dot]).unwrap())
                    .for_each_with(sender, |s, x| s.send(x).unwrap());

                let mut counts: Vec<usize> =
                    iter::repeat(0).take(dist_dot_lookup.n_cells).collect();

                for idx in receiver.iter() {
                    counts[idx] += 1;
                }
                counts
            })
        } else {
            self._pairs_to_counts_ser(jobs, dist_dot_lookup)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_slice_eq(test: &[Precision], reference: &[Precision]) {
        let msg = format!("\ttest: {:?}\n\t ref: {:?}", test, reference);
        if test.len() != reference.len() {
            panic!("Slices have different length\n{}", msg);
        }

        for (test_val, ref_val) in test.iter().zip(reference.iter()) {
            if (test_val - ref_val).abs() > Precision::EPSILON {
                panic!("Slices mave mismatched values\n{}", msg)
            }
        }
    }

    #[test]
    fn test_logspace() {
        let base: Precision = 10.0;
        let count: usize = 5;
        assert_slice_eq(
            &logspace(base, 1.0, 5.0, count),
            &[
                (10f64).powf(1.0),
                (10f64).powf(2.0),
                (10f64).powf(3.0),
                (10f64).powf(4.0),
                (10f64).powf(5.0),
            ],
        );
    }
}
