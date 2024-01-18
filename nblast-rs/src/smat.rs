//! The crux of the NBLAST algorithm is converting `(distance, abs_dot)` point matches
//! into a meaningful score.
//!
//! The strategy here generates a lookup table whose indices are dist and dot bins.
//! A neuron match score is the sum of the values of the lookup table over every point match.
//!
//! The values in the table cells are the log2 odds ratio of a dist-dot in that cell
//! being from a matching over a non-matching neuron pair.
//! This is calculated by adding a pool of neurons, subsets of which should and should not be matches.
//! Calculate all the point matches between pairs of neurons in the matching sets,
//! and the nonmatching sets.
//! Normalise the counts falling into each cell by the total number of matching
//! and non-matching dist-dots, and calculate the log2 odds ratios.
//!
//! This is handled by the ScoreMatrixBuilder.
use fastrand::Rng;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::HashSet;
use std::ops::BitXor;
use thiserror::Error;

use crate::{BinLookup, NdBinLookup, RangeTable};
use crate::{DistDot, Precision, TargetNeuron};

const EPSILON: Precision = 1e-6;

type JobSet = HashSet<(usize, usize)>;

#[derive(Debug, Error)]
pub enum ScoreMatBuildErr {
    #[error("No matching sets given")]
    MatchingSets,
    #[error("No distance bin information given")]
    DistBins,
    #[error("No dot product bin information given")]
    DotBins,
}

struct PairSampler {
    sets: Vec<Vec<usize>>,
    cumu_weights: Vec<f64>,
    pub rng: Rng,
}

impl PairSampler {
    fn new(sets: Vec<Vec<usize>>, seed: Option<u64>) -> Self {
        let rng = Rng::new();
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
    pub fn sample_n(&mut self, n: usize) -> Result<JobSet, JobSet> {
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

fn make_rng(seed: Option<u64>) -> Rng {
    let rng = Rng::new();
    if let Some(s) = seed {
        rng.seed(s);
    }
    rng
}

pub struct TrainingSampler {
    seed: Option<u64>,
    matching_sets: Vec<HashSet<usize>>,
    nonmatching_sets: Option<Vec<HashSet<usize>>>,
    n_neurons: usize,
}

impl TrainingSampler {
    pub fn new(n_neurons: usize, seed: Option<u64>) -> Self {
        Self {
            seed,
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
    #[allow(clippy::vec_init_then_push)]
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
    /// Which jobs are selected is deterministic with the seed and number of jobs requested, but the order is not.
    pub fn make_jobs(
        &self,
        n_matching: Option<usize>,
        n_nonmatching: Option<usize>,
    ) -> (JobSet, JobSet) {
        let seed = self.seed.map(|s1| {
            let mut s = s1;
            if let Some(s2) = n_matching {
                s = s.bitxor(s2 as u64);
            }
            if let Some(s3) = n_nonmatching {
                s = s.bitxor(s3 as u64);
            }
            s
        });
        let rng = make_rng(seed);

        let mut match_sampler =
            PairSampler::from_sets(&self.matching_sets, Some(rng.u64(0..u64::MAX)));

        let matching_jobs: JobSet = match n_matching {
            Some(n) => match match_sampler.sample_n(n) {
                Ok(jobs) => jobs,
                Err(jobs) => jobs,
            },
            None => match_sampler.exhaust().into_iter().collect(),
        };

        // if nonmatching not given, use all neurons
        let nonmatch_seed = Some(rng.u64(0..u64::MAX));
        let mut nonmatch_sampler = self.nonmatching_sets.as_ref().map_or_else(
            || {
                let v = vec![(0..self.n_neurons).collect()];
                PairSampler::from_sets(&v, nonmatch_seed)
            },
            |s| PairSampler::from_sets(s, nonmatch_seed),
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
        .flat_map(|(q, t)| neurons[*q].query_dist_dots(&neurons[*t], use_alpha))
        .collect()
}

/// Uses global thread pool by default. To use your own threadpool, use
///
/// ```ignore
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

#[derive(Clone, Debug)]
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
#[allow(dead_code)]
pub struct ScoreMatrixBuilder<T: TargetNeuron> {
    // Data set of neuron point clouds
    neurons: Vec<T>,
    sampler: TrainingSampler,
    use_alpha: bool,
    // shows as dead code without parallel feature
    threads: Option<usize>,
    dist_bin_lookup: Option<LookupArgs>,
    dot_bin_lookup: Option<LookupArgs>,
    max_matching_pairs: Option<usize>,
    max_nonmatching_pairs: Option<usize>,
}

impl<T: TargetNeuron + Sync> ScoreMatrixBuilder<T> {
    /// `seed` is used when randomly selecting non-matching neurons to use as controls.
    pub fn new(neurons: Vec<T>, seed: u64) -> Self {
        let n_neurons = neurons.len();
        Self {
            neurons,
            sampler: TrainingSampler::new(n_neurons, Some(seed)),
            use_alpha: false,
            threads: None,
            dist_bin_lookup: None,
            dot_bin_lookup: None,
            max_matching_pairs: None,
            max_nonmatching_pairs: None,
        }
    }

    /// Add a set of neurons which are considered to be mutually matching,
    /// as indices into the `ScoreMatrixBuilder.neurons`.
    /// This can be done several times with different sets.
    /// Matching sets should be small subsets of the total neuron count.
    ///
    /// Indices outside of the neuron vec will be silently filtered out.
    /// Filtered sets with length <2 will also be silently ignored.
    pub fn add_matching_set(&mut self, matching: &[usize]) -> &mut Self {
        self.sampler.add_matching_set(matching);
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
    pub fn add_nonmatching_set(&mut self, nonmatching: &[usize]) -> &mut Self {
        self.sampler.add_nonmatching_set(nonmatching);
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
    /// `None` means the calculation is done in serial (default);
    /// for `Some(n)`, `n = 0` means use all available CPUs (default),
    /// and any other value uses that many cores.
    #[cfg(feature = "parallel")]
    pub fn set_threads(&mut self, threads: Option<usize>) -> &mut Self {
        self.threads = threads;
        self
    }

    /// Manually set the lookup bins for distances.
    ///
    /// Consider `BinLookup::new_exp` here.
    pub fn set_dist_lookup(&mut self, lookup: BinLookup<Precision>) -> &mut Self {
        self.dist_bin_lookup = Some(LookupArgs::Lookup(lookup));
        self
    }

    /// Automatically generate distance bins by by using distances
    /// from matching sets.
    pub fn set_n_dist_bins(&mut self, n_bins: usize) -> &mut Self {
        self.dist_bin_lookup = Some(LookupArgs::NBins(n_bins));
        self
    }

    /// Manually set the lookup bins for dot products.
    ///
    /// Consider `BinLookup::new_linear(0.0, 1.0, n_bins, (true, true))` here.
    pub fn set_dot_lookup(&mut self, lookup: BinLookup<Precision>) -> &mut Self {
        self.dot_bin_lookup = Some(LookupArgs::Lookup(lookup));
        self
    }

    /// Automatically generate abs dot product bins by using dot products
    /// from matching sets.
    pub fn set_n_dot_bins(&mut self, n_bins: usize) -> &mut Self {
        self.dot_bin_lookup = Some(LookupArgs::NBins(n_bins));
        self
    }

    /// Draw at most this many matching pairs from the given matching sets.
    /// May be fewer if there aren't that many combinations.
    /// By default, uses all possible pairs.
    pub fn set_max_matching_pairs(&mut self, n_pairs: usize) -> &mut Self {
        self.max_matching_pairs = Some(n_pairs);
        self
    }

    /// Draw at most this many nonmatching pairs from the given nonmatching sets.
    /// May be fewer if there aren't that many combinations.
    /// By default, uses as many pairs as for matching.
    pub fn set_max_nonmatching_pairs(&mut self, n_pairs: usize) -> &mut Self {
        self.max_nonmatching_pairs = Some(n_pairs);
        self
    }

    /// Unwrap lookup args (`BinLookup` or number of bins) with informative error if not given.
    fn _get_lookup_args(&self) -> Result<(LookupArgs, LookupArgs), ScoreMatBuildErr> {
        let dist_lookup_args = match &self.dist_bin_lookup {
            Some(lookup) => lookup.clone(),
            None => return Err(ScoreMatBuildErr::DistBins),
        };
        let dot_lookup_args = match &self.dot_bin_lookup {
            Some(lookup) => lookup.clone(),
            None => return Err(ScoreMatBuildErr::DotBins),
        };
        Ok((dist_lookup_args, dot_lookup_args))
    }

    /// Unwrap or calculate `BinLookup`s.
    fn _get_lookup(
        &self,
        match_distdots: &[DistDot],
    ) -> Result<NdBinLookup<Precision>, ScoreMatBuildErr> {
        let (dist_lookup_args, dot_lookup_args) = self._get_lookup_args()?;

        let dist_bin_lookup = match dist_lookup_args {
            LookupArgs::Lookup(lookup) => Ok(lookup),
            LookupArgs::NBins(n) => {
                let mut dists: Vec<_> = match_distdots.iter().map(|dd| dd.dist).collect();
                dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                BinLookup::new_n_quantiles(&dists, n, (true, true))
                    .map_err(|_| ScoreMatBuildErr::DistBins)
            }
        }?;
        let dot_bin_lookup = match dot_lookup_args {
            LookupArgs::Lookup(lookup) => Ok(lookup),
            LookupArgs::NBins(n) => {
                let mut dots: Vec<_> = match_distdots.iter().map(|dd| dd.dot).collect();
                dots.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                BinLookup::new_n_quantiles(&dots, n, (true, true))
                    .map_err(|_| ScoreMatBuildErr::DotBins)
            }
        }?;

        Ok(NdBinLookup::new(vec![dist_bin_lookup, dot_bin_lookup]))
    }

    /// Return a tuple of `(dist_bins, dot_bins, cells)` in the forms accepted by
    /// `table_to_fn`.
    pub fn build(&self) -> Result<RangeTable<Precision, Precision>, ScoreMatBuildErr> {
        if self.sampler.matching_sets.is_empty() {
            return Err(ScoreMatBuildErr::MatchingSets);
        }

        let (match_jobs, nonmatch_jobs) = self
            .sampler
            .make_jobs(self.max_matching_pairs, self.max_nonmatching_pairs);

        let match_distdots = self._all_distdots(&match_jobs.into_iter().collect::<Vec<_>>());

        let dist_dot_lookup = self._get_lookup(&match_distdots)?;

        let match_counts = cell_counts(&dist_dot_lookup, match_distdots);

        let nonmatch_distdots = self._all_distdots(&nonmatch_jobs.into_iter().collect::<Vec<_>>());

        let nonmatch_counts = cell_counts(&dist_dot_lookup, nonmatch_distdots);

        let cells = log_odds_ratio(match_counts, nonmatch_counts);

        Ok(RangeTable {
            bins_lookup: dist_dot_lookup,
            cells,
        })
    }

    /// Calculate all distdots for given index pairs.
    #[cfg(not(feature = "parallel"))]
    fn _all_distdots(&self, jobs: &[(usize, usize)]) -> Vec<DistDot> {
        all_distdots(&self.neurons, jobs, self.use_alpha)
    }

    /// Calculate all distdots for given index pairs, possibly in parallel.
    #[cfg(feature = "parallel")]
    fn _all_distdots(&self, jobs: &[(usize, usize)]) -> Vec<DistDot> {
        if let Some(t) = self.threads {
            // todo: avoid building this pool twice
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| all_distdots_par(&self.neurons, jobs, self.use_alpha))
        } else {
            all_distdots(&self.neurons, jobs, self.use_alpha)
        }
    }
}

/// Find the counts of distdots which fall into each cell in an NdBinLookup (useful for building a `RangeTable`).
fn cell_counts(lookup: &NdBinLookup<Precision>, distdots: Vec<DistDot>) -> Vec<Precision> {
    let mut counts = vec![0.0; lookup.n_cells];
    for dd in distdots {
        if let Ok(idx) = lookup.to_linear_idx(&[dd.dist, dd.dot]) {
            counts[idx] += 1.0;
        }
    }
    counts
}

/// Only need this with floats, even though counts should be usize.
fn log_odds_ratio(match_counts: Vec<Precision>, nonmatch_counts: Vec<Precision>) -> Vec<Precision> {
    let match_total: Precision = match_counts.iter().sum();
    let nonmatch_total: Precision = nonmatch_counts.iter().sum();

    match_counts
        .into_iter()
        .zip(nonmatch_counts.into_iter())
        .map(|(match_count, nonmatch_count)| {
            let p_match = match_count / match_total;
            let p_nonmatch = nonmatch_count / nonmatch_total;
            // N.B. these brackets are not shown in the original paper,
            // but that is probably a formatting error
            ((p_match + EPSILON) / (p_nonmatch + EPSILON)).log2()
        })
        .collect()
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
}
