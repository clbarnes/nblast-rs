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
use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand_pcg::Pcg32;
use rayon::prelude::*;
use std::collections::HashSet;
use std::error;
use std::fmt;
use std::iter;
use std::sync::mpsc::channel;

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
            nonmatching: None,
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
        let mut matching_len = 0;
        let mut matching_jobs = JobSet::default();

        // for every matching set
        for matching_set in self.matching_sets.iter() {
            // for every possible query idx
            for q_idx in matching_set.iter() {
                // if the idx is in the given neurons
                let q_len = match self.neurons.get(*q_idx) {
                    Some(n) => n.len(),
                    None => continue,
                };
                // for every possible target idx
                for t_idx in matching_set.iter() {
                    // if t!=q, t exists, and the pair is not already addressed
                    if t_idx != q_idx
                        && t_idx < &self.neurons.len()
                        && matching_jobs.insert((*q_idx, *t_idx))
                    {
                        // keep track of how many distdots we're producing
                        matching_len += q_len
                    }
                }
            }
        }

        // if nonmatching not given, use all neurons
        let nonmatching_idxs = self
            .nonmatching
            .as_ref()
            .cloned() // ? is this avoidable
            .or_else(|| Some((0..self.neurons.len()).collect()))
            .unwrap();

        if matching_jobs.len() > nonmatching_idxs.len() * (nonmatching_idxs.len() - 1) {
            panic!("Not enough non-matching neurons")
        }

        let range = Uniform::new(0, nonmatching_idxs.len());

        let mut rng = Pcg32::seed_from_u64(self.seed);
        let mut nonmatching_jobs: HashSet<(usize, usize)> = HashSet::default();

        // randomly pick nonmatching pairs until we have requested as many distdots
        // as we did for matching
        while matching_len > 0 {
            let q_idx = nonmatching_idxs[range.sample(&mut rng)];
            let t_idx = nonmatching_idxs[range.sample(&mut rng)];

            let key = (q_idx, t_idx);
            if q_idx != t_idx && !matching_jobs.contains(&key) && nonmatching_jobs.insert(key) {
                matching_len -= self.neurons[q_idx].len()
            }
        }

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
