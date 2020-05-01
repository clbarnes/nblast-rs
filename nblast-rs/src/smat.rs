use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand_pcg::Pcg32;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::error;
use std::fmt;
use std::iter;

use crate::{DistDot, Precision, TargetNeuron};

const EPSILON: Precision = 1e-6;

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

fn match_nonmatch_distdots<T: TargetNeuron + Sync>(
    neurons: &[T],
    matching_sets: &[HashSet<usize>],
    non_matching_set: Option<Vec<usize>>,
    use_alpha: bool,
    seed: u64,
    threads: Option<usize>,
) -> (Vec<DistDot>, Vec<DistDot>) {
    let mut matching_jobs: HashMap<(usize, usize), usize> = HashMap::default();
    for matching_set in matching_sets {
        for q_idx in matching_set.iter() {
            let q_len = match neurons.get(*q_idx) {
                Some(n) => n.len(),
                None => continue,
            };
            for t_idx in matching_set.iter() {
                if t_idx == q_idx || t_idx >= &neurons.len() {
                    continue;
                }
                matching_jobs.insert((*q_idx, *t_idx), q_len);
            }
        }
    }

    let neurons_comb = neurons.len() * (neurons.len() - 1);
    if matching_jobs.len() > neurons_comb {
        panic!("Not enough non-matching neurons")
    }

    let matching_len = matching_jobs.iter().fold(0, |total, (_, v)| total + v);

    let mut rng = Pcg32::seed_from_u64(seed);
    let mut nonmatching_remaining = matching_len;
    let mut nonmatching_jobs: HashSet<(usize, usize)> = HashSet::default();

    let nonmatching_idxs = non_matching_set
        .or_else(|| Some((0..neurons.len()).collect()))
        .unwrap();
    let range = Uniform::new(0, nonmatching_idxs.len());

    while nonmatching_remaining > 0 {
        let q_idx = nonmatching_idxs[range.sample(&mut rng)];
        let t_idx = nonmatching_idxs[range.sample(&mut rng)];

        let key = (q_idx, t_idx);
        if q_idx != t_idx || nonmatching_jobs.insert(key) {
            nonmatching_remaining -= neurons[q_idx].len()
        }
    }

    let matching_dd = pairs_to_distdots(
        &neurons,
        matching_jobs.drain().map(|(k, _)| k).collect(),
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

fn calculate_cells(
    dist_thresholds: &[Precision],
    dot_thresholds: &[Precision],
    matching_distdots: Vec<DistDot>,
    nonmatching_distdots: Vec<DistDot>,
) -> Vec<Precision> {
    let n_cells = dist_thresholds.len() * dot_thresholds.len();
    let mut matching_count: Vec<usize> = iter::repeat(0).take(n_cells).collect();

    let mut nonmatching_count = matching_count.clone();

    // There may be more nonmatching distdots than matching ones.
    // Therefore, the nonmatching count in each cell should be scaled up.
    let matching_factor =
        nonmatching_distdots.len() as Precision / matching_distdots.len() as Precision;

    for dd in matching_distdots {
        matching_count[dd.to_linear_idx(dist_thresholds, dot_thresholds)] += 1;
    }
    for dd in nonmatching_distdots {
        nonmatching_count[dd.to_linear_idx(dist_thresholds, dot_thresholds)] += 1;
    }

    matching_count
        .into_iter()
        .zip(nonmatching_count.into_iter())
        .map(|(m, nm)| {
            ((m as Precision * matching_factor + EPSILON) / (nm as Precision + EPSILON)).log2()
        })
        .collect()
}

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

#[derive(Debug)]
pub struct MissingBinsErr {}

impl fmt::Display for MissingBinsErr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bins not set")
    }
}

impl error::Error for MissingBinsErr {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}

impl<T: TargetNeuron + Sync> ScoreMatrixBuilder<T> {
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

    pub fn add_matching_set(&mut self, matching: HashSet<usize>) -> &mut Self {
        self.matching_sets.push(matching);
        self
    }

    pub fn set_nonmatching(&mut self, nonmatching: Vec<usize>) -> &mut Self {
        self.nonmatching = Some(nonmatching);
        self
    }

    pub fn set_use_alpha(&mut self, use_alpha: bool) -> &mut Self {
        self.use_alpha = use_alpha;
        self
    }

    pub fn set_threads(&mut self, threads: Option<usize>) -> &mut Self {
        self.threads = threads;
        self
    }

    pub fn set_dist_bins<'a>(&'a mut self, dist_bins_upper: Vec<Precision>) -> &mut Self {
        self.dist_bins_upper = Some(dist_bins_upper);
        self
    }

    pub fn set_n_dist_bins(
        &mut self,
        n_dist_bins: usize,
        greatest_lower_bound: Precision,
        base: Precision,
    ) -> &Self {
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

    pub fn set_dot_bins<'a>(&'a mut self, dot_bins_upper: Vec<Precision>) -> &'a Self {
        self.dot_bins_upper = Some(dot_bins_upper);
        self
    }

    pub fn set_n_dot_bins(&mut self, n_dot_bins: usize) -> &Self {
        let step = 1.0 / n_dot_bins as Precision;
        self.set_dot_bins(
            (1..(n_dot_bins + 1))
                .map(|n| step * n as Precision)
                .collect(),
        )
    }

    pub fn build(self) -> Result<(Vec<Precision>, Vec<Precision>, Vec<Precision>), MissingBinsErr> {
        let dist_bins = self.dist_bins_upper.ok_or(MissingBinsErr {})?;
        let dot_bins = self.dot_bins_upper.ok_or(MissingBinsErr {})?;

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
