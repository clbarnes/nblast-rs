use std::fs::File;
use std::iter::repeat_with;
use std::path::PathBuf;

use bencher::{benchmark_group, benchmark_main, Bencher};
use csv::ReaderBuilder;
use fastrand::Rng;

#[cfg(feature = "bosque")]
use nblast::neurons::bosque::BosqueNeuron;
#[cfg(feature = "kiddo")]
use nblast::neurons::kiddo::{ExactKiddoNeuron, KiddoNeuron};
#[cfg(feature = "nabo")]
use nblast::neurons::nabo::NaboNeuron;
use nblast::neurons::rstar::RstarNeuron;
use nblast::{
    BinLookup, NblastArena, NblastNeuron, Point3, Precision, QueryNeuron, RangeTable, ScoreCalc,
    ScoreMatrixBuilder, TangentAlpha,
};

const NAMES: [&str; 20] = [
    "ChaMARCM-F000586_seg002",
    "FruMARCM-F000085_seg001",
    "FruMARCM-F000188_seg001",
    "FruMARCM-F000270_seg001",
    "FruMARCM-F000706_seg001",
    "FruMARCM-F001115_seg002",
    "FruMARCM-F001494_seg002",
    "FruMARCM-F001929_seg001",
    "FruMARCM-M000115_seg001",
    "FruMARCM-M000842_seg002",
    "FruMARCM-M001051_seg002",
    "FruMARCM-M001205_seg002",
    "FruMARCM-M001339_seg001",
    "GadMARCM-F000050_seg001",
    "GadMARCM-F000071_seg001",
    "GadMARCM-F000122_seg001",
    "GadMARCM-F000142_seg002",
    "GadMARCM-F000423_seg001",
    "GadMARCM-F000442_seg002",
    "GadMARCM-F000476_seg001",
];

const N_NEIGHBORS: usize = 5;

fn data_dir() -> PathBuf {
    // crate dir
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .expect("couldn't resolve");

    // workspace dir
    let has_parent = d.pop();

    if !has_parent {
        panic!("Couldn't find parent directory");
    }

    d.push("data");
    d
}

fn to_path(name: &str) -> PathBuf {
    let mut d = data_dir();
    d.push("points");
    d.push(format!("{}.csv", name));
    d
}

type Record = (usize, f64, f64, f64);

fn read_points(name: &str) -> Vec<Point3> {
    let fpath = to_path(name);
    let f =
        File::open(fpath.clone()).unwrap_or_else(|_| panic!("couldn't find file at {:?}", fpath));
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(f);
    let mut out = Vec::default();

    for result in reader.deserialize() {
        let record: Record = result.expect("Could not deserialise");
        out.push([record.1, record.2, record.3]);
    }

    out
}

fn read_all_points() -> Vec<Vec<Point3>> {
    NAMES.iter().map(|n| read_points(n)).collect()
}

pub type MatchNonmatch = (Vec<Vec<Point3>>, Vec<Vec<usize>>, Vec<Vec<usize>>);

fn match_nonmatch(repeats: usize) -> MatchNonmatch {
    let n_csvs = NAMES.len();
    let mut out_points = Vec::with_capacity(n_csvs * repeats);
    let all_points = read_all_points();

    let mut matches: Vec<Vec<usize>> = repeat_with(|| Vec::with_capacity(repeats))
        .take(n_csvs)
        .collect();
    let mut nonmatches: Vec<Vec<usize>> = repeat_with(|| Vec::with_capacity(n_csvs))
        .take(repeats)
        .collect();

    let rng = Rng::with_seed(1991);

    for (rep, nm) in nonmatches.iter_mut().enumerate() {
        let aug = PointAug::new_random(
            0.5,  // 500nm
            0.02, // 20nm
            &rng,
        );
        let offset = rep * repeats;

        for (p_idx, ps) in all_points.iter().enumerate() {
            let global_p_idx = offset + p_idx;
            out_points.push(aug.augment_all(ps));
            matches[p_idx].push(global_p_idx);
            nm.push(global_p_idx);
        }
    }

    (out_points, matches, nonmatches)
}

struct PointAug<'a> {
    rng: &'a Rng,
    translation: Point3,
    jitter_stdev: Precision,
}

fn box_muller_sin(u1: Precision, u2: Precision) -> Precision {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin()
}

fn random_translation(rng: &Rng, stdev: Precision) -> Point3 {
    let u1 = rng.f64();
    let u2 = rng.f64();
    let u3 = rng.f64();
    [
        box_muller_sin(u1, u2) * stdev,
        box_muller_sin(u2, u3) * stdev,
        box_muller_sin(u1, u3) * stdev,
    ]
}

impl<'a> PointAug<'a> {
    fn new(translation: Point3, jitter_stdev: Precision, rng: &'a Rng) -> Self {
        Self {
            rng,
            translation,
            jitter_stdev,
        }
    }

    fn new_random(translation_stdev: Precision, jitter_stdev: Precision, rng: &'a Rng) -> Self {
        Self::new(
            random_translation(rng, translation_stdev),
            jitter_stdev,
            rng,
        )
    }

    fn augment(&self, orig: &Point3) -> Point3 {
        let t2 = random_translation(self.rng, self.jitter_stdev);
        [
            orig[0] + self.translation[0] + t2[0],
            orig[1] + self.translation[1] + t2[1],
            orig[2] + self.translation[2] + t2[2],
        ]
    }

    fn augment_all(&self, orig: &[Point3]) -> Vec<Point3> {
        orig.iter().map(|p| self.augment(p)).collect()
    }
}

fn parse_interval(s: &str) -> (f64, f64) {
    let substr = &s[1..s.len() - 1];
    let split_idx = substr.find(',').expect("no comma");
    let l_r = substr.split_at(split_idx);
    let r = l_r.1.split_at(1);
    (
        l_r.0
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("lower bound not float: '{}'", l_r.0)),
        r.1.parse::<f64>()
            .unwrap_or_else(|_| panic!("upper bound not float: '{}'", r.1)),
    )
}

/// dist thresholds, dot thresholds, cells
fn read_smat() -> (Vec<Precision>, Vec<Precision>, Vec<Precision>) {
    let mut d = data_dir();
    d.push("smat_fcwb.csv");

    let f = File::open(d).expect("file not found");
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(f);

    let mut dot_thresholds = Vec::new();

    let mut results = reader.records();
    let first_row = results.next().expect("no first row").expect("bad parse");

    // drop first (empty) column
    let mut first_row_iter = first_row.iter();
    first_row_iter.next();
    let mut is_first = true;
    for dot_interval_str in first_row_iter {
        let dot_interval = parse_interval(dot_interval_str);
        if is_first {
            dot_thresholds.push(dot_interval.0);
            is_first = false;
        }
        dot_thresholds.push(dot_interval.1);
    }

    let mut dist_thresholds = Vec::new();
    let mut cells = Vec::new();
    is_first = true;
    for result in reader.records() {
        let record = result.expect("failed record");
        let mut record_iter = record.iter();
        let dist_interval_str = record_iter.next().expect("No first item");
        let dist_interval = parse_interval(dist_interval_str);
        if is_first {
            dist_thresholds.push(dist_interval.0);
            is_first = false;
        }
        dist_thresholds.push(dist_interval.1);

        for cell in record_iter {
            cells.push(cell.parse::<f64>().expect("cell not float"));
        }
    }
    (dist_thresholds, dot_thresholds, cells)
}

fn get_score_fn() -> ScoreCalc {
    let args = read_smat();
    // table_to_fn(args.0, args.1, args.2)
    // ScoreCalc::Table(RangeTable::new(args.0, args.1, args.2))
    let rtable =
        RangeTable::new_from_bins(vec![args.0, args.1], args.2).expect("Invalid score table");
    ScoreCalc::Table(rtable)
}

fn bench_query_rstar(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = RstarNeuron::new(&read_points(NAMES[0]), N_NEIGHBORS).expect("couldn't parse");
    let target = RstarNeuron::new(&read_points(NAMES[1]), N_NEIGHBORS).expect("couldn't parse");

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_construction_rstar(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| RstarNeuron::new(&points, N_NEIGHBORS).expect("couldn't parse"));
}

fn bench_construction_nabo(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| NaboNeuron::new(points.clone(), N_NEIGHBORS))
}

fn bench_construction_kiddo(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| KiddoNeuron::new(points.clone(), N_NEIGHBORS))
}

fn bench_construction_exact_kiddo(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| ExactKiddoNeuron::new(points.clone(), N_NEIGHBORS))
}

fn bench_construction_bosque(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| BosqueNeuron::new(points.clone(), N_NEIGHBORS))
}

fn bench_query_nabo(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = NaboNeuron::new(read_points(NAMES[0]), N_NEIGHBORS);
    let target = NaboNeuron::new(read_points(NAMES[1]), N_NEIGHBORS);

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_query_kiddo(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = KiddoNeuron::new(read_points(NAMES[0]), N_NEIGHBORS).unwrap();
    let target = KiddoNeuron::new(read_points(NAMES[1]), N_NEIGHBORS).unwrap();

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_query_exact_kiddo(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = ExactKiddoNeuron::new(read_points(NAMES[0]), N_NEIGHBORS).unwrap();
    let target = ExactKiddoNeuron::new(read_points(NAMES[1]), N_NEIGHBORS).unwrap();

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_query_bosque(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = BosqueNeuron::new(read_points(NAMES[0]), N_NEIGHBORS).unwrap();
    let target = BosqueNeuron::new(read_points(NAMES[1]), N_NEIGHBORS).unwrap();

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_construction_with_tangents_rstar(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    let neuron = RstarNeuron::new(&points, N_NEIGHBORS).expect("couldn't parse");
    let tangents_alphas: Vec<_> = neuron
        .tangents()
        .zip(neuron.alphas())
        .map(|(t, a)| TangentAlpha {
            tangent: t,
            alpha: a,
        })
        .collect();
    b.iter(|| RstarNeuron::new_with_tangents_alphas(&points, tangents_alphas.clone()));
}

fn bench_arena_construction(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let pointtangents: Vec<_> = NAMES
        .iter()
        .map(|n| RstarNeuron::new(&read_points(n), N_NEIGHBORS).expect("couldn't parse"))
        .collect();
    b.iter(|| {
        let mut arena = NblastArena::new(score_fn.clone(), false);
        for dp in pointtangents.iter().cloned() {
            arena.add_neuron(dp);
        }
    })
}

fn bench_arena_query(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let p0 = read_points(NAMES[0]);
    let idx0 = arena.add_neuron(RstarNeuron::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 = arena.add_neuron(RstarNeuron::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, false, &None));
}

fn bench_arena_query_norm(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let p0 = read_points(NAMES[0]);
    let idx0 = arena.add_neuron(RstarNeuron::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 = arena.add_neuron(RstarNeuron::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, true, &None));
}

fn bench_arena_query_geom(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let p0 = read_points(NAMES[0]);
    let idx0 = arena.add_neuron(RstarNeuron::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 = arena.add_neuron(RstarNeuron::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, false, &Some(nblast::Symmetry::GeometricMean)));
}

fn bench_arena_query_norm_geom(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let p0 = read_points(NAMES[0]);
    let idx0 = arena.add_neuron(RstarNeuron::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 = arena.add_neuron(RstarNeuron::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, true, &Some(nblast::Symmetry::GeometricMean)));
}

fn bench_all_to_all_serial_rstar(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(
            arena.add_neuron(RstarNeuron::new(&points, N_NEIGHBORS).expect("couldn't parse")),
        );
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

fn bench_all_to_all_serial_nabo(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(arena.add_neuron(NaboNeuron::new(points, N_NEIGHBORS)));
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

fn bench_all_to_all_serial_kiddo(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(arena.add_neuron(KiddoNeuron::new(points, N_NEIGHBORS).unwrap()));
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

fn bench_all_to_all_serial_exact_kiddo(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(arena.add_neuron(ExactKiddoNeuron::new(points, N_NEIGHBORS).unwrap()));
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

fn bench_all_to_all_serial_bosque(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(arena.add_neuron(BosqueNeuron::new(points, N_NEIGHBORS).unwrap()));
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

#[cfg(feature = "parallel")]
fn bench_all_to_all_parallel(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn(), false).with_threads(0);
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(
            arena.add_neuron(RstarNeuron::new(&points, N_NEIGHBORS).expect("couldn't parse")),
        );
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, None));
}

fn make_smatb_kiddo() -> ScoreMatrixBuilder<ExactKiddoNeuron> {
    let (all_points, matches, nonmatches) = match_nonmatch(10);
    let neurons: Vec<_> = all_points
        .into_iter()
        .map(|ps| ExactKiddoNeuron::new(ps, N_NEIGHBORS).unwrap())
        .collect();

    let mut smatb = ScoreMatrixBuilder::new(neurons.clone(), 1991);
    for m in matches.iter() {
        smatb.add_matching_set(m);
    }
    for nm in nonmatches.iter() {
        smatb.add_nonmatching_set(nm);
    }
    smatb
}

fn bench_smatbuild_ser(b: &mut Bencher) {
    let mut smatb = make_smatb_kiddo();
    let (dist, dot, _cells) = read_smat();
    smatb.set_dot_lookup(BinLookup::new(dot, (true, true)).unwrap());
    smatb.set_dist_lookup(BinLookup::new(dist, (true, true)).unwrap());

    b.iter(|| smatb.build())
}

fn bench_smatbuild_par(b: &mut Bencher) {
    let mut smatb = make_smatb_kiddo();
    let (dist, dot, _cells) = read_smat();

    smatb.set_dot_lookup(BinLookup::new(dot, (true, true)).unwrap());
    smatb.set_dist_lookup(BinLookup::new(dist, (true, true)).unwrap());

    smatb.set_threads(Some(0));

    b.iter(|| smatb.build())
}

fn bench_smatbuild_quantiles(b: &mut Bencher) {
    let mut smatb = make_smatb_kiddo();
    let (dist, dot, _cells) = read_smat();

    smatb.set_n_dot_bins(dot.len() - 1);
    smatb.set_n_dist_bins(dist.len() - 1);

    b.iter(|| smatb.build())
}

benchmark_group!(
    impl_rstar,
    bench_construction_rstar,
    bench_construction_with_tangents_rstar,
    bench_query_rstar,
    bench_all_to_all_serial_rstar,
);

#[cfg(feature = "nabo")]
benchmark_group!(
    impl_nabo,
    bench_construction_nabo,
    bench_query_nabo,
    bench_all_to_all_serial_nabo,
);

#[cfg(feature = "kiddo")]
benchmark_group!(
    impl_kiddo,
    bench_construction_kiddo,
    bench_query_kiddo,
    bench_all_to_all_serial_kiddo,
);

#[cfg(feature = "bosque")]
benchmark_group!(
    impl_bosque,
    bench_construction_bosque,
    bench_query_bosque,
    bench_all_to_all_serial_bosque,
);

#[cfg(feature = "kiddo")]
benchmark_group!(
    impl_exact_kiddo,
    bench_construction_exact_kiddo,
    bench_query_exact_kiddo,
    bench_all_to_all_serial_exact_kiddo,
);

benchmark_group!(
    arena,
    bench_arena_query,
    bench_arena_query_norm,
    bench_arena_query_geom,
    bench_arena_query_norm_geom,
    bench_all_to_all_parallel,
    bench_arena_construction,
);

#[cfg(feature = "kiddo")]
benchmark_group!(
    smat,
    bench_smatbuild_ser,
    bench_smatbuild_par,
    bench_smatbuild_quantiles,
);

benchmark_main!(
    impl_rstar,
    impl_nabo,
    impl_exact_kiddo,
    impl_bosque,
    arena,
    smat
);
