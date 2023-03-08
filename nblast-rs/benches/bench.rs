use std::fs::File;
use std::path::PathBuf;

use bencher::{benchmark_group, benchmark_main, Bencher};
use csv::ReaderBuilder;

use nblast::{
    NblastArena, Neuron, Point3, Precision, QueryNeuron, RStarTangentsAlphas, RangeTable,
    ScoreCalc, TangentAlpha,
};

#[cfg(feature = "nabo")]
use nblast::NaboTangentsAlphas;

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
    let query =
        RStarTangentsAlphas::new(&read_points(NAMES[0]), N_NEIGHBORS).expect("couldn't parse");
    let target =
        RStarTangentsAlphas::new(&read_points(NAMES[1]), N_NEIGHBORS).expect("couldn't parse");

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_construction_rstar(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| RStarTangentsAlphas::new(&points, N_NEIGHBORS).expect("couldn't parse"));
}

fn bench_construction_nabo(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    b.iter(|| NaboTangentsAlphas::new(points.clone(), N_NEIGHBORS))
}

fn bench_query_nabo(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let query = NaboTangentsAlphas::new(read_points(NAMES[0]), N_NEIGHBORS);
    let target = NaboTangentsAlphas::new(read_points(NAMES[1]), N_NEIGHBORS);

    b.iter(|| query.query(&target, false, &score_fn))
}

fn bench_construction_with_tangents_rstar(b: &mut Bencher) {
    let points = read_points(NAMES[0]);
    let neuron = RStarTangentsAlphas::new(&points, N_NEIGHBORS).expect("couldn't parse");
    let tangents_alphas: Vec<_> = neuron
        .tangents()
        .iter()
        .zip(neuron.alphas().iter())
        .map(|(t, a)| TangentAlpha {
            tangent: *t,
            alpha: *a,
        })
        .collect();
    b.iter(|| RStarTangentsAlphas::new_with_tangents_alphas(&points, tangents_alphas.clone()));
}

fn bench_arena_construction(b: &mut Bencher) {
    let score_fn = get_score_fn();
    let pointtangents: Vec<_> = NAMES
        .iter()
        .map(|n| RStarTangentsAlphas::new(&read_points(n), N_NEIGHBORS).expect("couldn't parse"))
        .collect();
    b.iter(|| {
        let mut arena = NblastArena::new(score_fn.clone());
        for dp in pointtangents.iter().cloned() {
            arena.add_neuron(dp);
        }
    })
}

fn bench_arena_query(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let p0 = read_points(NAMES[0]);
    let idx0 =
        arena.add_neuron(RStarTangentsAlphas::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 =
        arena.add_neuron(RStarTangentsAlphas::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, false, &None, false));
}

fn bench_arena_query_norm(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let p0 = read_points(NAMES[0]);
    let idx0 =
        arena.add_neuron(RStarTangentsAlphas::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 =
        arena.add_neuron(RStarTangentsAlphas::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| arena.query_target(idx0, idx1, true, &None, false));
}

fn bench_arena_query_geom(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let p0 = read_points(NAMES[0]);
    let idx0 =
        arena.add_neuron(RStarTangentsAlphas::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 =
        arena.add_neuron(RStarTangentsAlphas::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| {
        arena.query_target(
            idx0,
            idx1,
            false,
            &Some(nblast::Symmetry::GeometricMean),
            false,
        )
    });
}

fn bench_arena_query_norm_geom(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let p0 = read_points(NAMES[0]);
    let idx0 =
        arena.add_neuron(RStarTangentsAlphas::new(&p0, N_NEIGHBORS).expect("couldn't parse"));
    let p1 = read_points(NAMES[1]);
    let idx1 =
        arena.add_neuron(RStarTangentsAlphas::new(&p1, N_NEIGHBORS).expect("couldn't parse"));

    b.iter(|| {
        arena.query_target(
            idx0,
            idx1,
            true,
            &Some(nblast::Symmetry::GeometricMean),
            false,
        )
    });
}

fn bench_all_to_all_serial_rstar(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(
            arena.add_neuron(
                RStarTangentsAlphas::new(&points, N_NEIGHBORS).expect("couldn't parse"),
            ),
        );
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, false, None, None));
}

fn bench_all_to_all_serial_nabo(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(arena.add_neuron(NaboTangentsAlphas::new(points, N_NEIGHBORS)));
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, false, None, None));
}

#[cfg(feature = "parallel")]
fn bench_all_to_all_parallel(b: &mut Bencher) {
    let mut arena = NblastArena::new(get_score_fn());
    let mut idxs = Vec::new();
    for name in NAMES.iter() {
        let points = read_points(name);
        idxs.push(
            arena.add_neuron(
                RStarTangentsAlphas::new(&points, N_NEIGHBORS).expect("couldn't parse"),
            ),
        );
    }

    b.iter(|| arena.queries_targets(&idxs, &idxs, false, &None, false, Some(0), None));
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

benchmark_group!(
    arena,
    bench_arena_query,
    bench_arena_query_norm,
    bench_arena_query_geom,
    bench_arena_query_norm_geom,
    bench_all_to_all_parallel,
    bench_arena_construction
);

benchmark_main!(impl_rstar, impl_nabo, arena);
