use fastrand::Rng;
use std::fs::File;
use std::path::PathBuf;

pub mod bosque;
pub mod kiddo;
pub mod nabo;
pub mod rstar;

use csv::ReaderBuilder;

pub type Precision = f64;
pub type Point3 = [Precision; 3];
pub const DIM: usize = 3;

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

pub trait SpatialArena: Default {
    fn add_points(&mut self, p: Vec<Point3>) -> usize;

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)>;

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn all_locals(&self, neighborhood: usize) -> Vec<Vec<Vec<usize>>> {
        (0..self.len())
            .map(|idx| self.local_query(idx, neighborhood))
            .collect()
    }

    fn all_v_all(&self) -> Vec<Vec<Vec<(usize, Precision)>>> {
        let len = self.len();
        (0..len)
            .map(|q| (0..len).map(move |t| self.query_target(q, t)).collect())
            .collect()
    }
}

#[derive(Default)]
pub struct ArenaWrapper<S: SpatialArena>(S);

impl<S: SpatialArena> SpatialArena for ArenaWrapper<S> {
    fn add_points(&mut self, p: Vec<Point3>) -> usize {
        self.0.add_points(p)
    }

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)> {
        self.0.query_target(q, t)
    }

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>> {
        self.0.local_query(q, neighborhood)
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

struct PointAug<'a> {
    rng: &'a mut Rng,
    translation: Point3,
    jitter_stdev: Precision,
}

fn box_muller_sin(u1: Precision, u2: Precision) -> Precision {
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin()
}

fn random_translation(rng: &mut Rng, stdev: Precision) -> Point3 {
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
    pub fn new(translation: Point3, jitter_stdev: Precision, rng: &'a mut Rng) -> Self {
        Self {
            rng,
            translation,
            jitter_stdev,
        }
    }

    pub fn new_random(
        translation_stdev: Precision,
        jitter_stdev: Precision,
        rng: &'a mut Rng,
    ) -> Self {
        Self::new(
            random_translation(rng, translation_stdev),
            jitter_stdev,
            rng,
        )
    }

    pub fn augment(&mut self, orig: &Point3) -> Point3 {
        let t2 = random_translation(self.rng, self.jitter_stdev);
        [
            orig[0] + self.translation[0] + t2[0],
            orig[1] + self.translation[1] + t2[1],
            orig[2] + self.translation[2] + t2[2],
        ]
    }

    pub fn augment_all(&mut self, orig: &[Point3]) -> Vec<Point3> {
        orig.iter().map(|p| self.augment(p)).collect()
    }
}

pub fn read_augmented(
    n: usize,
    rng: &mut Rng,
    translation_stdev: Precision,
    jitter_stdev: Precision,
) -> Vec<Vec<Point3>> {
    let orig = read_all_points();
    let mut aug = PointAug::new_random(translation_stdev, jitter_stdev, rng);
    orig.iter()
        .cycle()
        .take(n)
        .map(|p| aug.augment_all(p))
        .collect()
}
