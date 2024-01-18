use std::fs::File;
use std::path::PathBuf;

use csv::ReaderBuilder;

pub type Precision = f64;
pub type Point3 = [Precision; 3];

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
