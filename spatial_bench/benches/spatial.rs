use std::iter::repeat_with;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fastrand::Rng;
use spatial_bench::{
    bosque::BosqueArena, kiddo::KiddoArena, nabo::NaboArena, read_augmented, rstar::RstarArena,
    Point3, SpatialArena,
};

const N_NEURONS: usize = 1000;

fn make_arena<S: SpatialArena>(pts: Vec<Vec<Point3>>) -> S {
    let mut ar = S::default();
    for p in pts.into_iter() {
        ar.add_points(p);
    }
    ar
}

fn random_pairs(max: usize, n: usize, rng: &mut Rng) -> Vec<(usize, usize)> {
    repeat_with(|| (rng.usize(0..max), rng.usize(0..max)))
        .take(n)
        .collect()
}

fn pair_queries<S: SpatialArena>(arena: &S, pairs: &[(usize, usize)]) {
    for (q, t) in pairs {
        arena.query_target(*q, *t);
    }
}

fn read_augmented_fixed() -> Vec<Vec<Point3>> {
    let mut rng = fastrand::Rng::with_seed(1991);
    read_augmented(N_NEURONS, &mut rng, 20.0, 1.0)
}

pub fn bench_construction(c: &mut Criterion) {
    let points = read_augmented_fixed();

    let mut group = c.benchmark_group("construction");
    group.bench_function("bosque", |b| {
        b.iter(|| black_box(make_arena::<BosqueArena>(points.clone())))
    });
    group.bench_function("kiddo", |b| {
        b.iter(|| black_box(make_arena::<KiddoArena>(points.clone())))
    });
    group.bench_function("nabo", |b| {
        b.iter(|| black_box(make_arena::<NaboArena>(points.clone())))
    });
    group.bench_function("rstar", |b| {
        b.iter(|| black_box(make_arena::<RstarArena>(points.clone())))
    });
}

pub fn bench_queries(c: &mut Criterion) {
    let points = read_augmented_fixed();
    let n_pairs = 1_000;
    let mut rng = fastrand::Rng::with_seed(1991);
    let pairs = random_pairs(points.len(), n_pairs, &mut rng);
    let mut group = c.benchmark_group("pairwise query");

    let ar = make_arena::<BosqueArena>(points.clone());
    group.bench_function("bosque", |b| {
        b.iter(|| {
            pair_queries(&ar, &pairs);
            black_box(())
        })
    });

    let ar = make_arena::<KiddoArena>(points.clone());
    group.bench_function("kiddo", |b| b.iter(|| {
        pair_queries(&ar, &pairs);
        black_box(())
    }));

    let ar = make_arena::<NaboArena>(points.clone());
    group.bench_function("nabo", |b| b.iter(|| {
        pair_queries(&ar, &pairs);
        black_box(())
    }));

    let ar = make_arena::<RstarArena>(points.clone());
    group.bench_function("rstar", |b| b.iter(|| {
        pair_queries(&ar, &pairs);
        black_box(())
    }));
}

criterion_group!(benches, bench_construction, bench_queries);
criterion_main!(benches);
