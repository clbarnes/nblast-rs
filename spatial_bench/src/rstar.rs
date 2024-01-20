use rstar::{primitives::GeomWithData, PointDistance, RTree};

use crate::{Point3, Precision, SpatialArena};

pub type RsPoint = GeomWithData<Point3, usize>;

#[derive(Debug, Default)]
pub struct RstarArena {
    trees: Vec<rstar::RTree<RsPoint>>,
}

impl SpatialArena for RstarArena {
    fn add_points(&mut self, p: Vec<Point3>) -> usize {
        let t = RTree::bulk_load(
            p.into_iter()
                .enumerate()
                .map(|(idx, p)| RsPoint::new(p, idx))
                .collect(),
        );
        let idx = self.len();
        self.trees.push(t);
        idx
    }

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)> {
        let t_tree = self.trees.get(t).unwrap();

        self.trees
            .get(q)
            .unwrap()
            .iter()
            .map(|p| {
                let neighb = t_tree.nearest_neighbor(p.geom()).unwrap();
                let dist = p.distance_2(neighb.geom()).sqrt();
                (p.data, dist)
            })
            .collect()
    }

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>> {
        let tree = self.trees.get(q).unwrap();
        tree.iter()
            .map(|p| {
                tree.nearest_neighbor_iter(p.geom())
                    .take(neighborhood)
                    .map(|n| n.data)
                    .collect()
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.trees.len()
    }
}
