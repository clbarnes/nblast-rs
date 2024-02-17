use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::{Point3, Precision, SpatialArena};

#[derive(Default)]
pub struct KiddoArena {
    trees: Vec<ImmutableKdTree<Precision, 3>>,
}

impl SpatialArena for KiddoArena {
    fn add_points(&mut self, p: Vec<Point3>) -> usize {
        let idx = self.len();
        self.trees.push(p.as_slice().into());
        idx
    }

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)> {
        let tgt = self.trees.get(t).unwrap();
        self.trees
            .get(q)
            .unwrap()
            .iter()
            .map(|(_, p)| {
                let nn = tgt.nearest_one::<SquaredEuclidean>(&p);
                (nn.item as usize, nn.distance.sqrt())
            })
            .collect()
    }

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>> {
        let tgt = self.trees.get(q).unwrap();
        self.trees
            .get(q)
            .unwrap()
            .iter()
            .map(|(_, p)| {
                tgt.nearest_n::<SquaredEuclidean>(&p, neighborhood)
                    .into_iter()
                    .map(|nn| nn.item as usize)
                    .collect()
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.trees.len()
    }
}
