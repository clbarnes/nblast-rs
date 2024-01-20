use crate::{Point3, Precision, SpatialArena};

#[derive(Default)]
pub struct BosqueArena {
    trees: Vec<Vec<Point3>>,
    idxs: Vec<Vec<usize>>,
}

impl SpatialArena for BosqueArena {
    fn add_points(&mut self, mut p: Vec<Point3>) -> usize {
        let mut idxs: Vec<_> = (0..(p.len() as u32)).collect();
        bosque::tree::build_tree_with_indices(p.as_mut(), idxs.as_mut());
        let idx = self.len();
        self.trees.push(p);
        self.idxs
            .push(idxs.into_iter().map(|i| i as usize).collect());
        idx
    }

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)> {
        let tgt = self.trees.get(t).unwrap();
        let tgt_idxs = self.idxs.get(t).unwrap();
        self.trees
            .get(q)
            .unwrap()
            .iter()
            .map(|p| {
                let (d, idx) = bosque::tree::nearest_one(tgt.as_slice(), p);
                (tgt_idxs[idx], d)
            })
            .collect()
    }

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>> {
        let t = self.trees.get(q).unwrap();
        let idxs = self.idxs.get(q).unwrap();
        t.iter()
            .map(|p| {
                bosque::tree::nearest_k(t.as_slice(), p, neighborhood)
                    .into_iter()
                    .map(|(_d, i)| idxs[i])
                    .collect()
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.trees.len()
    }
}
