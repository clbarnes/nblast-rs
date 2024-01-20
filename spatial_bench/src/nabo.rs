use crate::{Point3, Precision, SpatialArena};
use nabo::{KDTree, NotNan, Point};

#[derive(Clone, Copy, Debug, Default)]
struct NaboPointWithIndex {
    pub point: [NotNan<Precision>; 3],
    pub index: usize,
}

impl NaboPointWithIndex {
    pub fn new(point: &Point3, index: usize) -> Self {
        Self {
            point: [
                NotNan::new(point[0]).unwrap(),
                NotNan::new(point[1]).unwrap(),
                NotNan::new(point[2]).unwrap(),
            ],
            index,
        }
    }
}

impl From<NaboPointWithIndex> for Point3 {
    fn from(p: NaboPointWithIndex) -> Self {
        [*p.point[0], *p.point[1], *p.point[2]]
    }
}

impl Point<Precision> for NaboPointWithIndex {
    fn get(&self, i: u32) -> nabo::NotNan<Precision> {
        self.point[i as usize]
    }

    fn set(&mut self, i: u32, value: nabo::NotNan<Precision>) {
        self.point[i as usize] = value;
    }

    const DIM: u32 = 3;
}

#[derive(Default)]
pub struct NaboArena {
    trees: Vec<KDTree<Precision, NaboPointWithIndex>>,
    points: Vec<Vec<NaboPointWithIndex>>,
}

impl SpatialArena for NaboArena {
    fn add_points(&mut self, p: Vec<Point3>) -> usize {
        let all_p: Vec<_> = p
            .iter()
            .enumerate()
            .map(|(idx, p)| NaboPointWithIndex::new(p, idx))
            .collect();
        let t = KDTree::new(&all_p);
        let idx = self.len();
        self.trees.push(t);
        self.points.push(all_p);
        idx
    }

    fn query_target(&self, q: usize, t: usize) -> Vec<(usize, Precision)> {
        let t_tree = self.trees.get(t).unwrap();
        self.points
            .get(q)
            .unwrap()
            .iter()
            .map(|p| {
                let n = t_tree.knn(1, p).pop().unwrap();
                (n.point.index, n.dist2.sqrt())
            })
            .collect()
    }

    fn local_query(&self, q: usize, neighborhood: usize) -> Vec<Vec<usize>> {
        let t = self.trees.get(q).unwrap();
        self.points
            .get(q)
            .unwrap()
            .iter()
            .map(|p| {
                t.knn(neighborhood as u32, p)
                    .into_iter()
                    .map(|n| n.point.index)
                    .collect()
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.trees.len()
    }
}
