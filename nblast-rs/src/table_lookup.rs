//! Utilities relating to converting point match values (distance and dot product of tangents) into a score using a lookup table.
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;

use thiserror::Error;

use crate::Precision;

#[derive(Debug, PartialEq, Eq, Error)]
pub enum OutOfBin {
    #[error("Value comes before the first bin boundary")]
    Before,
    #[error("Value comes after the last bin ({0:?})")]
    After(usize),
}

#[derive(Debug, PartialEq, Eq)]
pub struct OutOfBins {
    outside: HashMap<usize, OutOfBin>,
}

impl fmt::Display for OutOfBins {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} value(s) falls outside of bins", self.outside.len())
    }
}

impl std::error::Error for OutOfBins {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum BinLookupBuildError {
    #[error("Bin boundary order is not ascending")]
    NotAscending,
    #[error("Bin boundaries must have >=2 values, got {0}")]
    NotEnough(usize),
}

fn is_ascending<T: PartialOrd>(values: &[T], strict: bool) -> bool {
    values
        .windows(2)
        .map(|w| w[1].partial_cmp(&w[0]))
        .all(|cmp| {
            if strict {
                cmp.map_or(false, |c| c.is_gt())
            } else {
                cmp.map_or(false, |c| c.is_ge())
            }
        })
}

#[derive(Debug, Clone)]
pub struct BinLookup<T: PartialOrd + Clone + Debug> {
    pub bin_boundaries: Vec<T>,
    pub snap: (bool, bool),
    pub n_bins: usize,
}

impl<T: PartialOrd + Copy + Debug> BinLookup<T> {
    /// `bin_boundaries` must be sorted ascending and have length > 2.
    /// `snap` is a tuple of whether to clamp values beyond the left and the right bin boundary respectively.
    pub fn new(bin_boundaries: Vec<T>, snap: (bool, bool)) -> Result<Self, BinLookupBuildError> {
        if bin_boundaries.len() < 2 {
            return Err(BinLookupBuildError::NotEnough(bin_boundaries.len()));
        }
        if !is_ascending(&bin_boundaries, true) {
            return Err(BinLookupBuildError::NotAscending);
        }
        let n_bins = bin_boundaries.len() - 1;

        Ok(Self {
            bin_boundaries,
            snap,
            n_bins,
        })
    }

    /// If snap = (true, true), result will always be `Ok`
    pub fn to_idx(&self, val: &T) -> Result<usize, OutOfBin> {
        // this implementation could be much simpler
        // if it will only ever need to support ascending sorts,
        // which is already all it supports anyway
        match self
            .bin_boundaries
            .binary_search_by(|bound| bound.partial_cmp(val).expect("Could not compare"))
        {
            Ok(idx) => {
                // exactly on boundary
                if idx >= self.n_bins {
                    if self.snap.1 {
                        Ok(self.n_bins - 1)
                    } else {
                        Err(OutOfBin::After(self.n_bins - 1))
                    }
                } else {
                    Ok(idx)
                }
            }
            Err(idx) => {
                if idx == 0 {
                    if self.snap.0 {
                        Ok(0)
                    } else {
                        Err(OutOfBin::Before)
                    }
                } else if idx > self.n_bins {
                    if self.snap.1 {
                        Ok(self.n_bins - 1)
                    } else {
                        Err(OutOfBin::After(self.n_bins - 1))
                    }
                } else {
                    Ok(idx - 1)
                }
            }
        }
    }
}

impl BinLookup<Precision> {
    /// Create a BinLookup whose boundaries are linearly spaced between `min` and `max`.
    /// `snap` is as used in `BinLookup::new`.
    pub fn new_linear(
        min: Precision,
        max: Precision,
        n_bins: usize,
        snap: (bool, bool),
    ) -> Result<Self, BinLookupBuildError> {
        let n_bounds = n_bins + 1;
        let step = (max - min) / n_bounds as f64;
        Self::new(
            (0..n_bounds).map(|n| min + step * n as Precision).collect(),
            snap,
        )
    }

    /// Create a BinLookup whose boundaries are logarithmically spaced
    /// between `base^min_exp` (which should be small) and `base^max_exp`.
    /// `snap` is as in `BinLookup::new`.
    pub fn new_exp(
        base: Precision,
        min_exp: Precision,
        max_exp: Precision,
        n_bins: usize,
        snap: (bool, bool),
    ) -> Result<Self, BinLookupBuildError> {
        let n_bounds = n_bins + 1;
        let step = (max_exp - min_exp) / n_bins as Precision;

        let bounds = (0..n_bounds)
            .map(|idx| base.powf(min_exp + idx as Precision * step))
            .collect();
        Self::new(bounds, snap)
    }

    /// Create a BinLookup whose boundaries are the quantiles of some data,
    /// which must be sorted in ascending order (this is checked).
    /// `quantiles` must be within `[0, 1]` and sorted ascending (also checked).
    /// `snap` is as in `BinLookup::new`.
    pub fn new_quantiles(
        sorted_data: &[Precision],
        quantiles: &[Precision],
        snap: (bool, bool),
    ) -> Result<BinLookup<Precision>, BinLookupBuildError> {
        if !is_ascending(quantiles, true) {
            return Err(BinLookupBuildError::NotAscending);
        }
        // todo: much more efficient with an approximate streaming implementation
        if !is_ascending(sorted_data, false) {
            return Err(BinLookupBuildError::NotAscending);
        }
        let len = sorted_data.len() as f64 - 1.0;
        let bounds = quantiles
            .iter()
            .map(|q| {
                let frac_idx = len * q;
                let fl = frac_idx.floor();
                let min = sorted_data[fl as usize];
                if fl == frac_idx {
                    return min;
                }
                let ce = frac_idx.ceil();
                let max = sorted_data[ce as usize];
                interp(min, max, frac_idx - fl)
            })
            .collect();
        Self::new(bounds, snap)
    }

    /// Create a BinLookup whose boundaries are the quantiles of some (sorted ascending) data,
    /// evenly spaced within `[0, 1]`.
    /// `snap` is as in `BinLookup::new`.
    pub fn new_n_quantiles(
        data: &[Precision],
        n_bins: usize,
        snap: (bool, bool),
    ) -> Result<BinLookup<Precision>, BinLookupBuildError> {
        let n_bounds = n_bins + 1;
        let step = 1.0 / n_bounds as f64;
        let quantiles: Vec<_> = (0..n_bounds).map(|n| step * n as Precision).collect();
        Self::new_quantiles(data, &quantiles, snap)
    }
}

fn interp(min: Precision, max: Precision, portion: Precision) -> Precision {
    (max - min) * portion + min
}

#[derive(Debug, Clone)]
pub struct NdBinLookup<T: PartialOrd + Clone + Debug> {
    pub lookups: Vec<BinLookup<T>>,
    idx_mult: Vec<usize>,
    pub n_cells: usize,
}

impl<T: PartialOrd + Copy + Debug> NdBinLookup<T> {
    pub fn new(lookups: Vec<BinLookup<T>>) -> Self {
        let n_cells = lookups.iter().map(|lu| lu.n_bins).product();
        let mut remaining_cells = n_cells;

        let mut idx_mult = Vec::with_capacity(lookups.len());
        for lookup in lookups.iter() {
            remaining_cells /= lookup.n_bins;
            idx_mult.push(remaining_cells);
        }

        Self {
            lookups,
            idx_mult,
            n_cells,
        }
    }

    pub fn to_idxs(&self, vals: &[T]) -> Result<Vec<usize>, OutOfBins> {
        assert_eq!(vals.len(), self.lookups.len());
        let mut idxs = Vec::with_capacity(vals.len());
        let mut outside = HashMap::with_capacity(vals.len());

        for (dim_idx, (val, lookup)) in vals.iter().zip(self.lookups.iter()).enumerate() {
            match lookup.to_idx(val) {
                Ok(idx) => idxs.push(idx),
                Err(oob) => {
                    outside.insert(dim_idx, oob);
                }
            };
        }

        if outside.is_empty() {
            Ok(idxs)
        } else {
            Err(OutOfBins { outside })
        }
    }

    pub fn to_linear_idx(&self, vals: &[T]) -> Result<usize, OutOfBins> {
        self.to_idxs(vals).map(|idxs| {
            idxs.iter()
                .zip(self.idx_mult.iter())
                .fold(0, |sum, (idx, mult)| sum + idx * mult)
        })
    }
}

#[derive(Debug, Clone)]
pub struct RangeTable<I: PartialOrd + Clone + Debug, T> {
    pub bins_lookup: NdBinLookup<I>,
    pub cells: Vec<T>,
}

#[derive(Error, Debug)]
pub enum InvalidRangeTable {
    #[error("Illegal bin boundaries")]
    IllegalBinBoundaries(#[from] BinLookupBuildError),
    #[error("Mismatched cell count: expected {expected:?} cells and got {got:?}")]
    MismatchedCellCount { expected: usize, got: usize },
}

impl<I: PartialOrd + Copy + Debug, T> RangeTable<I, T> {
    pub fn new(bins_lookup: NdBinLookup<I>, cells: Vec<T>) -> Result<Self, InvalidRangeTable> {
        if bins_lookup.n_cells == cells.len() {
            Ok(Self { bins_lookup, cells })
        } else {
            Err(InvalidRangeTable::MismatchedCellCount {
                expected: bins_lookup.n_cells,
                got: cells.len(),
            })
        }
    }

    pub fn new_from_bins(bins: Vec<Vec<I>>, cells: Vec<T>) -> Result<Self, InvalidRangeTable> {
        let mut lookups = Vec::with_capacity(bins.len());
        for b in bins {
            let lookup = BinLookup::new(b, (true, true))?;
            lookups.push(lookup);
        }

        Self::new(NdBinLookup::new(lookups), cells)
    }

    pub fn lookup(&self, vals: &[I]) -> &T {
        let idx = self.bins_lookup.to_linear_idx(vals).expect("Out of bounds");
        self.cells.get(idx).expect("Index out of bounds")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type Precision = f64;

    #[test]
    fn inner_lookup() {
        let bins = BinLookup::new(vec![1.0, 2.0, 3.0], (false, false)).expect("invalid bins");
        assert_eq!(bins.to_idx(&1.0), Ok(0));
        assert_eq!(bins.to_idx(&1.5), Ok(0));
        assert_eq!(bins.to_idx(&2.0), Ok(1));
        assert_eq!(bins.to_idx(&2.5), Ok(1));
    }

    #[test]
    fn outside_err() {
        let bins = BinLookup::new(vec![1.0, 2.0, 3.0], (false, false)).expect("invalid bins");
        assert_eq!(bins.to_idx(&0.9), Err(OutOfBin::Before));
        assert_eq!(bins.to_idx(&3.0), Err(OutOfBin::After(1)));
        assert_eq!(bins.to_idx(&3.1), Err(OutOfBin::After(1)));
    }

    #[test]
    fn below_snaps() {
        let bins = BinLookup::new(vec![1.0, 2.0, 3.0], (true, false)).expect("invalid bins");
        assert_eq!(bins.to_idx(&0.9), Ok(0));
        assert_eq!(bins.to_idx(&1.0), Ok(0));
        assert_eq!(bins.to_idx(&1.5), Ok(0));
        assert_eq!(bins.to_idx(&2.0), Ok(1));
        assert_eq!(bins.to_idx(&2.5), Ok(1));
        assert_eq!(bins.to_idx(&3.1), Err(OutOfBin::After(1)));
        assert_eq!(bins.to_idx(&3.1), Err(OutOfBin::After(1)));
    }

    #[test]
    fn above_snaps() {
        let bins = BinLookup::new(vec![1.0, 2.0, 3.0], (false, true)).expect("invalid bins");
        assert_eq!(bins.to_idx(&0.9), Err(OutOfBin::Before));
        assert_eq!(bins.to_idx(&1.0), Ok(0));
        assert_eq!(bins.to_idx(&1.5), Ok(0));
        assert_eq!(bins.to_idx(&2.0), Ok(1));
        assert_eq!(bins.to_idx(&2.5), Ok(1));
        assert_eq!(bins.to_idx(&3.1), Ok(1));
        assert_eq!(bins.to_idx(&3.1), Ok(1));
    }

    #[test]
    fn both_snaps() {
        let bins = BinLookup::new(vec![1.0, 2.0, 3.0], (true, true)).expect("invalid bins");
        assert_eq!(bins.to_idx(&0.9), Ok(0));
        assert_eq!(bins.to_idx(&1.0), Ok(0));
        assert_eq!(bins.to_idx(&1.5), Ok(0));
        assert_eq!(bins.to_idx(&2.0), Ok(1));
        assert_eq!(bins.to_idx(&2.5), Ok(1));
        assert_eq!(bins.to_idx(&3.1), Ok(1));
        assert_eq!(bins.to_idx(&3.1), Ok(1));
    }

    #[test]
    fn non_monotonic() {
        assert!(BinLookup::new(vec![1.0, 3.0, 2.0], (true, true)).is_err());
    }

    #[test]
    fn insufficient_bounds() {
        assert!(BinLookup::new(vec![1.0], (true, true)).is_err());
    }

    // #[test]
    // fn rev_sort() {
    //     let bins = BinLookup::new(vec![3.0, 2.0, 1.0], (true, true)).unwrap();
    //     assert_eq!(bins.to_idx(&3.1), Ok(0));
    //     assert_eq!(bins.to_idx(&3.0), Ok(0));
    //     assert_eq!(bins.to_idx(&2.5), Ok(0));
    //     assert_eq!(bins.to_idx(&2.0), Ok(1));
    //     assert_eq!(bins.to_idx(&1.5), Ok(1));
    //     assert_eq!(bins.to_idx(&1.0), Ok(1));
    //     assert_eq!(bins.to_idx(&0.9), Ok(1));
    // }

    fn assert_2d_bins(bins: &NdBinLookup<Precision>, vals: &[Precision], expected: &[usize]) {
        if let Ok(idxs) = bins.to_idxs(vals) {
            assert_eq!(idxs.len(), 2);
            assert_eq!(idxs[0], expected[0]);
            assert_eq!(idxs[1], expected[1]);
        } else {
            panic!("Got error result");
        }
    }

    #[test]
    fn nd() {
        let bins = NdBinLookup::new(vec![
            BinLookup::new(vec![0.0, 0.25, 0.5, 0.75, 1.0], (true, true)).unwrap(),
            BinLookup::new(vec![0.0, 10.0, 100.0, 1000.0], (true, true)).unwrap(),
        ]);
        assert_eq!(bins.n_cells, 12);
        assert_2d_bins(&bins, &[0.0, 0.0], &[0, 0]);
        assert_2d_bins(&bins, &[0.3, 150.0], &[1, 2]);
        assert_2d_bins(&bins, &[1.1, 1001.0], &[3, 2]);
    }

    #[test]
    fn nd_linear() {
        let bins = NdBinLookup::new(vec![
            BinLookup::new(vec![0.0, 0.25, 0.5, 0.75, 1.0], (true, true)).unwrap(),
            BinLookup::new(vec![0.0, 10.0, 100.0, 1000.0], (true, true)).unwrap(),
        ]);

        assert_eq!(bins.to_linear_idx(&[0.0, 0.0]), Ok(0));
        assert_eq!(bins.to_linear_idx(&[0.3, 150.0]), Ok(5));
        assert_eq!(bins.to_linear_idx(&[1.1, 1001.0]), Ok(11));
    }

    #[test]
    fn range_table() {
        let table = RangeTable::new_from_bins(
            vec![
                vec![0.0, 0.25, 0.5, 0.75, 1.0],
                vec![0.0, 10.0, 100.0, 1000.0],
            ],
            (0..12).map(|x| x as Precision).collect(),
        )
        .unwrap();

        assert!((table.lookup(&[0.0, 0.0]) - 0.0).abs() < Precision::EPSILON);
        assert!((table.lookup(&[0.3, 150.0]) - 5.0).abs() < Precision::EPSILON);
        assert!((table.lookup(&[1.1, 1001.0]) - 11.0).abs() < Precision::EPSILON);
    }
}
