from typing import List, Optional, Set, Union

import numpy as np
from numpy.typing import NDArray

from .arena import DEFAULT_K, DEFAULT_THREADS
from .score_matrix import ScoreMatrix
from .pynblast import build_score_matrix


class ScoreMatrixBuilder:
    """Class for training your own score matrix from data.

    1. Create the builder
    2. Add some point clouds to it with `.add_points()`, which returns an index for each
    3. Use those indices to designate groups of neurons which should match each other with `.add_matching_set()`
    4. Optionally designate groups which should not match each other `.add_nonmatching_set()`.
    5. Set the dist and dot bins (`.set_{dist,dot}_bins()`).
       These can be a list of N-1 inner boundaries for N bins,
       or just the integer N,
       in which case they will be determined from the data.
    6. Optionally, set the maximum number of matching and/or nonmatching pairs with `.set_max_pairs()`.
    7. `.build()`
    """

    def __init__(self, seed: int, k: int = DEFAULT_K, use_alpha: bool = False):
        self.seed = seed
        self.k = k
        self.use_alpha = use_alpha

        self.neurons: List[NDArray[np.float64]] = []

        self.matching_sets: List[List[int]] = []
        self.nonmatching_sets: Optional[List[List[int]]] = None

        self.dist_n_bins: Optional[int] = None
        self.dist_inner_bounds: Optional[List[float]] = None
        self.dot_n_bins: Optional[int] = None
        self.dot_inner_bounds: Optional[List[float]] = None

        self.max_matching_pairs: Optional[int] = None
        self.max_nonmatching_pairs: Optional[int] = None

    def add_points(self, points: np.ndarray) -> int:
        points = np.asarray(points)
        if len(points) < self.k:
            raise ValueError(f"Neuron does not have enough points (needs {self.k})")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Not an Nx3 array")
        idx = len(self.neurons)
        self.neurons.append(points)
        return idx

    def add_matching_set(self, ids: Set[int]):
        self.matching_sets.append(sorted(ids))
        return self

    def add_nonmatching_set(self, ids: Set[int]):
        if self.nonmatching_sets is None:
            self.nonmatching_sets = []
        self.nonmatching_sets.append(sorted(ids))
        return self

    def set_dist_bins(self, bins: Union[int, List[float]]):
        """Number of bins, or list of inner boundaries of bins"""
        if isinstance(bins, int):
            self.dist_n_bins = bins
            self.dist_inner_bounds = None
        else:
            self.dist_inner_bounds = sorted(bins)
            self.dist_n_bins = None
        return self

    def set_dot_bins(self, bins: Union[int, List[float]]):
        """Number of bins, or list of inner boundaries of bins"""
        if isinstance(bins, int):
            self.dot_n_bins = bins
            self.dot_inner_bounds = None
        else:
            self.dot_inner_bounds = sorted(bins)
            self.dot_n_bins = None
        return self

    def set_max_pairs(self, matching: Optional[int], nonmatching: Optional[int]):
        self.max_matching_pairs = matching
        self.max_nonmatching_pairs = nonmatching
        return self

    def build(self, threads: Optional[int] = DEFAULT_THREADS):
        dist_bins, dot_bins, cells = build_score_matrix(
            self.neurons,
            self.k,
            self.seed,
            self.use_alpha,
            self.matching_sets,
            self.nonmatching_sets,
            self.dist_n_bins,
            self.dot_inner_bounds,
            self.dot_n_bins,
            self.dot_inner_bounds,
            self.max_matching_pairs,
            self.max_nonmatching_pairs,
            threads,
        )
        values = np.array(cells).reshape((len(dist_bins) - 1, len(dot_bins) - 1))
        return ScoreMatrix(dist_bins, dot_bins, values)
