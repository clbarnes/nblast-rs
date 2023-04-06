from typing import List, Optional, Set, Union

import numpy as np

from pynblast.arena import DEFAULT_K, DEFAULT_THREADS
from .pynblast import build_score_matrix


class ScoreMatrixBuilder:
    def __init__(self, seed: int, k: int = DEFAULT_K, use_alpha: bool = False):
        self.seed = seed
        self.k = k
        self.use_alpha = use_alpha

        self.neurons: List[List[List[float]]] = []

        self.matching_sets: List[List[int]] = []
        self.nonmatching_sets: Optional[List[List[int]]] = None

        self.dist_n_bins: Optional[int] = None
        self.dist_inner_bounds: Optional[List[float]] = None
        self.dot_n_bins: Optional[int] = None
        self.dot_inner_bounds: Optional[List[float]] = None

        self.max_matching_pairs: Optional[int] = None
        self.max_nonmatching_pairs: Optional[int] = None

    def add_neuron(self, points: np.ndarray) -> int:
        points = np.asarray(points)
        if len(points) < self.k:
            raise ValueError(f"Neuron does not have enough points (needs {self.k})")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Not an Nx3 array")
        idx = len(self.neurons)
        self.neurons.append(points.tolist())
        return idx

    def add_matching_set(self, ids: Set[int]):
        self.matching_sets.append(sorted(ids))

    def add_nonmatching_set(self, ids: Set[int]):
        if self.nonmatching_sets is None:
            self.nonmatching_sets = []
        self.nonmatching_sets.append(sorted(ids))

    def set_dist_bins(self, bins: Union[int, List[float]]):
        """Number of bins, or list of inner boundaries of bins"""
        if isinstance(bins, int):
            self.dist_n_bins = bins
            self.dist_inner_bounds = None
        else:
            self.dist_inner_bounds = sorted(bins)
            self.dist_n_bins = None

    def set_dot_bins(self, bins: Union[int, List[float]]):
        """Number of bins, or list of inner boundaries of bins"""
        if isinstance(bins, int):
            self.dot_n_bins = bins
            self.dot_inner_bounds = None
        else:
            self.dot_inner_bounds = sorted(bins)
            self.dot_n_bins = None

    def set_max_pairs(self, matching: Optional[int], nonmatching: Optional[int]):
        self.max_matching_pairs = matching
        self.max_nonmatching_pairs = nonmatching

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
        values = np.array(cells).reshape((len(dist_bins), len(dot_bins)))
        return dist_bins, dot_bins, values
