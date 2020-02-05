# -*- coding: utf-8 -*-

"""Top-level package for nblast-rs."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"
__version__ = "0.1.0"
__version_info__ = tuple(int(n) for n in __version__.split("."))

from typing import NewType, List, Dict, Tuple, Iterator

import numpy as np

from .pynblast import ArenaWrapper

__all__ = ["NblastArena"]

Idx = NewType("Idx", int)


class NblastArena:
    def __init__(
        self, dist_bins: List[float], dot_bins: List[float], score_mat: np.ndarray
    ):
        if score_mat.shape != (len(dist_bins), len(dot_bins)):
            raise ValueError("Bin thresholds do not match score matrix")
        score_vec = score_mat.flatten().tolist()
        self._impl = ArenaWrapper(dist_bins, dot_bins, score_vec)

    def add_points(self, points: np.ndarray) -> Idx:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape Nx3")
        return self._impl.add_points(points.tolist())

    def query_target(
        self, query_idx: Idx, target_idx: Idx, normalise=False, symmetric=False
    ) -> float:
        return self._impl.query_target(query_idx, target_idx, bool(normalise), bool(symmetric))

    def queries_targets(
        self,
        query_idxs: List[Idx],
        target_idxs: List[Idx],
        normalise=False,
        symmetric=False,
    ) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.queries_targets(query_idxs, target_idxs, bool(normalise), bool(symmetric))

    def all_v_all(
        self, normalise=False, symmetric=False
    ) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.all_v_all(bool(normalise), bool(symmetric))

    def __len__(self) -> int:
        return self._impl.len()

    def __iter__(self) -> Iterator[Idx]:
        for idx in range(len(self)):
            yield Idx(idx)
