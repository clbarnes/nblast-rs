from typing import List, Tuple, Dict, Iterator

import numpy as np

from .pynblast import ArenaWrapper
from .util import Idx, raise_if_none, rectify_tangents


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

    def add_points_tangents(self, points: np.ndarray, tangents: np.ndarray) -> Idx:
        if points.shape != tangents.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points and tangents must have the same shape, Nx3")

        return self._impl.add_points_tangents(points.tolist(), tangents.tolist())

    def query_target(
        self, query_idx: Idx, target_idx: Idx, normalize=False, symmetry=None
    ) -> float:
        out = self._impl.query_target(query_idx, target_idx, bool(normalize), symmetry)
        return raise_if_none(out, query_idx, target_idx)

    def queries_targets(
        self,
        query_idxs: List[Idx],
        target_idxs: List[Idx],
        normalize=False,
        symmetry=None,
    ) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.queries_targets(
            query_idxs, target_idxs, bool(normalize), symmetry
        )

    def all_v_all(self, normalize=False, symmetry=None) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.all_v_all(bool(normalize), symmetry)

    def __len__(self) -> int:
        return self._impl.len()

    def __iter__(self) -> Iterator[Idx]:
        for idx in range(len(self)):
            yield Idx(idx)

    def self_hit(self, idx) -> float:
        out = self._impl.self_hit(idx)
        return raise_if_none(out, idx)

    def points(self, idx) -> np.ndarray:
        return np.array(raise_if_none(self._impl.points(idx), idx))

    def tangents(self, idx, rectify=False) -> np.ndarray:
        out = np.array(raise_if_none(self._impl.tangents(idx), idx))
        return rectify_tangents(out, True) if rectify else out
