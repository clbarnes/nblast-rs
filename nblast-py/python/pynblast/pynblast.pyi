"""Type stubs for classes to be used directly from pynblast"""
from __future__ import annotations
from typing import List, Optional, Set, Tuple
from .util import Idx

class ResamplingArbor:
    def __init__(self, table: List[Tuple[int, Optional[int], float, float, float]]): ...
    def prune_at(self, ids: List[int]) -> int: ...
    def prune_branches_containing(self, ids: List[int]) -> int: ...
    def prune_below_strahler(self, threshold: int) -> int: ...
    def prune_beyond_branches(self, threshold: int) -> int: ...
    def prune_beyond_steps(self, threshold: int) -> int: ...
    def prune_twigs(self, threshold: float) -> int: ...
    def prune_beyond_distance(self, threshold: float) -> int: ...
    def cable_length(self) -> float: ...
    def copy(self) -> ResamplingArbor: ...
    def points(self) -> List[Tuple[float, float, float]]: ...
    def skeleton(self) -> List[Tuple[int, Optional[int], float, float, float]]: ...

class ArenaWrapper:
    def __init__(
        self,
        dist_thresholds: list[float],
        dot_thresholds: list[float],
        cells: list[float],
        k: int,
        use_alpha: bool,
        threads: Optional[int],
    ) -> None: ...
    def add_points(self, points: list[list[float]]) -> Idx: ...
    def add_points_tangents_alphas(
        self, points: list[list[float]], tangents: list[float], alphas: list[float]
    ) -> Idx: ...
    def query_target(
        self,
        query_idx: Idx,
        target_idx: Idx,
        normalize: bool,
        symmetry: Optional[str],
    ) -> Optional[float]: ...
    def queries_targets(
        self,
        query_idxs: list[Idx],
        target_idxs: list[Idx],
        normalize: bool,
        symmetry: Optional[str],
        max_centroid_dist: Optional[float],
    ) -> dict[tuple[Idx, Idx], float]: ...
    def all_v_all(
        self,
        normalize: bool,
        symmetry: Optional[str],
        max_centroid_dist: Optional[float],
    ) -> dict[tuple[Idx, Idx], float]: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def self_hit(self, idx: Idx) -> Optional[float]: ...
    def points(self, idx: Idx) -> Optional[list[list[float]]]: ...
    def tangents(self, idx: Idx) -> Optional[list[list[float]]]: ...
    def alphas(self, idx: Idx) -> Optional[list[float]]: ...

def build_score_matrix(
    points: List[List[List[float]]],
    k: int,
    seed: int,
    use_alpha: bool,
    threads: Optional[int],
    matching_sets: List[List[int]],
    nonmatching_sets: Optional[List[List[int]]],
    dist_n_bins: Optional[int],
    dist_inner_bounds: Optional[List[float]],
    dot_n_bins: Optional[int],
    dot_inner_bounds: Optional[List[float]],
    max_matching_pairs: Optional[int],
    max_nonmatching_pairs: Optional[int],
) -> Tuple[List[float], List[float], List[float]]: ...
