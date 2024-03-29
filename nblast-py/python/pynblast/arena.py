from typing import List, Tuple, Dict, Iterator, Optional
import warnings
from copy import copy

import numpy as np

from .score_matrix import ScoreMatrix
from .pynblast import ArenaWrapper
from .util import Idx, raise_if_none, rectify_tangents, Symmetry
import pandas as pd

DEFAULT_THREADS = 0
DEFAULT_K = 20


class NblastArena:
    """
    Class for creating and keeping track of many neurons for comparison with NBLAST.

    Create the arena with a score matrix.
    Then use `arena.add_points()` to add point clouds and return indices used for querying
    (if you have already calculated tangents and alphas *using the same neighborhood size `k`*,
    use `arena.add_points_tangents_alphas()`).
    Use `arena.query_target()`, `arena.queries_targets()`, and
    `arena.all_v_all()` to perform queries.

    You can retrieve the points, tangents, alpha values, or everything about a neuron
    with `arena.points()`, `arena.tangents()`, `arena.alphas()`, and `arena.neuron_table()`.
    """

    def __init__(
        self,
        score_mat: ScoreMatrix,
        use_alpha: bool = False,
        threads: Optional[int] = DEFAULT_THREADS,
        k=DEFAULT_K,
    ):
        """
        The required arguments describe a lookup table which is used to convert
        ``(distance, abs_dot_product)`` tuples into a score for a single
        point match.
        Queries are clamped to the domain of the lookup
        (i.e. the first and last values are effectively replaced with -inf and +inf respectively).

        ``k`` gives the number of points to use when calculating tangents.

        ``threads`` sets the number of threads to use.
        If it is 0, use the number of threads available.
        If it is > 0, use at most that many.
        If it is ``None``, run in serial.
        """
        self.use_alpha = use_alpha
        self.threads = threads
        self.k = k

        self._impl = ArenaWrapper(
            score_mat.dist_thresholds.tolist(),
            score_mat.dot_thresholds.tolist(),
            score_mat._flat_values().tolist(),
            self.k,
            self.use_alpha,
            self.threads,
        )

    def add_points(self, points: np.ndarray) -> Idx:
        """Add an Nx3 point cloud to the arena representing a neuron.
        Tangents will be calculated automatically.

        Returns an integer index which is used to refer to that neuron later,
        for queries etc..
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape Nx3")
        return self._impl.add_points(points)

    def add_points_tangents_alphas(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        alphas: np.ndarray,
    ) -> Idx:
        """Add an Nx3 point cloud representing a neuron, with pre-calculated tangents.
        Tangents must be unit-length and in the same order as the points.
        If this arena is not using alphas,
        you can give `None` for the `alphas` argument.

        Returns an integer index which is used to refer to that neuron later.
        """
        points = np.asarray(points)
        tangents = np.asarray(tangents)

        if points.shape != tangents.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points and tangents must have the same shape, Nx3")

        if alphas is None:
            if self.use_alpha:
                raise ValueError(
                    "Alpha values not given, but this NblastArena uses alpha weighting"
                )
            else:
                warnings.warn(
                    "Alpha values should be given, even if they are not used",
                    PendingDeprecationWarning,
                )
                alphas = np.full(len(points), 1.0)
        else:
            alphas = np.asarray(alphas)
            if alphas.shape != (len(points),):
                raise ValueError("Alphas must be 1D and have same length as points")

        return self._impl.add_points_tangents_alphas(points, tangents, alphas)

    def query_target(
        self,
        query_idx: Idx,
        target_idx: Idx,
        normalize: bool = False,
        symmetry: Optional[Symmetry] = None,
    ) -> float:
        """Query one neuron against another,
        using the indices generated when they were added.

        If ``normalize`` is true, divide the raw score by the query neuron's
        self-hit score.

        If ``symmetry`` is not ``None``, additionally calculate the score from target
        to query (normalizing if requested), and apply some function to make the
        operation commutative/ symmetric
        (see the ``Symmetry`` enum for available functions).
        """
        out = self._impl.query_target(query_idx, target_idx, bool(normalize), symmetry)
        return raise_if_none(out, query_idx, target_idx)

    def queries_targets(
        self,
        query_idxs: List[Idx],
        target_idxs: List[Idx],
        normalize: bool = False,
        symmetry: Optional[Symmetry] = None,
        max_centroid_dist: Optional[float] = None,
    ) -> Dict[Tuple[Idx, Idx], float]:
        """Query all combinations of some query neurons against some target neurons.

        ``max_centroid_dist`` pre-filters neurons based on their centroid location:
        if the centroids are too far apart, they will not be queried.

        See the ``query_target`` method for more details on
        ``normalize`` and ``symmetry``.
        See the ``__init__`` method for more details on ``threads``.
        """
        return self._impl.queries_targets(
            query_idxs,
            target_idxs,
            bool(normalize),
            symmetry,
            max_centroid_dist,
        )

    def all_v_all(
        self,
        normalize=False,
        symmetry=None,
        max_centroid_dist: Optional[float] = None,
    ) -> Dict[Tuple[Idx, Idx], float]:
        """Query all loaded neurons against each other.

        ``max_centroid_dist`` pre-filters neurons based on their centroid location:
        if the centroids are too far apart, they will not be queried.

        See the ``query_target`` method for more details on
        ``normalize`` and ``symmetry``.
        See the ``__init__`` method for more details on ``threads``.
        """
        return self._impl.all_v_all(bool(normalize), symmetry, max_centroid_dist)

    def __len__(self) -> int:
        return self._impl.len()

    def __iter__(self) -> Iterator[Idx]:
        for idx in range(len(self)):
            yield Idx(idx)

    def self_hit(self, idx) -> float:
        """Get the raw score for querying the given neuron against itself.

        N.B. this is much faster than ``arena.query_target(n, n)``.
        """
        out = self._impl.self_hit(idx)
        return raise_if_none(out, idx)

    def points(self, idx) -> np.ndarray:
        """Return a copy of the points associated with the indexed neuron.

        Order is arbitrary.
        """
        return np.asarray(raise_if_none(self._impl.points(idx), idx))

    def tangents(self, idx, rectify=False) -> np.ndarray:
        """Return a copy of the tangents associated with the indexed neuron.

        Order is arbitrary, but consistent with the order returned by the
        ``.points`` method.
        """
        out = np.asarray(raise_if_none(self._impl.tangents(idx), idx))
        return rectify_tangents(out, True) if rectify else out

    def alphas(self, idx) -> np.ndarray:
        """Return a copy of the alpha values associated with the indexed neuron.

        Order is arbitrary, but consistent with the order returned by the
        ``.points`` method.
        """
        return np.asarray(raise_if_none(self._impl.alphas(idx), idx))

    def neuron_table(self, idx: Idx) -> pd.DataFrame:
        """Return a neuron's points, tangents, and alphas as a dataframe."""
        arr = raise_if_none(self._impl.neuron_array(idx), idx)
        return pd.DataFrame(
            arr, columns=["x", "y", "z", "tangent_x", "tangent_y", "tangent_z", "alpha"]
        )

    # def serialize_neuron(
    #     self, idx: Idx, write_to: Optional[BytesIO], format: Format
    # ) -> Optional[bytes]:
    #     b = raise_if_none(self._impl.serialize_neuron(idx, format), idx)
    #     if write_to is None:
    #         return b
    #     else:
    #         write_to.write(b)
    #     return None

    # def add_serialized_neuron(self, read_from: BytesIO, format: Format) -> Idx:
    #     # todo: slower than adding p,t,a. Need better serialization?
    #     b = read_from.read()
    #     return self._impl.add_serialized_neuron(b, format)

    def copy(self, deep=True):
        out = copy(self)
        if deep:
            out._impl = out._impl.deepcopy()
        return out

    def __deepcopy__(self):
        return self.copy()
