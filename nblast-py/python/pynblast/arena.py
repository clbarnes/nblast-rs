from typing import List, Tuple, Dict, Iterator, Optional

import numpy as np

from .pynblast import ArenaWrapper
from .util import Idx, raise_if_none, rectify_tangents, Symmetry

DEFAULT_THREADS = 0
DEFAULT_K = 20


class NblastArena:
    """
    Class for creating and keeping track of many neurons for comparison with NBLAST.
    """

    def __init__(
        self,
        dist_bins: List[float],
        dot_bins: List[float],
        score_mat: np.ndarray,
        threads: Optional[int] = DEFAULT_THREADS,
        k=DEFAULT_K,
        use_alpha=False,
    ):
        """
        The required arguments describe a lookup table which is used to convert
        ``(distance, abs_dot_product)`` tuples into a score for a single
        point match.
        The ``*_bins`` arguments describe the bounds of the bins:
        N bounds make for N-1 bins.
        Queries are clamped to the domain of the lookup.
        ``score_mat`` is the table of values, in dist-major order.

        For example, if the lookup table was stored as a pandas dataframe,
        where the distance bins were in the left margin and the absolute dot
        product bins in the top margin, the object would be instantiated by

        >>> arena = NblastArena(df.index, df.columns, df.to_numpy())

        See the ``ScoreMatrix`` namedtuple for convenience.

        ``k`` gives the number of points to use when calculating tangents.

        ``threads`` sets the default number of threads to use.
        If `threads` < 0 (default), use the class' ``threads`` attribute (default 0).
        If it is 0, use the number of threads available.
        If it is > 0, use at most that many.
        If it is ``None``, run in serial.
        """
        self.threads = threads
        self.k = k

        if score_mat.shape != (len(dist_bins) - 1, len(dot_bins) - 1):
            raise ValueError("Bin thresholds do not match score matrix")
        score_vec = score_mat.flatten().tolist()
        self._impl = ArenaWrapper(dist_bins, dot_bins, score_vec, self.k)

    def add_points(self, points: np.ndarray) -> Idx:
        """Add an Nx3 point cloud to the arena representing a neuron.
        Tangents will be calculated automatically.

        Returns an integer index which is used to refer to that neuron later,
        for queries etc..
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape Nx3")
        return self._impl.add_points(points.tolist())

    def add_points_tangents_alphas(
        self, points: np.ndarray, tangents: np.ndarray, alphas: np.ndarray
    ) -> Idx:
        """Add an Nx3 point cloud representing a neuron, with pre-calculated tangents.
        Tangents are assumed to be unit-length and in the same order as the points.

        Returns an integer index which is used to refer to that neuron later.
        """
        if points.shape != tangents.shape or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points and tangents must have the same shape, Nx3")

        return self._impl.add_points_tangents_alphas(
            points.tolist(), tangents.tolist(), list(alphas)
        )

    def _parse_threads(self, threads: Optional[int]) -> Optional[int]:
        if threads is None:
            return None
        elif threads < 0:
            return self.threads
        else:
            return int(threads)

    def query_target(
        self,
        query_idx: Idx,
        target_idx: Idx,
        normalize: bool = False,
        symmetry: Optional[Symmetry] = None,
        use_alpha: bool = False,
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
        out = self._impl.query_target(
            query_idx, target_idx, bool(normalize), symmetry, use_alpha
        )
        return raise_if_none(out, query_idx, target_idx)

    def queries_targets(
        self,
        query_idxs: List[Idx],
        target_idxs: List[Idx],
        normalize: bool = False,
        symmetry: Optional[Symmetry] = None,
        use_alpha: bool = False,
        threads: Optional[int] = -1,
    ) -> Dict[Tuple[Idx, Idx], float]:
        """Query all combinations of some query neurons against some target neurons.

        See the ``query_target`` method for more details on
        ``normalize`` and ``symmetry``.
        See the ``__init__`` method for more details on ``threads``.
        """
        threads = self._parse_threads(threads)
        return self._impl.queries_targets(
            query_idxs, target_idxs, bool(normalize), symmetry, use_alpha, threads
        )

    def all_v_all(
        self,
        normalize=False,
        symmetry=None,
        use_alpha: bool = False,
        threads: Optional[int] = -1,
    ) -> Dict[Tuple[Idx, Idx], float]:
        """Query all loaded neurons against each other.

        See the ``query_target`` method for more details on
        ``normalize`` and ``symmetry``.
        See the ``__init__`` method for more details on ``threads``.
        """
        threads = self._parse_threads(threads)
        return self._impl.all_v_all(bool(normalize), symmetry, use_alpha, threads)

    def __len__(self) -> int:
        return self._impl.len()

    def __iter__(self) -> Iterator[Idx]:
        for idx in range(len(self)):
            yield Idx(idx)

    def self_hit(self, idx, use_alpha: bool = False) -> float:
        """Get the raw score for querying the given neuron against itself.

        N.B. this is much faster that ``arena.query_target(n, n)``.
        """
        out = self._impl.self_hit(idx, use_alpha)
        return raise_if_none(out, idx)

    def points(self, idx) -> np.ndarray:
        """Return a copy of the points associated with the indexed neuron.

        Point order is arbitrary, but consistent with the order returned by the
        ``.tangents`` method.
        """
        return np.array(raise_if_none(self._impl.points(idx), idx))

    def tangents(self, idx, rectify=False) -> np.ndarray:
        """Return a copy of the tangents associated with the indexed neuron.

        Tangent order is arbitrary, but consistent with the order returned by the
        ``.points`` method.
        """
        out = np.array(raise_if_none(self._impl.tangents(idx), idx))
        return rectify_tangents(out, True) if rectify else out

    def alphas(self, idx) -> np.ndarray:
        return np.array(raise_if_none(self._impl.alphas(idx), idx))
