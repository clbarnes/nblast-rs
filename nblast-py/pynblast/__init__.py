# -*- coding: utf-8 -*-

"""Top-level package for nblast-rs."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"
__version__ = "0.1.0"
__version_info__ = tuple(int(n) for n in __version__.split("."))

from typing import NewType, List, Dict, Tuple, Iterator, NamedTuple
import csv

import numpy as np

from .pynblast import ArenaWrapper

__all__ = ["NblastArena", "ScoreMatrix"]

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
        return self._impl.query_target(
            query_idx, target_idx, bool(normalise), bool(symmetric)
        )

    def queries_targets(
        self,
        query_idxs: List[Idx],
        target_idxs: List[Idx],
        normalise=False,
        symmetric=False,
    ) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.queries_targets(
            query_idxs, target_idxs, bool(normalise), bool(symmetric)
        )

    def all_v_all(
        self, normalise=False, symmetric=False
    ) -> Dict[Tuple[Idx, Idx], float]:
        return self._impl.all_v_all(bool(normalise), bool(symmetric))

    def __len__(self) -> int:
        return self._impl.len()

    def __iter__(self) -> Iterator[Idx]:
        for idx in range(len(self)):
            yield Idx(idx)


def parse_interval(s):
    no_brackets = s.strip("([]) ")
    if not no_brackets:
        return None
    low_high = no_brackets.split(",")
    if len(low_high) == 1:
        return float(low_high)
    else:
        return float(low_high[-1])


class ScoreMatrix(NamedTuple):
    dist_thresholds: List[float]
    dot_thresholds: List[float]
    values: np.ndarray

    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.values, self.dist_thresholds, self.dot_thresholds)

    @classmethod
    def read(cls, fpath, **csv_kwargs):
        """Read a precomputed score matrix from a CSV.

        Assumes the matrix has distance bins on the row index,
        and dot product bins on the column index.
        Assumes the col/row indices are strings describing
        the upper bound of the bin,
        or the interval that bin covers, e.g. "(0.75,1.5]"
        (N.B. all brackets are ignored; upper value is assumed to be closed
        and lower value is assumed to be 0 or the upper value of the previous bin).

        Returns a tuple of distance thresholds, dot product thresholds,
        and the array of values.
        """
        with open(fpath) as f:
            reader = csv.reader(f, **csv_kwargs)
            dot_bins = [parse_interval(s) for s in next(reader)[1:]]

            dist_bins = []
            data = []
            for idx, row in enumerate(reader, 1):
                dist_bins.append(parse_interval(row[0]))
                row_data = [float(s) for s in row[1:]]
                if len(row_data) != len(dot_bins):
                    raise ValueError(
                        f"Line {idx} has {len(row_data)} values; expected {len(dot_bins)}"
                    )
                data.append(row_data)

        return cls(dist_bins, dot_bins, np.array(data))
