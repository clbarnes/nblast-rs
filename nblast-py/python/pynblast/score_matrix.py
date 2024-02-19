import csv

import numpy as np
from numpy.typing import ArrayLike


def parse_interval(s) -> tuple[float, float]:
    no_brackets = s.strip("([]) ")
    # if not no_brackets:
    #     return None
    low, high = (float(i) for i in no_brackets.split(","))
    return (low, high)


def intervals_to_bins(intervals: list[tuple[float, float]]):
    it = iter(intervals)
    out = list(next(it))
    for lower, upper in it:
        if lower != out[-1]:
            raise ValueError("Bins are not abutting")
        out.append(upper)
    return out


class ScoreMatrix:
    """Representation of a lookup table for point match scores.

    N thresholds represent N-1 bins.
    The values are in dot-major order
    (i.e. values in the same dist bin are next to each other).
    """

    def __init__(
        self,
        dist_thresholds: ArrayLike,
        dot_thresholds: ArrayLike,
        values: ArrayLike,
    ) -> None:
        self.dist_thresholds = np.asarray(dist_thresholds, np.float64).flatten()
        self.dot_thresholds = np.asarray(dot_thresholds, np.float64).flatten()
        self.values = np.asarray(values, np.float64)

        exp_shape = (len(self.dist_thresholds) - 1, len(self.dot_thresholds) - 1)
        if self.values.shape != exp_shape:
            raise ValueError(
                "For N dist_thresholds and M dot_thresholds, values must be (N-1)x(M-1)"
            )

    def _flat_values(self):
        if self.values.flags["F_CONTIGUOUS"]:
            return self.values.T.flatten()
        else:
            return self.values.flatten()

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

            dot_bins = intervals_to_bins([parse_interval(s) for s in next(reader)[1:]])
            intervals = []
            data = []
            for idx, row in enumerate(reader, 1):
                intervals.append(parse_interval(row[0]))
                row_data = [float(s) for s in row[1:]]
                if len(row_data) != len(dot_bins) - 1:
                    raise ValueError(
                        f"Line {idx} has {len(row_data)} values; "
                        f"expected {len(dot_bins)}"
                    )
                data.append(row_data)

        dist_bins = intervals_to_bins(intervals)

        return cls(dist_bins, dot_bins, np.array(data))
