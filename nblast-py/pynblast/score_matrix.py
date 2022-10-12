from typing import List, NamedTuple
import csv

import numpy as np


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
