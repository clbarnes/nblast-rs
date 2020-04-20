from typing import List, NamedTuple
import csv

import numpy as np


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
                        f"Line {idx} has {len(row_data)} values; "
                        f"expected {len(dot_bins)}"
                    )
                data.append(row_data)

        return cls(dist_bins, dot_bins, np.array(data))
