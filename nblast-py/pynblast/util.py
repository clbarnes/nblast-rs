from typing import NewType
import enum

import numpy as np


class StrEnum(str, enum.Enum):
    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, (str, enum.auto)):
                raise TypeError(
                    "Values of StrEnums must be strings: {} is a {}".format(
                        repr(arg), type(arg)
                    )
                )
        return super().__new__(cls, *args)

    def __str__(self):
        return self.value

    # The first argument to this function is documented to be the name of the
    # enum member, not `self`:
    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, *_):
        return name


Idx = NewType("Idx", int)


def raise_if_none(result, *idxs):
    if result is None:
        raise IndexError(f"Index(es) not in arena: {idxs}")
    return result


class Symmetry(StrEnum):
    """Enum of strategies for making an NBLAST query symmetric.

    i.e. the name of a function to apply to the forward and reverse scores
    together to generate a comparable score.
    """
    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MIN = "min"
    MAX = "max"


def rectify_tangents(orig: np.ndarray, inplace=False) -> np.ndarray:
    """Normalises orientation of tangents.

    Makes the first nonzero element positive.
    """
    if not inplace:
        orig = orig.copy()

    prev_zero = np.full(len(orig), True, dtype=bool)
    for this_col in orig.T:
        to_flip = np.logical_and(prev_zero, this_col < 0)
        orig[to_flip, :] *= -1
        prev_zero = np.logical_and(prev_zero, this_col == 0)
        if not prev_zero.any():
            break

    return orig
