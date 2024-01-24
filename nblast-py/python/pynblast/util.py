from typing import NewType
import enum
import sys

import numpy as np

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum


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


class Format(StrEnum):
    """Enum of serialization formats for neurons.

    JSON is plain text, simple and easily introspectable.
    CBOR is binary, faster and smaller, but needs different tools to debug.
    """

    # todo: fix JSON
    # JSON = "json"
    CBOR = "cbor"


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
