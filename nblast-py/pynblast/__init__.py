# -*- coding: utf-8 -*-

"""Top-level package for nblast-rs."""

__author__ = """Chris L. Barnes"""
__email__ = "chrislloydbarnes@gmail.com"

from .pynblast import get_version as _get_version, ResamplingArbor

__version__ = _get_version()
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .util import rectify_tangents, Idx, Symmetry
from .arena import NblastArena
from .score_matrix import ScoreMatrix

__all__ = [
    "NblastArena",
    "ScoreMatrix",
    "Symmetry",
    "rectify_tangents",
    "Idx",
    "ResamplingArbor",
]
