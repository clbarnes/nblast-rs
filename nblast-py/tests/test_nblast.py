#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nblast` package."""
from typing import Tuple, Dict

import pytest
import pandas as pd
import numpy as np

from pynblast import NblastArena, Idx, ScoreMatrix


def test_read_smat(smat_path):
    dist_bins, dot_bins, arr = ScoreMatrix.read(smat_path)
    assert arr.shape == (len(dist_bins), len(dot_bins))
    df = pd.read_csv(smat_path, index_col=0, header=0)
    assert np.allclose(arr, df.to_numpy())


def test_construction(score_mat_tup):
    NblastArena(*score_mat_tup)


def test_insertion(arena: NblastArena, points):
    df = points[0][1]
    arena.add_points(df.to_numpy())


def test_query(arena_points: Tuple[NblastArena, Dict[int, str]]):
    arena, points = arena_points

    result = arena.query_target(Idx(0), Idx(1))
    assert result


def all_v_all(arena, points, normalise=False, symmetric=False):
    q = list(points)
    return arena.queries_targets(q, q, normalise, symmetric)


def test_all_v_all(arena_points: Tuple[NblastArena, Dict[int, str]]):
    out = all_v_all(*arena_points)
    assert len(out) == len(arena_points[1]) ** 2


def test_normed(arena_points):
    out = all_v_all(*arena_points, normalise=True)
    for (q, t), v in out.items():
        if q == t:
            assert v == 1


def test_self_hit(arena, points):
    """Check that the self-hit results are correct.

    Ensures that explicit self-hits (with speed optimisations) return
    practically the same result as implicit self-hits (testing two
    identical neurons against each other).

    As a side-effect, proves that tangent vectors are unit-length, as
    this is assumed by the self-hit optimisations.
    """
    _name, df = points[0]
    idx1 = arena.add_points(df.to_numpy())
    idx2 = arena.add_points(df.to_numpy())
    true_self = arena.query_target(idx1, idx1)
    different_self = arena.query_target(idx1, idx2)
    assert different_self == pytest.approx(true_self, 0.0001)


def test_normed_calc(arena_points):
    arena, points = arena_points
    self_hit = arena.query_target(0, 0)
    non_norm = arena.query_target(0, 1)
    norm = arena.query_target(0, 1, normalise=True)

    assert non_norm / self_hit == norm


def test_symmetric(arena_points):
    out = all_v_all(*arena_points, symmetric=True)
    for (q, t), v in out.items():
        assert out[(t, q)] == v
