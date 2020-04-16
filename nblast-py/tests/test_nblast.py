#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nblast` package."""
from typing import Tuple, Dict

import pytest
import pandas as pd
import numpy as np

from pynblast import NblastArena, Idx, ScoreMatrix, rectify_tangents

EPSILON = 0.001


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


def test_points_conserved(arena: NblastArena, points):
    p1 = points[0][1].to_numpy()
    idx = arena.add_points(p1)
    p2 = arena.points(idx)
    assert np.allclose(p1, p2)


@pytest.mark.parametrize(
    ["test", "expected"],
    [
        ([1, 1, 1], [1, 1, 1]),
        ([-1, -1, -1], [1, 1, 1]),
        ([-1, 1, 1], [1, -1, -1]),
        ([0, 1, 1], [0, 1, 1]),
        ([0, -1, 1], [0, 1, -1]),
        ([0, 0, 1], [0, 0, 1]),
        ([0, 0, -1], [0, 0, 1]),
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_rectify_tangents(test, expected):
    out = rectify_tangents(np.array([test]))
    expected = np.array([expected])
    assert np.allclose(out, expected)


def test_tangents(arena, dotprops):
    df = dotprops[0][1]
    points = df[["points." + d for d in "XYZ"]].to_numpy()
    expected_tangents = rectify_tangents(
        df[[f"vect.{n}" for n in range(1, 4)]].to_numpy()
    )
    idx = arena.add_points(points)
    tangents = arena.tangents(idx, True)
    assert np.allclose(tangents, expected_tangents)


def test_query(arena_names: Tuple[NblastArena, Dict[int, str]]):
    arena, _ = arena_names

    result = arena.query_target(Idx(0), Idx(1))
    assert result


def all_v_all(arena, names, normalize=False, symmetric=False):
    q = list(names)
    return arena.queries_targets(q, q, normalize, symmetric)


def test_all_v_all(arena_names: Tuple[NblastArena, Dict[int, str]]):
    out = all_v_all(*arena_names)
    assert len(out) == len(arena_names[1]) ** 2


def test_normed(arena_names):
    out = all_v_all(*arena_names, normalize=True)
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


def test_self_hit_calc(arena_names):
    arena, _ = arena_names
    explicit = arena.self_hit(0)
    implicit = arena.query_target(0, 0)
    assert explicit == pytest.approx(implicit, abs=EPSILON)


def test_normed_calc(arena_names):
    arena, _ = arena_names
    self_hit = arena.query_target(0, 0)
    non_norm = arena.query_target(0, 1)
    norm = arena.query_target(0, 1, normalize=True)

    assert non_norm / self_hit == pytest.approx(norm)


def test_symmetric(arena_names):
    out = all_v_all(*arena_names, symmetric=True)
    for (q, t), v in out.items():
        assert out[(t, q)] == v
