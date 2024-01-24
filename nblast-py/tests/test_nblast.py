#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nblast` package."""
from typing import Tuple, Dict

import pytest
import pandas as pd
import numpy as np

from pynblast import NblastArena, Idx, ScoreMatrix, rectify_tangents, Symmetry

EPSILON = 0.001


ArenaNames = tuple[NblastArena, dict[Idx, str]]


def test_read_smat(smat_path):
    smat = ScoreMatrix.read(smat_path)
    dist_bins = smat.dist_thresholds
    dot_bins = smat.dot_thresholds
    arr = smat.values
    assert arr.shape == (len(dist_bins) - 1, len(dot_bins) - 1)
    df = pd.read_csv(smat_path, index_col=0, header=0)
    assert np.allclose(arr, df.to_numpy())


def test_construction(score_mat):
    NblastArena(score_mat)


def test_shallow_copy(arena: NblastArena):
    arena2 = arena.copy(False)
    pts = np.random.random((100, 3))
    arena.add_points(pts)
    assert len(arena) == len(arena2)


def test_deep_copy(arena: NblastArena):
    arena2 = arena.copy(True)
    pts = np.random.random((100, 3))
    arena.add_points(pts)
    assert len(arena) != len(arena2)


def test_insertion(arena: NblastArena, points):
    df = points[0][1]
    arena.add_points(df.to_numpy())


def sort3(p: np.ndarray) -> np.ndarray:
    lst = p.tolist()
    lst.sort()
    return np.asarray(lst)


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


def test_tangents(arena: NblastArena, dotprops):
    df = dotprops[0][1]
    points = df[["points." + d for d in "XYZ"]].to_numpy()
    expected_tangents = rectify_tangents(
        df[[f"vect.{n}" for n in range(1, 4)]].to_numpy()
    )
    idx = arena.add_points(points)
    tangents = arena.tangents(idx, True)
    assert np.allclose(tangents, expected_tangents)


def test_alphas(arena, dotprops):
    df = dotprops[0][1]
    points = df[["points." + d for d in "XYZ"]].to_numpy()
    expected_alpha = df.alpha
    idx = arena.add_points(points)
    alpha = arena.alphas(idx)
    assert np.allclose(alpha, expected_alpha)


def test_query(arena_names: Tuple[NblastArena, Dict[int, str]]):
    arena, _ = arena_names

    result = arena.query_target(Idx(0), Idx(1), False)
    assert result


def all_v_all(arena: NblastArena, names, normalize=False, symmetry=None):
    return arena.all_v_all(normalize, symmetry)
    # q = list(names)

    # return arena.queries_targets(q, q, normalize, symmetry=symmetry)


def test_all_v_all(arena_names: Tuple[NblastArena, Dict[int, str]]):
    out = all_v_all(*arena_names)
    assert len(out) == len(arena_names[1]) ** 2


def test_qs_ts_sym(arena_names: ArenaNames):
    arena, names = arena_names
    out = arena.queries_targets(list(names), list(names), symmetry="arithmetic_mean")
    assert len(out) == len(names) ** 2


def test_all_v_all_sym(arena_names: Tuple[NblastArena, Dict[int, str]]):
    out = all_v_all(*arena_names, symmetry="arithmetic_mean")
    assert len(out) == len(arena_names[1]) ** 2


def test_all_v_all_par(arena_names_factory):
    out = []

    for threads in [None, 1, 2]:
        arena, names = arena_names_factory(False, threads)
        out.append(all_v_all(arena, names))

    assert out[0] == out[1]
    assert out[0] == out[2]


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
    out = all_v_all(*arena_names, symmetry="arithmetic_mean")
    for (q, t), v in out.items():
        assert out[(t, q)] == v


@pytest.mark.parametrize("sym", list(Symmetry))
def test_symmetry_enum(arena_names, sym):
    arena, _ = arena_names
    out1 = arena.query_target(0, 1, symmetry=sym)
    out2 = arena.query_target(1, 0, symmetry=sym)
    assert out1 == out2


def test_prepop_tangents(points, arena):
    p = points[0][1].to_numpy()
    idx0 = arena.add_points(p)
    tangents = arena.tangents(idx0)
    alphas = arena.alphas(idx0)
    idx1 = arena.add_points_tangents_alphas(p, tangents, alphas)
    assert arena.query_target(idx0, idx1, normalize=True) == pytest.approx(
        1.0, abs=EPSILON
    )


def test_neuron_table(points, arena: NblastArena):
    p = points[0][1].to_numpy()
    idx1 = arena.add_points(p)
    df = arena.neuron_table(idx1)

    points = df[list("xyz")].to_numpy()
    assert np.allclose(points, arena.points(idx1))

    tangents = df[["tangent_" + d for d in "xyz"]].to_numpy()
    assert np.allclose(tangents, arena.tangents(idx1))

    assert np.allclose(df["alpha"].to_numpy(), arena.alphas(idx1))


# @pytest.mark.parametrize(
#     ["format"],
#     [
#         # (Format.JSON,),
#         (Format.CBOR,)
#     ],
# )
# def test_ser_roundtrip(points, arena: NblastArena, format):
#     p = points[0][1].to_numpy()
#     idx1 = arena.add_points(p)
#     b: bytes = arena.serialize_neuron(idx1, None, format)
#     idx2 = arena.add_serialized_neuron(BytesIO(b), format)

#     df1 = arena.neuron_table(idx1)
#     df2 = arena.neuron_table(idx2)

#     assert np.allclose(df1.to_numpy(), df2.to_numpy())
