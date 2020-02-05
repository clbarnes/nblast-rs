#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nblast` package."""
from typing import Tuple, Dict

import pytest

from pynblast import NblastArena, Idx


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
