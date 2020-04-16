#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `nblast` package."""
import pytest
import pandas as pd
import numpy as np

EPSILON = 0.001


def dict_to_df(idxs_to_score, idx_to_name):
    names = sorted(idx_to_name.values())
    size = len(idx_to_name)
    df = pd.DataFrame(
        np.full((size, size), np.nan),
        index=names, columns=names
    )
    for (q_idx, t_idx), score in idxs_to_score.items():
        q_name = idx_to_name[q_idx]
        t_name = idx_to_name[t_idx]
        df.loc[q_name][t_name] = score

    assert not np.isnan(df.to_numpy()).any()
    return df


def test_vs_r(arena_names, expected_nblast):
    arena, idx_to_name = arena_names
    idxs_to_score = arena.all_v_all()

    for (q_idx, t_idx), score in idxs_to_score.items():
        q_name = idx_to_name[q_idx]
        t_name = idx_to_name[t_idx]
        score = idxs_to_score[(q_idx, t_idx)]
        expected_score = expected_nblast.loc[q_name][t_name]
        assert score == pytest.approx(expected_score, abs=EPSILON)
