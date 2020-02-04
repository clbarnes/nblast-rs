from typing import Tuple, List, Dict
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

import pynblast

from .constants import DATA_DIR


@pytest.fixture
def score_mat_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "smat_jefferis.csv", sep=" ", header=0, index_col=0)


def parse_interval(s):
    return float(s.split(",")[-1][:1])


@pytest.fixture
def score_mat_tup(
    score_mat_df: pd.DataFrame,
) -> Tuple[List[float], List[float], np.ndarray]:
    dist_bins = [parse_interval(item) for item in score_mat_df.index]
    dot_bins = [parse_interval(item) for item in score_mat_df.columns]
    data = score_mat_df.to_numpy()
    return dist_bins, dot_bins, data


def parse_points(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=",", usecols=[1, 2, 3], header=0)
    df.columns = [c.lower() for c in df.columns]
    return df


@pytest.fixture
def points() -> List[Tuple[str, pd.DataFrame]]:
    p_dir = DATA_DIR / "points"
    return [(fpath.stem, parse_points(fpath)) for fpath in sorted(p_dir.iterdir())]


@pytest.fixture
def arena(score_mat_tup):
    return pynblast.NblastArena(*score_mat_tup)


@pytest.fixture
def arena_points(arena, points) -> Tuple[pynblast.NblastArena, Dict[int, str]]:
    idx_to_name = dict()
    for name, df in points:
        idx = arena.add_points(df.to_numpy())
        idx_to_name[idx] = name
    return arena, idx_to_name
