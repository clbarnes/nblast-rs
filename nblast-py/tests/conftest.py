from typing import Tuple, List, Dict
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

import pynblast

from .constants import DATA_DIR


@pytest.fixture
def smat_path() -> Path:
    return DATA_DIR / "smat_fcwb.csv"


@pytest.fixture
def score_mat_tup(smat_path) -> pynblast.ScoreMatrix:
    return pynblast.ScoreMatrix.read(smat_path)


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
