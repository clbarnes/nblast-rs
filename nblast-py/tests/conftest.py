from typing import Callable, Tuple, List, Dict, Optional
from pathlib import Path
import json
from pynblast.smat_builder import ScoreMatrixBuilder

import pytest
import pandas as pd

import pynblast

from .constants import DATA_DIR


@pytest.fixture
def smat_path() -> Path:
    return DATA_DIR / "smat_fcwb.csv"


@pytest.fixture
def score_mat(smat_path) -> pynblast.ScoreMatrix:
    return pynblast.ScoreMatrix.read(smat_path)


def parse_points(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=",", usecols=[1, 2, 3], header=0)
    df.columns = [c.lower() for c in df.columns]
    return df


@pytest.fixture
def points() -> List[Tuple[str, pd.DataFrame]]:
    p_dir = DATA_DIR / "points"
    return sorted((fpath.stem, parse_points(fpath)) for fpath in p_dir.glob("*.csv"))


@pytest.fixture
def arena(score_mat):
    return pynblast.NblastArena(score_mat, k=5)


@pytest.fixture
def arena_names_factory(
    score_mat, points
) -> Callable[[bool, Optional[int]], Tuple[pynblast.NblastArena, dict[int, str]]]:
    def fn(use_alpha, threads):
        arena = pynblast.NblastArena(
            score_mat, k=5, use_alpha=use_alpha, threads=threads
        )

        idx_to_name = dict()
        for name, df in points:
            idx = arena.add_points(df.to_numpy())
            idx_to_name[idx] = name
        return arena, idx_to_name

    return fn


@pytest.fixture
def arena_names(arena, points) -> Tuple[pynblast.NblastArena, Dict[int, str]]:
    """Tuple of (arena, {idx: "name"})"""
    idx_to_name = dict()
    for name, df in points:
        idx = arena.add_points(df.to_numpy())
        idx_to_name[idx] = name
    return arena, idx_to_name


@pytest.fixture
def expected_nblast() -> pd.DataFrame:
    """
    Query on row index, target on column index.
    """
    p_dir = DATA_DIR / "points"
    names = sorted(p.stem for p in p_dir.glob("*.csv"))
    df = pd.read_csv(DATA_DIR / "kcscores.csv", index_col=0)
    df = df[names]
    df = df.loc[names]
    return df.T


@pytest.fixture
def dotprops() -> List[Tuple[str, pd.DataFrame]]:
    p_dir = DATA_DIR / "dotprops"
    return sorted((p.stem, pd.read_csv(p, index_col=0)) for p in p_dir.glob("*.csv"))


@pytest.fixture
def skeleton() -> List[Tuple[int, Optional[int], float, float, float]]:
    with open(DATA_DIR / "arbors" / "example.json") as f:
        return [tuple(row) for row in json.load(f)["data"]]


@pytest.fixture
def resampler(skeleton) -> pynblast.ResamplingArbor:
    return pynblast.ResamplingArbor(skeleton)


@pytest.fixture
def smatbuilder(points):
    matching_sets = {"Fru": set(), "Gad": set()}
    builder = ScoreMatrixBuilder(1991)
    for name, pts in points:
        idx = builder.add_points(pts.to_numpy())
        for k, v in matching_sets.items():
            if name.startswith(k):
                v.add(idx)

    for ms in matching_sets.values():
        builder.add_matching_set(ms)

    builder.set_dist_bins(6).set_dot_bins(4)
    return builder
