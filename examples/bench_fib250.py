#!/usr/bin/env python
import zipfile as zf
from pathlib import Path
import logging
import datetime as dt
from contextlib import contextmanager
import os

from tqdm import tqdm
import pandas as pd
import numpy as np

from pynblast import NblastArena, ScoreMatrix

logger = logging.getLogger(__name__)
time_log = logging.getLogger(__name__ + ".time")

here = Path(__file__).resolve().parent


def get_threads():
    val = os.environ.get("NBLAST_THREADS", 0) or 0
    try:
        return int(val)
    except ValueError:
        if val.lower() == "none":
            return None
        raise


THREADS = get_threads()
logger.warning("Running with THREADS = %s", THREADS)

URL_PREFIX = "https://github.com/clbarnes/nblast-rs/files"
SCORES_NAME = "fib250.aba.csv.zip"

SCORES_URL = f"{URL_PREFIX}/4567582/{SCORES_NAME}"

SCORES_FPATH = here / SCORES_NAME


def df_to_pt_tan_a(df):
    pt = df[["X", "Y", "Z"]].to_numpy()
    tan = df[["i", "j", "k"]].to_numpy()
    a = df["alpha"].to_numpy()
    return pt, tan, a


def ingest_dotprops():
    DOTPROPS_NAME = "fib250.csv.zip"
    DOTPROPS_URL = f"{URL_PREFIX}/4567531/{DOTPROPS_NAME}"
    DOTPROPS_FPATH = here / DOTPROPS_NAME

    if not DOTPROPS_FPATH.is_file():
        raise ValueError(f"Download necessary data from\n\t{DOTPROPS_URL}")

    with zf.ZipFile(DOTPROPS_FPATH) as z:
        with z.open(DOTPROPS_FPATH.name[:-4]) as f:
            dotprops = pd.read_csv(f)

    skids = []
    pt_tan_as = []

    for skid in np.unique(dotprops.id):
        skids.append(skid)
        df = dotprops.loc[dotprops.id == skid]
        pt_tan_as.append(df_to_pt_tan_a(df))

    return skids, pt_tan_as


def get_smat():
    SMAT_FPATH = here.parent / "data" / "smat_fcwb.csv"
    return ScoreMatrix.read(SMAT_FPATH)


@contextmanager
def timer(name):
    started = dt.datetime.now()
    time_log.warning("%s started at %s", name, started)
    yield
    finished = dt.datetime.now()
    time_log.warning(
        "%s finished at %s, taking %s seconds",
        name,
        finished,
        (finished - started).total_seconds(),
    )


def populate_arena(skids, pt_tan_as):
    idx_to_skid = dict()
    smat = get_smat()
    arena = NblastArena(*smat, threads=THREADS)

    with timer("Arena population"):
        for skid, pt_tan_a in tqdm(
            zip(skids, pt_tan_as),
            desc="Loading arena (building spatial trees)",
            total=len(skids),
        ):
            idx = arena.add_points_tangents_alphas(*pt_tan_a)
            idx_to_skid[idx] = skid

    return arena, idx_to_skid


def all_by_all(arena):
    with timer("All v all"):
        results = arena.all_v_all()
    return results


def main():
    skids, pt_tan_as = ingest_dotprops()
    arena, idx_to_skid = populate_arena(skids, pt_tan_as)
    return all_by_all(arena)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
