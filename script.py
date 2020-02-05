from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from pynblast import NblastArena, Idx

def parse_interval(s):
    return float(s.split(",")[-1][:1])


def parse_points(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=",", usecols=[1, 2, 3], header=0)
    df.columns = [c.lower() for c in df.columns]
    return df


def dict_to_df(d, idx_to_names):
    index_names = []
    index_idxs = dict()
    col_names = []
    col_idxs = dict()
    for (q, t), v in sorted(d.items()):
        if q not in index_idxs:
            index_names.append(idx_to_names[q])
            index_idxs[q] = len(index_idxs)
        if t not in col_idxs:
            col_names.append(idx_to_names[t])
            col_idxs[t] = len(col_idxs)

    data = np.full((len(index_names), len(col_names)), np.nan)
    for (q, t), v in d.items():
        data[index_idxs[q], col_idxs[t]] = v

    return pd.DataFrame(data=data, index=index_names, columns=col_names)


smat = pd.read_csv("data/smat_jefferis.csv", sep=" ", header=0, index_col=0)

dist_bins = [parse_interval(item) for item in smat.index]
dot_bins = [parse_interval(item) for item in smat.columns]
smat_arr = smat.to_numpy()

points = [(fpath.stem, parse_points(fpath)) for fpath in sorted(Path("data/points").iterdir())]

arena = NblastArena(dist_bins, dot_bins, smat_arr)
idx_to_name = {arena.add_points(df.to_numpy()): name for name, df in points}


def heatmaps(arena, idx_to_name):
    for normalised, symmetric in product([0, 1], repeat=2):
        title = f"norm = {normalised}, symm = {symmetric}"
        print(title)
        fig, ax = plt.subplots()
        ax.set_title(title)
        d = arena.all_v_all(bool(normalised), bool(symmetric))
        df = dict_to_df(d, idx_to_name)
        sns.heatmap(df, annot=True, ax=ax)

    plt.show()



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)



def cluster(arena):
    d = arena.all_v_all(True, True)
    distances = 1 - dict_to_df(d, idx_to_name)
    assert list(distances.index) == list(distances.columns)

    agg_clus = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage="average", affinity="precomputed")
    agg_clus = agg_clus.fit(distances)

    fig, ax = plt.subplots()

    dend = plot_dendrogram(agg_clus, ax=ax, p=3, truncate_mode="level")

    plt.show()


cluster(arena)
