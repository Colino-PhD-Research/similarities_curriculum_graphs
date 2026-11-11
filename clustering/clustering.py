"""
Complete-linkage hierarchical clustering of similarity methods
(Soundarajan, Eliassi-Rad & Gallagher, "A Guide to Selecting a Network Similarity Method").

Input is a precomputed, symmetric distance matrix, e.g. the methods x methods Kendall-Tau
distance from rankings.ranking_comparisons.pairwise_correlations(...)["kendall"] or
average_matrix(results, "kendall").

Output is a linkage matrix in scipy's format, so scipy.cluster.hierarchy.dendrogram,
fcluster, leaves_list, etc. work on it directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def complete_linkage(D: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Agglomerative clustering where the distance between two clusters is the largest
    distance between any member of one and any member of the other.

    Returns Z, shape (n - 1, 4). Row i merges clusters Z[i, 0] and Z[i, 1] at distance Z[i, 2]
    into a new cluster of Z[i, 3] items, which gets id n + i. Ids 0..n-1 are the original items.
    """
    D = np.asarray(D, dtype=float)
    n = len(D)
    if D.shape != (n, n) or not np.allclose(D, D.T):
        raise ValueError("D must be a square, symmetric distance matrix")
    if np.isnan(D).any():
        raise ValueError("D contains NaN")

    # cluster id -> (member indices); dist[a][b] = complete-linkage distance between clusters
    members = {i: [i] for i in range(n)}
    dist = {i: {j: D[i, j] for j in range(n) if j != i} for i in range(n)}
    Z = np.zeros((n - 1, 4))

    for step in range(n - 1):
        # closest pair of active clusters
        a, b = min(((a, b) for a in dist for b in dist[a] if a < b), key=lambda p: dist[p[0]][p[1]])
        new = n + step
        Z[step] = [a, b, dist[a][b], len(members[a]) + len(members[b])]

        # Lance-Williams update for complete linkage: d(a u b, c) = max(d(a, c), d(b, c))
        dist[new] = {}
        for c in dist:
            if c in (a, b, new):
                continue
            d = max(dist[a][c], dist[b][c])
            dist[new][c] = dist[c][new] = d
            del dist[c][a], dist[c][b]
        del dist[a], dist[b]
        members[new] = members.pop(a) + members.pop(b)

    return Z


def cut(Z: np.ndarray, labels: list[str], k: int) -> list[list[str]]:
    """Cut the dendrogram into k clusters by undoing the last k - 1 merges."""
    n = len(labels)
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}")
    members = {i: [labels[i]] for i in range(n)}
    for step in range(n - k):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        members[n + step] = members.pop(a) + members.pop(b)
    return list(members.values())


def plot_dendrogram(D: pd.DataFrame, title: str = "", ax=None):
    """Complete-linkage dendrogram of a labeled distance matrix (labels = D.index)."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    dendrogram(complete_linkage(D), labels=list(D.index), ax=ax, leaf_rotation=90)
    ax.set_ylabel("complete-linkage distance")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax
