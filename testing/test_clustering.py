"""
Tests for clustering/clustering.py
Run from the repo root:  python -m pytest testing/test_clustering.py -v
"""
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from clustering.clustering import complete_linkage, cut


@pytest.fixture(scope="module")
def D():
    # 4 methods: A-B close, C-D close, the two pairs far apart
    names = ["A", "B", "C", "D"]
    X = np.array([[0.0, 0.1, 0.8, 0.9],
                  [0.1, 0.0, 0.7, 0.6],
                  [0.8, 0.7, 0.0, 0.2],
                  [0.9, 0.6, 0.2, 0.0]])
    return pd.DataFrame(X, index=names, columns=names)


def test_merge_order_and_heights(D):
    Z = complete_linkage(D)
    # A+B at 0.1, then C+D at 0.2, then the two at max over all cross pairs = 0.9
    assert np.allclose(Z, [[0, 1, 0.1, 2], [2, 3, 0.2, 2], [4, 5, 0.9, 4]])


def test_uses_max_not_min_or_mean(D):
    # single linkage would give 0.6, average 0.75
    assert complete_linkage(D)[-1, 2] == pytest.approx(0.9)


@pytest.mark.parametrize("seed", range(5))
def test_matches_scipy(seed):
    X = np.random.default_rng(seed).random((15, 3))
    expected = linkage(pdist(X), method="complete")
    assert np.allclose(complete_linkage(squareform(pdist(X))), expected)


def test_heights_non_decreasing():
    X = np.random.default_rng(0).random((20, 4))
    assert np.all(np.diff(complete_linkage(squareform(pdist(X)))[:, 2]) >= 0)


def test_cut(D):
    Z = complete_linkage(D)
    labels = list(D.index)
    assert sorted(map(sorted, cut(Z, labels, 2))) == [["A", "B"], ["C", "D"]]
    assert cut(Z, labels, 1) == [["A", "B", "C", "D"]]
    assert sorted(cut(Z, labels, 4)) == [["A"], ["B"], ["C"], ["D"]]


def test_rejects_asymmetric():
    with pytest.raises(ValueError):
        complete_linkage(np.array([[0, 1], [2, 0]]))
