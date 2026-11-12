"""
Tests for clustering/clustering.py
Run from the repo root:  python -m pytest testing/test_clustering.py -v
"""
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from clustering.clustering import CompleteLinkageClustering
from clustering.utils import cut, plot_dendrogram


def complete_linkage(D):
    return CompleteLinkageClustering(D).complete_linkage()


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
        CompleteLinkageClustering(np.array([[0, 1], [2, 0]]))


def test_rejects_nan():
    with pytest.raises(ValueError):
        CompleteLinkageClustering(np.array([[0, np.nan], [np.nan, 0]]))


@pytest.fixture(scope="module")
def D_ties():
    # every off-diagonal distance equal, so every merge is a tie
    return np.ones((5, 5)) - np.eye(5)


def test_ties_broken_at_random(D_ties):
    firsts = {tuple(complete_linkage(D_ties)[0, :2]) for _ in range(20)}
    assert len(firsts) > 1


def test_ties_still_valid_linkage(D_ties):
    # whichever tied pair is picked, every merge is at distance 1 and sizes add up to n
    Z = complete_linkage(D_ties)
    assert np.allclose(Z[:, 2], 1.0)
    assert Z[-1, 3] == 5


def test_deterministic_without_ties(D):
    assert np.array_equal(complete_linkage(D), complete_linkage(D))


def test_plot_dendrogram(D):
    import matplotlib
    matplotlib.use("Agg")
    clusterer = CompleteLinkageClustering(D)
    ax = plot_dendrogram(clusterer.complete_linkage(), clusterer.labels, title="test")
    assert ax.get_title() == "test"


def test_modal_tree_no_ties(D):
    # no ties -> every run gives the same tree
    clusterer = CompleteLinkageClustering(D)
    sig, frac, n_distinct, Z = clusterer.modal_tree(runs=20)
    assert frac == 1.0
    assert n_distinct == 1
    assert sig == {frozenset("AB"), frozenset("CD"), frozenset("ABCD")}
    assert np.allclose(Z, complete_linkage(D))


def test_modal_tree_Z_matches_signature(D_ties):
    clusterer = CompleteLinkageClustering(D_ties)
    sig, _, _, Z = clusterer.modal_tree(runs=50)
    assert clusterer._tree_signature(Z, clusterer.labels) == sig


def test_modal_tree_with_ties(D_ties):
    # all distances tied -> many different trees, and the modal one is no more than all of them
    sig, frac, n_distinct, Z = CompleteLinkageClustering(D_ties).modal_tree(runs=200)
    assert n_distinct > 1
    assert 0 < frac < 1
    assert len(sig) == 4  # n - 1 internal nodes for 5 leaves


def test_modal_tree_picks_most_common_tree():
    # A-B tied with B-C at 0.1, everything else far: the tree is {A,B} or {B,C} first,
    # each about half the time; the modal one must be one of those two
    names = ["A", "B", "C", "D"]
    X = np.array([[0.0, 0.1, 0.5, 0.9],
                  [0.1, 0.0, 0.1, 0.9],
                  [0.5, 0.1, 0.0, 0.9],
                  [0.9, 0.9, 0.9, 0.0]])
    sig, frac, n_distinct, _ = CompleteLinkageClustering(pd.DataFrame(X, index=names, columns=names)).modal_tree(runs=400)
    assert n_distinct == 2
    assert 0.4 < frac < 0.65
    assert frozenset("AB") in sig or frozenset("BC") in sig


def test_plot_dendrogram_from_modal_tree(D_ties):
    import matplotlib
    matplotlib.use("Agg")
    clusterer = CompleteLinkageClustering(D_ties)
    _, frac, _, Z = clusterer.modal_tree(runs=50)
    ax = plot_dendrogram(Z, clusterer.labels, title=f"modal tree ({frac:.0%} of runs)")
    assert ax.get_title().startswith("modal tree")
    # every leaf is drawn exactly once
    assert sorted(t.get_text() for t in ax.get_xticklabels()) == sorted(map(str, clusterer.labels))
