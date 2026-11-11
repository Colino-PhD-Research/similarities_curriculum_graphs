"""
Tests for similarity_measures/similarity_measures.py
Run from the repo root:  python -m pytest testing/test_all_measures.py -v
"""
import itertools
import math
import random

import networkx as nx
import numpy as np
import pytest
from netrd.distance.netsimile import feature_extraction

from similarity_measures.similarity_measures import SimilarityMeasures, Similarity


# ---------- graphs used across tests ----------

@pytest.fixture(scope="module")
def sm():
    return SimilarityMeasures()


@pytest.fixture(scope="module")
def s():
    return Similarity()


@pytest.fixture(scope="module")
def G1():
    # 5-cycle A-B-C-D-E plus chord A-C: one triangle (A, B, C), string labels
    return nx.Graph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"), ("A", "C")])


@pytest.fixture(scope="module")
def BA():
    return nx.barabasi_albert_graph(1000, 3, seed=1)


@pytest.fixture(scope="module")
def BA2():
    return nx.barabasi_albert_graph(1000, 3, seed=2)


@pytest.fixture(scope="module")
def ER():
    return nx.gnp_random_graph(1000, 6 / 1000, seed=1)


@pytest.fixture(scope="module")
def two_triangles():
    # two components of 3 nodes: no community of size >= 4 exists
    return nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))


# ---------- netsimile ----------

def test_netsimile_vector_length_35(sm, BA):
    assert sm.netsimile(BA).shape == (35,)


def test_netsimile_vector_deterministic(sm, BA):
    assert np.array_equal(sm.netsimile(BA), sm.netsimile(BA))

# Test that netsimile with 'svm' returns a matrix of shape (min(300, n), 7)
def test_netsimile_svm_samples_300_rows(sm, BA):
    assert sm.netsimile(BA, "svm").shape == (300, 7)


def test_netsimile_svm_small_graph_uses_every_node(sm):
    assert sm.netsimile(nx.cycle_graph(12), "svm").shape == (12, 7)


def test_netsimile_svm_reproducible_and_seeded(sm, BA):
    assert np.array_equal(sm.netsimile(BA, "svm", seed=0), sm.netsimile(BA, "svm", seed=0))
    assert not np.array_equal(sm.netsimile(BA, "svm", seed=0), sm.netsimile(BA, "svm", seed=1))


def test_netsimile_match_same_samples_as_svm(sm, BA):
    # paper: NetSimile-Dist uses the same neighborhoods as NetSimileSVM
    assert np.array_equal(sm.netsimile(BA, "match"), sm.netsimile(BA, "svm"))


def test_netsimile_node_A_features(G1):
    # feature_extraction rows follow sorted(G.nodes()), so row 0 is A; hand-computed in our walkthrough
    A = feature_extraction(G1)[0]
    assert np.allclose(A, [3, 1 / 3, 7 / 3, 4 / 9, 4, 2, 1])


def test_netsimile_invalid_type_raises(sm, G1):
    with pytest.raises(Exception):
        sm.netsimile(G1, "bad")


# ---------- rw_distance ----------

def test_rw_distance_length_20(sm, BA):
    assert sm.rw_distance(BA, rng=random.Random(0)).shape == (20,)


def test_rw_distance_same_rng_seed_same_result(sm, BA):
    assert np.array_equal(sm.rw_distance(BA, rng=random.Random(0)), sm.rw_distance(BA, rng=random.Random(0)))


@pytest.mark.xfail(reason="default rng=random.Random(0) is shared across calls, so repeat calls differ")
def test_rw_distance_default_rng_repeatable(sm, BA):
    assert np.array_equal(sm.rw_distance(BA), sm.rw_distance(BA))


# ---------- infomap ----------

@pytest.mark.parametrize("infomap_type, length", [("in", 1), ("known", 1), ("in_and_known", 2)])
def test_infomap_shapes(sm, BA, infomap_type, length):
    f = sm.infomap(BA, infomap_type)
    assert f.shape == (length,)
    assert ((0 <= f) & (f <= 1)).all()


def test_infomap_barbell(sm):
    # two K5s joined by one edge: Infomap finds both cliques
    # IMIn: 8 nodes have all neighbors inside (1.0), the 2 bridge nodes have 4 of 5 (0.8) -> 0.96
    # IMKnown: every node is adjacent to its whole clique -> 1.0
    assert np.allclose(sm.infomap(nx.barbell_graph(5, 0), "in_and_known"), [0.96, 1.0])


def test_infomap_string_labels(sm, G1):
    assert sm.infomap(G1, "in_and_known").shape == (2,)


def test_infomap_invalid_type_raises(sm, G1):
    with pytest.raises(Exception):
        sm.infomap(G1, "bad")


# ---------- degree, density, transitivity ----------

def test_degree(sm, G1):
    assert sm.degree(G1) == pytest.approx(12 / 5)       # 6 edges, 5 nodes


def test_density(sm, G1):
    assert sm.density(G1) == pytest.approx(6 / 10)      # 6 of C(5,2) edges


def test_transitivity(sm, G1):
    # 1 triangle; 2-hop paths = sum C(d,2) over degrees 3,2,3,2,2 = 9
    assert sm.transitivity(G1) == pytest.approx(3 / 9)


# ---------- eigenvalues ----------

def test_eigenvalues_match_numpy(sm):
    G = nx.cycle_graph(12)
    expected = np.sort(np.linalg.eigvalsh(nx.to_numpy_array(G)))[::-1][:10]
    assert np.allclose(sm.eigenvalues(G), expected)


def test_eigenvalues_sorted_descending(sm, BA):
    ev = sm.eigenvalues(BA)
    assert ev.shape == (10,)
    assert (np.diff(ev) <= 0).all()


# ---------- lbd ----------

@pytest.mark.parametrize("G, expected", [
    (nx.star_graph(5), [1, 0, 0]),                # max centralization, no triangles, every edge pair shares the center
    (nx.complete_graph(4), [0, 1, 0]),            # all degrees equal, all triangles, no independent edge pairs
    (nx.Graph([(0, 1), (2, 3)]), [0, 0, 1]),      # 2K2: exactly one independent dipole, denominator 1
    (nx.path_graph(2), [0, 0, 0]),                # n < 3 and n < 4 guards
])
def test_lbd_known_graphs(sm, G, expected):
    assert np.allclose(sm.lbd(G), expected)


def test_lbd_diversity_matches_brute_force(sm):
    # diversity^2 * denom should equal a brute-force count of independent dipoles
    G = nx.gnp_random_graph(12, 0.3, seed=3)
    brute = sum(
        1 for (a, b), (c, d) in itertools.combinations(G.edges(), 2)
        if len({a, b, c, d}) == 4 and not any(G.has_edge(x, y) for x in (a, b) for y in (c, d))
    )
    denom = math.comb(6, 2) * math.comb(6, 2)
    assert sm.lbd(G)[2] ** 2 * denom == pytest.approx(brute)


def test_lbd_ignores_self_loops(sm):
    G = nx.star_graph(5)
    G.add_edge(0, 0)
    assert np.allclose(sm.lbd(G), [1, 0, 0])


# ---------- community_features ----------

def test_community_features_path_hand_check(sm):
    # community {5,6,7} in path(10) is the path 5-6-7, fringe nodes 4 and 8
    expected = [
        3, 2, 2, 2 / 3, 1 / 3, 0, 0,          # 1-7: n, m, diameter, density, conductance, transitivity, triangle density
        1, 1, 1.5, 2,                         # 8-11: shortest paths [1,1,2]
        2 / 3, 2 / 3, 2 / 3, 2 / 3,           # 12-15: edge betweenness
        0, 0, 0, 0.5, 1,                      # 16-20: node betweenness [0,1,0]
        1, 1, 1, 1, 1,                        # 21-25: alpha, fringe nodes 4 and 8
        1, 1, 1, 1.5, 2,                      # 26-30: beta, degrees [1,2,1]
        1 / 3,                                # 31: treesum, 1 tree / 3^1
        1, 1, 1, 5 / 4, 3 / 2,                # 32-36: Stephenson-Zelen information centrality n / sum_j R_ij = [1, 3/2, 1]
    ]
    assert np.allclose(sm._community_features(nx.path_graph(10), {5, 6, 7}), expected)


def test_community_features_clique(sm):
    f = sm._community_features(nx.complete_graph(6), {0, 1, 2, 3})
    assert f[5] == pytest.approx(1)    # transitivity
    assert f[6] == pytest.approx(1)    # triangle density
    assert f[30] == pytest.approx(1)   # treesum: K4 has all of K4's spanning trees


def test_community_features_treesum_matches_exact_count(sm, BA):
    C = sm._bfs_community(BA, 12, random.Random(0))
    exact = nx.number_of_spanning_trees(BA.subgraph(C)) / 12 ** 10
    assert sm._community_features(BA, C)[30] == pytest.approx(exact)


def test_community_features_string_labels(sm):
    G = nx.relabel_nodes(nx.path_graph(10), {i: f"n{i}" for i in range(10)})
    assert np.allclose(sm._community_features(G, {"n5", "n6", "n7"}),
                       sm._community_features(nx.path_graph(10), {5, 6, 7}))


def test_community_features_large_community_no_overflow(sm, BA):
    f = sm._community_features(BA, sm._bfs_community(BA, 200, random.Random(0)))
    assert f.shape == (36,)
    assert np.isfinite(f).all()


# ---------- community finders ----------

def is_consecutive(nodes):
    nodes = sorted(nodes)
    return nodes == list(range(nodes[0], nodes[0] + len(nodes)))


@pytest.mark.parametrize("finder", ["_bfs_community", "_ab_community", "_rw_community"])
def test_finder_size_and_connected(sm, BA, finder):
    rng = random.Random(0)
    for _ in range(50):
        C = getattr(sm, finder)(BA, 20, rng)
        assert len(C) == 20
        assert nx.is_connected(BA.subgraph(C))


@pytest.mark.parametrize("finder", ["_bfs_community", "_ab_community", "_rw_community"])
def test_finder_reproducible(sm, BA, finder):
    f = getattr(sm, finder)
    assert f(BA, 15, random.Random(7)) == f(BA, 15, random.Random(7))


@pytest.mark.parametrize("finder", ["_bfs_community", "_ab_community", "_rw_community"])
def test_finder_on_path_is_consecutive(sm, finder):
    rng = random.Random(0)
    for _ in range(20):
        assert is_consecutive(getattr(sm, finder)(nx.path_graph(10), 4, rng))


@pytest.mark.parametrize("finder", ["_bfs_community", "_ab_community", "_rw_community"])
def test_finder_component_too_small(sm, two_triangles, finder):
    # can't reach 5 nodes inside a triangle: returns fewer than k instead of hanging
    assert len(getattr(sm, finder)(two_triangles, 5, random.Random(0))) == 3


def test_bfs_star_from_center_takes_center_plus_leaves(sm):
    # star_graph(6): center 0, leaves 1-6; any BFS community of size 4 must include the center
    rng = random.Random(0)
    for _ in range(20):
        assert 0 in sm._bfs_community(nx.star_graph(6), 4, rng)


def test_ab_barbell_finds_a_clique(sm):
    # two K10s joined by one edge: swaps should pull the BFS set into one clique
    left, right = set(range(10)), set(range(10, 20))
    rng = random.Random(0)
    for _ in range(20):
        assert sm._ab_community(nx.barbell_graph(10, 0), 10, rng) in (left, right)


def test_ab_returns_alpha_beta_communities(sm, BA):
    # [7]: beta(S) (member neighbors in C, +1 self-loop) > alpha(S) (fringe neighbors in C)
    rng = random.Random(0)
    for _ in range(50):
        C = sm._ab_community(BA, 20, rng)
        beta = min(sum(u in C for u in BA.neighbors(w)) + 1 for w in C)
        alpha = max(sum(u in C for u in BA.neighbors(v)) for v in nx.node_boundary(BA, C))
        assert beta > alpha


def test_rw_isolated_start_node(sm):
    G = nx.empty_graph(1)
    assert sm._rw_community(G, 3, random.Random(0)) == {0}


def test_rwr_restart_one_never_leaves_start(sm, BA):
    # restart=1 jumps back every step, so only the start node is ever visited
    assert len(sm._rw_community(BA, 10, random.Random(0), restart=1.0)) == 1


def test_rwr_more_compact_than_rw(sm, BA):
    # [4]: RW15 "forms more compact sets" than RW0
    rw = sm.RW(BA, n_samples=100, k=20)[:, 2].mean()       # feature 3 = diameter
    rwr = sm.RWR(BA, n_samples=100, k=20)[:, 2].mean()
    assert rwr < rw


# ---------- sample_communities: BFS, AB, RW, RWR ----------

METHODS = ["BFS", "AB", "RW", "RWR"]


@pytest.mark.parametrize("method", METHODS)
def test_method_shape(sm, BA, method):
    assert getattr(sm, method)(BA, n_samples=30).shape == (30, 36)


@pytest.mark.parametrize("method", METHODS)
def test_method_reproducible_and_seeded(sm, BA, method):
    m = getattr(sm, method)
    assert np.array_equal(m(BA, n_samples=20, seed=0), m(BA, n_samples=20, seed=0))
    assert not np.array_equal(m(BA, n_samples=20, seed=0), m(BA, n_samples=20, seed=1))


@pytest.mark.parametrize("method", METHODS)
def test_method_too_small_graph_raises(sm, two_triangles, method):
    with pytest.raises(ValueError):
        getattr(sm, method)(two_triangles, n_samples=5, k=5)


@pytest.mark.parametrize("method", METHODS)
def test_method_handles_self_loops(sm, BA, method):
    G = BA.copy()
    G.add_edges_from((v, v) for v in range(0, 1000, 10))
    assert getattr(sm, method)(G, n_samples=10).shape == (10, 36)


@pytest.mark.parametrize("method", METHODS)
def test_method_all_features_finite(sm, BA, method):
    assert np.isfinite(getattr(sm, method)(BA, n_samples=30)).all()


# ---------- find_similarity: vector ----------

@pytest.mark.parametrize("method, kwargs", [
    ("netsimile", {}),
    ("degree", {}),
    ("density", {}),
    ("transitivity", {}),
    ("eigenvalues", {}),
    ("lbd", {}),
    ("infomap", {"infomap_type": "in"}),
    ("infomap", {"infomap_type": "known"}),
    ("infomap", {"infomap_type": "in_and_known"}),
])
def test_vector_same_graph_is_1(sm, s, BA, method, kwargs):
    assert s.find_similarity(BA, BA, "vector", getattr(sm, method), **kwargs) == pytest.approx(1)


@pytest.mark.xfail(reason="rw_distance shares its default rng, so the two graphs get different walks")
def test_vector_rw_distance_same_graph_is_1(sm, s, BA):
    assert s.find_similarity(BA, BA, "vector", sm.rw_distance) == pytest.approx(1)


def test_vector_scalar_measures_work(sm, s, G1):
    # degree/density/transitivity return 0-d arrays; atleast_1d makes canberra accept them
    s.find_similarity(G1, nx.cycle_graph(12), "vector", sm.density)


def test_vector_similar_graphs_score_higher(sm, s, BA, BA2, ER):
    assert s.find_similarity(BA, BA2, "vector", sm.netsimile) > s.find_similarity(BA, ER, "vector", sm.netsimile)


# ---------- find_similarity: svm ----------

# Tests SVM netsimile similarity measure. The SVM is trained on a sample of up to 300 nodes from each graph, and the similarity score is based on how well the SVM can distinguish between the two graphs. A score of 1 indicates that the graphs are indistinguishable, while a score of -1 indicates perfect separation.
def test_svm_perfect_separation_is_minus_1(sm, s):
    sim = s.find_similarity(nx.cycle_graph(300), nx.complete_graph(300), "svm", sm.netsimile, netsimile_type="svm")
    assert sim == pytest.approx(-1)


def test_svm_identical_neighborhoods_is_1(sm, s):
    # every node of a cycle has the same 7 features, so the SVM can't tell them apart
    sim = s.find_similarity(nx.cycle_graph(5), nx.cycle_graph(12), "svm", sm.netsimile, netsimile_type="svm")
    assert sim == pytest.approx(1)


def test_svm_same_graph_different_samples_is_high(sm, s, BA):
    F1, F2 = sm.netsimile(BA, "svm", seed=0), sm.netsimile(BA, "svm", seed=1)
    sim = s.find_similarity(F1, F2, "svm", lambda F: F)    # pass the sample matrices straight through
    assert sim > 0.9


@pytest.mark.parametrize("method, kwargs", [("netsimile", {"netsimile_type": "svm"})] + [(m, {"n_samples": 50}) for m in METHODS])
def test_svm_similar_graphs_score_higher(sm, s, BA, BA2, ER, method, kwargs):
    m = getattr(sm, method)
    same_model = s.find_similarity(BA, BA2, "svm", m, **kwargs)
    different_model = s.find_similarity(BA, ER, "svm", m, **kwargs)
    assert -1 <= different_model < same_model <= 1


def test_svm_small_graph_fewer_rows_than_folds(sm, s, G1):
    # G1 has 5 rows; n_splits drops to fit
    sim = s.find_similarity(G1, nx.cycle_graph(12), "svm", sm.netsimile, netsimile_type="svm")
    assert -1 <= sim <= 1


# ---------- find_similarity: match ----------
# NetSimile-Match uses the same 300 sampled neighborhoods as NetSimileSVM (paper Sec. 3), so netsimile_type="svm"

MATCH_METHODS = [("netsimile", {"netsimile_type": "svm"})] + [(m, {"n_samples": 50}) for m in METHODS]


def passthrough(F):
    # lets a test hand find_similarity sample matrices directly instead of graphs
    return F


def test_match_hand_computed(s):
    # one row [1, 1] vs [3, 1]: canberra = |1-3|/(1+3) + 0 = 0.5
    assert s.find_similarity(np.array([[1.0, 1.0]]), np.array([[3.0, 1.0]]), "match", passthrough) == pytest.approx(0.5)


def test_match_picks_least_cost_pairing(s):
    # rows swapped between F1 and F2: the best matching pairs each row with its copy, cost 0
    F1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    F2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert s.find_similarity(F1, F2, "match", passthrough) == pytest.approx(1)


def test_match_each_row_used_once(s):
    # both F1 rows are closest to F2's first row, but only one can have it
    F1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    F2 = np.array([[1.0, 1.0], [3.0, 1.0]])
    assert s.find_similarity(F1, F2, "match", passthrough) == pytest.approx(1 - (0 + 0.5) / 2)


def test_match_symmetric(sm, s, BA, ER):
    F1, F2 = sm.netsimile(BA, "svm"), sm.netsimile(ER, "svm")
    assert s.find_similarity(F1, F2, "match", passthrough) == pytest.approx(s.find_similarity(F2, F1, "match", passthrough))


@pytest.mark.parametrize("method, kwargs", MATCH_METHODS)
def test_match_same_graph_same_seed_is_1(sm, s, BA, method, kwargs):
    # identical rows each pair with their copy at cost 0 (unlike the SVM, see test_svm_same_graph_different_samples_is_high)
    assert s.find_similarity(BA, BA, "match", getattr(sm, method), **kwargs) == pytest.approx(1)


@pytest.mark.parametrize("method, kwargs", [("netsimile", {"netsimile_type": "svm"})] + [(m, {"n_samples": 300}) for m in METHODS])
def test_match_similar_graphs_score_higher(sm, s, BA, BA2, ER, method, kwargs):
    # uses the paper's 300 samples: at 50, RW and BFS get the order wrong for about 1 seed in 5
    m = getattr(sm, method)
    assert s.find_similarity(BA, BA2, "match", m, **kwargs) > s.find_similarity(BA, ER, "match", m, **kwargs)


@pytest.mark.parametrize("method, kwargs, d", [(m, kw, 7 if m == "netsimile" else 36) for m, kw in MATCH_METHODS])
def test_match_range(sm, s, BA, ER, method, kwargs, d):
    # canberra between d-dim vectors is in [0, d], so similarity is in [1 - d, 1]
    sim = s.find_similarity(BA, ER, "match", getattr(sm, method), **kwargs)
    assert 1 - d <= sim <= 1


def test_match_cycle_vs_complete_lower_than_cycle_vs_cycle(sm, s):
    m = sm.netsimile
    cyc_cyc = s.find_similarity(nx.cycle_graph(300), nx.cycle_graph(200), "match", m, netsimile_type="svm")
    cyc_complete = s.find_similarity(nx.cycle_graph(300), nx.complete_graph(300), "match", m, netsimile_type="svm")
    assert cyc_cyc == pytest.approx(1)      # every cycle node has the same 7 features
    assert cyc_complete < cyc_cyc


def test_match_unequal_row_counts(sm, s, G1, BA):
    # G1 has 5 rows, BA has 300: only 5 pairs are matched, the rest of BA's rows are ignored
    F1, F2 = sm.netsimile(G1, "svm"), sm.netsimile(BA, "svm")
    assert F1.shape[0] == 5 and F2.shape[0] == 300
    assert -6 <= s.find_similarity(G1, BA, "match", sm.netsimile, netsimile_type="svm") <= 1
