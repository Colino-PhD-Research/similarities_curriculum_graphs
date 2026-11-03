"""
Compare network-similarity methods through the rankings they produce
(Soundarajan, Eliassi-Rad & Gallagher, "A Guide to Selecting a Network Similarity Method",
Section 4 and the cross-sectional / longitudinal experiments of Section 5).

Pipeline for one reference network G_r and comparison networks H_1..H_k:
    scores      methods x networks     similarity of each H_i to G_r under each method
    kendall     methods x methods      Kendall-Tau distance (0 = same ranking, 0.5 = unrelated, 1 = reversed)
    kendall_p   methods x methods      p-value of each Kendall-Tau correlation
    ndcg        methods x methods      nDCG, averaged over using each ranking as the 'true' one
    consensus   networks               Kemeny-Young consensus ranking
    to_consensus methods               Kendall-Tau distance of each method from the consensus
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import canberra
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import ndcg_score

FeatureFn = Callable[[nx.Graph], np.ndarray]          # e.g. SimilarityMeasures().netsimile
PairwiseFn = Callable[[nx.Graph, nx.Graph], float]    # e.g. a future SVM- or matching-based method


# --------------------------------------------------------------------------- scores

# hand-verified
def similarity_scores(
    reference: nx.Graph,
    comparisons: Mapping[str, nx.Graph],
    feature_methods: Mapping[str, FeatureFn] | None = None,
    pairwise_methods: Mapping[str, PairwiseFn] | None = None,
) -> pd.DataFrame:
    """Similarity of every comparison network to the reference, one row per method.

    feature_methods: vector-based methods (graph -> feature vector). Each graph's features are
        computed once, and similarity is 1 - Canberra(F_ref, F_H), the same formula as
        Similarity.find_similarity. Use functools.partial for keyword arguments, e.g.
        partial(sm.infomap, infomap_type="in").
    pairwise_methods: methods that score a pair of graphs directly (classifier- and
        matching-based methods), called as fn(reference, H).
    """
    rows: dict[str, dict[str, float]] = {}
    for name, fn in (feature_methods or {}).items():
        f_ref = np.atleast_1d(fn(reference)).astype(float)
        rows[name] = {
            h: 1.0 - canberra(f_ref, np.atleast_1d(fn(H)).astype(float))
            for h, H in comparisons.items()
        }
    for name, fn in (pairwise_methods or {}).items():
        rows[name] = {h: float(fn(reference, H)) for h, H in comparisons.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[list(comparisons)]


# hand-verified
def rankings(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank position of each network under each method (1 = most similar; ties share the best rank)."""
    return scores.apply(lambda row: pd.Series(rankdata(-row.to_numpy(), method="min"),
                                              index=row.index), axis=1).astype(int)


# --------------------------------------------------------------------------- pairwise comparisons

# hand-verified
def _ndcg(true_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
    """nDCG of one ranking against another treated as ground truth. Relevance is the rank
    position under the 'true' method: k for its most similar network down to 1 for its least."""
    if np.isnan(true_scores).any() or np.isnan(predicted_scores).any():
        return np.nan
    relevance = rankdata(true_scores)          # average ranks for ties
    return float(ndcg_score([relevance], [predicted_scores]))


# hand-verified
def pairwise_correlations(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Kendall-Tau distance, its p-value, and symmetric nDCG for every pair of methods."""
    names = list(scores.index)
    S = scores.to_numpy(dtype=float)
    m = len(names)
    dist = np.zeros((m, m))
    pval = np.full((m, m), np.nan)
    ndcg = np.ones((m, m))
    for a, b in combinations(range(m), 2):
        tau, p = kendalltau(S[a], S[b])
        dist[a, b] = dist[b, a] = (1.0 - tau) / 2.0
        pval[a, b] = pval[b, a] = p
        ndcg[a, b] = ndcg[b, a] = (_ndcg(S[a], S[b]) + _ndcg(S[b], S[a])) / 2.0

    def frame(X):
        return pd.DataFrame(X, index=names, columns=names)

    return {"kendall": frame(dist), "kendall_p": frame(pval), "ndcg": frame(ndcg)}


# --------------------------------------------------------------------------- Kemeny-Young consensus

# TODO: come back to
def kemeny_consensus(scores: pd.DataFrame) -> pd.Series:
    """Exact Kemeny-Young consensus of the methods' rankings, solved as an integer program.

    P[i, j] = number of methods ranking network i strictly above network j. The consensus is the
    ordering maximizing the sum of P[i, j] over pairs it places i above j. Variables y_ij (i < j)
    are 1 when i is above j; the triangle constraints rule out cycles, so any feasible solution
    is a total order. Returns the consensus position of each network (1 = most similar), sorted.
    """
    items = list(scores.columns)
    k = len(items)
    S = scores.to_numpy(dtype=float)
    P = (S[:, :, None] > S[:, None, :]).sum(axis=0)

    if k == 1:
        return pd.Series([1], index=items, name="consensus_rank")

    pairs = list(combinations(range(k), 2))
    var = {pair: idx for idx, pair in enumerate(pairs)}
    # maximize sum (P_ij - P_ji) y_ij  ->  minimize the negative
    c = np.array([-(P[i, j] - P[j, i]) for i, j in pairs], dtype=float)

    rows, cols, vals, upper = [], [], [], []
    r = 0
    for i, j, l in combinations(range(k), 3):
        # no cycle i>j>l>i:  y_ij + y_jl - y_il <= 1 ; no cycle i>l>j>i:  -y_ij - y_jl + y_il <= 0
        for coeffs, ub in (((1, 1, -1), 1), ((-1, -1, 1), 0)):
            for pair, coef in zip(((i, j), (j, l), (i, l)), coeffs):
                rows.append(r); cols.append(var[pair]); vals.append(coef)
            upper.append(ub)
            r += 1

    constraints = []
    if r:
        A = coo_matrix((vals, (rows, cols)), shape=(r, len(pairs)))
        constraints.append(LinearConstraint(A, -np.inf, np.array(upper, dtype=float)))

    res = milp(c, constraints=constraints, integrality=np.ones(len(pairs)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"Kemeny MILP failed: {res.message}")

    y = np.round(res.x).astype(int)
    wins = np.zeros(k, dtype=int)                 # number of networks each one is placed above
    for (i, j), idx in var.items():
        if y[idx]:
            wins[i] += 1
        else:
            wins[j] += 1
    position = k - wins                           # 1 = top of the consensus
    return pd.Series(position, index=items, name="consensus_rank").sort_values()


# hand-verified
def distance_to_consensus(scores: pd.DataFrame, consensus: pd.Series) -> pd.Series:
    """Kendall-Tau distance between each method's ranking and the consensus, smallest first."""
    # similarity- higher is more similar
    # consensus - lower is more similar
    target = -consensus.reindex(scores.columns).to_numpy(dtype=float)   #  make so higher = more similar
    out = {}
    for name, row in scores.iterrows():
        tau, _ = kendalltau(row.to_numpy(dtype=float), target)
        out[name] = (1.0 - tau) / 2.0
    return pd.Series(out, name="kendall_to_consensus").sort_values()


# --------------------------------------------------------------------------- one reference network

# hand-verified
@dataclass
class RankingComparison:
    scores: pd.DataFrame        # methods x networks
    kendall: pd.DataFrame       # methods x methods, Kendall-Tau distance
    kendall_p: pd.DataFrame     # methods x methods, p-values
    ndcg: pd.DataFrame          # methods x methods
    consensus: pd.Series        # network -> consensus position
    to_consensus: pd.Series     # method -> Kendall-Tau distance from consensus

    @property
    def ranks(self) -> pd.DataFrame:
        return rankings(self.scores)

    def significant(self, alpha: float = 0.1) -> pd.DataFrame:
        """Boolean methods x methods matrix of correlations significant at alpha (paper: 0.1)."""
        return self.kendall_p < alpha

    def closest_to_consensus(self, n: int = 5) -> list[str]:
        """The n methods nearest the consensus: one column of the paper's Table 5."""
        return list(self.to_consensus.index[:n])


# hand-verified
def compare_methods(
    reference: nx.Graph,
    comparisons: Mapping[str, nx.Graph],
    feature_methods: Mapping[str, FeatureFn] | None = None,
    pairwise_methods: Mapping[str, PairwiseFn] | None = None,
) -> RankingComparison:
    scores = similarity_scores(reference, comparisons, feature_methods, pairwise_methods)
    corr = pairwise_correlations(scores)
    consensus = kemeny_consensus(scores)
    return RankingComparison(scores=scores, consensus=consensus,
                             to_consensus=distance_to_consensus(scores, consensus), **corr)


# --------------------------------------------------------------------------- experiment helpers

def make_baselines(G: nx.Graph, frac: float = 0.05, seed: int = 0) -> dict[str, nx.Graph]:
    """Section 5.3 baselines: delete a random frac of edges, and rewire a random frac of edges
    with degree-preserving double edge swaps (each swap rewires two edges)."""
    rng = random.Random(seed)
    m = G.number_of_edges()

    deleted = G.copy()
    deleted.remove_edges_from(rng.sample(list(G.edges()), int(round(frac * m))))

    rewired = G.copy()
    nswap = max(1, math.ceil(frac * m / 2))
    nx.double_edge_swap(rewired, nswap=nswap, max_tries=100 * nswap, seed=seed)
    return {"deleted": deleted, "rewired": rewired}


def cross_sectional_experiment(
    networks: Mapping[str, nx.Graph],
    feature_methods: Mapping[str, FeatureFn] | None = None,
    pairwise_methods: Mapping[str, PairwiseFn] | None = None,
    frac: float = 0.05,
    seed: int = 0,
) -> dict[str, RankingComparison]:
    """Use each network in turn as the reference; rank the others plus its two baselines."""
    results = {}
    for ref_name, G_ref in networks.items():
        comparisons = {name: H for name, H in networks.items() if name != ref_name}
        for kind, H in make_baselines(G_ref, frac, seed).items():
            comparisons[f"{ref_name}-{kind}"] = H
        results[ref_name] = compare_methods(G_ref, comparisons, feature_methods, pairwise_methods)
    return results


def _upper(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy()[np.triu_indices(len(df), k=1)]


def summarize(results: Mapping[str, RankingComparison], alpha: float = 0.1) -> pd.DataFrame:
    """Statistics pooled over every method pair (upper triangle) of every reference network,
    as in Section 5.3: mean/std Kendall-Tau distance, mean/std nDCG, fraction significant."""
    kd = np.concatenate([_upper(r.kendall) for r in results.values()])
    nd = np.concatenate([_upper(r.ndcg) for r in results.values()])
    pv = np.concatenate([_upper(r.kendall_p) for r in results.values()])
    return pd.DataFrame({
        "mean": [np.nanmean(kd), np.nanmean(nd), np.nanmean(pv < alpha)],
        "std": [np.nanstd(kd), np.nanstd(nd), np.nan],
    }, index=["kendall_distance", "ndcg", f"fraction_significant_p<{alpha}"])


def average_matrix(results: Mapping[str, RankingComparison], field: str = "kendall") -> pd.DataFrame:
    """Method x method matrix averaged over reference networks (e.g. RW vs RWR across all networks)."""
    frames = [getattr(r, field) for r in results.values()]
    return sum(frames) / len(frames)


def consensus_table(results: Mapping[str, RankingComparison], n: int = 5) -> pd.DataFrame:
    """Table 5: the n methods closest to the consensus, one column per reference network."""
    return pd.DataFrame({ref: r.closest_to_consensus(n) for ref, r in results.items()})


def consensus_distance_table(results: Mapping[str, RankingComparison]) -> pd.DataFrame:
    """Figure 7: methods x reference networks, Kendall-Tau distance from each consensus."""
    return pd.DataFrame({ref: r.to_consensus for ref, r in results.items()})


# --------------------------------------------------------------------------- plotting

def plot_heatmap(matrix: pd.DataFrame, title: str = "", order: list[str] | None = None,
                 vmin: float = 0.0, vmax: float = 1.0, cmap: str = "jet", ax=None):
    """Heatmap in the style of the paper's figures. Pass order to fix the method order."""
    import matplotlib.pyplot as plt

    if order is not None:
        matrix = matrix.loc[order]                      # reorder rows (methods)
        if set(order) == set(matrix.columns):           # square method x method matrix
            matrix = matrix[order]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax)
    ax.figure.tight_layout()
    return ax


def plot_significance(result: RankingComparison, alpha: float = 0.1, title: str = "",
                      order: list[str] | None = None, ax=None):
    """Black = significant at alpha, white = not, as in the technical report's figures."""
    return plot_heatmap(result.significant(alpha).astype(float), title=title, order=order,
                        cmap="gray_r", ax=ax)


# --------------------------------------------------------------------------- example

if __name__ == "__main__":
    from functools import partial
    from similarity_measures import SimilarityMeasures

    sm = SimilarityMeasures()
    feature_methods = {
        "NetSim": sm.netsimile,
        "d-RWDist": sm.rw_distance,
        "IMIn": partial(sm.infomap, infomap_type="in"),
        "IMKnown": partial(sm.infomap, infomap_type="known"),
        "IM-In&Known": partial(sm.infomap, infomap_type="in_and_known"),
        "Degs": sm.degree,
        "Dens": sm.density,
        "Trans": sm.transitivity,
        "Eigs": sm.eigenvalues,
        "LBD": sm.lbd,
    }

    networks = {   # replace with Grad, Undergrad, Amazon, DBLP, LJ1, LJ2, Email
        "ER": nx.gnp_random_graph(300, 0.05, seed=1),
        "BA": nx.barabasi_albert_graph(300, 7, seed=2),
        "WS": nx.watts_strogatz_graph(300, 14, 0.1, seed=3),
        "SBM": nx.stochastic_block_model([100, 100, 100], [[0.12, 0.01, 0.01],
                                                           [0.01, 0.12, 0.01],
                                                           [0.01, 0.01, 0.12]], seed=4),
    }

    results = cross_sectional_experiment(networks, feature_methods)
    print(summarize(results), "\n")
    print(consensus_table(results), "\n")

    import matplotlib.pyplot as plt
    plot_heatmap(results["BA"].kendall, "BA: Kendall-Tau distances")
    plot_heatmap(results["BA"].ndcg, "BA: nDCG")
    plot_significance(results["BA"], title="BA: significant at p = 0.1")
    plt.show()