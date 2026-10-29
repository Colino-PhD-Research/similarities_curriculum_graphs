from typing import Callable
import networkx as nx
from netrd.distance.netsimile import feature_extraction, graph_signature
import numpy as np
from scipy.spatial.distance import canberra
from scipy.stats import moment
from scipy.sparse.linalg import eigsh
import infomap
import random

class SimilarityMeasures:
    def __init__(self):
        pass

    def netsimile(self, G):
        # undirected, simple
        G = nx.Graph(G)  
        # length 35 = 7 features × 5 aggregates                    
        return graph_signature(feature_extraction(G))   

    # claude generated, hand-verified
    def rw_distance(self, G, ds=(10, 20, 50, 100), walks=100, rng=random.Random(0)):
        nodes = [v for v in G if G.degree(v) > 0]
        feats = []
        for d in ds:
            dists = []
            for _ in range(walks):
                start = cur = rng.choice(nodes)
                for _ in range(d):
                    cur = rng.choice(list(G.neighbors(cur)))
                dists.append(nx.shortest_path_length(G, start, cur))
            x = np.array(dists, dtype=float)
            feats += [np.median(x), x.mean(), x.var(), moment(x, 3), moment(x, 4)]
        return np.array(feats)    # length 20

    def infomap(self, G, type):
        # type must be "in", "known", or "in_and_known"
        pass

    def degree(self, G):
        return np.array(sum(d for n, d in G.degree()) / G.number_of_nodes())

    def density(self, G):
        return np.array(nx.density(G))

    def transitivity(self, G):
        return np.array(nx.transitivity(G))

    def eigenvalues(self, G, k=10):
        A = nx.adjacency_matrix(G).astype(float)
        eigenvalues = eigsh(A, k=k, which="LA", return_eigenvectors=False) # NOTE: assumes symmetric matrix, not true if we do directed graph
        return np.sort(eigenvalues)[::-1] # line suggestion from claude

    def lbd(self, G):
        pass

class Similarity:
    def __init__(self):
        pass

    # claude used for metho syntax and **kwargs
    def find_similarity(self, G1, G2, method: Callable[..., np.ndarray], **kwargs) -> float:
        # method is a SimilarityMeasures function, e.g. SimilarityMeasures().netsimile
        # kwargs are forwarded to method, e.g. type="in" for infomap
        # canberra needs 1-D vectors; scalar measures return 0-d arrays
        F1 = np.atleast_1d(method(G1, **kwargs))
        F2 = np.atleast_1d(method(G2, **kwargs))

        # canberra is in [0, len(F)], so divide by len to keep similarity in [0, 1]
        return 1 - canberra(F1, F2) # / len(F1) # suggested by claude but not in paper

    