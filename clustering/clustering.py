"""
Complete-linkage hierarchical clustering of similarity methods
(Soundarajan, Eliassi-Rad & Gallagher, "A Guide to Selecting a Network Similarity Method").

Input is a precomputed, symmetric distance matrix, e.g. the methods x methods Kendall-Tau
distance from rankings.ranking_comparisons.pairwise_correlations(...)["kendall"] or
average_matrix(results, "kendall").

Output is a linkage matrix in scipy's format, so scipy.cluster.hierarchy.dendrogram,
fcluster, leaves_list, etc. work on it directly. This is run for each network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

class CompleteLinkageClustering:
    def __init__(self, D: np.ndarray):
        """
        Complete-linkage agglomerative clustering.

        D   : (n, n) symmetric distance matrix with zero diagonal.
        """
        self.D = np.asarray(D, dtype=float)
        self.labels = list(D.index) if hasattr(D, "index") else list(range(len(D)))
        self.n = len(D)
        if self.D.shape != (self.n, self.n) or not np.allclose(self.D, self.D.T):
            raise ValueError("D must be a square, symmetric distance matrix")
        if np.isnan(self.D).any():
            raise ValueError("D contains NaN")


        # cluster id -> (member indices); dist[a][b] = complete-linkage distance between clusters
        #self.members = {i: [i] for i in range(self.n)} # Initially each cluster starts with one member (one similarity method)
        #self.dist = {i: {j: self.D[i, j] for j in range(self.n) if j != i} for i in range(self.n)} # distances between clusters
        #self.Z = np.zeros((self.n - 1, 4))



    # ---------------------------------------------------------------------------
    # 1. The algorithm
    # ---------------------------------------------------------------------------
    def complete_linkage(self) -> np.ndarray:
        """
        Complete-linkage agglomerative clustering.
    
        D   : (n, n) symmetric distance matrix with zero diagonal.
    
        Returns a SciPy-style linkage matrix Z of shape (n-1, 4). Row t says that
        at step t clusters Z[t,0] and Z[t,1] merged at distance Z[t,2], producing
        a cluster of size Z[t,3] whose id is n + t. Leaves have ids 0..n-1.
        Ties between equally close cluster pairs are broken at random.
        """
        
        n = len(self.D)
        
        rng = np.random.default_rng()

        # cluster id -> (member indices); dist[a][b] = complete-linkage distance between clusters
        members = {i: [i] for i in range(n)} # Initially each cluster starts with one member (one similarity method)
        dist = {(i, j): self.D[i, j] for i in range(n) for j in range(i + 1, n)} # distances between clusters
        Z = np.zeros((n - 1, 4))

        for step in range(n - 1):
            # --- find the closest pair of active clusters ---
            best = min(dist.values())
            tol = 1e-12
            candidates = [pair for pair, d in dist.items() if d <= best + tol]
            a, b = rng.choice(candidates) 
            
            # --- merge them into a new cluster ---
            new = n + step
            members[new] = members.pop(a) + members.pop(b)
            Z[step] = [a, b, best, len(members[new])]

            # Lance-Williams update for complete linkage: d(a u b, c) = max(d(a, c), d(b, c))
            # --- update distances: COMPLETE linkage = take the max ---
            # (Lance-Williams update: d(k, a∪b) = max(d(k, a), d(k, b)).
            #  Single linkage would use min, average linkage a size-weighted mean.)
            for k in members:
                if k == new:
                    continue
                d_ka = dist.pop((min(k, a), max(k, a)))
                d_kb = dist.pop((min(k, b), max(k, b)))
                dist[(k, new)] = max(d_ka, d_kb)
            dist.pop((a, b))

        return Z



    # ---------------------------------------------------------------------------
    # 2. Turning a tree into something you can count
    # ---------------------------------------------------------------------------
    # Note: The paper says "treat each dendrogram as a tree “without information about clustering order"; Claude implemented this as the set of leaf-sets of every internal node. Two dendrograms with the same nested groups get the same signature even if their merges happened in a different order or height.
    @staticmethod
    def _tree_signature(Z, labels):
        """
        The tree 'without information about clustering order': the set of leaf-sets
        of every internal node. Two dendrograms with the same nested groups get the
        same signature even if their merges happened in a different order or height.
        """
        n = len(labels)
        leaves = {i: frozenset([labels[i]]) for i in range(n)}
        groups = []
        for t, (a, b, _, _) in enumerate(Z):
            leaves[n + t] = leaves[int(a)] | leaves[int(b)]
            groups.append(leaves[n + t])
        return frozenset(groups)

    # ---------------------------------------------------------------------------
    # 3. The paper's procedure for one reference network
    # ---------------------------------------------------------------------------
    def modal_tree(self, runs=1000):
        """
        Cluster `runs` times with random tie-breaking; return the most common tree.

        Returns (signature, fraction of runs that produced it, number of distinct
        trees seen, Z), where Z is one linkage matrix that produced the modal tree,
        so it can be plotted with real merge heights.
        """
        counts = Counter()
        example_Z = {}  # signature -> first Z that produced it
        for _ in range(runs):
            Z = self.complete_linkage()
            sig = self._tree_signature(Z, self.labels)
            counts[sig] += 1
            example_Z.setdefault(sig, Z)
        signature, freq = counts.most_common(1)[0]
        return signature, freq / runs, len(counts), example_Z[signature]





