"""
Deterministic sampler that carves an "LJ2"-like subgraph out of SNAP's
com-LiveJournal undirected edge list (lj_whole.txt: 3,997,962 nodes,
34,681,189 edges), matching Table 2 of Soundarajan, Eliassi-Rad &
Gallagher, "A Guide to Selecting a Network Similarity Method":

    LJ2 [paper]: 500K nodes, 11M edges, avg deg 43, max deg 12,796,
        edge density 0.0001, transitivity 0.08, 1 connected component
    
    Output: 4999,990 nodes, 11,011,991 edges, avg degree 44.0, max deg 12,803,
        edge densitity 0.000088, transitivity 0.079, 1 connected compent

Method (no randomness, so every run gives the same graph):
  1. Load the edge list with NumPy (never touches networkx).
  2. BFS from the highest-degree node; keep the first M visited nodes.
  3. Within that ball, keep the K highest-degree nodes (degree measured in
     the full graph, ties broken by node ID).
  4. Take the induced subgraph and keep its largest connected component.

Defaults (seed = max-degree node, M = 730,000, K = 500,000) were tuned
so the result lands on LJ2's node count, edge count and max degree.

Usage:
    python sample_lj.py lj_whole.txt lj2_sample.txt      # write edge list
    # or from Python:
    from sample_lj import sample_lj
    G = sample_lj("lj_whole.txt", as_networkx=True)

Requires numpy, scipy, pandas (and networkx only if as_networkx=True).
Peak memory is roughly 3 GB while loading the full file.
"""
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components


def load_edges(path):
    df = pd.read_csv(path, sep="\t", comment="#", header=None,
                     dtype=np.int32, engine="c")
    return df.values


def sample_lj2(path, k=500_000, ball_size=730_000, seed_rank=0,
              as_networkx=False):
    """Return the sampled graph.

    seed_rank: 0 = BFS from the highest-degree node, 1 = second-highest, ...
    Returns an (m, 2) array of edges using the ORIGINAL node IDs, or a
    networkx.Graph if as_networkx=True.
    """
    edges = load_edges(path)
    # ids - sorted list of IDs that occur
    # inv - each entry of edges its position in the original list of edges
    ids, inv = np.unique(edges, return_inverse=True)   # sorted -> deterministic
    e = inv.reshape(-1, 2).astype(np.int32) # shape m x 2
    del edges, inv
    n = len(ids)

    A = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
                      shape=(n, n)).tocsr()
    del e
    # file lists each edge once with, u < v, need to add transpose to make symmetric, undirected graph
    A = (A + A.T).tocsr()
    # next line makes sure if there is a duplicate edge it doesn't sum to 2
    A.data[:] = 1
    # make matrix ordering deterministic so each time we run this we get the same result
    A.sort_indices()

    """
    Regarding the next uncommented line that calculates deg. Here's a toy example
    of why this works

          col: 0  1  2  3
row 0:   [ 0  1  1  0 ]   neighbors of 0: 1, 2
row 1:   [ 1  0  1  0 ]   neighbors of 1: 0, 2
row 2:   [ 1  1  0  1 ]   neighbors of 2: 0, 1, 3
row 3:   [ 0  0  1  0 ]   neighbors of 3: 2

indices = [1, 2,   0, 2,   0, 1, 3,   2]
           row 0   row 1   row 2      row 3

indptr  = [0, 2, 4, 7, 8]

Reading indptr:

Row 0 is indices[0:2], so neighbors 1 and 2.
Row 1 is indices[2:4], so neighbors 0 and 2.
Row 2 is indices[4:7], so neighbors 0, 1 and 3.
Row 3 is indices[7:8], so neighbor 2.

np.diff subtracts each element from the next one:

np.diff([0, 2, 4, 7, 8]) = [2-0, 4-2, 7-4, 8-7] = [2, 2, 3, 1]

Each difference is the length of one row’s slice, which is that node’s number of neighbors. Node 2 has degree 3 and the others match the picture too.

    """
    deg = np.diff(A.indptr)

    # Get node with highest degree (if seed rank is 0 else whatever seed rank is)
    seed = np.argsort(-deg, kind="stable")[seed_rank]
    # RUn BFS from seed and return every reachable node i the order visited
    order = breadth_first_order(A, int(seed), directed=False,
                                return_predecessors=False)

    # keep first ball_size nodes visited that are connected to seed
    ball = order[:ball_size]
    # keep the highest-degree nodes in ball
    keep = np.sort(ball[np.argsort(-deg[ball], kind="stable")[:k]])

    S = A[keep][:, keep].tocsr()
    # lab gives each node a component id
    _, lab = connected_components(S, directed=False)
    # np.bincount(lab).argmax() finds the largest component and lcc holds the positions of its nodes
    lcc = np.flatnonzero(lab == np.bincount(lab).argmax())

    # update S and keep with our new constraint of largest component nodes kept
    S = S[lcc][:, lcc]
    keep = keep[lcc]

    # S is symmetric so each edge appears twice, we want to keep just one in our representation
    # let's keep upper triangle so each edge appears once
    C = sp.triu(S, k=1).tocoo() # k=1 makes sure to not include the main diagonal
    # map labels back into original data format of "node_id1 node_id2"
    out = np.column_stack((ids[keep[C.row]], ids[keep[C.col]]))
    # take above output and sort by edges first point, then second, so output file is identical on every run
    out = out[np.lexsort((out[:, 1], out[:, 0]))]

    if as_networkx:
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(map(tuple, out.tolist()))
        return G
    return out


def table2_stats(edges):
    """Table-2 statistics computed with scipy (fast, low memory)."""
    ids, inv = np.unique(edges, return_inverse=True)
    e = inv.reshape(-1, 2)
    n = len(ids)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    d = np.diff(A.indptr)
    # triangles: orient each edge from lower to higher (degree, id) rank
    rank = np.empty(n, np.int64)
    # sort order by degree from lowest to highest
    rank[np.lexsort((np.arange(n), d))] = np.arange(n)

    # A stores each edge twice (u,v) and (v, u)
    C = A.tocoo()
    # mask to keep only copy from lower-ranked end point to higher
    m = rank[C.row] < rank[C.col]
    L = sp.csr_matrix((np.ones(m.sum()), (C.row[m], C.col[m])), shape=(n, n)) # directed

    # count triangles 
    """
       L @ L counts two step paths (e.g. a->x, x->c meaning a->x->c). However this may include
       paths that are reachable but not triangles

       To keep only triangles, we then do .multiply which is element-wise multiplication with the original L giving (L @ L).multiply(L).sum() (sum will count the number of triangles found)

       The above uses too much memory so we add chunking to fix this: rows i to j of L @ L is the same as L[i:j] @ L

    """
    tri = sum((L[i:i + 50_000] @ L).multiply(L[i:i + 50_000]).sum()
              for i in range(0, n, 50_000))
    # sum of i=1:n of degree choose 2
    triples = (d.astype(np.float64) * (d - 1)).sum() / 2
    ncc, lab = connected_components(A, directed=False)
    return {
        "nodes": n, "edges": len(e), "avg_degree": d.mean(),
        "max_degree": int(d.max()), "density": 2 * len(e) / (n * (n - 1)),
        "transitivity": 3 * tri / triples, "num_cc": ncc,
        "frac_in_lcc": np.bincount(lab).max() / n,
    }


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "lj2_whole.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "lj2_sample.txt"
    E = sample_lj2(src)
    with open(dst, "w") as f:
        f.write("# LJ2-like sample of lj2_whole (see sample_lj.py)\n")
        f.write(f"# Nodes: {len(np.unique(E))} Edges: {len(E)}\n")
        np.savetxt(f, E, fmt="%d", delimiter="\t")
    for key, val in table2_stats(E).items():
        print(f"{key:>13}: {val}")