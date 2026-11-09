"""
Deterministic sampler that carves an "LJ1"-like subgraph out of SNAP's
soc-LiveJournal1 directed edge list (soc-LiveJournal1.txt: 4,847,571 nodes,
68,993,773 edges), matching Table 2 of Soundarajan, Eliassi-Rad &
Gallagher, "A Guide to Selecting a Network Similarity Method":

    LJ1 [paper]: 500K nodes, 11M edges, avg deg 43, max deg 16,365,
        edge density 0.0001, transitivity 0.04, 1 connected component

    Output: 498,546 nodes, 10,682,120 edges, avg degree 42.9, max deg 16,348,
        edge density 0.000086, transitivity 0.045, 1 connected component

Why not lj_whole.txt: its max degree is 14,815 < 16,365, and sampling can
only lower a node's degree. soc-LiveJournal1 (treated as undirected, self
loops dropped) has one node of degree 20,333, so LJ1 must contain that node
with ~80% of its neighbors.

Method (seeded RNG, so every run gives the same graph):
  1. Load the edge list with NumPy, symmetrize, drop self loops.
  2. BFS from the highest-degree node; keep the first M visited nodes.
  3. Within that ball, draw K nodes without replacement with probability
     proportional to degree**beta (degree measured in the full graph).
  4. Take the induced subgraph and keep its largest connected component.

Unlike lj2_sample.py, step 3 is a weighted random draw instead of top-K by
degree: top-K makes the sample too dense (avg deg ~50) by the time the hub
is trimmed down to 16K neighbors. M sets the hub's degree, beta the density.

Defaults (seed = max-degree node, M = 635,000, K = 500,000, beta = 0.35,
rng_seed = 0) were tuned so the result lands on LJ1's node count, average
degree and max degree.

Usage:
    python lj1_sample.py soc-LiveJournal1.txt lj1_sample.txt   # write edge list
    # or from Python:
    from lj1_sample import sample_lj1
    G = sample_lj1("soc-LiveJournal1.txt", as_networkx=True)

Requires numpy, scipy, pandas (and networkx only if as_networkx=True).
Peak memory is roughly 6 GB while loading the full file.
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, connected_components

try:
    from load_data.lj2_sample import load_edges, table2_stats
except ImportError:
    from lj2_sample import load_edges, table2_stats


def sample_lj1(path, k=500_000, ball_size=635_000, beta=0.35, seed_rank=0,
               rng_seed=0, as_networkx=False):
    """Return the sampled graph.

    seed_rank: 0 = BFS from the highest-degree node, 1 = second-highest, ...
    beta: 0 = uniform draw from the ball, larger = favor high-degree nodes
    Returns an (m, 2) array of edges using the ORIGINAL node IDs, or a
    networkx.Graph if as_networkx=True.
    """
    edges = load_edges(path)
    # ids - sorted list of IDs that occur
    # inv - each entry of edges its position in the original list of edges
    ids, inv = np.unique(edges, return_inverse=True)   # sorted -> deterministic
    e = inv.reshape(-1, 2).astype(np.int32)
    del edges, inv
    n = len(ids)

    A = sp.coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])),
                      shape=(n, n)).tocsr()
    del e
    # file lists each edge once with, u < v, need to add transpose to make symmetric, undirected graph
    A = (A + A.T).tocsr()      # directed -> undirected
    # next line makes sure if there is a duplicate edge it doesn't sum to 2
    A.data[:] = 1

    A.setdiag(0)               # LiveJournal1 has self loops
    A.eliminate_zeros() # needed to make sure storage holds only real nonzeros after assignment

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

    # weighted random sampling without replacement (based on the Efraimidis-Spirakis method):
    # keep the k largest u**(1/w), computed as log(u)/w
    # this results in k nodes being kept our of the total in ball and while the pick
    # is random, a high-degree node is more likely to be kept
    rng = np.random.default_rng(rng_seed)
    # key is a score for each node
    # numerator, draw one random number (u) between 0 and 1 for each node in ball
    # denominator each node's weight (w) -> it's degree in full graph raised to power beta
    # higher degree -> larger key
    key = np.log(rng.random(len(ball))) / deg[ball].astype(np.float64) ** beta 
    # sort highest to lowest
    keep = np.sort(ball[np.argsort(-key, kind="stable")[:k]])

    S = A[keep][:, keep].tocsr()
    # lab gives each node a component id
    _, lab = connected_components(S, directed=False)
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


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "lj1_whole.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "lj1_sample.txt"
    E = sample_lj1(src)
    with open(dst, "w") as f:
        f.write("# LJ1-like sample of lj1_whole (see lj1_sample.py)\n")
        f.write(f"# Nodes: {len(np.unique(E))} Edges: {len(E)}\n")
        np.savetxt(f, E, fmt="%d", delimiter="\t")
    for key, val in table2_stats(E).items():
        print(f"{key:>13}: {val}")
