"""
Collection of similarity measures for graphs.
"""

from typing import Callable
import networkx as nx
from netrd.distance.netsimile import feature_extraction, graph_signature
import numpy as np
import scipy
from scipy.spatial.distance import canberra
from scipy.stats import moment
from scipy.sparse.linalg import eigsh
from infomap import Infomap
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import math
from collections import deque   # add with the other imports
from collections import Counter    # for counting each node's neighbors in S



class SimilarityMeasures:
    def __init__(self):
        pass

    def netsimile(self, G, netsimile_type="vector", seed=0):
        """NetSimile features for G (Berlingerio et al.), computed via netrd.

        G is converted to a simple undirected graph, then 7 local features are
        extracted per node.

        Args:
            G: networkx graph.
            netsimile_type: "vector" returns the aggregated graph signature,
                shape (35,) = 7 features x 5 aggregates (median, mean, std,
                skew, kurtosis). "svm" or "match" return the raw per-node
                feature rows for a random sample of up to 300 nodes,
                shape (min(300, n), 7).
            seed: seed for the node sample (ignored for "vector").

        Raises:
            Exception: if netsimile_type is not "vector", "svm", or "match".
        """
       
        G = nx.Graph(G)   # undirected, simple G
        f = feature_extraction(G) # Extracts 7 features per node, shape (n, 7)

        if netsimile_type not in ["vector", "svm", "match"]:
            raise Exception(f"Infomap feature type {netsimile_type} invalid. Must be vector, svm, match")
        
        # Returns a vector of length 35 = 7 features × 5 aggregates 
        if netsimile_type == "vector":                   
            return graph_signature(f)    # vector length 35

        # Returns a matrix of shape (min(300, n), 7) for a sample of up to 300 nodes (or all nodes if less)
        if netsimile_type in ["svm", "match"]:
            rng = random.Random(seed)
            idx = rng.sample(range(G.number_of_nodes()), min(300, G.number_of_nodes()))
            # feature_extraction rows are ordered by sorted(G.nodes()), so idx picks those nodes' rows
            # Returns a matrix of shape (min(n_samples, n), 7)
            return f[idx]  

        # Returns features for all nodes
        return f    

     
    # claude generated, hand-verified
    def rw_distance(self, G, ds=(10, 20, 50, 100), walks=100, rng=None):
        """Random-walk distance features: how far a d-step random walk ends from where it started.

        For each walk length d, runs `walks` random walks from uniformly chosen
        non-isolated nodes and records the shortest-path distance between each
        walk's start and end node. The distances are summarized by 5 statistics:
        median, mean, variance, 3rd central moment, 4th central moment.

        Args:
            G: networkx graph (assumed undirected; a walk only stays within its
                start node's connected component).
            ds: walk lengths to use.
            walks: number of walks per walk length.
            rng: random.Random used for start nodes and steps; defaults to
                random.Random(0) so results are reproducible.

        Returns:
            np.ndarray of length 5 * len(ds) (20 with the defaults), ordered by
            walk length, then by statistic.
        """
        rng = rng or random.Random(0)

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

    # claude generated, hand verified
    def infomap(self, G, infomap_type, seed=42, trials=10):
        """Community features from a two-level Infomap partition of G.

        G is converted to a simple undirected graph with integer node ids, then
        partitioned with Infomap. Two node-averaged scores are computed:
            IMIn: fraction of a node's neighbors that are in its own community
                (nodes with no neighbors are skipped).
            IMKnown: fraction of a node's community (excluding itself) that it
                is adjacent to (nodes in singleton communities are skipped).

        Args:
            G: networkx graph.
            infomap_type: "in" returns [IMIn], "known" returns [IMKnown],
                "in_and_known" returns [IMIn, IMKnown].
            seed: Infomap random seed.
            trials: number of Infomap runs; the best partition is kept.

        Returns:
            np.ndarray of length 1 or 2, values in [0, 1].

        Raises:
            Exception: if infomap_type is not "in", "known", or "in_and_known".
        """

        def infomap_partition(G: nx.Graph, seed: int = 42, trials: int = 10) -> np.ndarray:
            """Run two-level, undirected Infomap. Returns module label per node index (0..n-1)."""
            im = Infomap(f"--two-level --silent --seed {seed} --num-trials {trials}")
            mapping = im.add_networkx_graph(G)          # {infomap_id: original node}
            im.run()
            labels = np.full(G.number_of_nodes(), -1, dtype=np.int64)
            for infomap_id, module_id in im.get_modules().items():
                labels[mapping[infomap_id]] = module_id
            assert (labels >= 0).all(), "some nodes received no module from Infomap"
            return labels

        if infomap_type not in ["in", "known", "in_and_known"]:
            raise Exception(f"Infomap feature type {infomap_type} invalid. Must be in, known, or in_and_known")
        
        # TODO: this assumes undirected graph
        # Simple, undirected, no self-loops, integer node ids 0..n-1
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
    
        labels = infomap_partition(G, seed, trials)
    
        edges = np.array(G.edges(), dtype=np.int64).reshape(-1, 2)
        u, v = edges[:, 0], edges[:, 1]
        same = labels[u] == labels[v]
    
        deg = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)
        # k_in[i] = number of i's neighbors in i's own community
        k_in = np.bincount(u[same], minlength=n) + np.bincount(v[same], minlength=n)
        comm_size = np.bincount(labels)[labels]           # |C(i)| for each node i
    
        # IMIn: fraction of each node's neighbors in its community, averaged over nodes
        has_nbrs = deg > 0
        im_in = float(np.mean(k_in[has_nbrs] / deg[has_nbrs]))
    
        # IMKnown: fraction of the node's community that is adjacent to it, averaged
        # (denominator |C|-1 excludes the node itself; singleton communities skipped)
        non_singleton = comm_size > 1
        im_known = float(np.mean(k_in[non_singleton] / (comm_size[non_singleton] - 1)))
    
        if infomap_type == "in":
            return np.array([im_in])
        elif infomap_type == "known":
            return np.array([im_known])
        else:
            return np.array([im_in, im_known])  

    def degree(self, G):
        """Average node degree of G (2|E| / |V| for an undirected graph), as a 0-d np.ndarray."""
        return np.array(sum(d for n, d in G.degree()) / G.number_of_nodes())

    def density(self, G):
        """Edge density of G: |E| / (|V|(|V|-1)/2) for an undirected graph, in [0, 1], as a 0-d np.ndarray."""
        return np.array(nx.density(G))

    def transitivity(self, G):
        """Global clustering coefficient of G: 3 * triangles / connected triples, in [0, 1], as a 0-d np.ndarray."""
        return np.array(nx.transitivity(G))

    def eigenvalues(self, G, k=10):
        """The k largest adjacency-matrix eigenvalues of G, sorted in descending order.

        Uses eigsh, so G must be undirected (symmetric adjacency) and k < |V|.
        Returns an np.ndarray of length k.
        """
        A =nx.adjacency_matrix(G).astype(float)
        eigenvalues = eigsh(A, k=k, which="LA", return_eigenvectors=False) # NOTE: assumes symmetric matrix, not true if we do directed graph
        return np.sort(eigenvalues)[::-1] # line suggestion from claude

    def lbd(self, G):
        """Leadership, bonding, diversity (LBD) features of G.

        G is converted to a simple undirected graph (self-loops removed).
            Leadership: Freeman degree centralization; 1 for a star, 0 when all degrees are equal.
            Bonding: global transitivity (fraction of connected triples that close into triangles).
            Diversity: sqrt(#independent dipoles / C(n/2, 2)^2); a dipole is a pair of
                edges with no shared node and no edge between them.

        Returns:
            np.ndarray [leadership, bonding, diversity]. Graphs with fewer than 3
            (leadership) or 4 (diversity) nodes get 0 for that feature.
        """
        def leadership(G: nx.Graph) -> float:
            """Freeman degree centralization: sum(d_max - d_i) / ((n-1)(n-2)). Star graph -> 1."""
            n = G.number_of_nodes()
            if n < 3:
                return 0.0
            deg = np.fromiter((d for _, d in G.degree()), dtype=np.int64, count=n)
            return float((deg.max() - deg).sum() / ((n - 1) * (n - 2)))


        def bonding(G: nx.Graph) -> float:
            """6 * triangles / paths of length two  ==  global transitivity."""
            return nx.transitivity(G)


        def count_independent_dipoles(G: nx.Graph) -> int:
            """Number of unordered pairs of edges {a,b}, {c,d} that share no vertex and have
            no edge between them (an induced 2K2; equivalently an induced square in the complement).
            
            TODO: assumes simple, undirected graph
            """
            m = G.number_of_edges()
            adj = {v: set(G[v]) for v in G}
            total = 0
            for u, v in G.edges():
                S = adj[u] | adj[v]                     # contains u and v themselves
                touching = sum(len(adj[w]) for w in S) - sum(len(adj[w] & S) for w in S) // 2
                total += m - touching                   # edges with no endpoint in N[u] ∪ N[v]
            return total // 2                           # each pair counted once from each edge


        def diversity(G: nx.Graph) -> float:
            """
            Claude generated and hand verified
            D = sqrt(#independent dipoles) / C(n/2, 2)^2
            https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5590790 
            ½ · (n/2) · (n/2 − 1) is C(n/2, 2)
            """
            n = G.number_of_nodes()
            if n < 4:
                return 0.0
            denom = math.comb(n // 2, 2) * math.comb(n - n // 2, 2)   # floor/ceil for odd n
            return math.sqrt(count_independent_dipoles(G) / denom)

        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return np.array([leadership(G), bonding(G), diversity(G)])

    def _community_features(self, G, C):
        """
        36 structural features of community C (a set of nodes in G), from [4] Table 1.
        These are the features used for classifier-based similarity (SVM) and matching-based similarity (Hungarian algorithm).
        Returns a length-36 vector.
        """
        C = set(C)
        G.remove_edges_from(nx.selfloop_edges(G))
        H = G.subgraph(C)

        # calculate the number of nodes and edges in the subgraph H
        n, m = H.number_of_nodes(), H.number_of_edges()

        def q_max(x):
            return [*np.percentile(x, [25, 50, 75]), max(x)]          # 4 values: 25th percentile, median, 75th percentile, max

        def min_q_max(x):
            return [min(x), *np.percentile(x, [25, 50, 75]), max(x)]  # 5 values

        feats = [] # empty list to hold the features

        # Compute feature 1-6
        feats += [n, m, nx.diameter(H), nx.density(H)] # number of nodes, number of edges, diameter, density
        feats += [m / sum(G.degree(v) for v in C)]  # fraction of edges in H relative to the sum of degrees of nodes in C
        feats += [nx.transitivity(H)] 

        # Feature 7: number of triangles in H divided by the number of possible triangles in a complete graph of n nodes
        triangles = (sum(nx.triangles(H).values()) / 3) /math.comb(n, 3) if n >= 3 else 0.0
        feats += [triangles]

        # Feature 8-11: Compute shortest path lengths
        # each pair once: skip targets already used as a source (and u itself)
        # counting both directions duplicates values and shifts the interpolated quartiles
        dists, done = [], set()
        for u, targets in nx.all_pairs_shortest_path_length(H):
            done.add(u)
            dists += [d for v, d in targets.items() if v not in done]
        feats += q_max(dists)

        # 12-15: edge betweenness - using nx function to compute edge betweenness
        feats += q_max(list(nx.edge_betweenness_centrality(H).values()))

        # 16-20: node betweenness - uses nx function to compute node betweenness
        feats += min_q_max(list(nx.betweenness_centrality(H).values()))

        # 21-25: α, for each fringe node, its number of neighbors in C (uses G)
        fringe = nx.node_boundary(G, C) # get fringe nodes of C in G
        alphas = [sum(1 for u in G.neighbors(v) if u in C) for v in fringe] # for each fridge node, count neighbors in C
        feats += min_q_max(alphas) if alphas else [0] * 5

        # 26-30: β, for each member, its number of neighbors in C - simple degree of each node in H counting only neighbors in H
        feats += min_q_max([d for _, d in H.degree()])

        # 31: treesum, spanning trees of H / spanning trees of K_n (n^(n-2), Cayley)
        # matrix-tree theorem: #trees = det of the Laplacian with one row and column removed
        # done in logs because n^(n-2) overflows for large n
        # Unclear if papers use log ratio or ratio; we use log ratio to avoid overflow for large n. If H is disconnected, the number of spanning trees is 0, and we set the feature to 0.0.
        # NOTE: Do we need to check if H is connected? If not, the number of spanning trees is 0, and log(0) is undefined. We might want to handle that case explicitly.
        L = nx.laplacian_matrix(H, weight=None).toarray().astype(float)
        sign, log_trees = np.linalg.slogdet(L[1:, 1:])
        feats += [math.exp(log_trees - (n - 2) * math.log(n)) if sign > 0 else 0.0] #log ratio of spanning trees of H to spanning trees of K_n, using Cayley's formula for K_n

        # 32-36: Stephenson and Zelen's information centrality (H must be connected)
        info = [n * c for c in nx.information_centrality(H).values()] # multiply by n to get the sum of the information centrality of each node in H, as per the definition in [4] Table 1
        feats += min_q_max(info)

        return np.array(feats)    # length 36


    def _bfs_community(self, G, k, rng):
        """
        Sample a community of exactly k nodes using BFS.
        
        Args:
            G: networkx graph (assumed undirected).
            k: desired community size. Note: K is not explicitly mentioned in the papers
            rng: random.Random instance for reproducibility.    
            
        Returns:
            A set of exactly k nodes forming a connected community.
        """

        start = rng.choice(list(G.nodes())) # choose a random starting node
        visited = {start} # initialize visited set with the starting node
        queue = deque([start]) # initialize queue with the starting node

        while queue and len(visited) < k: # continue until we have k nodes or the queue is empty
            node = queue.popleft() # get the next node from the queue
            neighbors = list(G.neighbors(node))
            rng.shuffle(neighbors) # shuffle neighbors to ensure randomness in BFS traversal
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= k: # stop if we have reached k nodes
                        break

        return visited # NOTE: this may return fewer than k nodes if the graph is too small or disconnected; the caller should handle this case.


    def _ab_community(self, G, k, rng, max_swaps=1000):
        # Placeholder for AB community sampling method
        # This is a placeholder implementation; the actual AB method from He et al. is more complex
        C = self._bfs_community(G, k, rng)  # Using BFS to get starting nodes; this is from the Abrahao et al. paper, but the actual AB method from He et al does not 
        if len(C) < k:
            return C

        def swapping(G, S, rng, max_swaps=1000):
            """Algorithm 2: while beta(S) < alpha(S), swap a max-alpha outsider in
            for a min-beta member. Returns S plus alpha, A, beta, B for Algorithm 1."""
             
            for _ in range(max_swaps):  # cap swaps so the loop can't run forever
                cnt = Counter(v for u in S for v in G[u] if v != u)  # cnt[v] = number of v's neighbors in S (ignoring self-loops in G)
                out = {v: c for v, c in cnt.items() if v not in S}   # keep only non-members: their alpha(v) values
                alpha = max(out.values(), default=0)                 # alpha(S): most links any outsider has into S (0 if none)
                A = [v for v, c in out.items() if c == alpha]        # A: outsiders achieving alpha(S) (Alg. 2, line 2)
                beta = min(cnt[w] + 1 for w in S)                    # beta(S): fewest in-S links of any member, +1 for self-loop
                B = [w for w in S if cnt[w] + 1 == beta]             # B: members achieving beta(S) (Alg. 2, line 3)
                if beta >= alpha:                                    # Alg. 2, line 1: stop once beta(S) is no longer < alpha(S)
                    break                                            # exit with the current state
                S = (S - {rng.choice(B)}) | {rng.choice(A)}          # Alg. 2, lines 4-5: swap a random b in B out, a random a in A in
            return S, alpha, A, beta, B 

        for _ in range(max_swaps):                           # iteration cap (the paper used 1,000)
            C, alpha, A, beta, B = swapping(G, C, rng)      # line 3: run Algorithm 2; lines 4-5: get A and B
            if beta > alpha:                                # line 2: beta(S) > alpha(S) means S is a community
                return C                                    # line 19: done
            rng.shuffle(A)                                  # randomize which a gets picked below
            rng.shuffle(B)                                  # randomize which b gets picked below
            pair = next(((a, b) for a in A for b in B       # line 6: look for a pair (a, b), a in A, b in B,
                        if not G.has_edge(a, b)), None)    #         that are NOT adjacent (None if no such pair)
            if pair:                                        # line 6: such a pair exists
                C = (C - {pair[1]}) | {pair[0]}             # lines 7-8: swap b out, a in
                continue                                    # next iteration (the paper's while loop)
            a = next((a for a in A                          # line 9: look for an a in A that is adjacent
                    if not any(G.has_edge(a, x)           #         to no other member of A
                                for x in A if x != a)), None)
            if a is not None:                               # line 9: such an a exists
                C = C | {a}                                 # lines 10-11: add a to C
                continue                                    # next iteration (the paper's while loop)
            b = next((b for b in B                          # line 12: look for a b in B that is adjacent
                    if not any(G.has_edge(b, x)           #          to no other member of B
                                for x in B if x != b)), None)
            if b is not None:                               # line 12: such a b exists
                C = C - {b}                                 # lines 13-14: remove b from C
            else:                                           # line 15: none of the above applied
                C = C | set(A)                              # line 16: add all of A to C
        return None                                         # no community found within max_iter


    def _rw_community(self, G, k, rng, restart=0.0, max_steps=None):
        """
        Random-walk community of size k ([4] Sec. 3.2.1): walk from a random node until
        k distinct nodes are visited. restart=0 is RW (RW0), restart=0.15 is RWR (RW15).
        """
        start = cur = rng.choice(list(G.nodes())) # choose a random starting node
        visited = {start}
        max_steps = max_steps or 100 * k          # stops a walk trapped in a component smaller than k

        for _ in range(max_steps):
            if len(visited) == k: # K nodes have been visited, return the community
                break
            if rng.random() < restart:
                cur = start                       # RWR: jump back to the start node
                continue
            neighbors = list(G.neighbors(cur))
            if not neighbors:
                break                             # isolated start node, nowhere to go
            cur = rng.choice(neighbors)           # uniform step to a neighbor
            visited.add(cur)

        return visited         # fewer than k if the walk got stuck; caller rejects


    def _sample_communities(self, G, finder, n_samples=300, k=10, seed=0, **finder_kwargs):
        """
        Shared sampling loop for BFS, AB, RW, RWR.
        finder is a community finder, e.g. self._bfs_community, called as finder(G, k, rng, **finder_kwargs).
        Keeps communities of exactly k nodes that are connected ([4] discards multi-component ones).
        Returns a matrix of shape (n_samples, 36), one row of community_features per community.
        """
        G = nx.Graph(G)  # Ensure the graph is undirected and simple
        G.remove_edges_from(nx.selfloop_edges(G))
        rng = random.Random(seed)

        communities = []

        attempts = 0
        while len(communities) < n_samples:
            community = finder(G, k, rng, **finder_kwargs)
            if len(community) == k and nx.is_connected(G.subgraph(community)):
                communities.append(community)
            attempts += 1
            if attempts > 100 * n_samples:  # Avoid infinite loop if not enough communities can be found
                raise ValueError(f"could only find {len(communities)} valid communities of size {k} with {finder.__name__}")

        return np.array([self._community_features(G, community) for community in communities]) # returns a matrix of shape (n_samples, 36), one row of community_features per community


    def BFS(self, G, n_samples=300, k=10, seed=0):
        return self._sample_communities(G, self._bfs_community, n_samples, k, seed)

    def AB(self, G, n_samples=300, k=10, seed=0):
        return self._sample_communities(G, self._ab_community, n_samples, k, seed)

    def RW(self, G, n_samples=300, k=10, seed=0):
        return self._sample_communities(G, self._rw_community, n_samples, k, seed)

    def RWR(self, G, n_samples=300, k=10, seed=0):
        # [4]: restart to the start node with 0.15 probability at each step
        return self._sample_communities(G, self._rw_community, n_samples, k, seed, restart=0.15)

    

class Similarity:
    def __init__(self):
        pass

    # claude used for metho syntax and **kwargs
    def find_similarity(self, G1, G2, type, method: Callable[..., np.ndarray], **kwargs) -> float:
        # method is a SimilarityMeasures function, e.g. SimilarityMeasures().netsimile
        # kwargs are forwarded to method, e.g. type="in" for infomap
        # canberra needs 1-D vectors; scalar measures return 0-d arrays
        F1 = np.atleast_1d(method(G1, **kwargs))
        F2 = np.atleast_1d(method(G2, **kwargs))

        def svm(F1, F2, n_splits=5, seed=0): # I dont think these are updatable from the outside, but they could be if we wanted to make them more flexible
            """
            Classifier-based similarity (paper Sec. 3): can an SVM tell F1's samples from F2's?
            F1, F2 are sample matrices, one row per neighborhood/community, e.g. netsimile(G, "svm")
            Returns 1 - canberra(P1, P2) in [-1, 1]: 1 = indistinguishable, -1 = perfectly separated
            Note that the paper does not define the splits used for CV
            """
            X = np.vstack([F1, F2])
            y = np.array([0] * len(F1) + [1] * len(F2))

            # every row is predicted by a model that never trained on it
            # scaler is inside the pipeline so it's only fit on the training folds
            # paper doesn't give folds, kernel, or scaling: 5 folds, RBF kernel, standardized features
            # StratifiedKFold needs at least n_splits rows from each graph
            n_splits = min(n_splits, len(F1), len(F2))
            folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            pred = cross_val_predict(make_pipeline(StandardScaler(), SVC()), X, y, cv=folds)

            # paper builds these per cross-validation round; we pool predictions over all rounds
            # P1 = [fraction of F1's rows predicted F1, fraction predicted F2], same for P2
            P1 = np.array([np.mean(pred[y == 0] == 0), np.mean(pred[y == 0] == 1)])
            P2 = np.array([np.mean(pred[y == 1] == 0), np.mean(pred[y == 1] == 1)])
            P1 = np.atleast_1d(P1)
            P2 = np.atleast_1d(P2)

            return P1, P2

        def match(F1, F2):
            """
            Matching-based similarity (paper Sec. 3): least-cost pairing of F1's rows with F2's rows.
            Edge weight = Canberra distance; similarity = 1 - mean cost of the matched pairs.
            """
            cost = scipy.spatial.distance.cdist(F1, F2, "canberra") # step 1: 300X300 cost matrix representing the bipartite graph of all possible pairings
            rows, cols = scipy.optimize.linear_sum_assignment(cost)     # step 2: Hungarian algorithm finds the minimum-cost matching of rows to columns
            return 1 - cost[rows, cols].mean()           # step 3: similarity = 1 - mean cost of the matched pairs, in [0, 1]

        
        if type == "svm":
            # P1 = [fraction of F1's rows predicted G1, fraction predicted G2], same for P2
            P1, P2 = svm(F1, F2)
            return 1 - canberra(P1, P2)  # / len(P1) # suggested by claude but not in paper

        if type == "match":
            return match(F1, F2)  # already in [0, 1]
        
        # canberra is in [0, len(F)]
        return 1 - canberra(F1, F2) # / len(F1) # divide by len to keep similarity in [0, 1]suggested by claude but not in paper

    