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


class SimilarityMeasures:
    def __init__(self):
        pass

    def netsimile(self, G, netsimile_type="vector", seed=0):
        # undirected, simple
        G = nx.Graph(G)  
        f = feature_extraction(G)

        if netsimile_type not in ["vector", "svm", "match"]:
            raise Exception(f"Infomap feature type {netsimile_type} invalid. Must be vector, svm, match")
        
        # length 35 = 7 features × 5 aggregates 
        if netsimile_type == "vector":                   
            return graph_signature(f)    # length 35

        if netsimile_type in ["svm", "match"]:
            rng = random.Random(seed)
            idx = rng.sample(range(G.number_of_nodes()), min(300, G.number_of_nodes()))
            # feature_extraction rows are ordered by sorted(G.nodes()), so idx picks those nodes' rows
            # shape (min(n_samples, n), 7)
            return f[idx]  

        return f    

     
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

    # claude generated, hand verified
    def infomap(self, G, infomap_type, seed=42, trials=10):
        # type must be "in", "known", or "in_and_known"
        
        def infomap_partition(G: nx.Graph, seed: int = 42, trials: int = 10) -> np.ndarray:
            """Run two-level, undirected Infomap. Returns module label per node index (0..n-1)."""
            im = Infomap(f"--two-level --silent --seed {seed} --num-trials {trials}")
            mapping = im.add_networkx_graph(G)          # {infomap_id: original node}
            im.run()
            labels = np.empty(G.number_of_nodes(), dtype=np.int64)
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

    def community_features(self, G, C):
        """
        36 structural features of community C (a set of nodes in G), from [4] Table 1.
        Returns a length-36 vector.
        """
        H = G.subgraph(C)
        # calculate the number of nodes and edges in the subgraph H
        n, m = H.number_of_nodes(), H.number_of_edges()

        def q_max(x):
            return [*np.percentile(x, [25, 50, 75]), max(x)]          # 4 values

        def min_q_max(x):
            return [min(x), *np.percentile(x, [25, 50, 75]), max(x)]  # 5 values

        feats = []

        # Compute feature 1-5
        feats += [n, m, nx.diameter(H), nx.density(H)]
        feats += [m / sum(G.degree(v) for v in C)]  
        feats += [nx.transitivity(H)]

        # compute triangles
        triangles = (sum(nx.triangles(H).values()) / 3) /math.comb(n, 3) if n >= 3 else 0.0
        feats += [triangles]

        # Compute shortest path lengths
        # each pair once: skip targets already used as a source (and u itself)
        # counting both directions duplicates values and shifts the interpolated quartiles
        dists, done = [], set()
        for u, targets in nx.all_pairs_shortest_path_length(H):
            done.add(u)
            dists += [d for v, d in targets.items() if v not in done]
        feats += q_max(dists)

        # edge betweenness
        # 12-15: edge betweenness
        feats += q_max(list(nx.edge_betweenness_centrality(H).values()))

        # 16-20: node betweenness
        feats += min_q_max(list(nx.betweenness_centrality(H).values()))

        # 21-25: α, for each fringe node, its number of neighbors in C (uses G)
        fringe = nx.node_boundary(G, C)
        alphas = [sum(1 for u in G.neighbors(v) if u in C) for v in fringe]
        feats += min_q_max(alphas) if alphas else [0] * 5

        # 26-30: β, for each member, its number of neighbors in C
        feats += min_q_max([d for _, d in H.degree()])

        # 31: treesum, spanning trees of H / spanning trees of K_n (n^(n-2), Cayley)
        # matrix-tree theorem: #trees = det of the Laplacian with one row and column removed
        # done in logs because n^(n-2) overflows for large n
        L = nx.laplacian_matrix(H).toarray().astype(float)
        sign, log_trees = np.linalg.slogdet(L[1:, 1:])
        feats += [math.exp(log_trees - (n - 2) * math.log(n)) if sign > 0 else 0.0]

        # 32-36: Stephenson and Zelen's information centrality (H must be connected)
        feats += min_q_max(list(nx.information_centrality(H).values()))

        return np.array(feats)    # length 36


    def bfs_community(self, G, k, rng):

        start = rng.choice(list(G.nodes()))
        visited = {start}
        queue = deque([start])

        while queue and len(visited) < k:
            node = queue.popleft()
            neighbors = list(G.neighbors(node))
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if len(visited) >= k:
                        break
        return visited


    def sample_communities(self, G, finder, n_samples, k, seed, **finder_kwargs):
        """
        Shared sampling loop for BFS, AB, RW, RWR.
        finder is a community finder, e.g. self.bfs_community, called as finder(G, k, rng, **finder_kwargs).
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

        return np.array([self.community_features(G, community) for community in communities])

    def BFS(self, G, n_samples=100, k=10, seed=0):
        return self.sample_communities(G, self.bfs_community, n_samples, k, seed)

    def ab_community(self, G, k, rng, max_swaps=1000):
        # Placeholder for AB community sampling method
        C = self.bfs_community(G, k, rng)  # Using BFS to start
        if len(C) < k:
            return C

        for _ in range(max_swaps):
            fridge = nx.node_boundary(G, C)
            if not fridge:
                break

            beta = {w: sum(1 for u in G.neighbors(w) if u in C) + 1 for w in C}
            alpha = {v: sum(1 for u in G.neighbors(v) if u in C) + 1 for v in fridge}
            beta_S, alpha_S = min(beta.values()), max(alpha.values())

            if beta_S >= alpha_S:
                break  # No more swaps possible

            # ties broken randomly ([7]: "pick a vertex a ∈ A and a vertex b ∈ B")
            b = rng.choice([w for w in C if beta[w] == beta_S])
            a = rng.choice([v for v in fridge if alpha[v] == alpha_S])
            C.remove(b)
            C.add(a)

        return C

    def AB(self, G, n_samples=100, k=10, seed=0):
        return self.sample_communities(G, self.ab_community, n_samples, k, seed)

    def rw_community(self, G, k, rng, restart=0.0, max_steps=None):
        """
        Random-walk community of size k ([4] Sec. 3.2.1): walk from a random node until
        k distinct nodes are visited. restart=0 is RW (RW0), restart=0.15 is RWR (RW15).
        """
        start = cur = rng.choice(list(G.nodes()))
        visited = {start}
        max_steps = max_steps or 100 * k          # stops a walk trapped in a component smaller than k

        for _ in range(max_steps):
            if len(visited) == k:
                break
            if rng.random() < restart:
                cur = start                       # RWR: jump back to the start node
                continue
            neighbors = list(G.neighbors(cur))
            if not neighbors:
                break                             # isolated start node, nowhere to go
            cur = rng.choice(neighbors)           # uniform step to a neighbor
            visited.add(cur)

        return visited                            # fewer than k if the walk got stuck; caller rejects

    def RW(self, G, n_samples=100, k=10, seed=0):
        return self.sample_communities(G, self.rw_community, n_samples, k, seed)

    def RWR(self, G, n_samples=100, k=10, seed=0):
        # [4]: restart to the start node with 0.15 probability at each step
        return self.sample_communities(G, self.rw_community, n_samples, k, seed, restart=0.15)

    

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

        def svm(F1, F2, n_splits=5, seed=0):
            """
            Classifier-based similarity (paper Sec. 3): can an SVM tell F1's samples from F2's?
            F1, F2 are sample matrices, one row per neighborhood/community, e.g. netsimile(G, "svm")
            Returns 1 - canberra(P1, P2) in [-1, 1]: 1 = indistinguishable, -1 = perfectly separated
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
            cost = scipy.spatial.distance.cdist(F1, F2, "canberra")          # step 1
            rows, cols = scipy.optimize.linear_sum_assignment(cost)     # step 2
            return 1 - cost[rows, cols].mean()           # step 3

        
        if type == "svm":
            # kwargs belong to method, not svm
            P1, P2 = svm(F1, F2)
            return 1 - canberra(P1, P2)  # / len(P1) # suggested by claude but not in paper

        if type == "match":
            return match(F1, F2)  # already in [0, 1]
        
        # canberra is in [0, len(F)], so divide by len to keep similarity in [0, 1]
        return 1 - canberra(F1, F2) # / len(F1) # suggested by claude but not in paper

    