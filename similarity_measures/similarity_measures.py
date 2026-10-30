from typing import Callable
import networkx as nx
from netrd.distance.netsimile import feature_extraction, graph_signature
import numpy as np
from scipy.spatial.distance import canberra
from scipy.stats import moment
from scipy.sparse.linalg import eigsh
from infomap import Infomap
import random
import math

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

    