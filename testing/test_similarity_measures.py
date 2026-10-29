import networkx as nx
from similarity_measures.similarity_measures import SimilarityMeasures, Similarity

# 1. Initialize an empty undirected graph
G1 = nx.Graph()

# 2. Add nodes individually or from a list
G1.add_node("A")
G1.add_nodes_from(["B", "C", "D", "E"])

# 3. Add edges (connections) between nodes
G1.add_edges_from([
    ("A", "B"), 
    ("B", "C"), 
    ("C", "D"), 
    ("D", "E"), 
    ("E", "A"), 
    ("A", "C")
])

# NOTE: can't test eigenvalue metric on G1 because graph too small

G2 = nx.cycle_graph(12)

sm = SimilarityMeasures()
vector = sm.eigenvalues(G2)
print(vector)

s = Similarity()
number = s.find_similarity(G1, G1, sm.rw_distance)
print(number)

print("hi")