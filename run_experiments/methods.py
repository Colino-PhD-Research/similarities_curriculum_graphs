"""
The 20 similarity methods of Soundarajan, Eliassi-Rad & Gallagher (2014), Table 1,
keyed by the names the paper uses.

FEATURE_METHODS:  vector-based, graph -> feature vector (similarity = 1 - Canberra)
PAIRWISE_METHODS: classifier- and matching-based, (G_ref, H) -> similarity score
"""
from functools import partial

from similarity_measures.similarity_measures import SimilarityMeasures, Similarity

sm, s = SimilarityMeasures(), Similarity()

FEATURE_METHODS = {
    "NetSimile":    sm.netsimile,
    "d-RW-Dist":    sm.rw_distance,
    "IMIn":         partial(sm.infomap, infomap_type="in"),
    "IMKnown":      partial(sm.infomap, infomap_type="known"),
    "IMIn&Known":   partial(sm.infomap, infomap_type="in_and_known"),
    "Degree":       sm.degree,
    "Density":      sm.density,
    "Transitivity": sm.transitivity,
    "Eigenvalues":  sm.eigenvalues,
    "LBD":          sm.lbd,
}


def _svm(method, **kw):
    return lambda G1, G2: s.find_similarity(G1, G2, "svm", method, **kw)


def _match(method, **kw):
    return lambda G1, G2: s.find_similarity(G1, G2, "match", method, **kw)


PAIRWISE_METHODS = {
    "NetSimileSVM":   _svm(sm.netsimile, netsimile_type="svm"),
    "NetSimile-Dist": _match(sm.netsimile, netsimile_type="match"),
}
for _name, _finder in {"AB": sm.AB, "BFS": sm.BFS, "RW": sm.RW, "RWR": sm.RWR}.items():
    PAIRWISE_METHODS[_name] = _svm(_finder)              # classifier-based
    PAIRWISE_METHODS[f"{_name}-Dist"] = _match(_finder)  # matching-based

# fixed order for heatmaps, grouped micro / mezzo / macro as in Table 1
METHOD_ORDER = [
    "NetSimile", "NetSimileSVM", "NetSimile-Dist",
    "d-RW-Dist", "IMIn", "IMKnown", "IMIn&Known",
    "AB", "BFS", "RW", "RWR", "AB-Dist", "BFS-Dist", "RW-Dist", "RWR-Dist",
    "Degree", "Density", "Transitivity", "Eigenvalues", "LBD",
]
