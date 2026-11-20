"""
The 20 similarity methods of Soundarajan, Eliassi-Rad & Gallagher (2014), Table 1,
keyed by the names the paper uses.

FEATURE_METHODS:  vector-based, graph -> feature vector (similarity = 1 - Canberra)
PAIRWISE_METHODS: classifier- and matching-based, (G_ref, H) -> similarity score
                  (built from SAMPLERS + PAIRWISE_SPECS: which samples, scored how)
"""
from functools import partial

from similarity_measures.similarity_measures import (SimilarityMeasures, match_similarity,
                                                     svm_similarity)

sm = SimilarityMeasures()

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

# The samples each pairwise method is built on: graph -> sample matrix (one row per sampled
# node / community). Each sampler is seeded, so a graph always gets the same samples, and the
# SVM and matching versions of a method share them (e.g. BFS and BFS-Dist both use "BFS").
# netsimile's "svm" and "match" types return the same node sample, so either works here.
SAMPLERS = {
    "NetSimile": partial(sm.netsimile, netsimile_type="svm"),
    "AB": sm.AB, "BFS": sm.BFS, "RW": sm.RW, "RWR": sm.RWR,
}

# pairwise method name -> (key into SAMPLERS, scorer(F_ref, F_H) -> similarity)
PAIRWISE_SPECS = {"NetSimileSVM": ("NetSimile", svm_similarity),
                  "NetSimile-Dist": ("NetSimile", match_similarity)}
for _name in ["AB", "BFS", "RW", "RWR"]:
    PAIRWISE_SPECS[_name] = (_name, svm_similarity)              # classifier-based
    PAIRWISE_SPECS[f"{_name}-Dist"] = (_name, match_similarity)  # matching-based


def _pairwise(sampler, scorer):
    # same as Similarity.find_similarity(G1, G2, "svm" / "match", sampler)
    return lambda G1, G2: scorer(sampler(G1), sampler(G2))


# (G_ref, H) -> similarity, sampling both graphs on every call. cross_sectional_experiments.py
# builds cached versions from SAMPLERS and PAIRWISE_SPECS instead.
PAIRWISE_METHODS = {name: _pairwise(SAMPLERS[key], scorer)
                    for name, (key, scorer) in PAIRWISE_SPECS.items()}

# fixed order for heatmaps, grouped micro / mezzo / macro as in Table 1
METHOD_ORDER = [
    "NetSimile", "NetSimileSVM", "NetSimile-Dist",
    "d-RW-Dist", "IMIn", "IMKnown", "IMIn&Known",
    "AB", "BFS", "RW", "RWR", "AB-Dist", "BFS-Dist", "RW-Dist", "RWR-Dist",
    "Degree", "Density", "Transitivity", "Eigenvalues", "LBD",
]
