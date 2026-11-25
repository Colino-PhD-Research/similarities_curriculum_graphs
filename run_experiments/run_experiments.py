"""
Cross-sectional (§5.3) and longitudinal (§5.4) experiments from Soundarajan, Eliassi-Rad &
Gallagher 2014, end to end.

Cross-sectional (default): for each reference network, score every other network plus the
reference's two baselines (5% edges deleted, 5% rewired) with all 20 methods, then compute
Kendall-Tau, p-values, nDCG and the Kemeny consensus. Afterwards: Table 5, Figure 7, heatmaps,
the 1000-run modal dendrogram per reference, and the clusters that recur across references
(Table 4).

Longitudinal (--longitudinal): each dataset is a time series of snapshots; one reference
snapshot per dataset is compared against every other snapshot plus its two baselines, with
the same analysis. Also writes tables/references.csv, Figure 22 and the top-5 overlap
(Figure 11). See the "Longitudinal experiment" section below for the 10 datasets.

Each reference's (or dataset's) RankingComparison is pickled as soon as it finishes, and
finished ones are skipped on rerun, so an interrupted run can be resumed.

Run from the repo root:
    python -m run_experiments.run_experiments --toy     # 4 synthetic graphs, ~5 min
    python -m run_experiments.run_experiments           # amazon dblp lj1 lj2 email
    python -m run_experiments.run_experiments --networks amazon email --out small
    python -m run_experiments.run_experiments --network-set my_package.my_module:my_function

    python -m run_experiments.run_experiments --longitudinal --toy
    python -m run_experiments.run_experiments --longitudinal \\
        --reference replies=2009-06-14 --reference retweets=2009-07 --reference yahoo=2008-04-08
    python -m run_experiments.run_experiments --longitudinal --auto-reference \\
        --datasets yahoo yahoo-4day

The cross-sectional graphs come from a "network set": any function returning
{name: networkx.Graph} (see NETWORK_SETS below). Everything is written under
<repo root>/results/<out>/ (default results/cross_sectional/ for the paper's set, results/toy/
with --toy, otherwise results/<set name>/; results/longitudinal/ or results/longitudinal_toy/
with --longitudinal).
"""
import argparse
import importlib
import inspect
import json
import os
import pickle
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # save figures, never open windows (must be set before pyplot is imported)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from clustering.clustering import CompleteLinkageClustering
from clustering.utils import plot_dendrogram
from load_data.load_data import load_cross_sectional_network, longitudinal_networks, snapshots
from rankings.ranking_comparisons import (consensus_distance_table, consensus_table,
                                          cross_sectional_reference, plot_heatmap,
                                          plot_significance)
from run_experiments.methods import FEATURE_METHODS, METHOD_ORDER, PAIRWISE_SPECS, SAMPLERS

# every output goes under <repo root>/results/, wherever the script is run from
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ---------------------------------------------------------------------------------------------
# Progress output: every line starts with the clock time, and flush=True makes lines show up
# immediately, including in a log file when run with nohup ... > log
# ---------------------------------------------------------------------------------------------
def log(msg):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def fmt_duration(seconds):
    """e.g. 3725 -> '1h 02m 05s', 65 -> '1m 05s', 4.2 -> '4.2s'"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def toy_networks():
    """Four small synthetic graphs with different structure, for a quick end-to-end test.

    Erdos-Renyi (random), Barabasi-Albert (hubs), Watts-Strogatz (high clustering, short paths)
    and powerlaw-cluster (hubs + clustering). Seeded, so every run gives the same graphs.
    """
    return {
        "ER": nx.gnp_random_graph(400, 0.03, seed=1),
        "BA": nx.barabasi_albert_graph(400, 6, seed=2),
        "WS": nx.watts_strogatz_graph(400, 12, 0.1, seed=3),
        "PL": nx.powerlaw_cluster_graph(400, 6, 0.3, seed=4),
    }

# networks used as both references and comparisons (keys of load_data.cross_sectional_files)
# add "grad", "undergrad" once they have loaders (see docs/reproducing_experiments.md §2.3)
CROSS_SECTIONAL = ["amazon", "dblp", "lj1", "lj2", "email"]


def cross_sectional_networks(names=CROSS_SECTIONAL):
    """Load the networks used in Soundarajan, Eliassi-Rad & Gallagher 2014, §5.3."""
    networks = {}
    for i, name in enumerate(names, 1):
        start = time.perf_counter()
        networks[name] = load_cross_sectional_network(name)   # lj1 / lj2 take ~20 s each
        G = networks[name]
        log(f"loaded {name} ({i}/{len(names)}): {G.number_of_nodes():,} nodes, "
            f"{G.number_of_edges():,} edges in {fmt_duration(time.perf_counter() - start)}")
    return networks


# ---------------------------------------------------------------------------------------------
# Network sets: which graphs the experiment runs on.
#
# A network set is any function that returns {name: networkx.Graph}. Every graph is used as a
# reference in turn and compared with all the others. To use your own graphs, either
#   - add the function here and run with --network-set <key>, or
#   - leave it in any module and run with --network-set some.module:function_name, or
#   - call run_experiment(your_function(), out_dir) from Python.
# If the function takes a `names` argument, --networks passes the chosen names to it (so only
# those are loaded); otherwise --networks just keeps those names from what it returns.
# ---------------------------------------------------------------------------------------------
NETWORK_SETS = {
    "paper": cross_sectional_networks,
    "toy": toy_networks,
}

# default output folder (under results/) per set; any other set writes to results/<its name>/
DEFAULT_OUT = {"paper": "cross_sectional", "toy": "toy"}


def resolve_network_set(spec):
    """Turn a --network-set value into (function, default output folder name).
    spec is a key of NETWORK_SETS or an import path "package.module:function"."""
    if spec in NETWORK_SETS:
        return NETWORK_SETS[spec], DEFAULT_OUT.get(spec, spec)
    if ":" not in spec:
        raise SystemExit(f"unknown network set {spec!r}: use one of {list(NETWORK_SETS)} "
                         "or an import path like my_package.my_module:my_function")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name), func_name


class FeatureCache:
    """Computes each method's features (or samples) for each graph once, then reuses them.

    Without it, every network's features are recomputed for every reference it is compared
    against, and pairwise methods resample both graphs on every call (and BFS / BFS-Dist etc.
    draw the same samples twice). All methods are seeded, so reusing features gives exactly
    the same scores.

    Graphs are told apart by id(). The networks in `networks` live for the whole run, so their
    entries are kept across references. A reference's two baselines are new graphs each time,
    so their entries are dropped by new_reference() (they're only compared with that
    reference); each entry also holds the graph itself, so its id can't be reused by another
    graph while the entry exists.
    """

    def __init__(self, networks):
        self.names = {id(G): name for name, G in networks.items()}
        self.kept = {}        # (key, id(G)) -> (G, features), for graphs in `networks`
        self.baselines = {}   # same, for the current reference's baselines

    def new_reference(self):
        self.baselines.clear()

    def name(self, G):
        return self.names.get(id(G), "baseline")

    def get(self, key, fn, G):
        """fn(G), computed only the first time; returns (features, was_cached)."""
        store = self.kept if id(G) in self.names else self.baselines
        k = (key, id(G))
        if k in store:
            return store[k][1], True
        features = fn(G)
        store[k] = (G, features)
        return features, False


def cached_methods(tag, cache):
    """Versions of all 20 methods that read features from `cache`, and print a line after
    every call: which method (i of 20), which graph it scored, how long it took, and
    "(cached)" when no features had to be computed.

    Vector methods are called once on the reference and once per comparison graph; pairwise
    (SVM / matching) methods once per comparison graph, as fn(reference, H).
    """
    n_methods = len(FEATURE_METHODS) + len(PAIRWISE_SPECS)

    def report(i, name, G, start, cached):
        note = " (cached)" if cached else ""
        log(f"{tag}   method {i:>2}/{n_methods} {name:<14} on {cache.name(G):<10} "
            f"{fmt_duration(time.perf_counter() - start)}{note}")

    def vector_method(i, name, fn):
        def run(G):
            start = time.perf_counter()
            features, cached = cache.get(("vector", name), fn, G)
            report(i, name, G, start, cached)
            return features
        return run

    def pairwise_method(i, name, sampler_key, scorer):
        sampler = SAMPLERS[sampler_key]
        def run(G_ref, H):
            start = time.perf_counter()
            # key on the sampler, so e.g. BFS and BFS-Dist share one set of samples per graph
            # ("samples", ...) keeps them apart from the vector methods: the NetSimile vector
            # and the NetSimile node samples are different things with the same name
            F_ref, ref_cached = cache.get(("samples", sampler_key), sampler, G_ref)
            F_H, h_cached = cache.get(("samples", sampler_key), sampler, H)
            score = scorer(F_ref, F_H)   # the SVM / matching step itself always runs
            report(i, name, H, start, ref_cached and h_cached)
            return score
        return run

    feature_methods = {name: vector_method(i, name, fn)
                       for i, (name, fn) in enumerate(FEATURE_METHODS.items(), 1)}
    pairwise_methods = {name: pairwise_method(i, name, key, scorer)
                        for i, (name, (key, scorer))
                        in enumerate(PAIRWISE_SPECS.items(), len(FEATURE_METHODS) + 1)}
    return feature_methods, pairwise_methods


def ranking_path(out_dir, key):
    return os.path.join(out_dir, "rankings", f"{key}.pkl")


def load_ranking(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_references(networks, out_dir, frac, seed, references=None):
    """One RankingComparison per reference, pickled to out_dir/rankings/<key>.pkl.

    references: {result key: name of the reference graph in `networks`}. Default: every
    network is a reference, keyed by its own name (the cross-sectional design). The
    longitudinal experiment passes one {dataset: reference snapshot} per call; the key is
    separate from the graph name because snapshot labels repeat across datasets.

    This is the expensive step: every method scores every (reference, comparison) pair.
    Features are computed once per (method, graph) and reused across references (FeatureCache).
    Each reference runs the same per-reference step that
    ranking_comparisons.cross_sectional_experiment loops over (cross_sectional_reference),
    called here one reference at a time so each result can be saved as soon as it finishes:
    compare against every other network + the reference's two baselines (§5.3), then
    scores (methods x networks) -> Kendall-Tau, p-values, nDCG, Kemeny consensus.
    """
    references = references or {name: name for name in networks}
    os.makedirs(os.path.join(out_dir, "rankings"), exist_ok=True)
    results = {}
    run_start = time.perf_counter()
    cache = FeatureCache(networks)   # shared by all references, so each network is featurized once
    for i, (key, ref) in enumerate(references.items(), 1):
        tag = f"[{key} {i}/{len(references)}]"   # which reference we're on, e.g. [amazon 2/5]

        # resume support: if this reference already finished on a previous run, reuse it
        # (delete out_dir/rankings/ after changing a method, or stale results get reused)
        path = ranking_path(out_dir, key)
        if os.path.exists(path):
            log(f"{tag} already done, loading {path}")
            results[key] = load_ranking(path)
            continue

        log(f"{tag} scoring {len(networks) - 1} networks + 2 baselines with "
            f"{len(FEATURE_METHODS) + len(PAIRWISE_SPECS)} methods...")
        ref_start = time.perf_counter()
        # same as one iteration of ranking_comparisons.cross_sectional_experiment:
        # compare against every other network + the reference's two baselines (§5.3), then
        # scores (methods x networks) -> Kendall-Tau, p-values, nDCG, Kemeny consensus.
        # the methods read features from the shared cache and print a line per call
        cache.new_reference()   # the previous reference's baselines are never seen again
        feature_methods, pairwise_methods = cached_methods(tag, cache)
        results[key] = cross_sectional_reference(ref, networks, feature_methods,
                                                 pairwise_methods, frac, seed)

        # save right away so a crash on a later reference doesn't lose this one
        with open(path, "wb") as f:
            pickle.dump(results[key], f)
        log(f"{tag} done in {fmt_duration(time.perf_counter() - ref_start)} "
            f"(total so far {fmt_duration(time.perf_counter() - run_start)}), saved {path}")
    return results


# output file names (under tables/) for each experiment's cross-reference tables
CROSS_SECTIONAL_TABLES = {
    "consensus": "table5_closest_to_consensus",   # Table 5
    "distance": "fig7_distance_to_consensus",     # Figure 7
    "clusters": "table4_recurring_clusters",      # Table 4
}
LONGITUDINAL_TABLES = {
    "consensus": "closest_to_consensus",          # Table 5, longitudinal columns
    "distance": "fig22_distance_to_consensus",    # Figure 22
    "clusters": "recurring_clusters",
}


def write_tables(results, out_dir, alpha, names=CROSS_SECTIONAL_TABLES):
    """Save the cross-reference tables and each reference's matrices as CSVs."""
    tables = {
        # Table 5: for each reference, the 5 methods whose ranking is closest to the consensus
        names["consensus"]: consensus_table(results, n=5),
        # Figure 7 / 22: methods x references, Kendall-Tau distance of each method from the consensus
        names["distance"]: consensus_distance_table(results),
    }
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, "tables", f"{name}.csv"))
    write_reference_tables(results, out_dir)

    # print Table 5 so there's something to look at without opening the CSVs
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n", tables[names["consensus"]], "\n")


def write_reference_tables(results, out_dir):
    """Per-reference outputs, one CSV each."""
    tables = {}
    for ref, r in results.items():
        tables[f"scores_{ref}"] = r.scores                    # methods x networks, raw similarity
        tables[f"ranks_{ref}"] = r.ranks                      # methods x networks, 1 = most similar
        tables[f"kendall_{ref}"] = r.kendall                  # methods x methods, Kendall-Tau distance
        tables[f"kendall_p_{ref}"] = r.kendall_p              # methods x methods, p-values
        tables[f"ndcg_{ref}"] = r.ndcg                        # methods x methods, symmetric nDCG
        tables[f"consensus_{ref}"] = r.consensus.to_frame()   # network -> consensus position

    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, "tables", f"{name}.csv"))


def write_heatmaps(results, out_dir, alpha):
    """Kendall-Tau, nDCG and significance heatmaps for each reference, saved as PNGs."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ref, r in results.items():
        # same method order (micro / mezzo / macro) in every figure so they can be compared
        order = [m for m in METHOD_ORDER if m in r.kendall.index]
        for kind, ax in [
            ("kendall", plot_heatmap(r.kendall, f"{ref}: Kendall-Tau distance", order=order)),
            ("ndcg", plot_heatmap(r.ndcg, f"{ref}: nDCG", order=order)),
            # black = Kendall-Tau correlation significant at alpha (paper uses 0.1)
            ("significance", plot_significance(r, alpha, f"{ref}: significant at p={alpha}",
                                               order=order)),
        ]:
            ax.figure.savefig(os.path.join(fig_dir, f"{kind}_{ref}.png"), dpi=150)
            plt.close(ax.figure)   # free memory; otherwise every figure stays open


def cluster(results, out_dir, runs, min_refs, table_name="table4_recurring_clusters"):
    """Modal dendrogram per reference, then the clusters found for >= min_refs references,
    saved to tables/<table_name>.csv."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    counts = Counter()   # cluster (frozenset of method names) -> number of references it appears in
    rows = []            # how stable each reference's modal tree is
    for ref, r in results.items():
        # kendalltau returns NaN when a method gives every network the same score;
        # complete linkage can't handle NaN distances, so skip this reference and say why
        if r.kendall.isna().any().any():
            bad = r.scores.index[r.scores.nunique(axis=1) <= 1].tolist()
            print(f"[{ref}] skipping clustering: NaN Kendall-Tau (constant scores from {bad})")
            continue

        # §4.2: cluster `runs` times with random tie-breaking, keep the most common tree
        # sig = the tree as a set of clusters, freq = fraction of runs that produced it,
        # n_distinct = how many different trees showed up, Z = linkage matrix for plotting
        sig, freq, n_distinct, Z = CompleteLinkageClustering(r.kendall).modal_tree(runs=runs)
        rows.append({"reference": ref, "modal_tree_freq": freq, "distinct_trees": n_distinct})
        print(f"[{ref}] modal tree in {freq:.1%} of {runs} runs, {n_distinct} distinct trees")

        ax = plot_dendrogram(Z, list(r.kendall.index), title=f"{ref} (modal tree, {freq:.1%})")
        ax.figure.savefig(os.path.join(fig_dir, f"dendrogram_{ref}.png"), dpi=150)
        plt.close(ax.figure)

        # every internal node of the tree is a cluster; count each one for this reference.
        # single methods (leaves) and the all-methods root are trivially in every tree, so skip them
        n_methods = len(r.kendall)
        counts.update(g for g in sig if 1 < len(g) < n_methods)

    # Table 4: clusters that appear in the modal tree of at least min_refs references,
    # smallest clusters first, then most common first
    table4 = pd.DataFrame(
        [{"cluster": ", ".join(sorted(g)), "size": len(g), "n_references": c}
         for g, c in counts.items() if c >= min_refs],
        columns=["cluster", "size", "n_references"],
    ).sort_values(["size", "n_references"], ascending=[True, False])

    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    # low modal_tree_freq / high distinct_trees means the dendrogram depends on tie-breaking
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "tables", "dendrogram_stability.csv"),
                              index=False)
    table4.to_csv(os.path.join(out_dir, "tables", f"{table_name}.csv"), index=False)
    print(f"\nclusters found for >= {min_refs} references:\n", table4.to_string(index=False))




def default_min_refs(n_references):
    """Table 4 threshold: the paper keeps clusters found for >= 4 of its 7 references; keep the
    same proportion for however many references we have (at least 2, or "recurring" means
    nothing)."""
    return max(2, round(4 / 7 * n_references))


def analyze(results, out_dir, alpha, runs, min_refs, names=CROSS_SECTIONAL_TABLES):
    """Everything after scoring: tables, heatmaps, dendrograms and recurring clusters."""
    log("writing tables")
    write_tables(results, out_dir, alpha, names)          # CSVs, Table 5, Figure 7 / 22
    log("drawing heatmaps")
    write_heatmaps(results, out_dir, alpha)               # PNG heatmaps
    log(f"clustering ({runs} runs per reference)")
    cluster(results, out_dir, runs, min_refs, table_name=names["clusters"])   # Table 4


def run_experiment(networks, out_dir, frac=0.05, seed=0, alpha=0.1, runs=1000, min_refs=None):
    """The whole cross-sectional pipeline on any {name: graph} dict; everything is written to
    out_dir.

    From Python, e.g.:
        from run_experiments.run_experiments import run_experiment, RESULTS_DIR
        run_experiment(my_networks(), os.path.join(RESULTS_DIR, "my_run"))
    """
    if not networks:
        raise SystemExit("no networks to compare")

    start = time.perf_counter()
    log(f"scoring {len(networks)} references ({', '.join(networks)})")
    results = run_references(networks, out_dir, frac, seed)   # slow: all 20 methods
    analyze(results, out_dir, alpha, runs, min_refs or default_min_refs(len(networks)))
    log(f"finished in {fmt_duration(time.perf_counter() - start)}, wrote everything to {out_dir}/")
    return results


# =============================================================================================
# Longitudinal experiment (§5.4), run with --longitudinal.
#
# Each dataset is a time series of graphs (one per day / month / window). For each dataset:
# pick one reference snapshot, compare it against every other snapshot plus the reference's
# two baselines -- the same per-reference step as above (run_references with a single
# reference), with snapshots in place of separate networks.
#
# Datasets (10):
#     original        replies (30 daily), retweets (5 monthly), yahoo (28 daily)
#     horizontal      replies-7day, replies-3day, yahoo-7day, yahoo-4day
#                     reference = the window containing the original reference snapshot
#     vertical        replies-cumulative, retweets-cumulative, yahoo-cumulative
#                     (snapshot t = everything up to t), reference = the last snapshot
#
# Reference snapshots: the paper uses the snapshot Berlingerio et al. (2013) found anomalous.
# Fill those in REFERENCE_DATES below, pass them with --reference, or use --auto-reference.
# =============================================================================================

# short name -> key in load_data.longitudinal_networks (which holds the snapshots() settings)
BASES = {"replies": "twitter replies", "retweets": "twitter retweets", "yahoo": "yahoo"}

# The reference snapshot of each original dataset, as a snapshot label:
# "YYYY-MM-DD" for replies and yahoo (daily), "YYYY-MM" for retweets (monthly).
# Paper: the snapshot found anomalous by Berlingerio et al. 2013. Fill these in once known.
REFERENCE_DATES = {"replies": None, "retweets": None, "yahoo": None}

# dataset -> (base dataset, extra snapshots() arguments, how its reference is chosen)
#   "original": the reference date itself
#   "window":   the window that contains the original reference date
#   "last":     the final snapshot (for cumulative datasets)
DATASETS = {
    "replies":             ("replies",  {},                   "original"),
    "retweets":            ("retweets", {},                   "original"),
    "yahoo":               ("yahoo",    {},                   "original"),
    "replies-7day":        ("replies",  {"days": 7},          "window"),
    "replies-3day":        ("replies",  {"days": 3},          "window"),
    "yahoo-7day":          ("yahoo",    {"days": 7},          "window"),
    "yahoo-4day":          ("yahoo",    {"days": 4},          "window"),
    "replies-cumulative":  ("replies",  {"cumulative": True}, "last"),
    "retweets-cumulative": ("retweets", {"cumulative": True}, "last"),
    "yahoo-cumulative":    ("yahoo",    {"cumulative": True}, "last"),
}


def build_snapshots(dataset):
    """{label: graph} for one dataset, in time order."""
    base, extra, _ = DATASETS[dataset]
    return snapshots(**{**longitudinal_networks[BASES[base]], **extra})


def toy_series():
    """Eight small 'daily' graphs from the same model, except day t5, which is denser
    and less clustered: a stand-in anomaly to use as the reference."""
    graphs = {f"t{i}": nx.powerlaw_cluster_graph(300, 4, 0.3, seed=i) for i in range(8)}
    graphs["t5"] = nx.gnp_random_graph(300, 0.04, seed=5)
    return graphs


def auto_reference(graphs):
    """Fallback when the paper's anomalous snapshot isn't known: the snapshot whose edge count
    is furthest from the median, on a log scale. Report it as a deviation if you use it."""
    labels = list(graphs)
    log_m = np.log([max(G.number_of_edges(), 1) for G in graphs.values()])
    return labels[int(np.argmax(np.abs(log_m - np.median(log_m))))]


def resolve_original_references(bases, cli_refs, auto):
    """Reference label of each original dataset: --reference, else REFERENCE_DATES,
    else (with --auto-reference) auto_reference. Returns {base: (label, how chosen)}."""
    refs, missing = {}, []
    for base in bases:
        if base in cli_refs:
            refs[base] = (cli_refs[base], "--reference")
        elif REFERENCE_DATES[base] is not None:
            refs[base] = (REFERENCE_DATES[base], "REFERENCE_DATES")
        elif auto:
            label = auto_reference(build_snapshots(base))
            refs[base] = (label, "auto: edge count furthest from median")
        else:
            missing.append(base)
    if missing:
        raise SystemExit(
            f"No reference snapshot for {missing}. Set REFERENCE_DATES in "
            f"{os.path.basename(__file__)}, pass e.g. --reference {missing[0]}=<label>, "
            "or use --auto-reference.")
    return refs


def pick_reference(graphs, rule, original):
    """Reference label for one dataset, given its rule and its base's original reference."""
    labels = list(graphs)
    if rule == "last":
        return labels[-1]
    if rule == "window":
        # bins are labelled by their first date, so the window containing `original` is the
        # last bin starting on or before it (labels are ISO dates, so string order = time order)
        earlier = [l for l in labels if l <= original]
        if not earlier:
            raise ValueError(f"reference {original} is before the first window {labels[0]}")
        return earlier[-1]
    if original not in graphs:
        raise ValueError(f"reference {original!r} is not a snapshot; labels look like "
                         f"{labels[:3]} ... {labels[-1]}")
    return original


def top5_overlap(r, n=5):
    """Figure 11: methods x methods, fraction of the n most similar snapshots two methods share.
    Ties at the cut-off are broken by snapshot order."""
    top = {m: set(r.ranks.loc[m].nsmallest(n).index) for m in r.ranks.index}
    return pd.DataFrame({a: {b: len(top[a] & top[b]) / n for b in top} for a in top})


def run_longitudinal(datasets, original_refs, out_dir, frac=0.05, seed=0, alpha=0.1,
                     runs=1000, min_refs=None, toy=False):
    """The whole longitudinal pipeline. One RankingComparison per dataset, pickled to
    out_dir/rankings/<dataset>.pkl (plus <dataset>_reference.json, which snapshot was the
    reference and why); finished datasets are loaded instead of recomputed.

    original_refs: {base: (label, how chosen)}, from resolve_original_references.
    toy=True runs the single synthetic series from toy_series() (datasets should be ["toy"]).
    """
    start = time.perf_counter()
    results, ref_rows = {}, []
    for i, dataset in enumerate(datasets, 1):
        path = ranking_path(out_dir, dataset)
        ref_path = os.path.join(out_dir, "rankings", f"{dataset}_reference.json")
        # checked here rather than left to run_references, so finished datasets don't have
        # their snapshots rebuilt (slow for the large and cumulative ones)
        if os.path.exists(path) and os.path.exists(ref_path):
            log(f"[{dataset}] already done, loading {path}")
            results[dataset] = load_ranking(path)
            with open(ref_path) as f:
                ref_rows.append(json.load(f))
            continue

        if toy:
            graphs, ref, how = toy_series(), "t5", "toy anomaly"
        else:
            base, _, rule = DATASETS[dataset]
            original, how = original_refs.get(base, (None, None))
            graphs = build_snapshots(dataset)
            ref = pick_reference(graphs, rule, original)
            if rule == "last":
                how = "last snapshot (cumulative)"
            elif rule == "window":
                how = f"window rule ({base} reference {original})"

        log(f"dataset {i}/{len(datasets)}: {dataset}, reference {ref} of {len(graphs)} snapshots")
        results.update(run_references(graphs, out_dir, frac, seed, references={dataset: ref}))
        ref_row = {"dataset": dataset, "reference": ref, "chosen_by": how,
                   "n_snapshots": len(graphs)}
        ref_rows.append(ref_row)
        with open(ref_path, "w") as f:
            json.dump(ref_row, f)
        del graphs   # cumulative datasets are large; free them before the next one

    references = pd.DataFrame(ref_rows)
    tables_dir, fig_dir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    references.to_csv(os.path.join(tables_dir, "references.csv"), index=False)

    analyze(results, out_dir, alpha, runs, min_refs or default_min_refs(len(datasets)),
            LONGITUDINAL_TABLES)

    # Figure 11: top-5 overlap, as a CSV and a heatmap per dataset
    log("drawing top-5 overlap")
    for dataset, r in results.items():
        overlap = top5_overlap(r)
        overlap.to_csv(os.path.join(tables_dir, f"top5_overlap_{dataset}.csv"))
        order = [m for m in METHOD_ORDER if m in overlap.index]
        ax = plot_heatmap(overlap, f"{dataset}: top-5 overlap", order=order)
        ax.figure.savefig(os.path.join(fig_dir, f"top5_overlap_{dataset}.png"), dpi=150)
        plt.close(ax.figure)

    print("\n", references.to_string(index=False), "\n")
    log(f"finished in {fmt_duration(time.perf_counter() - start)}, wrote everything to {out_dir}/")
    return results, references


# =============================================================================================
# Command line
# =============================================================================================

def main():
    # ---- command-line options ----
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network-set", default=None, metavar="SET",
                   help=f"which graphs to use: one of {list(NETWORK_SETS)} (default paper), "
                        "or an import path like my_package.my_module:my_function that "
                        "returns {name: networkx.Graph}")
    p.add_argument("--networks", nargs="+", default=None,
                   help="only use these networks from the set (default: all of them)")
    p.add_argument("--toy", action="store_true",
                   help="shortcut for --network-set toy; with --longitudinal, one synthetic "
                        "8-snapshot series instead of the real datasets")
    p.add_argument("--out", default=None,
                   help="subfolder of results/ to write to (default: cross_sectional for "
                        "paper, toy for toy, otherwise the set's name; longitudinal or "
                        "longitudinal_toy with --longitudinal)")
    p.add_argument("--frac", type=float, default=0.05, help="fraction of edges deleted/rewired")
    p.add_argument("--seed", type=int, default=0, help="seed for the baselines")
    p.add_argument("--alpha", type=float, default=0.1, help="significance level (paper: 0.1)")
    p.add_argument("--runs", type=int, default=1000, help="clustering runs per reference")
    p.add_argument("--min-refs", type=int, default=None,
                   help="Table 4 threshold (default: paper's 4 of 7, scaled to #references)")

    lon = p.add_argument_group("longitudinal experiment (§5.4)")
    lon.add_argument("--longitudinal", action="store_true",
                     help="one reference snapshot per time-series dataset, instead of every "
                          "network as a reference")
    lon.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS),
                     metavar="DATASET",
                     help=f"subset to run (default all): {', '.join(DATASETS)}")
    lon.add_argument("--reference", action="append", default=[], metavar="BASE=LABEL",
                     help="reference snapshot of an original dataset, e.g. yahoo=2008-04-08; "
                          "overrides REFERENCE_DATES; repeat for each dataset")
    lon.add_argument("--auto-reference", action="store_true",
                     help="for datasets with no reference given, use the snapshot whose edge "
                          "count is furthest from the median")
    args = p.parse_args()

    if args.longitudinal:
        if args.network_set or args.networks:
            p.error("--network-set / --networks don't apply with --longitudinal "
                    "(use --datasets)")
        run_longitudinal_cli(args)
        return
    if args.datasets or args.reference or args.auto_reference:
        p.error("--datasets / --reference / --auto-reference need --longitudinal")

    # ---- load the networks ----
    # each set writes to its own folder, so results from different sets never mix
    # (or get resumed as each other's)
    network_fn, default_out = resolve_network_set(
        "toy" if args.toy else (args.network_set or "paper"))
    out_dir = os.path.join(RESULTS_DIR, args.out or default_out)

    # call the network-set function, restricted to --networks if given
    if args.networks is None:
        networks = network_fn()
    elif "names" in inspect.signature(network_fn).parameters:
        networks = network_fn(names=args.networks)   # function loads only what's asked for
    else:
        networks = network_fn()                      # load everything, keep only what's asked for
        missing = set(args.networks) - set(networks)
        if missing:
            raise SystemExit(f"--networks {sorted(missing)} not in this set: {list(networks)}")
        networks = {name: networks[name] for name in args.networks}
    if len(networks) < 2:
        raise SystemExit("need at least 2 networks to compare")

    # ---- run the pipeline ----
    run_experiment(networks, out_dir, args.frac, args.seed, args.alpha, args.runs, args.min_refs)


def run_longitudinal_cli(args):
    out_dir = os.path.join(RESULTS_DIR,
                           args.out or ("longitudinal_toy" if args.toy else "longitudinal"))
    if args.toy:
        datasets, original_refs = ["toy"], {}
    else:
        datasets = args.datasets or list(DATASETS)
        cli_refs = dict(r.split("=", 1) for r in args.reference)
        unknown = set(cli_refs) - set(BASES)
        if unknown:
            raise SystemExit(f"--reference must name one of {list(BASES)}, got {unknown}")
        # only the bases of the datasets being run need a reference
        bases = sorted({DATASETS[d][0] for d in datasets if DATASETS[d][2] != "last"})
        original_refs = resolve_original_references(bases, cli_refs, args.auto_reference)

    run_longitudinal(datasets, original_refs, out_dir, args.frac, args.seed, args.alpha,
                     args.runs, args.min_refs, toy=args.toy)


if __name__ == "__main__":
    main()
