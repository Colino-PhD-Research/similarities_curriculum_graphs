"""
Cross-sectional experiment (Soundarajan, Eliassi-Rad & Gallagher 2014, §5.3), end to end.

For each reference network: score every other network plus the reference's two baselines
(5% edges deleted, 5% rewired) with all 20 methods, then compute Kendall-Tau, p-values,
nDCG and the Kemeny consensus. Afterwards: Table 5, Figure 7, heatmaps,
the 1000-run modal dendrogram per reference, and the clusters that recur across
references (Table 4).

Each reference's RankingComparison is pickled as soon as it finishes, and finished
references are skipped on rerun, so an interrupted run can be resumed.

Run from the repo root:
    python -m run_experiments.cross_sectional_experiments --toy     # 4 synthetic graphs, ~5 min
    python -m run_experiments.cross_sectional_experiments           # amazon dblp lj1 lj2 email
    python -m run_experiments.cross_sectional_experiments --networks amazon email --out small
    python -m run_experiments.cross_sectional_experiments --network-set my_package.my_module:my_function

The graphs come from a "network set": any function returning {name: networkx.Graph}
(see NETWORK_SETS below). Everything is written under <repo root>/results/<out>/
(default results/cross_sectional/ for the paper's set, results/toy/ with --toy, otherwise
results/<set name>/).
"""
import argparse
import importlib
import inspect
import os
import pickle
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # save figures, never open windows (must be set before pyplot is imported)
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from clustering.clustering import CompleteLinkageClustering
from clustering.utils import plot_dendrogram
from load_data.load_data import load_cross_sectional_network
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


def run_references(networks, out_dir, frac, seed):
    """One RankingComparison per reference, pickled to out_dir/rankings/<ref>.pkl.

    This is the expensive step: every method scores every (reference, comparison) pair.
    Features are computed once per (method, graph) and reused across references (FeatureCache).
    Each reference runs the same per-reference step that
    ranking_comparisons.cross_sectional_experiment loops over (cross_sectional_reference),
    called here one reference at a time so each result can be saved as soon as it finishes:
    compare against every other network + the reference's two baselines (§5.3), then
    scores (methods x networks) -> Kendall-Tau, p-values, nDCG, Kemeny consensus.
    """
    os.makedirs(os.path.join(out_dir, "rankings"), exist_ok=True)
    results = {}
    run_start = time.perf_counter()
    cache = FeatureCache(networks)   # shared by all references, so each network is featurized once
    for i, ref in enumerate(networks, 1):
        tag = f"[{ref} {i}/{len(networks)}]"   # which reference we're on, e.g. [amazon 2/5]

        # resume support: if this reference already finished on a previous run, reuse it
        # (delete out_dir/rankings/ after changing a method, or stale results get reused)
        path = os.path.join(out_dir, "rankings", f"{ref}.pkl")
        if os.path.exists(path):
            log(f"{tag} already done, loading {path}")
            with open(path, "rb") as f:
                results[ref] = pickle.load(f)
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
        results[ref] = cross_sectional_reference(ref, networks, feature_methods,
                                                 pairwise_methods, frac, seed)

        # save right away so a crash on a later reference doesn't lose this one
        with open(path, "wb") as f:
            pickle.dump(results[ref], f)
        log(f"{tag} done in {fmt_duration(time.perf_counter() - ref_start)} "
            f"(total so far {fmt_duration(time.perf_counter() - run_start)}), saved {path}")
    return results


def write_tables(results, out_dir, alpha):
    """Save the cross-reference tables and each reference's matrices as CSVs."""
    tables = {
        # Table 5: for each reference, the 5 methods whose ranking is closest to the consensus
        "table5_closest_to_consensus": consensus_table(results, n=5),
        # Figure 7: methods x references, Kendall-Tau distance of each method from the consensus
        "fig7_distance_to_consensus": consensus_distance_table(results),
    }
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(out_dir, "tables", f"{name}.csv"))
    write_reference_tables(results, out_dir)

    # print Table 5 so there's something to look at without opening the CSVs
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n", tables["table5_closest_to_consensus"], "\n")


def write_reference_tables(results, out_dir):
    """Per-reference outputs, one CSV each (also used by longitudinal_experiments.py)."""
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


def run_experiment(networks, out_dir, frac=0.05, seed=0, alpha=0.1, runs=1000, min_refs=None):
    """The whole pipeline on any {name: graph} dict; everything is written to out_dir.

    From Python, e.g.:
        from run_experiments.cross_sectional_experiments import run_experiment, RESULTS_DIR
        run_experiment(my_networks(), os.path.join(RESULTS_DIR, "my_run"))
    """
    # paper keeps clusters found for >= 4 of its 7 references; keep the same proportion
    # for however many references we have (at least 2, or "recurring" means nothing)
    min_refs = min_refs or max(2, round(4 / 7 * len(networks)))

    if not networks:
        raise SystemExit("no networks to compare")

    start = time.perf_counter()
    log(f"step 1/4: scoring {len(networks)} references ({', '.join(networks)})")
    results = run_references(networks, out_dir, frac, seed)   # slow: all 20 methods
    log("step 2/4: writing tables")
    write_tables(results, out_dir, alpha)                    # CSVs, Table 5, Figure 7
    log("step 3/4: drawing heatmaps")
    write_heatmaps(results, out_dir, alpha)                  # PNG heatmaps
    log(f"step 4/4: clustering ({runs} runs per reference)")
    cluster(results, out_dir, runs, min_refs)                # dendrograms, Table 4
    log(f"finished in {fmt_duration(time.perf_counter() - start)}, wrote everything to {out_dir}/")
    return results


def main():
    # ---- command-line options ----
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network-set", default="paper", metavar="SET",
                   help=f"which graphs to use: one of {list(NETWORK_SETS)} (default paper), "
                        "or an import path like my_package.my_module:my_function that "
                        "returns {name: networkx.Graph}")
    p.add_argument("--networks", nargs="+", default=None,
                   help="only use these networks from the set (default: all of them)")
    p.add_argument("--toy", action="store_true", help="shortcut for --network-set toy")
    p.add_argument("--out", default=None, help="subfolder of results/ to write to (default: "
                                               "cross_sectional for paper, toy for toy, "
                                               "otherwise the set's name)")
    p.add_argument("--frac", type=float, default=0.05, help="fraction of edges deleted/rewired")
    p.add_argument("--seed", type=int, default=0, help="seed for the baselines")
    p.add_argument("--alpha", type=float, default=0.1, help="significance level (paper: 0.1)")
    p.add_argument("--runs", type=int, default=1000, help="clustering runs per reference")
    p.add_argument("--min-refs", type=int, default=None,
                   help="Table 4 threshold (default: paper's 4 of 7, scaled to #references)")
    args = p.parse_args()

    # ---- load the networks ----
    # each set writes to its own folder, so results from different sets never mix
    # (or get resumed as each other's)
    network_fn, default_out = resolve_network_set("toy" if args.toy else args.network_set)
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


if __name__ == "__main__":
    main()
