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

Everything is written under <repo root>/results/<out>/ (default results/cross_sectional/,
or results/toy/ with --toy).
"""
import argparse
import os
import pickle
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
from run_experiments.methods import FEATURE_METHODS, METHOD_ORDER, PAIRWISE_METHODS

# every output goes under <repo root>/results/, wherever the script is run from
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# networks used as both references and comparisons (keys of load_data.cross_sectional_files)
# add "grad", "undergrad" once they have loaders (see docs/reproducing_experiments.md §2.3)
CROSS_SECTIONAL = ["amazon", "dblp", "lj1", "lj2", "email"]


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


def run_references(networks, out_dir, frac, seed):
    """One RankingComparison per reference, pickled to out_dir/rankings/<ref>.pkl.

    This is the expensive step: every method is run on every (reference, comparison) pair.
    """
    os.makedirs(os.path.join(out_dir, "rankings"), exist_ok=True)
    results = {}
    for ref in networks:
        # resume support: if this reference already finished on a previous run, reuse it
        # (delete out_dir/rankings/ after changing a method, or stale results get reused)
        path = os.path.join(out_dir, "rankings", f"{ref}.pkl")
        if os.path.exists(path):
            print(f"[{ref}] already done, loading {path}", flush=True)
            with open(path, "rb") as f:
                results[ref] = pickle.load(f)
            continue

        print(f"[{ref}] scoring {len(networks) - 1} networks + 2 baselines...", flush=True)
        # same as one iteration of ranking_comparisons.cross_sectional_experiment:
        # compare against every other network + the reference's two baselines (§5.3), then
        # scores (methods x networks) -> Kendall-Tau, p-values, nDCG, Kemeny consensus
        results[ref] = cross_sectional_reference(ref, networks, FEATURE_METHODS,
                                                 PAIRWISE_METHODS, frac, seed)

        # save right away so a crash on a later reference doesn't lose this one
        with open(path, "wb") as f:
            pickle.dump(results[ref], f)
    return results


def write_tables(results, out_dir, alpha):
    """Save the cross-reference tables and each reference's matrices as CSVs."""
    tables = {
        # Table 5: for each reference, the 5 methods whose ranking is closest to the consensus
        "table5_closest_to_consensus": consensus_table(results, n=5),
        # Figure 7: methods x references, Kendall-Tau distance of each method from the consensus
        "fig7_distance_to_consensus": consensus_distance_table(results),
    }
    # per-reference outputs, one CSV each
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

    # print Table 5 so there's something to look at without opening the CSVs
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n", tables["table5_closest_to_consensus"], "\n")


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


def cluster(results, out_dir, runs, min_refs):
    """Modal dendrogram per reference, then the clusters found for >= min_refs references."""
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
    table4.to_csv(os.path.join(out_dir, "tables", "table4_recurring_clusters.csv"), index=False)
    print(f"\nclusters found for >= {min_refs} references:\n", table4.to_string(index=False))


def main():
    # ---- command-line options ----
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--networks", nargs="+", default=CROSS_SECTIONAL,
                   help="names from load_data.cross_sectional_files")
    p.add_argument("--toy", action="store_true", help="use 4 small synthetic graphs instead")
    p.add_argument("--out", default=None, help="subfolder of results/ to write to (default "
                                               "cross_sectional, or toy with --toy)")
    p.add_argument("--frac", type=float, default=0.05, help="fraction of edges deleted/rewired")
    p.add_argument("--seed", type=int, default=0, help="seed for the baselines")
    p.add_argument("--alpha", type=float, default=0.1, help="significance level (paper: 0.1)")
    p.add_argument("--runs", type=int, default=1000, help="clustering runs per reference")
    p.add_argument("--min-refs", type=int, default=None,
                   help="Table 4 threshold (default: paper's 4 of 7, scaled to #references)")
    args = p.parse_args()

    # ---- load the networks ----
    # toy results go in their own folder so they never mix with (or get resumed as) real results
    out_dir = os.path.join(RESULTS_DIR, args.out or ("toy" if args.toy else "cross_sectional"))
    if args.toy:
        networks = toy_networks()
    else:
        networks = {}
        for name in args.networks:
            print(f"loading {name}...", flush=True)   # lj1 / lj2 take ~20 s each
            networks[name] = load_cross_sectional_network(name)

    # paper keeps clusters found for >= 4 of its 7 references; keep the same proportion
    # for however many references we have (at least 2, or "recurring" means nothing)
    min_refs = args.min_refs or max(2, round(4 / 7 * len(networks)))

    # ---- run the pipeline ----
    results = run_references(networks, out_dir, args.frac, args.seed)   # slow: all 20 methods
    write_tables(results, out_dir, args.alpha)                         # CSVs, Table 5, Figure 7
    write_heatmaps(results, out_dir, args.alpha)                       # PNG heatmaps
    cluster(results, out_dir, args.runs, min_refs)                     # dendrograms, Table 4
    print(f"\nwrote everything to {out_dir}/")


if __name__ == "__main__":
    main()
