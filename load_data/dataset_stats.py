"""
Table-2 and Table-3 statistics for the networks in load_data.py
(Soundarajan, Eliassi-Rad & Gallagher, "A Guide to Selecting a Network
Similarity Method").

Each cross-sectional network is loaded with load_cross_sectional_network and
each longitudinal snapshot is built with snapshots, so the stats describe
exactly the graphs the similarity measures see (undirected, repeated pairs
merged). Self loops are dropped before computing stats.

Usage (from the project root):
    python -m load_data.dataset_stats             # print both tables
    python -m load_data.dataset_stats out_dir     # also save them as csv
    # or from Python:
    from load_data.dataset_stats import cross_sectional_stats, longitudinal_stats
    df = cross_sectional_stats()                  # Table 2: one row per network
    df = longitudinal_stats()                     # Table 3: "min–max" per cell

Takes a few minutes; lj1, lj2 and tweet_graph.data dominate the load time.
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx
from utility.program_structure.build_graph_from_db import build_graph
from utility.universal_helpers import get_verified_ai_majors

try:
    from load_data.load_data import (cross_sectional_files, load_cross_sectional_network,
                                     longitudinal_networks, snapshots)
    from load_data.lj2_sample import table_stats
except ImportError:
    from load_data import (cross_sectional_files, load_cross_sectional_network,
                           longitudinal_networks, snapshots)
    from lj2_sample import table_stats

# how each stat is printed in the longitudinal ranges
STAT_FORMATS = {
    "nodes": "{:,.0f}", "edges": "{:,.0f}", "avg_degree": "{:.2f}",
    "max_degree": "{:,.0f}", "density": "{:.1e}", "transitivity": "{:.3f}",
    "num_cc": "{:,.0f}", "frac_in_lcc": "{:.3f}",
}


def network_stats(G):
    G.remove_edges_from(list(networkx.selfloop_edges(G)))
    edges = np.array(G.edges())
    if edges.dtype.kind not in "iu":
        # longitudinal node ids are usernames; table2_stats needs integers
        _, edges = np.unique(edges, return_inverse=True) # return inverse returns indices
        edges = edges.reshape(-1, 2)
    # table2_stats uses scipy, much faster than networkx.transitivity on lj1/lj2
    return table_stats(edges.astype(np.int64))


def _load_cross_sectional():
    for name in cross_sectional_files:
        print(f"loading {name}...", flush=True)
        yield name, load_cross_sectional_network(name)


def cross_sectional_stats(graphs=None):
    """Return a DataFrame with one row of Table-2 stats per graph.

    graphs: {name: networkx.Graph}. Default loads every network in
    cross_sectional_files, one at a time to keep memory down.
    """
    graphs = _load_cross_sectional() if graphs is None else graphs.items()
    return pd.DataFrame([{"network": name, **network_stats(G)} for name, G in graphs])


def longitudinal_stats():
    """Return a DataFrame like Table 3: one row per longitudinal network,
    each cell a "min–max" range of that stat over the network's snapshots."""
    rows = []
    for name, kwargs in longitudinal_networks.items():
        print(f"loading {name}...", flush=True)
        # one row per snapshot, labelled by its period
        per_period = cross_sectional_stats(snapshots(**kwargs))
        row = {"network": name, "periods": len(per_period)}
        for stat, fmt in STAT_FORMATS.items():
            lo, hi = per_period[stat].min(), per_period[stat].max()
            row[stat] = f"{fmt.format(lo)}–{fmt.format(hi)}"
        rows.append(row)
    return pd.DataFrame(rows)

def ai_major_stats():
    # get list of AI majors
    universities_programs = get_verified_ai_majors()

    # build graph for each major
    graphs = [(f"{university}:{program}", build_graph(university, program, allow_messy=True)) for university, program in universities_programs]

    # make dataframe calculating network statistics for each major (graph)
    df = pd.DataFrame([{"network": name, **network_stats(G)} for name, G in graphs])
    df.to_csv("ai_curriculum_non_aggregated_stats.csv")

    # aggregate the previous dataframe to match the output of the two other datasets from the paper
    row = {}
    for stat, fmt in STAT_FORMATS.items():
        lo, hi = df[stat].min(), df[stat].max()
        row[stat] = f"{fmt.format(lo)}–{fmt.format(hi)}"
    return pd.DataFrame(row, index=[0])

if __name__ == "__main__":
    tables = {"cross_sectional_stats": cross_sectional_stats(),
              "longitudinal_stats": longitudinal_stats(),
              "ai_majors": ai_major_stats()}
    for file_name, df in tables.items():
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df)
        if len(sys.argv) > 1:
            df.to_csv(os.path.join(sys.argv[1], f"{file_name}.csv"), index=False)
