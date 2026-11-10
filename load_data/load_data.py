import os
import networkx

cross_sectional_networks = ["amazon", "dblp", "livejournal", "email"] # split up livejournal? include rice?

longitudinal_networks = ["twitter replies", "twitter retweets", "yahoo"]

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

cross_sectional_files = {
    "amazon": "amazon0302.txt",
    "dblp": "dblp_undirected.txt",
    "lj1": "lj1_sample.txt",
    "lj2": "lj2_sample.txt",
    "email": "email-Enron.txt",
}

def load_cross_sectional_network(network_name):
    if network_name not in cross_sectional_networks:
        raise Exception(f"{network_name} invalid. Valid options are {cross_sectional_networks}")

    # SNAP edge list: "#" header lines, then one "FromNodeId<tab>ToNodeId" per line
    # loaded as undirected since the similarity measures assume undirected graphs
    path = os.path.join(DATASETS_DIR, cross_sectional_files[network_name])
    return networkx.read_edgelist(path, comments="#", nodetype=int, create_using=networkx.Graph)

def load_longitudinal_network(network_name):
    if network_name not in longitudinal_networks:
        raise Exception(f"{network_name} invalid. Valid options are {longitudinal_networks}")
