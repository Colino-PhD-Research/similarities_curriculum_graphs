import os
import networkx

"""
Main functions to use: 
    load_cross_sectional_network
    snapshots

Everything else is helper
"""


longitudinal_networks = ["twitter replies", "twitter retweets", "yahoo"]

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

# note lj1 and lj2 take a long time to load (~20 seconds)
cross_sectional_files = {
    "amazon": "amazon0302.txt",
    "dblp": "dblp_undirected.txt",
    "lj1": "lj1_sample.txt",
    "lj2": "lj2_sample.txt",
    "email": "email-Enron.txt",
}

# usage ex: G = load_cross_sectional_network("lj2")
def load_cross_sectional_network(network_name):
    if network_name not in cross_sectional_files.keys():
        raise Exception(f"{network_name} invalid. Valid options are {cross_sectional_files.keys()}")

    # SNAP edge list: "#" header lines, then one "FromNodeId<tab>ToNodeId" per line
    # loaded as undirected since the similarity measures assume undirected graphs
    path = os.path.join(DATASETS_DIR, cross_sectional_files[network_name])
    return networkx.read_edgelist(path, comments="#", nodetype=int, create_using=networkx.Graph)

# use for longitudinal data
def snapshots(source, freq="D", days=None, start=None, end=None, types=None,
              cumulative=False):
    """Split interactions into time periods and build one graph per period.

    source: For twitter, pass exact file (there's only two: one for tweets the other retweets). For yahoo, pass folder or list/tuple of individual files

    freq:   calendar period, e.g. "D" daily, "W" weekly (Mon-Sun), "M" monthly.
            Ignored when days is given.
    days:   fixed-width bins of this many days, counted from start (or from
            the first day in the data). Use this for the paper's 3-day,
            4-day and 7-day aggregations.
    start, end: optional date bounds, inclusive, e.g. "2009-06-01".
    types:  optional list of values to keep from the type column.
    cumulative: if True, each graph contains everything up to that period.

    Returns a dict {period label: networkx.Graph}, in time order. Labels are
    the period ("2009-06", "2009-06-01") or, with days=, the bin's first date.
    """
    df = load(source)

    if types is not None:
        df = df[df["type"].isin(types)]
    day = df["time"].dt.normalize()
    if start is not None:
        df, day = df[day >= pd.Timestamp(start)], day[day >= pd.Timestamp(start)]
    if end is not None:
        df, day = df[day <= pd.Timestamp(end)], day[day <= pd.Timestamp(end)]
    if df.empty:
        return {}

    if days is not None:
        origin = pd.Timestamp(start) if start is not None else day.min()
        bin_start = origin + pd.to_timedelta((day - origin).dt.days // days * days,
                                             unit="D")
        labels = bin_start.dt.strftime("%Y-%m-%d")
    else:
        labels = df["time"].dt.to_period(freq).astype(str)

    graphs, running = {}, nx.Graph()
    for label, group in df.groupby(labels, sort=True):
        if cumulative:
            running = add_edges(running.copy(), group)
            graphs[label] = running
        else:
            graphs[label] = to_graph(group)
    return graphs


######################## Longitudinal Helper Functions #######################################

"""
Load the longitudinal datasets from Soundarajan, Eliassi-Rad & Gallagher,
"A Guide to Selecting a Network Similarity Method", and build time-sliced
graphs from them.

Both loaders return a DataFrame with the same core columns, so the same
snapshot code works for either dataset:

    time  (datetime)   when the interaction happened
    src   (str)        source user
    dst   (str)        target user (or hashtag/URL for some Twitter types)
    type  (str)        interaction type

Twitter (.data CSV files)
    tweet_graph.data   types: message, mention_user, mention_topic, mention_url
    retweet_graph.data types: rt_auth_orig, rt_auth_quot, rt_int_orig

Yahoo! Messenger (Webscope G5, one space-separated .txt file per day)
    columns: day  time  sender  zip  receiver  in_contacts
    Day N is N days after March 31, 2008: D00-MON is Monday, March 31 and
    D24-THU is Thursday, April 24 (files cover D00-D27, Mar 31 - Apr 27).
    Extra columns kept: zip, in_contacts ('y', 'n' or '-').
    type is always 'message'.

Examples
    from longitudinal_data import load, snapshots

    # Twitter Retweets: 5 monthly graphs, May-Sep 2009
    rt = load("datasets/twitter/retweet_graph.data")
    retweets = snapshots(rt, freq="M", start="2009-05-01", end="2009-09-30")

    # Twitter Replies: 30 daily graphs, June 2009
    tw = load("datasets/twitter/tweet_graph.data")
    replies = snapshots(tw, freq="D", start="2009-06-01", end="2009-06-30")

    # Yahoo! IM: 28 daily graphs, then 4-day and 7-day bins as in the paper
    ym = load("datasets/yahoo")              # folder, glob, file or list
    daily   = snapshots(ym, freq="D")
    fourday = snapshots(ym, days=4)          # bins start at D00, Mar 31
    weekly  = snapshots(ym, days=7)
    cumul   = snapshots(ym, freq="D", cumulative=True)

    # Filtering: use types= for the type column, or filter the frame first
    contacts_only = snapshots(ym[ym["in_contacts"] == "y"], freq="D")
"""
import glob
import os

import networkx as nx
import pandas as pd

YAHOO_COLS = ["day", "clock", "src", "zip", "dst", "in_contacts"]
YAHOO_DAY0 = pd.Timestamp("2008-03-31")  # date of D00
WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_twitter(path):
    """Read a Twitter .data file (tweet_graph.data or retweet_graph.data)."""
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["start", "end", "src", "dst", "type"],
                     dtype=str, keep_default_na=False)
    # keep_default_na=False stops usernames like "NA" or "null" becoming NaN
    df["time"] = pd.to_datetime(df["start"], format="%Y-%m-%d %H:%M:%S")
    return df[["time", "src", "dst", "type"]]  # start == end on every row


def _yahoo_files(source):
    """Turn a folder, glob pattern, single file or list into a file list."""
    if isinstance(source, (list, tuple)):
        files = list(source)
    elif os.path.isdir(source):
        files = glob.glob(os.path.join(source, "ydata-ymessenger-*.txt"))
    else:
        files = glob.glob(source)
    if not files:
        raise FileNotFoundError(f"No Yahoo! Messenger files found in {source!r}")
    return sorted(files)


def load_yahoo(source):
    """Read one or more daily Yahoo! Messenger files into one DataFrame."""
    frames = [pd.read_csv(f, sep=" ", header=None, names=YAHOO_COLS,
                          dtype=str, keep_default_na=False)
              for f in _yahoo_files(source)]
    df = pd.concat(frames, ignore_index=True)

    # "D24-THU" -> day 24, weekday THU; day N is N days after D00 (Mar 31)
    parts = df["day"].str.extract(r"D(\d+)-([A-Z]{3})")
    day_num = parts[0].astype(int)
    date = YAHOO_DAY0 + pd.to_timedelta(day_num, unit="D")
    # check if day of week matches up with date
    bad = date.dt.dayofweek.map(lambda i: WEEKDAYS[i]) != parts[1]
    if bad.any():
        raise ValueError("Day labels don't match 2008 weekdays, e.g. "
                         f"{df.loc[bad, 'day'].iloc[0]}")

    df["time"] = date + pd.to_timedelta(df["clock"])
    df["type"] = "message"
    return df[["time", "src", "dst", "type", "zip", "in_contacts"]]


def load(source):
    """Detect the dataset from the path and call the matching loader.

        For twitter, pass exact file (there's only two: one for tweets the other retweets)

        For yahoo, pass folder or list/tuple of individual files
    
    """
    is_list = isinstance(source, (list, tuple))
    names = source if is_list else [source]
    first = os.path.basename(str(names[0]))
    is_dir = not is_list and os.path.isdir(source)
    if is_dir and not glob.glob(os.path.join(source, "ydata-ymessenger-*.txt")):
        # e.g. datasets/twitter: several .data files, so the file must be named
        raise ValueError(f"{source!r} is not a Yahoo! folder; pass a Twitter "
                         "file instead, e.g. tweet_graph.data or "
                         "retweet_graph.data")
    if (is_list or is_dir
            or first.startswith("ydata-ymessenger") or "*" in first):
        return load_yahoo(source)
    return load_twitter(source)


# --------------------------------------------------------------------------
# Graphs
# --------------------------------------------------------------------------

def add_edges(G, df):
    """Add df's interactions to G as undirected edges, skipping self-loops."""
    df = df[df["src"] != df["dst"]]
    G.add_edges_from(zip(df["src"], df["dst"]))
    return G


def to_graph(df):
    """Undirected simple graph: no self-loops, repeated pairs merged."""
    return add_edges(nx.Graph(), df)





