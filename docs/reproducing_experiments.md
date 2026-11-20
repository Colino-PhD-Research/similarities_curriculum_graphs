# Reproducing the experiments in Soundarajan, Eliassi-Rad & Gallagher (2014)

This guide covers how to rerun the paper's cross-sectional (§5.3) and longitudinal (§5.4)
experiments with the code in this repo. It covers environment setup, data preparation, the
20 methods, and the analysis that produces each of the paper's tables and figures.

For what the paper specifies (and leaves unspecified), see
[soundarajan_2014_methods.md](soundarajan_2014_methods.md). This guide only covers how to run the experiments.

---

## 0. Status: what exists and what's missing

| Piece | Where | Status |
|---|---|---|
| 20 similarity methods | `similarity_measures/similarity_measures.py` | Done and tested |
| Scores, rankings, Kendall-Tau, p-values, nDCG, Kemeny consensus | `rankings/ranking_comparisons.py` | Done |
| Baselines (5% delete, 5% rewire) | `ranking_comparisons.make_baselines` | Done |
| Cross-sectional driver | `ranking_comparisons.cross_sectional_experiment` | Done |
| Complete linkage + 1000-run modal tree | `clustering/clustering.py` | Done |
| Amazon, DBLP, LJ1, LJ2, Email loaders | `load_data/load_data.py` | Done |
| Twitter / Yahoo snapshot builders | `load_data/load_data.snapshots` | Done |
| **Grad / Undergrad (Rice Facebook)** | — | **Missing**: no loader, and the Grad/Undergrad split isn't possible from `socfb-Rice31.mtx` alone (see §2.3) |
| Longitudinal end-to-end script (10 datasets, Fig. 11 top-5 overlap, Fig. 22) | `run_experiments/longitudinal_experiments.py` | Done, tested with `--toy`; reference dates still to fill in |
| Recurring clusters across references (Table 4) | `cross_sectional_experiments.cluster` | Done (§5.2) |
| 20-method registry | `run_experiments/methods.py` | Done |
| Cross-sectional end-to-end script (scores → tables → heatmaps → dendrograms → Table 4) | `run_experiments/cross_sectional_experiments.py` | Done, tested with `--toy` |

---

## 1. Environment

```bash
cd similarities_curriculum_graphs
python -m venv .venv            # this repo already has one in .venv (Python 3.13)
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install pytest
```

> **Import gotcha:** `pyproject.toml` only lists `similarity_measures` and `clustering` under
> `packages`, so `rankings` and `load_data` are **not** installed. They only import when you run
> from the repo root. Either always run from the root (`python -m run_experiments.cross_sectional_experiments`)
> or set `PYTHONPATH=.`. The permanent fix is to change `pyproject.toml` to:
> ```toml
> packages = ["similarity_measures", "clustering", "rankings", "load_data"]
> ```
> and rerun `pip install -e .`.

Check the install:

```bash
python -m pytest testing -v
```

---

## 2. Data

Everything lives in `datasets/`, which is gitignored. The loaders expect this layout:

```
datasets/
├── amazon0302.txt            # SNAP Amazon co-purchasing, March 2 2003
├── dblp_undirected.txt       # DBLP co-authorship
├── email-Enron.txt           # SNAP Enron email
├── lj1_sample.txt            # generated, see 2.2
├── lj2_sample.txt            # generated, see 2.2
├── livejournal/
│   ├── lj1_whole.txt         # SNAP soc-LiveJournal1 (directed, 4.8M nodes)
│   └── lj2_whole.txt         # SNAP com-LiveJournal (undirected, 4.0M nodes)
├── socfb-Rice31/socfb-Rice31.mtx
├── twitter/
│   ├── tweet_graph.data      # Twitter Replies
│   └── retweet_graph.data    # Twitter Retweets
└── yahoo/
    └── ydata-ymessenger-user-communication-pattern-v1_0-d00.txt … d27.txt   # Yahoo! Webscope G5
```

### 2.1 Direct-download networks
Amazon, DBLP and Email are standard SNAP edge lists. Put them in `datasets/` under the
filenames in `load_data.cross_sectional_files`.

### 2.2 LJ1 and LJ2
The paper's LiveJournal graphs are 500K-node samples. The two samplers are deterministic and
tuned to match Table 2:

```bash
python load_data/lj1_sample.py datasets/livejournal/lj1_whole.txt datasets/lj1_sample.txt   # ~6 GB peak RAM
python load_data/lj2_sample.py datasets/livejournal/lj2_whole.txt datasets/lj2_sample.txt   # ~3 GB peak RAM
```

Each script prints its Table-2 stats when it finishes. The module docstrings list the expected
values: LJ1 has 498,546 nodes and 10.68M edges; LJ2 has ~500K nodes and 11.01M edges.

### 2.3 Grad and Undergrad (not done yet)
The paper uses two networks from Rice University's Facebook: graduate students and
undergraduates. `socfb-Rice31.mtx` is only the full Rice graph (4,087 nodes, 184,828 edges), with
no node attributes, so it can't be split. Two options:
1. Find the Facebook100 attribute file for Rice31 (the original `.mat` release has a
   student/faculty status column), split the nodes by status, and take each induced subgraph.
2. Run without Grad/Undergrad and report it as a deviation. That leaves 5 references and 4 + 2 = 6
   comparison networks per reference, instead of the paper's 8.

Whichever you choose, add the result to `cross_sectional_files` (or write a small loader for the
`.mtx`) so that `load_cross_sectional_network("grad")` works.

### 2.4 Check the data against Tables 2 and 3

```bash
python -m load_data.dataset_stats results/        # prints both tables, writes CSVs
```

Compare each row with the paper's Table 2 (cross-sectional) and Table 3 (longitudinal ranges)
before you run anything else. It takes a few minutes, most of it spent loading LJ1, LJ2 and the
Twitter files.

---

## 3. The 20 methods

This is in [run_experiments/methods.py](../run_experiments/methods.py), so every experiment uses
the same names. The names match the paper's Table 1. In short:

```python
from functools import partial
from similarity_measures.similarity_measures import SimilarityMeasures, Similarity

sm, s = SimilarityMeasures(), Similarity()

# Vector-based: graph -> feature vector. Similarity is 1 - Canberra.
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

# Classifier- and matching-based: (G_ref, H) -> score
def _svm(method, **kw):
    return lambda G1, G2: s.find_similarity(G1, G2, "svm", method, **kw)

def _match(method, **kw):
    return lambda G1, G2: s.find_similarity(G1, G2, "match", method, **kw)

PAIRWISE_METHODS = {
    "NetSimileSVM":   _svm(sm.netsimile, netsimile_type="svm"),
    "NetSimile-Dist": _match(sm.netsimile, netsimile_type="match"),
}
for name, finder in {"AB": sm.AB, "BFS": sm.BFS, "RW": sm.RW, "RWR": sm.RWR}.items():
    PAIRWISE_METHODS[name] = _svm(finder)            # classifier
    PAIRWISE_METHODS[f"{name}-Dist"] = _match(finder)  # matching
```

Defaults the paper doesn't fix, which you should report:
- 300 sampled neighborhoods or communities per graph. Communities have size `k=10`.
- SVM: RBF kernel, standardized features, 5-fold stratified CV, with predictions pooled across folds.
- Seeds: `seed=0` for sampling, Infomap `seed=42` with 10 trials, and `rw_distance` uses `random.Random(0)`.
- Canberra isn't normalized, so vector and SVM scores can fall below 0. This doesn't affect
  rankings.

---

## 4. Cross-sectional experiment (§5.3)

### Quick way: the script

```bash
python -m run_experiments.cross_sectional_experiments --toy        # smoke test on 4 synthetic graphs (~5 min of CPU)
python -m run_experiments.cross_sectional_experiments              # amazon dblp lj1 lj2 email -> results/cross_sectional/
python -m run_experiments.cross_sectional_experiments --networks amazon email --out small --runs 1000
```

**Using other graphs.** The graphs come from a *network set*: any function that returns
`{name: networkx.Graph}`. `paper` (`cross_sectional_networks`, the default) and `toy` are
registered in `NETWORK_SETS` at the top of the script. To run on your own graphs, do one of:
- add your function to `NETWORK_SETS` and run with `--network-set <key>`
- leave it in any module and run with `--network-set my_package.my_module:my_function`
- from Python, call `run_experiment(my_function(), out_dir)`

`--networks a b c` keeps only those graphs from the set. If your function takes a `names`
argument, only those graphs are loaded. Each set writes to its own folder,
`results/<set name>/`, except `paper`, which writes to `results/cross_sectional/`.

It writes:
- `rankings/<ref>.pkl`: one pickled `RankingComparison` per reference, saved as soon as that
  reference finishes. References that are already done get skipped, so rerunning after a crash
  picks up where it stopped. **Delete this folder if you change a method**, or the old results
  will be reused.
- `tables/*.csv`: `table5_closest_to_consensus`, `fig7_distance_to_consensus`,
  `table4_recurring_clusters`, `dendrogram_stability`, and per-reference scores, ranks, Kendall
  distances, p-values, nDCG and consensus.
- `figures/*.png`: Kendall, nDCG and significance heatmaps, plus the modal dendrogram for each
  reference.

The Table 4 threshold defaults to the paper's 4 of 7, scaled to the number of references
(`--min-refs` overrides it).

### By hand

The script does the following steps:

```python
import pickle
from load_data.load_data import load_cross_sectional_network
from rankings.ranking_comparisons import cross_sectional_experiment
from run_experiments.methods import FEATURE_METHODS, PAIRWISE_METHODS

NAMES = ["amazon", "dblp", "lj1", "lj2", "email"]     # add "grad", "undergrad" once they exist
networks = {n: load_cross_sectional_network(n) for n in NAMES}

results = cross_sectional_experiment(
    networks, FEATURE_METHODS, PAIRWISE_METHODS,
    frac=0.05,   # 5% edges deleted / rewired
    seed=0,
)
pickle.dump(results, open("results/cross_sectional.pkl", "wb"))
```

`cross_sectional_experiment` uses each network in turn as the reference. It compares that
reference with every other network plus its own `"<ref>-deleted"` and `"<ref>-rewired"`
baselines. The result is a `{reference_name: RankingComparison}` dict.

### 4.1 Paper outputs and the code that produces them

| Paper | Code (with `results` from above) |
|---|---|
| Scores and rankings per reference | `results[ref].scores`, `results[ref].ranks` |
| Kendall-Tau heatmap per reference | `plot_heatmap(results[ref].kendall, ref)` |
| nDCG heatmap per reference | `plot_heatmap(results[ref].ndcg, ref)` |
| Significance map per reference | `plot_significance(results[ref], alpha=0.1)` |
| Method × method matrix averaged over references | `average_matrix(results, "kendall")` |
| **Table 5**: top 5 methods closest to consensus | `consensus_table(results, n=5)` |
| **Figure 7**: distance from consensus, methods × references | `consensus_distance_table(results)` |
| Dendrograms, **Table 4** | §5 below |

Everything above is imported from `rankings.ranking_comparisons`. Pass the same `order=[...]` to
every heatmap so the figures can be compared side by side.

---

## 5. Clustering the methods (§4.2, Table 4)

### 5.1 Representative dendrogram per reference

```python
from clustering.clustering import CompleteLinkageClustering
from clustering.utils import plot_dendrogram

trees = {}
for ref, r in results.items():
    sig, freq, n_distinct, Z = CompleteLinkageClustering(r.kendall).modal_tree(runs=1000)
    trees[ref] = sig
    print(f"{ref}: modal tree in {freq:.1%} of runs, {n_distinct} distinct trees")
    plot_dendrogram(Z, list(r.kendall.index), title=ref)
```

Ties are broken randomly on each run, and two trees count as the same if they have the same
nested groups, whatever order the merges happened in. Report `freq` and `n_distinct`. In the
`--toy` run (20 methods, 6 comparison networks), **every one of 200 runs gave a different tree**
for all 4 references. With only 6 comparisons, Kendall distances fall on a coarse grid, so
there are many ties. With real data the "modal" tree may be just as weak. If it is, Table 4
mostly reflects which tie-break happened to win, and you should say so.

### 5.2 Clusters that recur across references (Table 4)
The paper lists the clusters found for at least 4 of its 7 references. One reading of that:

```python
from collections import Counter

counts = Counter()
for ref, sig in trees.items():
    n_methods = len(results[ref].kendall)
    counts.update(g for g in sig if 1 < len(g) < n_methods)   # drop leaves and the root

min_refs = 4   # paper: >= 4 of 7; scale it if you have fewer references
table4 = sorted((sorted(g) for g, c in counts.items() if c >= min_refs), key=len)
```

---

## 6. Longitudinal experiment (§5.4)

The script is [run_experiments/longitudinal_experiments.py](../run_experiments/longitudinal_experiments.py).
For each dataset it builds the snapshots, chooses a reference snapshot, adds that snapshot's two
baselines, and runs the same analysis as the cross-sectional experiment.

```bash
python -m run_experiments.longitudinal_experiments --toy        # synthetic 8-snapshot series, ~2 min
python -m run_experiments.longitudinal_experiments \
    --reference replies=2009-06-?? --reference retweets=2009-0? --reference yahoo=2008-0?-??
python -m run_experiments.longitudinal_experiments --auto-reference --datasets yahoo yahoo-4day
```

Output goes to `results/longitudinal/`, organized like the cross-sectional output. It saves one
results file per dataset in `rankings/` and resumes after a crash the same way, so **delete
`rankings/` after changing a method or a reference**. Its `tables/` folder also has
`references.csv`, `closest_to_consensus.csv` (the longitudinal columns of Table 5),
`fig22_distance_to_consensus.csv` and `top5_overlap_<dataset>.csv` (Figure 11). The top-5
overlap is also plotted as a heatmap in `figures/`.

Snapshot labels are `"YYYY-MM-DD"` for daily windows, `"YYYY-MM"` for monthly windows, and the
first date of each bin for `days=` windows.

### 6.1 Choosing the reference
The paper uses the snapshot that Berlingerio et al. (2013) found to be anomalous, and says
the exact choice isn't essential. The script takes each original dataset's reference, in this order:
1. `--reference BASE=LABEL` on the command line (`BASE` is `replies`, `retweets` or `yahoo`)
2. `REFERENCE_DATES` at the top of the script (all `None` for now)
3. `--auto-reference`: the snapshot whose edge count is furthest from the median, on a log scale

If none of these gives a reference, the script stops and says which one is missing. The
choice is saved to `tables/references.csv`. If you use the automatic rule, report it as a
deviation. For Yahoo it picks **2008-04-08**, the only day with all 100,000 users active
(about twice the edges of any other day), so that is probably the paper's anomaly too. For
Retweets it just picks the largest month (2009-09), since the network grows every month, so
the automatic rule isn't meaningful there.

### 6.2 Re-aggregated datasets (7 in total)

| `--datasets` name | Snapshots | Reference |
|---|---|---|
| `replies-7day` | 5 weekly windows from Jun 1 (the last one covers only Jun 29–30) | the window containing the replies reference |
| `replies-3day` | 10 three-day windows | the window containing the replies reference |
| `yahoo-7day` | 4 weekly windows from Mar 31 | the window containing the yahoo reference |
| `yahoo-4day` | 7 four-day windows | the window containing the yahoo reference |
| `replies-cumulative` | 30, where snapshot t = all of Jun 1..t | last snapshot |
| `retweets-cumulative` | 5 | last snapshot |
| `yahoo-cumulative` | 28 | last snapshot |

The split into 4 horizontal and 3 cumulative datasets is my reading of §5.4. Check it against
the paper.

### 6.3 Snapshot sizes (for estimating runtime)
- **Replies:** each day has 4–9K nodes and is very sparse, with only 1–6% of nodes in the
  largest component. The community samplers still find their 300 communities, in about 1 s each.
- **Retweets:** each month has 26–120K nodes.
- **Yahoo:** each day has 29–100K nodes.
- **Cumulative:** the final snapshot has 107K nodes for replies, 218K for retweets, and 100K
  nodes with 588K edges for yahoo.

Yahoo daily (28 snapshots) and the cumulative datasets will take the longest.

---

## 7. Practical notes

- **Runtime.** On four 400-node toy graphs, all 20 methods took about 5 minutes (4 references ×
  6 comparisons). Real networks are 3–5 orders of magnitude bigger, and NetSimile's per-node
  features, Infomap and the community features on LJ1/LJ2 will dominate. Run the methods
  **one reference at a time** and pickle each `RankingComparison`. Consider running LJ1/LJ2 on
  a cluster.
- **Repeated work.** `similarity_scores` computes the reference's samples again for every
  comparison network when it runs a pairwise method. With 8 comparisons that's 8× the work on
  the reference. If runtime becomes a problem, cache `method(G_ref)` and score from the cached
  feature matrices (`find_similarity` accepts any callable, so `lambda F: F` passes matrices
  through, as `testing/test_all_measures.py` does).
- **Memory.** Loading LJ1 or LJ2 into networkx takes about 20 s and several GB. Avoid having
  both loaded at once alongside their baselines (each baseline is a full copy).
- **NaN Kendall-Tau.** If a method gives every comparison network the same score, `kendalltau`
  returns NaN and `CompleteLinkageClustering` raises an error. Look at `r.scores` for
  constant rows before clustering.
- **Reproducibility.** Every sampler is seeded except the clustering tie-breaking, which uses
  `np.random.default_rng()` with no seed. Give it a seed if you need identical dendrogram
  counts from run to run.
- **Record the deviations** from §6 of the methods doc (SVM settings, sample sizes,
  unnormalized Canberra, rewiring by double-edge swaps, missing Grad/Undergrad) in any
  write-up.

---

## 8. Checklist

1. [ ] Fix `pyproject.toml` packages, then run `pip install -e .`
2. [ ] Run `pytest testing` and confirm it passes
3. [ ] Put the datasets in place. Generate `lj1_sample.txt` and `lj2_sample.txt`
4. [ ] Decide how to handle Grad/Undergrad (§2.3)
5. [ ] Run `python -m load_data.dataset_stats results/` and compare with Tables 2 and 3
6. [x] `run_experiments/methods.py` (§3)
7. [ ] Run `python -m run_experiments.cross_sectional_experiments --toy`, then run it on the real networks (§4).
   This covers the heatmaps, Table 5, Figure 7, the dendrograms and Table 4
8. [ ] Look up the anomalous snapshots (Berlingerio et al. 2013) and put them in `REFERENCE_DATES` (§6.1)
9. [ ] Run `python -m run_experiments.longitudinal_experiments` (all 10 datasets, Fig. 11, Fig. 22)
