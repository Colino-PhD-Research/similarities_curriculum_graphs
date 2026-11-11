# Methods: Soundarajan, Eliassi-Rad & Gallagher (2014), *A Guide to Selecting a Network Similarity Method*

This is a summary of what the paper specifies, plus notes on what it leaves unspecified.
Section numbers refer to the paper.

## 1. The task: network-similarity ranking (§1, §4)

You're given a reference network $G_r$ and comparison networks $H_1, \dots, H_k$. Each
similarity method scores $\text{sim}(G_r, H_i)$ and ranks the $H_i$ from most to least similar.
The paper compares methods by their **rankings**, not their raw scores, so methods whose scores
fall in different ranges can still be compared.

None of the methods need node correspondences between the two networks.

## 2. How the methods are categorized (§3)

Each method is classified on two axes.

**Level**: where the features come from.
- **Micro**: individual nodes or egonets (the 1-hop induced subgraph around a node).
- **Mezzo**: communities.
- **Macro**: the whole network.

**Comparison type**: how two networks are compared.

| Type | Procedure | Similarity |
|---|---|---|
| **Vector-based** | Build one feature vector per network: $F_1$, $F_2$. | $1 - \text{Canberra}(F_1, F_2)$ |
| **Classifier-based** | Sample a fixed number of structures (neighborhoods, communities, ...) in each network and compute a feature vector for each, labeled by its network. Cross-validate an SVM that tries to tell $G_1$'s vectors from $G_2$'s. For each network, build a length-2 vector $F_i$ = [fraction of its vectors classified as $G_1$, fraction classified as $G_2$]. | $1 - \text{Canberra}(F_1, F_2)$ |
| **Matching-based** | Use the same structures and feature vectors as the classifier-based version. Build a complete bipartite graph between $G_1$'s vectors and $G_2$'s vectors, with edge weight = Canberra distance. Find the least-cost matching. | $1 - \text{mean edge cost of the matching}$ |

Canberra distance: $\text{Canberra}(U, V) = \sum_{i=1}^{n} \frac{|U_i - V_i|}{|U_i| + |V_i|}$

Macro-level methods only produce one structure per network (the whole network), so they can only be vector-based.

## 3. The 20 methods (Table 1, §3)

|  | Micro | Mezzo | Macro |
|---|---|---|---|
| **Vector** | NetSimile | d-RW-Dist, IMIn, IMKnown, IMIn&Known | Degree, Density, Transitivity, Eigenvalues, LBD |
| **Classifier** | NetSimileSVM | AB, BFS, RW, RWR | – |
| **Matching** | NetSimile-Dist | AB-Dist, BFS-Dist, RW-Dist, RWR-Dist | – |

> **Naming:** Table 1 calls the matching methods "-Match" (NetSimile-Match, AB-Match, ...).
> The rest of the paper (method list, Tables 4–5, figures) calls them "-Dist". Don't mix up
> **RW-Dist** (random-walk *community* + matching) with **d-RW-Dist** (random-walk *distances*,
> a vector method).

### Micro level

**NetSimile** (vector, 35 features). Compute 7 features for each node:
1. degree
2. clustering coefficient
3. average degree of its neighbors
4. average clustering coefficient of its neighbors
5. number of edges in its egonet
6. number of edges leaving its egonet
7. number of nodes adjacent to its egonet

For each feature's distribution over all nodes, take the median and the first four moments.
That gives 7 features × 5 statistics = 35 values.

**NetSimileSVM** (classifier). Sample 300 node neighborhoods, compute the same 7 features for each, then run the classifier procedure.

**NetSimile-Dist** (matching). Same 300 neighborhoods and features as NetSimileSVM, then run the matching procedure.

### Mezzo level

**d-RW-Dist (Random Walk Distances)** (vector, 20 features). For each $d \in \{10, 20, 50, 100\}$:
1. Run 100 random walks of length $d$, each starting from a random node $n$ and ending at some node $m$.
2. Record the shortest-path distance from $n$ to $m$.
3. Take the median and first four moments of those 100 distances.

That gives 4 walk lengths × 5 statistics = 20 values.

**IMIn, IMKnown, IMIn&Known** (vector). Run Infomap community detection. For each node $n$ with community $C$:
- **IMIn** (1 feature): the fraction of $n$'s neighbors that are in $C$, averaged over all nodes.
- **IMKnown** (1 feature): the fraction of $C$'s nodes that are adjacent to $n$, averaged over all nodes.
- **IMIn&Known** (2 features): both of the above.

**AB, BFS, RW, RWR** (classifier). Find 300 communities per network, using one of:
- **AB**: the (α, β)-community swap algorithm
- **BFS**: breadth-first search
- **RW**: a random walk without restart
- **RWR**: a random walk that restarts at the start node with probability 0.15

For each community, compute a 36-feature vector (conductance, diameter, density, etc.). The
full list is in Abrahao et al., KDD 2012. Then run the classifier procedure.

**AB-Dist, BFS-Dist, RW-Dist, RWR-Dist** (matching). Same communities and features as above, then run the matching procedure.

### Macro level (all vector-based)

- **Degree**: average degree (1 feature).
- **Density**: edge density (1 feature).
- **Transitivity**: global transitivity (1 feature).
- **Eigenvalues (Eigs)**: the $k$ largest eigenvalues, with $k = 10$ (10 features).
- **LBD (Leadership–Bonding–Diversity)** (3 features, from Richards & Macindoe 2010):
  - **Leadership**: how much the highest-degree node dominates the network's connectivity. It's the average difference between the max degree and every other node's degree.
  - **Bonding**: transitivity, i.e. closed triads divided by all (open and closed) triads.
  - **Diversity**: the number of disjoint dipoles, normalized to a range similar to the other two statistics.
  - The paper compares two networks by "the difference" between their LBD vectors. The table in §2 suggests this is Canberra.

## 4. Comparing the methods (§4, Figure 1)

All of this runs **once per reference network**. With $m$ methods there are $m$ rankings of the $k$ comparison networks.

### 4.1 Rank correlation
- **Kendall-Tau distance** between every pair of rankings: the probability that two randomly chosen items appear in different orders in the two rankings. 0 means identical rankings, 0.5 means no correlation, 1 means one ranking is the reverse of the other.
- **p-value** for every Kendall-Tau correlation. The paper reports significance at $p = 0.1$.
- **nDCG** as a check on Kendall-Tau, since it gives more weight to the top of the list. Each pair is computed twice, treating each ranking as the "true" one in turn, and the two results are averaged.

### 4.2 Clustering the methods
- **Complete-linkage hierarchical clustering** on the $m \times m$ Kendall-Tau distance matrix. The paper chose it because it tends to produce many small, tight clusters.
- Run the clustering **$l = 1000$ times** for each reference network.
- Treat each resulting dendrogram as an **unordered tree**, ignoring the order of merges. Keep the tree that comes up most often as the **representative dendrogram**.
- Report the clusters that recur across reference networks (Table 4 lists those found for ≥ 4 of the 7 networks).

> **Implementation note:** complete linkage is deterministic. Repeated runs only differ when
> there are **ties**, which happen when two pairs of clusters are equally close. With $k = 8$
> comparison networks, Kendall-Tau distances can only take values in multiples of $1/28$, so
> ties are common. Reproducing the 1000-run procedure therefore requires breaking ties
> randomly (for example by shuffling the method order before each run). Merges at equal
> heights then count as the same tree. The paper doesn't explain how it breaks ties.

### 4.3 Consensus ranking
- **Kemeny–Young**: build a $k \times k$ preference matrix $P$, where $P_{ij}$ is the number of methods that rank network $i$ above network $j$. Score each possible ranking $R$ by summing $P_{ij}$ over every pair that $R$ places $i$ above $j$. The highest-scoring ranking is the consensus. If each method's ranking is treated as a noisy estimate of a true ranking, this consensus is the maximum-likelihood estimate.
- Rank each method by its Kendall-Tau distance to the consensus (Table 5 shows the top 5 per network; Figure 22 shows the longitudinal results).

## 5. Experimental setup (§5)

### Cross-sectional (§5.3)
- **7 reference networks**: Grad and Undergrad (Rice University Facebook), Amazon, DBLP, LJ1 and LJ2 (LiveJournal), Email (Enron).
- **2 baselines per reference network:**
  - delete a random 5% of edges
  - rewire a random 5% of edges, keeping the degree distribution
- For each reference, rank the other **8 networks** (6 other datasets + 2 baselines) with all 20 methods, then apply all of §4.

### Longitudinal (§5.4)
- **Datasets:**
  - Twitter Replies: 30 daily networks, June 2009
  - Twitter Retweets: 5 monthly networks, May–September 2009
  - Yahoo! IM: 28 daily networks, April 2008
- The **reference** is the network previously identified as anomalous in Berlingerio et al. 2013. The paper says the choice of reference isn't essential. Baselines are built by edge deletion and rewiring, as in the cross-sectional setup.
- **Extra analysis:** the overlap between the top 5 of each pair of rankings (Figure 11).
- **Re-aggregated datasets** (7 in total), to test whether the daily networks were too sparse to rank meaningfully:
  - *Horizontal*: windows of a week and of 3 days (Twitter Replies) or 4 days (Yahoo! IM). The reference is the window that contains the original reference network.
  - *Vertical (cumulative)*: network $t$ contains all data up to time $t$. The reference is the final network.
- On the re-aggregated datasets, rerun the correlations, clustering and consensus.

## 6. What the paper doesn't specify

| Detail | Paper says |
|---|---|
| SVM kernel, number of folds, feature scaling | not specified |
| How $F_1$/$F_2$ are combined across cross-validation rounds | built "in each round", but how rounds are combined isn't stated |
| How the 300 neighborhoods/communities are sampled, and their size | not specified |
| Community feature list (36 features) | deferred to Abrahao et al. 2012 |
| "First four moments" | mean, variance, skewness, kurtosis are implied but not defined |
| Directed or weighted graphs | out of scope (treated as undirected and unweighted) |
| Normalizing Canberra distance | none. $1 - \text{Canberra}$ can be negative when there's more than one feature, even though §5.2 says scores are in $[0, 1]$ |
| LBD diversity normalization | "a similar range" with no formula |
| Tie-breaking in clustering, and how to count identical trees | not specified (see §4.2) |
| How ties in rankings are handled | not specified |
