# RANNS: Density-aware Exploration

Experimental companion to **Reverse Approximate Nearest Neighbor Search: From Hardness to Density-aware Exploration**.

This project studies candidate exploration for high-dimensional reverse approximate nearest neighbor search (RANNS). The paper evaluates **Density-aware Exploration (DE)** against range- and hop-based strategies, and examines how density guidance and spatial bounding affect reverse-neighbor discovery under a limited candidate budget.

This README describes the experimental settings and results reported in the paper. Figure images are provided for reference; they do not replace the underlying measurements or experiment configurations.

## Overview

For a query $q$, reverse $k$-nearest neighbor search identifies data points that regard $q$ as one of their $k$ nearest neighbors. Let $r_k(p)$ be the distance from $p$ to its $k$-th nearest **nonself** neighbor. The exact reverse-neighbor set is
$$
\mathcal{R}_k(q)=\{p\in\mathcal{P}:\operatorname{dist}(p,q)\le r_k(p)\}.
$$

The number of reverse neighbors varies across queries and need not equal $k$. A distant point can still be a reverse neighbor when its own neighborhood radius is large enough to cover the query.

DE starts with forward ANN retrieval, then uses a local sparsity signal to guide graph exploration. A spatial bound limits expansion around the query. The candidate verification cap $B$ limits the number of distinct candidates passed to verification; it is separate from the index-entry access budget used in the paper's hardness analysis.

## Experimental Coverage

| Study | Setting | Paper reference |
| --- | --- | --- |
| QPS–reverse-recall trade-off | Five datasets, four graph indexes, $k=50$; non-graph RaBitQ reference | Figure 5 |
| Neighborhood-size comparison | GIST1M, $k\in\{25,50,75,100\}$, fixed candidate budget | Figure 6 |
| Candidate-budget comparison | SIFT1M, MNIST, DEEP1M, and GIST1M on HNSW; $B\in\{800,1600,3200,6400,12800\}$ | Table 2 |
| Component ablation | SIFT1M, four indexes, $k=50$, $B=3200$; Base, +DG, +SB, and Full | Figure 7 |
| Additional neighborhood-size comparison | SIFT1M, $k\in\{25,50,75,100\}$ | Appendix B, Figure 8 |
| Density-tolerance sensitivity | SIFT1M, $B=3200$, $R_{\max}=10\widehat r_k(q)$ | Appendix C.1, Figure 9 |
| Expansion-radius sensitivity | SIFT1M, $B=3200$, $\alpha=1$ | Appendix C.2, Figure 10 |
| Cumulative distant-neighbor discovery | First-discovery positions in ordered candidate lists | Appendix D.6, Figure 11 |
| Construction and query-stage costs | SIFT1M, $k=50$, $B=3200$, $\alpha=1$, $R_{\max}=10\widehat r_k(q)$ | Appendix E, Figure 12 and Table 4 |

## Methods and Graph Indexes

The graph-based comparisons evaluate candidate-exploration strategies based on the following methods:

| Method | Candidate exploration |
| --- | --- |
| **SFT** | Retrieves candidates from the query's forward neighborhood using range-based exploration. |
| **RDT** | Uses incremental forward-neighbor retrieval, intrinsic-dimensionality estimates, and local distance tests. |
| **HAMG** | Broadens candidate coverage through multi-hop graph exploration. |
| **DE** | Combines density guidance and spatial bounding to allocate the available candidate budget. |

A non-graph **RaBitQ** implementation with 1-bit quantization is included as a reference in Figure 5. It is not part of the component-ablation study.

The evaluated graph indexes are **HNSW, HAMG, NN-Descent, and NSG**. Index names describe the graph used for initialization and traversal; method names describe how candidates are explored. **HAMG** appears in both roles.

In the controlled SIFT1M studies, the HAMG graph label denotes the HNSW-derived implementation described in Appendix A. Comparisons within an index reuse the same graph; the ablation variants also share initialization and adjacency order.

## Data and Ground Truth

The main benchmark comparisons cover **MNIST, SIFT1M, DEEP1M, GIST1M, and SIFT10M**. Use the dataset variants and preprocessing specified for the corresponding experiment.

The existing ground-truth entry point is `Ground_truth.py`; the configuration described in the original project README uses $k=100$. Configure dataset paths and the required neighborhood ranks for your experiment before running the script. A ground-truth file for one value of $k$ should not be treated as a substitute for the candidate radii at another rank.

For each tested $k$, verification requires the corresponding candidate-specific $r_k(p)$. Reverse-neighbor truth uses the distance condition above, retaining boundary ties. Query self IDs must be excluded when queries are sampled from the base vectors.

### Two Evaluation Protocols

The main benchmark and controlled SIFT1M studies use different query protocols. Keep them separate when comparing results.

| Setting | Main benchmark comparisons | Controlled component and supplementary studies |
| --- | --- | --- |
| Dataset | Five benchmark datasets; SIFT1M/GIST1M for varying-$k$ plots | SIFT1M, raw 128-dimensional descriptors |
| Evaluation queries | 500 sampled from the corresponding query set | 500 distinct base-vector queries, sampled without replacement with seed 482 |
| Self matches | Apply the experiment's query/base identity policy | Query self IDs excluded |
| Neighborhood size | $k=50$ for Figure 5 and Table 2; varying $k$ for Figures 6 and 8 | $k=50$ |
| Parameter selection | Match the configuration for the benchmark being reproduced | 100 additional parameter-selection queries, disjoint from evaluation queries |

The controlled studies use Euclidean distance and independently established exact reverse-neighbor ground truth. Mean Rev-Recall and distant Rev-Recall use 499 and 484 nonempty query sets, respectively; search QPS includes all 500 evaluation queries.

## Evaluation Metrics

**Rev-Recall@$k$** measures the fraction of a query's exact reverse neighbors recovered:

$$
\operatorname{RevRecall}@k(q)=
\frac{|\mathcal{R}_k(q)\cap\widehat{\mathcal{R}}_k(q)|}
{|\mathcal{R}_k(q)|}.
$$

**Distant Rev-Recall** applies the same measure to true reverse neighbors outside the query's own exact $k$-NN radius:

$$
\mathcal{R}^{\mathrm{dist}}_k(q)=
\{p\in\mathcal{R}_k(q):\operatorname{dist}(p,q)>r_k(q)\}.
$$

**QPS** is query throughput. Report its timing scope explicitly. In the controlled studies, search timing includes native ANN initialization and DE exploration, but excludes graph construction, density preparation, truth evaluation, and candidate verification. These search-stage measurements should not be interpreted as full-query latency including verification.

**Budget utilization** is the fraction of the candidate cap actually used. The 50 initialization seeds count toward $B$ in the controlled studies; traversal stops at the cap or when the admissible frontier is exhausted. Equal caps therefore do not imply equal candidate counts or verification workloads.

## Component Ablation

The ablation isolates **density guidance (DG)** and **spatial bounding (SB)**:

| Variant | Density guidance | Spatial bounding |
| --- | --- | --- |
| **Base** | Disabled | Disabled |
| **+DG / Density only** | Enabled | Disabled |
| **+SB / Spatial only** | Disabled | Enabled |
| **Full / Full DE** | Enabled | Enabled |

Figure 7 uses SIFT1M, $k=50$, and $B=3200$. Within each index, the four variants share the native graph, initial retrieval of 50 distinct nonself candidates, and adjacency order. Density-enabled variants use $\alpha=1$; spatial-enabled variants use $R_{\max}=1.25\widehat r_k(q)$ so the bound is active. This is a diagnostic radius setting, distinct from the larger radius used in the tolerance and cost studies.

<img width="866" height="979" alt="image" src="https://github.com/user-attachments/assets/4fbb1884-210c-40fb-9d30-2aaa741c7f50" />


At the displayed budget, each component improves overall and distant reverse recall over Base across the four indexes, and Full gives the highest recall on both metrics. The benefit is not universal across budgets: on NN-Descent at $B=12800$, Full has lower Rev-Recall than +SB. Interpret component gains under their reported budget and radius settings.

## Parameter Sensitivity

Both sensitivity studies use the same offline ANN-derived 50-NN radius scores across indexes. The scores come from an HNSW reference and occupy one float per base vector, or 4.0 MB for SIFT1M. The controlled study selects the ANN-radius estimator and $\alpha=1$ using the disjoint parameter-selection queries.

### Density Tolerance

Fix $B=3200$ and $R_{\max}=10\widehat r_k(q)$, then vary $\alpha$. The transition rule allows a candidate $v$ from parent $u$ when $\widehat\rho(v)\le\alpha\widehat\rho(u)$, together with the spatial condition. At $\alpha=1$, surrogate density cannot increase along accepted transitions; larger finite tolerances permit temporary increases. The density-disabled reference is denoted by $\alpha=\infty$.

Figure 9 reports **absolute percentage-point gains** over density-disabled traversal on the same graph. These are differences in recall, not raw recall values or relative percentage improvements. Use the finite tolerance grid recorded with the experimental configuration.

<img width="853" height="930" alt="image" src="https://github.com/user-attachments/assets/66e489e9-9f89-4d7c-bae9-df5406510c4d" />


### Maximum Expansion Radius

Fix $B=3200$ and $\alpha=1$, then vary $R_{\max}=c\widehat r_k(q)$ for

$$
c\in\{1,1.25,1.5,2,3,5,10,\infty\}.
$$

Infinity disables the spatial cap. Figure 10 compares density-enabled and density-disabled traversal **at the same radius on the same index**. An excessively tight radius can restrict useful paths, so increasing pruning strength does not necessarily improve recall.

<img width="867" height="930" alt="image" src="https://github.com/user-attachments/assets/bdf961c9-03df-4a79-88dd-81417686d4fc" />


## Distant-Neighbor Discovery and Costs

### Cumulative Discovery

Record each true distant reverse neighbor's first position in the ordered candidate list from the longest traversal. Prefixes of that list measure cumulative discovery as the available candidate budget grows. The study checks the five budget endpoints against separately measured shorter runs.

<img width="1066" height="561" alt="image" src="https://github.com/user-attachments/assets/db22a569-647d-4888-bdc8-201feb050924" />


### Cost Breakdown

The cost study uses $k=50$, $B=3200$, $\alpha=1$, and $R_{\max}=10\widehat r_k(q)$. It separates target-graph construction, shared offline density preparation, native ANN initialization, and DE exploration.

<img width="863" height="971" alt="image" src="https://github.com/user-attachments/assets/15be2e0b-494d-48ef-86ab-336748e017f8" />


The figure compares DE integrations across graph indexes. It does not time all competing RANNS methods. NSG construction includes its NN-Descent input graph. The common HNSW reference and density scores are offline preparation costs; they are not repeated for each query.

Individual-query stage means and parallel batch QPS are different measurements. The stage means use 500 queries and three repetitions with eight workers and should not be interpreted as the inverse of parallel batch QPS.

## Main Benchmark Figures

### QPS–Rev-Recall@50 — Figure 5

The comparison covers five datasets and four graph indexes, with RaBitQ included as a non-graph reference.

<img width="4740" height="4356" alt="image" src="https://github.com/user-attachments/assets/1000ff0c-158d-48b3-9ac7-2b67e04621a3" />


### Neighborhood Size on SIFT1M and GIST1M — Figures 6 and 8

The combined figure presents the SIFT1M results in the top row and GIST1M results in the bottom row, varying `k` across the four graph indexes. These correspond to Figure 8 and Figure 6 in the paper, respectively.

<img width="1931" height="871" alt="image" src="https://github.com/user-attachments/assets/9a22aeed-0a76-4b3b-8b09-8caa46016df3" />


## Reproduction Workflow

1. Prepare the required base vectors and query set, and record the dataset variant, distance metric, and preprocessing.
2. Generate exact neighbor information with the ground-truth workflow, ensuring the tested $k$ values and candidate-specific radii match the experiment.
3. Construct the target graph and record its construction and initialization parameters. For controlled comparisons, reuse the same graph, seeds, and adjacency order across variants.
4. Prepare the selected density scores offline. Keep parameter-selection queries separate from evaluation queries.
5. Run the main comparisons or controlled variants using the budgets, tolerances, and radius settings specified above.
6. Aggregate query-level recall over the appropriate nonempty sets. Report the QPS timing scope, thread configuration, repetitions, and actual candidate counts separately.
7. Generate figures from the measured results. Keep raw measurements and configurations alongside the corresponding experiment outputs.

### Controlled-Study Environment

The paper reports **Windows 11**, an **AMD Ryzen 7 7800X3D**, and **32 GB RAM**. The controlled implementation uses **C++20**, **Faiss 1.14.1**, and **MSVC Release** with `/O2`, OpenMP, and AVX2; plotting uses **Python 3.11**. Measurements use eight query workers, one OpenBLAS thread, and three timing repetitions, with parallel batch QPS summarized by its median.

These are the reported environment settings for the controlled SIFT1M studies, rather than a claim that all main benchmark experiments share an identical implementation configuration. Consult the paper's Appendix A for the native graph settings and implementation details.
