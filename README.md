# RANNS

Experimental code and results accompanying the paper:

**Reverse Approximate Nearest Neighbor Search: From Hardness to Density-aware Exploration**

This repository presents the experimental evaluation of **Density-aware Exploration (DE)** for high-dimensional reverse approximate nearest neighbor search. It includes a ground-truth generation script and visualizations comparing DE with representative RANNS methods across different datasets and ANN indexes.

## Overview

Reverse approximate nearest neighbor search (RANNS) identifies data points that regard a query as one of their nearest neighbors. A point can be a valid reverse neighbor even when it is far from the query, provided that its own neighborhood radius is sufficiently large.

DE uses local density to guide candidate exploration toward promising sparse regions, while spatial bounding limits unnecessary expansion. The experiments examine how this strategy affects reverse recall and query throughput under a limited candidate verification budget.

## Datasets and Ground Truth

The experiments cover five datasets:

- SIFT1M
- GIST1M
- MNIST
- DEEP1M
- SIFT10M

The `Ground_truth.py` script computes k-nearest-neighbor ground truth for these datasets. The provided configuration uses **k = 100**.

The generated ground-truth results are available in the following Google Drive folder:

**[Download the ground-truth results](https://drive.google.com/drive/folders/1QWjLzQDn9FjC4_0aJ9ybQcXLcu3fgot2?usp=sharing)**

For reverse search, each candidate is evaluated using its own k-NN radius. Let $r_k(p)$ denote the distance from a data point $p$ to its k-th nearest neighbor. The ground-truth reverse-neighbor set of query $q$ is

$$ \mathcal{R}_k(q)=\{p\in\mathcal{P}:\operatorname{dist}(p,q)\leq r_k(p)\}. $$


For experiments with different values of k, verification uses the neighborhood radius at the corresponding rank. The value of k specifies the forward neighborhood size; the number of reverse neighbors can vary across queries.

## Compared Methods

The evaluation compares the following RANNS candidate exploration methods:

| Method | Candidate exploration strategy |
| --- | --- |
| **SFT** | Generates candidates from the query's forward neighborhood using range-based exploration. |
| **RDT** | Uses incremental forward-neighbor retrieval, intrinsic-dimensionality estimates, and local distance tests to reduce filtering and verification work. |
| **HAMG** | Broadens candidate coverage through multi-hop exploration along graph edges. |
| **DE (ours)** | Combines density-aware exploration with spatial bounding to prioritize promising sparse regions under a limited verification budget. |

A **non-graph implementation based on RaBitQ** is included as a reference in the QPS–reverse-recall comparison.

## ANN Indexes and RANNS Methods

The graph indexes evaluated in this work are **HNSW, HAMG, NN-Descent, and NSG**. They provide forward ANN retrieval and graph neighborhoods that can support reverse-search candidate exploration. **RaBitQ** is a vector quantization technique used in the non-graph reference implementation.

ANN indexes and RANNS methods play different roles. A RANNS method can use an ANN index to retrieve initial candidates and obtain neighborhood information for verification. In each graph-index experiment, the compared candidate exploration methods operate over the specified underlying graph.

The name **HAMG** appears in both contexts: it denotes the underlying graph structure and the associated hop-based RANNS exploration method. The RaBitQ reference provides a comparison with candidate retrieval that does not rely on graph traversal.

## Evaluation Metrics

- **Rev-Recall@k** measures the fraction of a query's ground-truth reverse neighbors recovered by a method.
- **QPS (queries per second)** measures query throughput. Higher QPS indicates faster query processing.
- **Candidate verification budget B** limits the number of candidates passed to reverse-neighbor verification.

Let $\widehat{\mathcal{R}}_k(q)$ be the returned reverse-neighbor set. Reverse recall is defined as

$$
\text{Rev-Recall}@k=
\frac{|\mathcal{R}_k(q)\cap\widehat{\mathcal{R}}_k(q)|}
{|\mathcal{R}_k(q)|}.
$$

QPS and Rev-Recall are considered together to evaluate the trade-off between search efficiency and result coverage.

## Experimental Results

The following visualizations report comparisons across datasets, neighborhood sizes, and index structures using the experimental ground truth.

### Reverse Recall across Neighborhood Sizes and Graph Indexes

Rev-Recall@k for the compared RANNS methods under different neighborhood sizes and graph indexes.

![Rev-Recall across neighborhood sizes and graph indexes](https://github.com/user-attachments/assets/fe533b7e-de6e-4c18-98eb-83ffe4f679db)

### Additional Reverse-Recall Comparisons

Additional reverse-recall results for the evaluated methods across datasets.

![Additional reverse-recall comparisons across datasets](https://github.com/user-attachments/assets/81d905b1-b8a2-4540-8b01-deafc2c14c2b)

### QPS–Reverse-Recall Trade-offs

QPS versus Rev-Recall@50 across the five datasets and four graph indexes. The RaBitQ implementation is included as a non-graph reference.

![QPS versus Rev-Recall at k equal to 50](https://github.com/user-attachments/assets/282c076e-ee06-402f-a7a2-6b9827b76614)

## Experimental Configuration

When comparing or reproducing results, use matching dataset and query sets, neighborhood size k, candidate verification budget B, and index settings. The paper reports the experimental configurations and provides the analysis behind DE.

