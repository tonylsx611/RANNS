#pragma once

/**
 * @file    search_algorithms.h
 * @brief   Range Exploration, Hop Exploration, and Density Exploration.
 */

#include <cstddef>
#include <vector>
#include <faiss/Index.h>

/// Range Exploration — search with increasing budget (factor × k).
void Range_Exploration(
    faiss::Index& ind,
    const std::vector<float>& que_vec,
    int num_queries,
    int k,
    const std::vector<std::vector<faiss::idx_t>>& gd);

/// Hop Exploration — multi-hop forward expansion on a precomputed KNN graph.
void Hop_Exploration(
    const std::vector<faiss::idx_t>& global_knn_I,
    int graph_k,
    int hop,
    int k,
    const std::vector<int>& que_index,
    int num_queries,
    const std::vector<std::vector<faiss::idx_t>>& gd);

/// Density Exploration — Hop Exploration with adaptive distance-ratio pruning.
void DE_Search(
    const std::vector<faiss::idx_t>& knn_graph_I,
    const std::vector<float>& knn_graph_D,
    int graph_k,
    int hop,
    int k,
    size_t dim,
    size_t num_vectors,
    const std::vector<int>& que_index,
    int num_queries,
    const std::vector<std::vector<faiss::idx_t>>& gd);
