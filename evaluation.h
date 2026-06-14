#pragma once

/**
 * @file    evaluation.h
 * @brief   Ground truth construction and recall/QPS metrics.
 */

#include <string>
#include <vector>
#include <faiss/Index.h>

/// Build reverse-ANN ground truth from precomputed KNN table.
std::vector<std::vector<faiss::idx_t>> search_ground_truth(
    const std::vector<faiss::idx_t>& I_gd,
    size_t num_vectors,
    size_t k_gd,
    const std::vector<int>& que_index);

/// Print method name, latency (ms), QPS, avg candidate size, and recall.
void print_metrics(
    const std::string& method_name,
    double duration,
    int num_queries,
    const std::vector<std::vector<int>>& candidates,
    const std::vector<std::vector<faiss::idx_t>>& ground_truth);
