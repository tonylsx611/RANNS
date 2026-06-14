/**
 * @file    evaluation.cpp
 * @brief   Ground truth construction and metrics computation.
 */

#include "../include/evaluation.h"

#include <iostream>
#include <unordered_set>
#include <vector>

#include <omp.h>

using namespace std;

// ============================================================================
//  Reverse-ANN ground truth
// ============================================================================

vector<vector<faiss::idx_t>> search_ground_truth(
    const vector<faiss::idx_t>& I_gd,
    size_t num_vectors,
    size_t k_gd,
    const vector<int>& que_index) {

    vector<vector<faiss::idx_t>> ground_truth(que_index.size());

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(que_index.size()); ++i) {
        faiss::idx_t target_idx = que_index[i];
        vector<faiss::idx_t> rev_ann;

        for (size_t row = 0; row < num_vectors; ++row) {
            size_t row_start = row * k_gd;
            faiss::idx_t row_self = I_gd[row_start];

            for (size_t col = 0; col < k_gd; ++col) {
                if (I_gd[row_start + col] == target_idx && target_idx != row_self) {
                    rev_ann.push_back(row_self);
                    break;
                }
            }
        }
        ground_truth[i] = rev_ann;
    }
    return ground_truth;
}

// ============================================================================
//  Metrics printer
// ============================================================================

void print_metrics(
    const string& method_name,
    double duration,
    int num_queries,
    const vector<vector<int>>& candidates,
    const vector<vector<faiss::idx_t>>& ground_truth) {

    double mean_recall = 0.0, avg_candidate_size = 0.0;

    for (size_t i = 0; i < ground_truth.size(); ++i) {
        const auto& b = ground_truth[i];
        avg_candidate_size += candidates[i].size();

        if (b.empty()) { mean_recall += 1.0; continue; }

        unordered_set<int> cs(candidates[i].begin(), candidates[i].end());
        size_t hits = 0;
        for (auto val : b)
            if (cs.count(val)) ++hits;

        mean_recall += static_cast<double>(hits) / b.size();
    }

    mean_recall        /= ground_truth.size();
    avg_candidate_size /= num_queries;

    cout << method_name
         << " | Time: " << duration << " ms"
         << " | QPS: "  << (num_queries / (duration / 1000.0))
         << " | Avg Candidates: " << avg_candidate_size
         << " | Recall: " << mean_recall << endl;
}
