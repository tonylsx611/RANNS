/**
 * @file    search_algorithms.cpp
 * @brief   Range Exploration, Hop Exploration, and Density Exploration.
 */

#include "../include/search_algorithms.h"
#include "../include/evaluation.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <omp.h>

using namespace std;
using namespace std::chrono;

// ============================================================================
//  1. Range Exploration
// ============================================================================

void Range_Exploration(
    faiss::Index& ind,
    const vector<float>& que_vec,
    int num_queries,
    int k,
    const vector<vector<faiss::idx_t>>& gd) {

    vector<int> expansion_factors = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    };

    for (int factor : expansion_factors) {
        int search_k = factor * k + 1;

        vector<float>         D(num_queries * search_k);
        vector<faiss::idx_t>  I(num_queries * search_k);

        auto start_time = high_resolution_clock::now();
        ind.search(num_queries, que_vec.data(), search_k,
                   D.data(), I.data());

        // Convert to vector<vector<int>> for print_metrics
        vector<vector<int>> candidate_sets(num_queries);
        for (int i = 0; i < num_queries; ++i) {
            candidate_sets[i].reserve(search_k);
            for (int j = 0; j < search_k; ++j) {
                candidate_sets[i].push_back(
                    static_cast<int>(I[i * search_k + j]));
            }
        }

        auto end_time = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(
            end_time - start_time).count() / 1000.0;

        string label = "Range Exploration (Budget="
                       + to_string(factor * k) + ")";
        print_metrics(label, duration, num_queries,
                      candidate_sets, gd);
    }
}

// ============================================================================
//  2. Hop Exploration
// ============================================================================

void Hop_Exploration(
    const vector<faiss::idx_t>& global_knn_I,
    int graph_k,
    int hop,
    int k,
    const vector<int>& que_index,
    int num_queries,
    const vector<vector<faiss::idx_t>>& gd) {

    vector<vector<int>> candidate_sets(num_queries);
    auto start_time = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_queries; ++i) {
        int q_idx = que_index[i];
        if (q_idx < 0) continue;

        unordered_set<int> query_candidates;
        vector<int> current_layer;

        int max_j = min(k, graph_k - 1);

        // Hop 1: direct neighbors
        for (int j = 1; j <= max_j; ++j) {
            int neighbor = global_knn_I[q_idx * graph_k + j];
            if (neighbor < 0) continue;
            query_candidates.insert(neighbor);
            current_layer.push_back(neighbor);
        }

        // Hop 2+ : expand from current layer
        for (int h = 1; h < hop; ++h) {
            vector<int> next_layer;
            for (int node_idx : current_layer) {
                if (node_idx < 0) continue;

                for (int j = 1; j <= max_j; ++j) {
                    int neighbor_idx =
                        global_knn_I[node_idx * graph_k + j];
                    if (neighbor_idx < 0) continue;

                    if (query_candidates.find(neighbor_idx)
                        == query_candidates.end()) {
                        query_candidates.insert(neighbor_idx);
                        next_layer.push_back(neighbor_idx);
                    }
                }
            }
            current_layer = move(next_layer);
        }

        candidate_sets[i] = vector<int>(query_candidates.begin(),
                                        query_candidates.end());
    }

    auto end_time = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::microseconds>(
        end_time - start_time).count() / 1000.0;

    string label = "Hop Exploration (Hop=" + to_string(hop) + ")";
    print_metrics(label, duration, num_queries, candidate_sets, gd);
}

// ============================================================================
//  3. Density Exploration (DE)
// ============================================================================

void DE_Search(
    const vector<faiss::idx_t>& knn_graph_I,
    const vector<float>& knn_graph_D,
    int graph_k,
    int hop,
    int k,
    size_t /* dim */,
    size_t num_vectors,
    const vector<int>& que_index,
    int num_queries,
    const vector<vector<faiss::idx_t>>& gd) {

    // Pruning threshold: larger beta = more candidates
    const float beta = 0.85f;

    vector<vector<int>> candidate_sets(num_queries);

    int max_threads = omp_get_max_threads();
    vector<vector<int>> thread_local_visited(
        max_threads,
        vector<int>(num_vectors, -1));

    auto start_time = high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < num_queries; ++qi) {
        int tid = omp_get_thread_num();
        int q_idx = que_index[qi];

        vector<int> current_layer;
        vector<int> next_layer;
        vector<int> candidates;

        // Hop 1: all graph_k seeds
        for (int j = 1; j < graph_k; ++j) {
            int v = knn_graph_I[q_idx * graph_k + j];
            if (v < 0) continue;

            thread_local_visited[tid][v] = qi;
            current_layer.push_back(v);
            candidates.push_back(v);
        }

        // Hop 2+: adaptive distance-ratio pruning
        for (int h = 1; h < hop; ++h) {
            next_layer.clear();
            if (current_layer.empty())
                break;

            for (int u : current_layer) {
                float rk_u = knn_graph_D[u * graph_k + k];

                for (int j = 1; j < graph_k; ++j) {
                    int v = knn_graph_I[u * graph_k + j];
                    if (v < 0) continue;

                    if (thread_local_visited[tid][v] == qi)
                        continue;

                    // Adaptive pruning
                    bool keep = true;
                    float rk_v = knn_graph_D[v * graph_k + k];
                    if (rk_v < beta * rk_u)
                        keep = false;

                    if (!keep) continue;

                    thread_local_visited[tid][v] = qi;
                    next_layer.push_back(v);
                    candidates.push_back(v);
                }
            }

            current_layer.swap(next_layer);
        }

        candidate_sets[qi] = move(candidates);
    }

    auto end_time = high_resolution_clock::now();
    double duration = duration_cast<microseconds>(
        end_time - start_time).count() / 1000.0;

    string label = "DE (Hop=" + to_string(hop) + ")";
    print_metrics(label, duration, num_queries, candidate_sets, gd);
}
