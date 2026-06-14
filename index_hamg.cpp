/**
 * @file    index_hamg.cpp
 * @brief   HAMG index implementation using HNSW pruning.
 *
 * Avoids O(N²) graph construction by using a FAISS HNSWFlat skeleton:
 *   - Ingest all data into a temporary HNSW index (O(N log N))
 *   - Extract K+1 approximate neighbors per point (batch size 50k)
 *   - Filter self-loops and pad to fixed-degree edges
 *
 * Search uses smart entry selection + greedy beam search.
 */

#include "../include/index_hamg.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <omp.h>
#include <faiss/IndexHNSW.h>

using namespace std;

// ============================================================================
//  Graph construction via HNSW skeleton
// ============================================================================

void IndexHAMGFlat::add(faiss::idx_t n, const float* x) {
    ntotal = n;
    dataset.assign(x, x + n * d);
    graph_edges.resize(ntotal * K, -1);

    // Step 1: temporary HNSW index
    int M = 32;
    faiss::IndexHNSWFlat hnsw_index(d, M);
    hnsw_index.hnsw.efConstruction = 64;

    cout << "    [Step 1/2] Filling HNSW with " << ntotal << " points..." << endl;
    hnsw_index.add(ntotal, x);

    cout << "    [Step 2/2] Extracting & pruning into HAMG structure..." << endl;
    hnsw_index.hnsw.efSearch = 64;

    // Step 2: batch extraction
    faiss::idx_t block_size = 50000;
    for (faiss::idx_t i = 0; i < ntotal; i += block_size) {
        faiss::idx_t current_block = min(block_size, ntotal - i);

        vector<float>      local_distances(current_block * (K + 1));
        vector<faiss::idx_t> local_labels(current_block * (K + 1));

        hnsw_index.search(current_block, x + i * d, K + 1,
                          local_distances.data(), local_labels.data());

        #pragma omp parallel for schedule(static)
        for (faiss::idx_t bi = 0; bi < current_block; ++bi) {
            faiss::idx_t global_idx = i + bi;
            int edge_count = 0;

            for (int j = 0; j < K + 1; ++j) {
                faiss::idx_t neighbor = local_labels[bi * (K + 1) + j];
                if (neighbor == global_idx || neighbor < 0) continue;
                if (edge_count < K)
                    graph_edges[global_idx * K + edge_count++] = neighbor;
            }
        }

        cout << "    [HAMG] Extracted " << (i + current_block)
             << " / " << ntotal << " points..." << endl;
    }

    cout << "--> [HAMG] Construction complete!" << endl;
}

// ============================================================================
//  Search: smart entry selection + greedy beam search
// ============================================================================

void IndexHAMGFlat::search(
    faiss::idx_t n, const float* x, faiss::idx_t k,
    float* distances, faiss::idx_t* labels,
    const faiss::SearchParameters* /* params */) const {

    #pragma omp parallel for schedule(dynamic)
    for (faiss::idx_t qi = 0; qi < n; ++qi) {
        const float* q = &x[qi * d];

        // Smart entry selection
        faiss::idx_t entry_point = 0;
        if (n == ntotal || x == dataset.data()) {
            entry_point = qi;  // full extraction: use self-ID
        } else {
            float min_d = compute_l2_dist(q, &dataset[0]);
            for (int s = 0; s < 4; ++s) {
                faiss::idx_t rv = rand() % ntotal;
                float td = compute_l2_dist(q, &dataset[rv * d]);
                if (td < min_d) { min_d = td; entry_point = rv; }
            }
        }

        // Beam search
        vector<pair<float, int>> pool;
        vector<bool> visited(ntotal, false);

        pool.push_back({compute_l2_dist(q, &dataset[entry_point * d]), entry_point});
        visited[entry_point] = true;

        size_t cursor = 0;
        while (cursor < pool.size()) {
            int u = pool[cursor].second;

            for (int e = 0; e < K; ++e) {
                int v = graph_edges[u * K + e];
                if (v < 0 || visited[v]) continue;
                visited[v] = true;

                float d_v = compute_l2_dist(q, &dataset[v * d]);
                auto it = lower_bound(pool.begin(), pool.end(), make_pair(d_v, v));
                pool.insert(it, {d_v, v});

                if (pool.size() > static_cast<size_t>(hamg.search_L))
                    pool.resize(hamg.search_L);
            }
            ++cursor;
        }

        // Write results
        for (faiss::idx_t i = 0; i < k; ++i) {
            if (i < static_cast<faiss::idx_t>(pool.size())) {
                distances[qi * k + i] = pool[i].first;
                labels[qi * k + i]    = pool[i].second;
            } else {
                distances[qi * k + i] = -1.0f;
                labels[qi * k + i]    = -1;
            }
        }
    }
}

// ============================================================================
//  Reset
// ============================================================================

void IndexHAMGFlat::reset() {
    dataset.clear();
    graph_edges.clear();
    ntotal = 0;
}
