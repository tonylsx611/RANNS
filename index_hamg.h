#pragma once

/**
 * @file    index_hamg.h
 * @brief   Custom HAMG index built via HNSW pruning.
 *
 * HAMG (Hierarchical Approximate Metric Graph) avoids O(N²) construction
 * by using a FAISS HNSW skeleton for fast approximate neighbor extraction,
 * then truncates/cleans the result into a fixed-degree graph.
 */

#include <vector>
#include <faiss/Index.h>

/// HAMG search parameters.
struct HAMGParam {
    int search_L = 128; ///< Beam width for search (larger = more accurate).
};

/// Custom HAMG flat index that implements add/search/reset.
class IndexHAMGFlat : public faiss::Index {
public:
    int K;                          ///< Max out-degree per node.
    HAMGParam hamg;                 ///< Search parameters.
    std::vector<float> dataset;     ///< Cached vectors.
    std::vector<int>   graph_edges; ///< Flattened edges: ntotal * K.

    IndexHAMGFlat(faiss::idx_t dim, int K);

    /// Squared L2 distance (inline, hot path).
    inline float compute_l2_dist(const float* a, const float* b) const {
        float sum = 0;
        for (faiss::idx_t i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /// Build graph via HNSW skeleton (M=32, efConstruction=64), batch size 50k.
    void add(faiss::idx_t n, const float* x) override;

    /// Greedy beam search with smart entry selection.
    void search(faiss::idx_t n, const float* x, faiss::idx_t k,
                float* distances, faiss::idx_t* labels,
                const faiss::SearchParameters* params = nullptr) const override;

    void reset() override;
};
