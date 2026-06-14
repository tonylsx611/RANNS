/**
 * @file    main.cpp
 * @brief   RANNS — Range-Adaptive Nearest Neighbor Search Framework
 *
 * Compares Range Exploration, Hop Exploration, and Density Exploration
 * across multiple graph types (HNSW, HAMG, NSG, NN-Descent) and also with a quantized RabitQ baseline.
 *
 * Compile:
 *   g++ -O3 -march=native -fopenmp -I include src/*.cpp -o ranns -L/path/to/faiss -lfaiss
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFRabitQ.h>
#include <faiss/Index.h>

#include "data_loader.h"
#include "evaluation.h"
#include "index_hamg.h"
#include "search_algorithms.h"

using namespace std;
using namespace std::chrono;

int main() {
    size_t num_vectors = 0;
    size_t dim         = 0;

    // ========================================================================
    //  1. Data loading (switch dataset by uncommenting)
    // ========================================================================
    vector<float> vec = read_fvec("sift/sift_base.fvecs", num_vectors, dim);
    vector<float> vec = read_fvec("gist/gist_base.fvecs", num_vectors, dim);
    vector<float> vec = read_fvec("rand/random_1M.fvecs", num_vectors, dim);
    vector<float> vec = read_fbin("deep/base.1M.fbin", num_vectors, dim);
    vector<float> vec = read_mnist_images("minst/train-images.idx3-ubyte", num_vectors, dim);

    int num_queries = 100;
    vector<int> k_list = {25, 50, 75, 100};

    // ========================================================================
    //  2. Build quantized index (compressed baseline)
    // ========================================================================

    //  2a. [Alternative] IVF-PQ (commented)
    //  size_t nlist = 256; size_t m = 8; size_t nbits = 8;
    //  faiss::IndexFlatL2 quantizer_pq(dim);
    //  faiss::IndexIVFPQ index_ivfpq(&quantizer_pq, dim, nlist, m, nbits);
    //  index_ivfpq.train(num_vectors, vec.data());
    //  index_ivfpq.add(num_vectors, vec.data());
    //  index_ivfpq.nprobe = 8;

    //  2b. IVF-RabitQ (active)
    cout << "\n--> Building IVF-RabitQ Index..." << endl;
    size_t nlist = 256;
    uint8_t m = 1;
    faiss::IndexFlatL2 quantizer(dim);
    faiss::IndexIVFRaBitQ index_rabitq(
        static_cast<faiss::Index*>(&quantizer),
        dim, nlist, faiss::METRIC_L2, true, m);
    index_rabitq.train(num_vectors, vec.data());
    index_rabitq.add(num_vectors, vec.data());
    index_rabitq.nprobe = 8;

    // ========================================================================
    //  3. Main experiment loop
    // ========================================================================

    for (int k : k_list) {
        cout << "\n==================== Testing K = " << k << " ====================" << endl;

        // 3a. Sample queries
        auto [que_index, que_vec] = select_query(vec, num_vectors, dim, num_queries);

        // 3b. Load ground truth
        cout << "--> Loading Pre-computed Ground Truth..." << endl;
        size_t gt_num_vectors = 0;
        size_t gt_k           = 0;

        vector<faiss::idx_t> I_gd = read_ivecs("sift/sift_100nn.ivecs", ...);
        vector<faiss::idx_t> I_gd = read_ivecs("minst/mnist_train_100nn.ivecs", ...);
        vector<faiss::idx_t> I_gd = read_ivecs("deep/deep1m_100nn.ivecs", ...);
        vector<faiss::idx_t> I_gd = read_ivecs("rand/random_1m_100nn.ivecs", ...);
        vector<faiss::idx_t> I_gd = read_ivecs("gist/gist1m_100nn.ivecs", gt_num_vectors, gt_k);

        int k_gd = k;
        if (gt_k < static_cast<size_t>(k_gd)) {
            cerr << "Error: ivecs K (" << gt_k << ") < required (" << k_gd << ")." << endl;
            exit(1);
        }

        auto gd = search_ground_truth(I_gd, num_vectors, gt_k, que_index);

        // 3c. IVF-RabitQ baseline
        //  Range_Exploration(index_ivfpq, que_vec, num_queries, k, gd);
        cout << "\n--- RabitQ Range Exploration (Baseline) ---" << endl;
        Range_Exploration(index_rabitq, que_vec, num_queries, k, gd);

        // ====================================================================
        //  Experiment A: HNSW graphs
        // ====================================================================

        int graph_k = k + 1;
        vector<float>         global_knn_D(num_vectors * graph_k);
        vector<faiss::idx_t>  global_knn_I(num_vectors * graph_k);

        // -- HNSW (high degree + high efSearch) --
        cout << "\n--> Building HNSW Graph..." << endl;
        int M = 32;
        faiss::IndexHNSWFlat index_hnsw(dim, M);
        index_hnsw.hnsw.efConstruction = 200;
        index_hnsw.add(num_vectors, vec.data());
        index_hnsw.hnsw.efSearch = 128;
        index_hnsw.search(num_vectors, vec.data(), graph_k, global_knn_D.data(), global_knn_I.data());

        // -- Evaluate on HNSW --
        cout << "\n--- 1. Range Exploration (Baseline) ---" << endl;
        Range_Exploration(index_hnsw, que_vec, num_queries, k, gd);

        cout << "\n--- 2. Hop Exploration ---" << endl;
        for (int h = 1; h <= 6; ++h) {
            Hop_Exploration(global_knn_I, graph_k, h, k, que_index, num_queries, gd);
        }

        cout << "\n--- 3. Density Exploration ---\n";
        for (int hop = 1; hop <= 9; ++hop) {
            DE_Search(global_knn_I, global_knn_D, graph_k, hop, k, dim, num_vectors, que_index, num_queries, gd);
        }

        // ====================================================================
        //  Experiment B: NSG graphs
        // ====================================================================

        cout << "\n--> Building NSG Graph..." << endl;
        int R = 32;
        faiss::IndexNSGFlat index_nsg(dim, R);
        index_nsg.nsg.search_L = 128;
        index_nsg.add(num_vectors, vec.data());
        index_nsg.search(num_vectors, vec.data(), graph_k, global_knn_D.data(), global_knn_I.data());


        cout << "\n--- 1. Range Exploration (Baseline) ---" << endl;
        Range_Exploration(index_nsg, que_vec, num_queries, k, gd);

        cout << "\n--- 2. Hop Exploration ---" << endl;
        for (int h = 1; h <= 6; ++h) {
            Hop_Exploration(global_knn_I, graph_k, h, k, que_index, num_queries, gd);
        }

        cout << "\n--- 3. Density Exploration ---\n";
        for (int hop = 1; hop <= 9; ++hop) {
            DE_Search(global_knn_I, global_knn_D, graph_k, hop, k, dim, num_vectors, que_index, num_queries, gd);
        }

        // ====================================================================
        //  Experiment C: NN-Descent graphs
        // ====================================================================

        cout << "\n--> Building NN-Descent Graph..." << endl;
        int K_nnd = 32;
        faiss::IndexNNDescentFlat index_nnd(dim, K_nnd);
        index_nnd.nndescent.L = 128;
        index_nnd.add(num_vectors, vec.data());
        index_nnd.search(num_vectors, vec.data(), graph_k, global_knn_D.data(), global_knn_I.data());


        cout << "\n--- 1. Range Exploration (Baseline) ---" << endl;
        Range_Exploration(index_nnd, que_vec, num_queries, k, gd);

        cout << "\n--- 2. Hop Exploration ---" << endl;
        for (int h = 1; h <= 6; ++h) {
            Hop_Exploration(global_knn_I, graph_k, h, k, que_index, num_queries, gd);
        }

        cout << "\n--- 3. Density Exploration ---\n";
        for (int hop = 1; hop <= 9; ++hop) {
            DE_Search(global_knn_I, global_knn_D, graph_k, hop, k, dim, num_vectors, que_index, num_queries, gd);
        }

        //====================================================================
        //  Experiment D: HAMG graphs 
        //====================================================================
        int K_hamg = 32;
        IndexHAMGFlat index_hamg(dim, K_hamg);
        index_hamg.hamg.search_L = 128;
        index_hamg.add(num_vectors, vec.data());
        
        vector<float>  global_knn_D_hamg(num_vectors * graph_k);
        vector<faiss::idx_t> global_knn_I_hamg(num_vectors * graph_k);
        index_hamg.search(num_vectors, vec.data(), graph_k, global_knn_D_hamg.data(), global_knn_I_hamg.data());
        

        cout << "\n--- 1. Range Exploration (Baseline) ---" << endl;
        Range_Exploration(index_hamg, que_vec, num_queries, k, gd);

        cout << "\n--- 2. Hop Exploration ---" << endl;
        for (int h = 1; h <= 6; ++h)
            Hop_Exploration(global_knn_I_hamg, graph_k, h, k, que_index, num_queries, gd);

        cout << "\n--- 3. Density Exploration ---\n";
        for (int hop = 1; hop <= 9; ++hop)
            DE_Search(global_knn_I_hamg, global_knn_D_hamg, graph_k, hop, k, dim, num_vectors, que_index, num_queries, gd);
    }

    return 0;
}
