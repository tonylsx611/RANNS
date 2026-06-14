/**
 * @file    data_loader.cpp
 * @brief   Implementation of data loading functions.
 */

#include "../include/data_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <omp.h>

using namespace std;

// ============================================================================
//  .fvecs reader
// ============================================================================

vector<float> read_fvec(const string& path, size_t& num_vectors, size_t& dim) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Error opening file: " << path << endl; exit(1); }

    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = d;

    in.seekg(0, ios::end);
    size_t file_size = in.tellg();
    num_vectors = file_size / (sizeof(int) + dim * sizeof(float));

    vector<float> vectors(num_vectors * dim);
    in.seekg(0, ios::beg);

    for (size_t i = 0; i < num_vectors; ++i) {
        in.seekg(sizeof(int), ios::cur);
        in.read(reinterpret_cast<char*>(&vectors[i * dim]), dim * sizeof(float));
    }

    cout << "Loaded dataset: (" << num_vectors << ", " << dim << ")" << endl;
    return vectors;
}

// ============================================================================
//  .fbin reader
// ============================================================================

vector<float> read_fbin(const string& path, size_t& num_vectors, size_t& dim) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Error opening file: " << path << endl; exit(1); }

    uint32_t n, d;
    in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    num_vectors = n;
    dim = d;

    vector<float> vectors(num_vectors * dim);
    in.read(reinterpret_cast<char*>(vectors.data()), num_vectors * dim * sizeof(float));

    cout << "Loaded fbin: (" << num_vectors << ", " << dim << ")" << endl;
    return vectors;
}

// ============================================================================
//  Endian swap (big → little)
// ============================================================================

uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) | ((val >> 8) & 0xff00) |
           ((val << 8) & 0xff0000) | ((val << 24) & 0xff000000);
}

// ============================================================================
//  MNIST IDX reader
// ============================================================================

vector<float> read_mnist_images(const string& path, size_t& num_vectors, size_t& dim) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Error opening MNIST file: " << path << endl; exit(1); }

    uint32_t magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    in.read(reinterpret_cast<char*>(&magic_number),     sizeof(magic_number));
    in.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    in.read(reinterpret_cast<char*>(&n_rows),           sizeof(n_rows));
    in.read(reinterpret_cast<char*>(&n_cols),           sizeof(n_cols));

    magic_number     = swap_endian(magic_number);
    number_of_images = swap_endian(number_of_images);
    n_rows           = swap_endian(n_rows);
    n_cols           = swap_endian(n_cols);

    if (magic_number != 2051) {
        cerr << "Invalid MNIST magic number! Got " << magic_number << endl;
        exit(1);
    }

    num_vectors = number_of_images;
    dim = n_rows * n_cols;
    cout << "Loading MNIST: " << num_vectors << " images, Dim: " << dim << endl;

    size_t total_pixels = num_vectors * dim;
    vector<uint8_t> byte_buffer(total_pixels);
    in.read(reinterpret_cast<char*>(byte_buffer.data()), total_pixels);

    vector<float> vectors(total_pixels);
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(total_pixels); ++i)
        vectors[i] = static_cast<float>(byte_buffer[i]) / 255.0f;

    cout << "Loaded MNIST successfully!" << endl;
    return vectors;
}

// ============================================================================
//  .ivecs Ground Truth reader
// ============================================================================

vector<faiss::idx_t> read_ivecs(const string& path, size_t& num_vectors, size_t& dim) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Error opening GT file: " << path << endl; exit(1); }

    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = d;

    in.seekg(0, ios::end);
    size_t file_size = in.tellg();
    num_vectors = file_size / (sizeof(int) + dim * sizeof(int));

    vector<faiss::idx_t> vectors(num_vectors * dim);
    in.seekg(0, ios::beg);

    vector<int> buffer(dim);
    for (size_t i = 0; i < num_vectors; ++i) {
        in.seekg(sizeof(int), ios::cur);
        in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(int));
        for (size_t j = 0; j < dim; ++j)
            vectors[i * dim + j] = static_cast<faiss::idx_t>(buffer[j]);
    }

    cout << "Loaded GT ivecs: (" << num_vectors << ", K=" << dim << ")" << endl;
    return vectors;
}

// ============================================================================
//  Query sampling
// ============================================================================

pair<vector<int>, vector<float>> select_query(
    const vector<float>& vec, size_t num_vectors, size_t dim, int num_queries) {

    vector<int>   que_index(num_queries);
    vector<float> que_vec(num_queries * dim);

    mt19937 rng(2001);
    uniform_int_distribution<int> dist(0, num_vectors - 1);

    for (int i = 0; i < num_queries; ++i) {
        int q = dist(rng);
        que_index[i] = q;
        copy(vec.begin() + q * dim, vec.begin() + (q + 1) * dim,
             que_vec.begin() + i * dim);
    }

    cout << "Queries selected: " << num_queries << endl;
    return {que_index, que_vec};
}
