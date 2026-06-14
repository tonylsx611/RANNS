#pragma once

/**
 * @file    data_loader.h
 * @brief   Load various binary vector formats and sample queries.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <faiss/Index.h>

/// Read .fvecs format: [dim(int)] [dim floats] per vector.
std::vector<float> read_fvec(const std::string& path, size_t& num_vectors, size_t& dim);

/// Read .fbin format: header (uint32_t N, uint32_t D) followed by N*D floats.
std::vector<float> read_fbin(const std::string& path, size_t& num_vectors, size_t& dim);

/// Big-endian to little-endian conversion for 32-bit unsigned int.
uint32_t swap_endian(uint32_t val);

/// Read MNIST IDX image file, normalize pixels to [0.0, 1.0].
std::vector<float> read_mnist_images(const std::string& path, size_t& num_vectors, size_t& dim);

/// Read .ivecs ground truth file (same layout as .fvecs but with ints).
std::vector<faiss::idx_t> read_ivecs(const std::string& path, size_t& num_vectors, size_t& dim);

/// Randomly select @p num_queries vectors from the dataset (seed = 482).
std::pair<std::vector<int>, std::vector<float>> select_query(
    const std::vector<float>& vec, size_t num_vectors, size_t dim, int num_queries = 100);
