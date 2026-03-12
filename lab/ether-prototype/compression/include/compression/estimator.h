#ifndef COMPRESSION_ESTIMATOR_H
#define COMPRESSION_ESTIMATOR_H

#include "compression/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Estimator Estimator;

// ── Lifecycle ───────────────────────────────────────────────────────────────

// Create/destroy estimator. use_gpu=1 to try GPU, falls back to CPU if unavailable.
Estimator* estimator_create(int use_gpu);
void estimator_destroy(Estimator* est);

// Load data into the estimator. Copies data; caller retains ownership.
int estimator_load(Estimator* est, const uint8_t* data, uint64_t size);

// Load from file
int estimator_load_file(Estimator* est, const char* path);

// Get backend info
int estimator_get_backend_info(const Estimator* est, BackendInfo* info);

// ── Size estimation ─────────────────────────────────────────────────────────

// Compute byte histogram
int estimator_histogram(const Estimator* est, ByteHistogram* out);

// Estimate compressed size for a single algorithm
int estimator_estimate(const Estimator* est, CompressionAlgorithm algo, EstimateResult* out);

// Estimate all algorithms at once
int estimator_estimate_all(const Estimator* est, EstimateAllResult* out);

// ── Actual compression ──────────────────────────────────────────────────────

// Compress data using the specified algorithm.
// Returns 0 on success, fills out with compressed data. Caller must free via compressed_buffer_free().
// level: compression level (1-12 for zstd, 1-9 for gzip/deflate, ignored for huffman/arithmetic/ans/lz4)
int estimator_compress(const Estimator* est, CompressionAlgorithm algo, int level, CompressedBuffer* out);

// Decompress data.
// src/src_size: compressed data. out_data/out_size: decompressed output (caller frees out_data).
// For algorithms that don't store the original size, expected_size must be provided.
int estimator_decompress(CompressionAlgorithm algo, const uint8_t* src, uint64_t src_size,
                         uint8_t** out_data, uint64_t* out_size, uint64_t expected_size);

// ── Parallel transformation slots ────────────────────────────────────────────

// Returns how many parallel transformation slots the GPU can handle for the loaded data.
// Each slot holds a mutable copy of the data for independent transformation + analysis.
uint32_t estimator_parallel_slots(const Estimator* est);

// ── Internal accessors (for GDExtension / GPU session integration) ────────

typedef struct GpuContext GpuContext;

// Get the GPU context from the estimator. Returns NULL if no GPU.
GpuContext* estimator_get_gpu(const Estimator* est);

// Get pointer to loaded data. Returns NULL if no data loaded.
const uint8_t* estimator_get_data(const Estimator* est, uint64_t* out_size);

// ── Verification ────────────────────────────────────────────────────────────

// Verify GPU estimates against CPU reference. Returns number of mismatches.
int estimator_verify(const Estimator* est);

#ifdef __cplusplus
}
#endif

#endif
