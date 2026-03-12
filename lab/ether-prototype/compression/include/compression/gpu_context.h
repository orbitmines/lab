#ifndef COMPRESSION_GPU_CONTEXT_H
#define COMPRESSION_GPU_CONTEXT_H

#include "compression/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;

// Create/destroy GPU context. Returns NULL if no GPU available.
GpuContext* gpu_context_create(void);
void gpu_context_destroy(GpuContext* ctx);

// Query backend info
int gpu_context_get_info(const GpuContext* ctx, BackendInfo* info);

// Returns how many parallel transformation slots fit for the given data size
uint32_t gpu_context_max_slots(const GpuContext* ctx, uint64_t data_size);

// Compute byte histogram on GPU. Returns 0 on success, -1 if GPU not available.
int gpu_context_histogram(GpuContext* ctx, const uint8_t* data, uint64_t size, ByteHistogram* out);

// Compute order-0 Shannon entropy on GPU (bits). Returns 0 on success.
int gpu_context_entropy_order0(GpuContext* ctx, const uint8_t* data, uint64_t size, double* out_bits);

// Compute order-1 Shannon entropy on GPU (bits). Returns 0 on success.
int gpu_context_entropy_order1(GpuContext* ctx, const uint8_t* data, uint64_t size, double* out_bits);

// GPU LZ match statistics. out_stats: [literal_bytes, match_bytes, match_count, 0]
int gpu_context_lz_stats(GpuContext* ctx, const uint8_t* data, uint64_t size, uint32_t out_stats[4]);

// GPU Huffman compress. Returns 0 on success.
int gpu_context_huffman_compress(GpuContext* ctx, const uint8_t* data, uint64_t size,
                                  CompressedBuffer* out);

// GPU Huffman size-only estimate. Returns total bits.
int gpu_context_huffman_size(GpuContext* ctx, const uint8_t* data, uint64_t size,
                              uint64_t* out_bits);

// GPU Huffman decompress. Returns 0 on success.
int gpu_context_huffman_decompress(GpuContext* ctx, const uint8_t* compressed, uint64_t comp_size,
                                    uint64_t original_size, uint8_t** out_data, uint64_t* out_size);

#ifdef __cplusplus
}
#endif

#endif
