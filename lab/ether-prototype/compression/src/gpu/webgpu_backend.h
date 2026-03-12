#ifndef WEBGPU_BACKEND_H
#define WEBGPU_BACKEND_H

#include "compression/types.h"
#include <webgpu/webgpu.h>

struct WebGpuBackend {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;

    // Histogram pipeline
    WGPUShaderModule histogram_shader;
    WGPUComputePipeline histogram_pipeline;
    WGPUBindGroupLayout histogram_bgl;

    // Entropy pipeline (order-0 from histogram)
    WGPUShaderModule entropy_shader;
    WGPUComputePipeline entropy_pipeline;
    WGPUBindGroupLayout entropy_bgl;

    // Bigram histogram pipeline (order-1)
    WGPUShaderModule bigram_shader;
    WGPUComputePipeline bigram_pipeline;
    WGPUBindGroupLayout bigram_bgl;

    // LZ match pipeline
    WGPUShaderModule lz_shader;
    WGPUComputePipeline lz_pipeline;
    WGPUBindGroupLayout lz_bgl;

    // Huffman encode pipeline
    WGPUShaderModule huffman_enc_shader;
    WGPUComputePipeline huffman_enc_pipeline;
    WGPUBindGroupLayout huffman_enc_bgl;

    // Huffman decode pipeline
    WGPUShaderModule huffman_dec_shader;
    WGPUComputePipeline huffman_dec_pipeline;
    WGPUBindGroupLayout huffman_dec_bgl;

    // Transform pipelines (shared BGL, per-type pipeline)
    WGPUBindGroupLayout transform_bgl;
    WGPUShaderModule transform_shaders[5];   // TRANSFORM_COUNT
    WGPUComputePipeline transform_pipelines[5];

    bool valid;
};

// Initialize WebGPU backend. Returns 0 on success.
int webgpu_backend_init(WebGpuBackend* backend);
void webgpu_backend_destroy(WebGpuBackend* backend);

// Get device info
void webgpu_backend_get_info(const WebGpuBackend* backend, BackendInfo* info);

// ── Buffer operations ──────────────────────────────────────────────────────

// Upload data to GPU buffer (storage + copy_dst)
WGPUBuffer webgpu_upload(WebGpuBackend* backend, const uint8_t* data, uint64_t size);

// Create a zero-initialized storage buffer
WGPUBuffer webgpu_create_buffer(WebGpuBackend* b, uint64_t size, WGPUBufferUsage usage);

// Free GPU buffer
void webgpu_free_buffer(WGPUBuffer buf);

// Read back data from GPU buffer
int webgpu_readback(WebGpuBackend* b, WGPUBuffer src, void* dst, uint64_t size);

// ── Compute operations ─────────────────────────────────────────────────────

// Compute byte histogram on GPU. out_histogram: 256 x uint32
int webgpu_histogram(WebGpuBackend* backend, WGPUBuffer data_buf, uint64_t data_size,
                     uint32_t* out_histogram);

// Compute order-0 Shannon entropy from histogram buffer (256 x u32).
// Returns entropy in bits.
int webgpu_entropy_order0(WebGpuBackend* b, WGPUBuffer hist_buf, uint32_t total,
                          float* out_bits);

// Compute bigram (order-1) histogram: 256x256 u32 co-occurrence matrix
int webgpu_bigram_histogram(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                            uint32_t* out_bigram);

// LZ match statistics for size estimation
// out_stats: [literal_bytes, match_bytes, match_count, 0]
int webgpu_lz_match_stats(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                          uint32_t out_stats[4]);

// ── GPU Huffman compression ────────────────────────────────────────────────

// Compress data using GPU Huffman (block-parallel)
// code_table: 256 entries, each (code << 8) | length
// Returns 0 on success
int webgpu_huffman_compress(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                            const uint32_t* code_table, CompressedBuffer* out);

// Estimate Huffman compressed size only (no actual encoding)
int webgpu_huffman_size(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                        const uint32_t* code_table, uint64_t* out_bits);

// Decompress GPU Huffman compressed data
int webgpu_huffman_decompress(WebGpuBackend* b, const uint8_t* compressed, uint64_t comp_size,
                              const uint32_t* decode_table, uint32_t max_code_len,
                              uint64_t original_size, uint8_t** out_data);

#endif
