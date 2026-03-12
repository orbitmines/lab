#ifndef COMPRESSION_TYPES_H
#define COMPRESSION_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    COMP_SHANNON_ORDER0,
    COMP_SHANNON_ORDER1,
    COMP_HUFFMAN,
    COMP_ARITHMETIC,
    COMP_ANS,
    COMP_GZIP_EST,
    COMP_ZSTD_EST,
    COMP_LZ4_EST,
    COMP_GZIP_EXACT,
    COMP_ZSTD_EXACT,
    COMP_LZ4_EXACT,
    COMP_DEFLATE_EXACT,
    COMP_ALGORITHM_COUNT
} CompressionAlgorithm;

typedef struct {
    CompressionAlgorithm algorithm;
    uint64_t estimated_bits;
    uint64_t estimated_bytes;
    double bits_per_byte;       // estimated_bits / input_size_bytes
    double ratio;               // estimated_bytes / input_size_bytes
} EstimateResult;

typedef struct {
    EstimateResult results[COMP_ALGORITHM_COUNT];
    int count;
    uint64_t input_size;
} EstimateAllResult;

typedef struct {
    uint64_t histogram[256];
    uint64_t total;
} ByteHistogram;

typedef enum {
    COMP_BACKEND_CPU,
    COMP_BACKEND_WEBGPU,
} CompressionBackend;

typedef struct {
    CompressionBackend backend;
    char device_name[256];
    char vendor[128];
    uint64_t max_buffer_size;
    uint32_t max_slots;
} BackendInfo;

// ── Transform types (for GPU-resident batch evaluation) ─────────────────────

typedef enum {
    TRANSFORM_DELTA,       // delta coding: out[i] = in[i] - in[i-1]
    TRANSFORM_XOR_PREV,    // XOR with previous: out[i] = in[i] ^ in[i-1]
    TRANSFORM_ROTATE_BITS, // byte rotation: out[i] = rotl(in[i], param0)
    TRANSFORM_SUB_MEAN,    // subtract running mean: out[i] = in[i] - mean(window)
    TRANSFORM_BYTE_SWAP,   // swap adjacent byte pairs
    TRANSFORM_COUNT
} TransformType;

typedef struct {
    TransformType type;
    uint32_t param0;   // transform-specific (rotation amount, window size, etc.)
    uint32_t param1;
    uint32_t param2;
} TransformDesc;

typedef struct {
    TransformType transform;
    uint32_t params[3];
    float entropy_o0_bpb;     // order-0 entropy in bits per byte
    double entropy_o0_total;  // total bits
} SlotScore;

// Compressed data buffer (caller must free .data when done)
typedef struct {
    uint8_t* data;
    uint64_t size;
    uint64_t capacity;
    CompressionAlgorithm algorithm;
} CompressedBuffer;

static inline void compressed_buffer_free(CompressedBuffer* buf) {
    if (buf && buf->data) { free(buf->data); buf->data = NULL; buf->size = 0; }
}

#ifdef __cplusplus
}
#endif

#endif
