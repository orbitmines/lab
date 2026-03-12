#ifndef REFERENCE_WRAPPERS_H
#define REFERENCE_WRAPPERS_H

#include "compression/types.h"

// ── Size-only (compress and return size) ────────────────────────────────────

uint64_t ref_gzip_size(const uint8_t* data, uint64_t size, int level);
uint64_t ref_deflate_size(const uint8_t* data, uint64_t size, int level);
uint64_t ref_zstd_size(const uint8_t* data, uint64_t size, int level);
uint64_t ref_lz4_size(const uint8_t* data, uint64_t size);

// ── Actual compress (returns CompressedBuffer) ──────────────────────────────

int ref_gzip_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out);
int ref_deflate_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out);
int ref_zstd_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out);
int ref_lz4_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out);

// ── Decompress ──────────────────────────────────────────────────────────────

int ref_gzip_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out, uint64_t* out_size);
int ref_deflate_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out, uint64_t* out_size, uint64_t expected_size);
int ref_zstd_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out, uint64_t* out_size);
int ref_lz4_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out, uint64_t* out_size, uint64_t expected_size);

#endif
