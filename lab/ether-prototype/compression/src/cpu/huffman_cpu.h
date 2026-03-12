#ifndef HUFFMAN_CPU_H
#define HUFFMAN_CPU_H

#include "compression/types.h"

// Compute Huffman coded size in bits from histogram.
double cpu_huffman_size(const ByteHistogram* hist, uint32_t* codelengths);

// Actual Huffman compression. Returns 0 on success.
int cpu_huffman_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out);

// Huffman decompression. Caller frees *out_data.
int cpu_huffman_decompress(const uint8_t* src, uint64_t src_size,
                           uint8_t** out_data, uint64_t* out_size);

#endif
