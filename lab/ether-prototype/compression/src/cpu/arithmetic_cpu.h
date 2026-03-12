#ifndef ARITHMETIC_CPU_H
#define ARITHMETIC_CPU_H

#include "compression/types.h"

// Arithmetic coding size estimate in bits.
double cpu_arithmetic_size(const ByteHistogram* hist);

// Actual arithmetic compression. Returns 0 on success.
int cpu_arithmetic_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out);

// Arithmetic decompression. Caller frees *out_data.
int cpu_arithmetic_decompress(const uint8_t* src, uint64_t src_size,
                              uint8_t** out_data, uint64_t* out_size);

#endif
