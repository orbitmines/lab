#ifndef ANS_CPU_H
#define ANS_CPU_H

#include "compression/types.h"

// rANS size estimate in bits.
double cpu_ans_size(const ByteHistogram* hist);

// Actual rANS compression. Returns 0 on success.
int cpu_ans_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out);

// rANS decompression. Caller frees *out_data.
int cpu_ans_decompress(const uint8_t* src, uint64_t src_size,
                       uint8_t** out_data, uint64_t* out_size);

#endif
