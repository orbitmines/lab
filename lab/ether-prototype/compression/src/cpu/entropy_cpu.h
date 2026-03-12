#ifndef ENTROPY_CPU_H
#define ENTROPY_CPU_H

#include "compression/types.h"

// Compute byte histogram from raw data
void cpu_histogram(const uint8_t* data, uint64_t size, ByteHistogram* out);

// Shannon entropy order-0: returns total bits needed
double cpu_shannon_order0(const ByteHistogram* hist);

// Shannon entropy order-1: returns total bits needed (uses bigram frequencies)
double cpu_shannon_order1(const uint8_t* data, uint64_t size);

#endif
