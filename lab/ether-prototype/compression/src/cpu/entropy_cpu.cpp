#include "entropy_cpu.h"
#include <cmath>
#include <cstring>

void cpu_histogram(const uint8_t* data, uint64_t size, ByteHistogram* out) {
    memset(out->histogram, 0, sizeof(out->histogram));
    out->total = size;
    for (uint64_t i = 0; i < size; i++) {
        out->histogram[data[i]]++;
    }
}

double cpu_shannon_order0(const ByteHistogram* hist) {
    if (hist->total == 0) return 0.0;
    double total = (double)hist->total;
    double bits = 0.0;
    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) continue;
        double p = (double)hist->histogram[i] / total;
        bits -= (double)hist->histogram[i] * log2(p);
    }
    return bits;
}

double cpu_shannon_order1(const uint8_t* data, uint64_t size) {
    if (size < 2) return 0.0;

    // Count bigram frequencies and unigram frequencies
    uint64_t bigram[256][256] = {};
    uint64_t unigram[256] = {};

    for (uint64_t i = 0; i < size - 1; i++) {
        bigram[data[i]][data[i + 1]]++;
        unigram[data[i]]++;
    }
    unigram[data[size - 1]]++;

    // H(X|Y) = -sum over all (y,x) of P(y,x) * log2(P(x|y))
    // P(x|y) = count(y,x) / count(y)
    // Total bits = (size-1) * H(X|Y)
    double bits = 0.0;
    uint64_t pair_total = size - 1;

    for (int y = 0; y < 256; y++) {
        if (unigram[y] == 0) continue;
        double ctx_count = 0;
        for (int x = 0; x < 256; x++) {
            ctx_count += bigram[y][x];
        }
        if (ctx_count == 0) continue;
        for (int x = 0; x < 256; x++) {
            if (bigram[y][x] == 0) continue;
            double p_cond = (double)bigram[y][x] / ctx_count;
            bits -= (double)bigram[y][x] * log2(p_cond);
        }
    }
    return bits;
}
