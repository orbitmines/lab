#include "ans_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── rANS codec ──────────────────────────────────────────────────────────────
// Uses 32-bit state with 16-bit precision (M = 1<<16).

static const uint32_t RANS_PRECISION = 15;
static const uint32_t RANS_M = 1u << RANS_PRECISION;  // 32768 — fits in uint16
static const uint32_t RANS_L = 1u << 23; // renorm threshold

struct RansSymbol {
    uint32_t freq;
    uint32_t cum_freq;
};

static void quantize_freqs(const ByteHistogram* hist, uint32_t* freq, uint32_t* cum) {
    double total = (double)hist->total;
    uint32_t sum = 0;

    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) {
            freq[i] = 0;
        } else {
            freq[i] = (uint32_t)((double)hist->histogram[i] / total * RANS_M);
            if (freq[i] == 0) freq[i] = 1;
            sum += freq[i];
        }
    }

    while (sum != RANS_M) {
        int best = -1;
        double best_err = -1;
        for (int i = 0; i < 256; i++) {
            if (hist->histogram[i] == 0) continue;
            double ideal = (double)hist->histogram[i] / total * RANS_M;
            double err;
            if (sum > RANS_M) {
                if (freq[i] <= 1) continue;
                err = (double)freq[i] - ideal;
            } else {
                err = ideal - (double)freq[i];
            }
            if (err > best_err) { best_err = err; best = i; }
        }
        if (best < 0) break;
        if (sum > RANS_M) { freq[best]--; sum--; }
        else { freq[best]++; sum++; }
    }

    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
}

double cpu_ans_size(const ByteHistogram* hist) {
    if (hist->total == 0) return 0.0;

    uint32_t freq[256], cum[257];
    quantize_freqs(hist, freq, cum);

    int num_symbols = 0;
    double bits = 0.0;
    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) continue;
        num_symbols++;
        double p_q = (double)freq[i] / RANS_M;
        bits -= (double)hist->histogram[i] * log2(p_q);
    }

    // Table overhead
    bits += num_symbols * 16.0;
    return bits;
}

// ── rANS encoder ────────────────────────────────────────────────────────────
// Encodes in reverse order, outputs bytes in reverse (standard rANS pattern).

int cpu_ans_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out) {
    if (!data || !out) return -1;

    ByteHistogram hist;
    memset(&hist, 0, sizeof(hist));
    hist.total = size;
    for (uint64_t i = 0; i < size; i++) hist.histogram[data[i]]++;

    uint32_t freq[256], cum[257];
    quantize_freqs(&hist, freq, cum);

    // Encode in reverse
    std::vector<uint8_t> stream;
    uint32_t state = RANS_L;

    for (int64_t i = (int64_t)size - 1; i >= 0; i--) {
        uint8_t s = data[i];
        uint32_t f = freq[s];
        uint32_t c = cum[s];

        // Renormalize: output bytes until state is small enough for encoding
        uint64_t x_max = ((uint64_t)(RANS_L >> RANS_PRECISION) << 8) * f;
        while (state >= x_max) {
            stream.push_back((uint8_t)(state & 0xFF));
            state >>= 8;
        }

        // Encode: state' = (state / f) * M + (state % f) + c
        state = ((state / f) << RANS_PRECISION) + (state % f) + c;
    }

    // Flush final state (4 bytes, LE)
    for (int i = 0; i < 4; i++) {
        stream.push_back((uint8_t)(state & 0xFF));
        state >>= 8;
    }

    // Reverse the stream (rANS outputs in reverse)
    std::reverse(stream.begin(), stream.end());

    // Build output: header + freq table + stream
    std::vector<uint8_t> header;
    // Original size (LE)
    for (int i = 0; i < 8; i++) header.push_back((uint8_t)(size >> (i * 8)));
    // Frequency table (256 x uint16 LE — freq values fit in 16 bits since M=1<<16)
    for (int i = 0; i < 256; i++) {
        header.push_back((uint8_t)(freq[i] & 0xFF));
        header.push_back((uint8_t)(freq[i] >> 8));
    }

    uint64_t total = header.size() + stream.size();
    out->data = (uint8_t*)malloc(total);
    if (!out->data) return -1;
    memcpy(out->data, header.data(), header.size());
    memcpy(out->data + header.size(), stream.data(), stream.size());
    out->size = total;
    out->capacity = total;
    out->algorithm = COMP_ANS;
    return 0;
}

int cpu_ans_decompress(const uint8_t* src, uint64_t src_size,
                       uint8_t** out_data, uint64_t* out_size) {
    if (!src || !out_data || !out_size) return -1;

    uint64_t header_size = 8 + 256 * 2;
    if (src_size < header_size + 4) return -1;

    uint64_t orig_size = 0;
    for (int i = 0; i < 8; i++) orig_size |= (uint64_t)src[i] << (i * 8);

    uint32_t freq[256], cum[257];
    for (int i = 0; i < 256; i++) {
        freq[i] = (uint32_t)src[8 + i * 2] | ((uint32_t)src[8 + i * 2 + 1] << 8);
    }
    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];

    // Build cum-to-symbol lookup table
    uint8_t cum2sym[RANS_M];
    for (int i = 0; i < 256; i++) {
        for (uint32_t j = cum[i]; j < cum[i + 1]; j++) {
            cum2sym[j] = (uint8_t)i;
        }
    }

    // Decode
    const uint8_t* stream = src + header_size;
    uint64_t slen = src_size - header_size;
    uint64_t spos = 0;

    // Read initial state (4 bytes BE — we reversed on compress)
    uint32_t state = 0;
    for (int i = 0; i < 4; i++) {
        if (spos >= slen) return -1;
        state = (state << 8) | stream[spos++];
    }

    uint8_t* output = (uint8_t*)malloc(orig_size);
    if (!output) return -1;

    for (uint64_t i = 0; i < orig_size; i++) {
        // Decode symbol from state
        uint32_t slot = state & (RANS_M - 1);
        uint8_t s = cum2sym[slot];
        output[i] = s;

        // Update state: state' = freq[s] * (state >> PRECISION) + (state & (M-1)) - cum[s]
        state = freq[s] * (state >> RANS_PRECISION) + (state & (RANS_M - 1)) - cum[s];

        // Renormalize
        while (state < RANS_L && spos < slen) {
            state = (state << 8) | stream[spos++];
        }
    }

    *out_data = output;
    *out_size = orig_size;
    return 0;
}
