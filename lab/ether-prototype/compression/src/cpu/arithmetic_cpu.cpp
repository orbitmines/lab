#include "arithmetic_cpu.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

double cpu_arithmetic_size(const ByteHistogram* hist) {
    if (hist->total == 0) return 0.0;
    double total = (double)hist->total;
    double bits = 0.0;
    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) continue;
        double c = (double)hist->histogram[i];
        bits -= c * log2(c / total);
    }
    bits += 2.0;
    return bits;
}

// ── Arithmetic coder using 64-bit state with 32-bit precision ───────────────
// Based on the range coder variant (no underflow counting needed).

static const uint32_t AC_TOP = 1u << 24;
static const uint32_t AC_BOT = 1u << 16;
static const uint32_t AC_PREC = 15;
static const uint32_t AC_TOTAL = 1u << AC_PREC;  // 32768 — fits in uint16

static void ac_quantize(const ByteHistogram* hist, uint32_t* freq, uint32_t* cum) {
    double total = (double)hist->total;
    uint32_t sum = 0;
    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) { freq[i] = 0; continue; }
        freq[i] = (uint32_t)((double)hist->histogram[i] / total * AC_TOTAL);
        if (freq[i] == 0) freq[i] = 1;
        sum += freq[i];
    }
    // Adjust to exactly AC_TOTAL
    while (sum != AC_TOTAL) {
        int best = -1; double best_err = -1;
        for (int i = 0; i < 256; i++) {
            if (hist->histogram[i] == 0) continue;
            double ideal = (double)hist->histogram[i] / total * AC_TOTAL;
            double err;
            if (sum > AC_TOTAL) { if (freq[i] <= 1) continue; err = (double)freq[i] - ideal; }
            else { err = ideal - (double)freq[i]; }
            if (err > best_err) { best_err = err; best = i; }
        }
        if (best < 0) break;
        if (sum > AC_TOTAL) { freq[best]--; sum--; } else { freq[best]++; sum++; }
    }
    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];
}

// ── Range encoder (byte-aligned variant) ────────────────────────────────────

struct RangeEncoder {
    std::vector<uint8_t> out;
    uint32_t lo = 0;
    uint32_t range = 0xFFFFFFFF;

    void encode(uint32_t cum_lo, uint32_t cum_hi, uint32_t total) {
        range /= total;
        lo += cum_lo * range;
        range = (cum_hi - cum_lo) * range;

        // Renormalize
        while (range < AC_BOT) {
            if ((lo ^ (lo + range)) >= AC_TOP) {
                // Range straddles a carry boundary — shrink range to avoid it
                range = (uint32_t)(-(int32_t)lo) & (AC_BOT - 1);
            }
            out.push_back((uint8_t)(lo >> 24));
            lo <<= 8;
            range <<= 8;
        }
    }

    void finish() {
        for (int i = 0; i < 4; i++) {
            out.push_back((uint8_t)(lo >> 24));
            lo <<= 8;
        }
    }
};

struct RangeDecoder {
    const uint8_t* data;
    uint64_t size;
    uint64_t pos = 0;
    uint32_t lo = 0;
    uint32_t range = 0xFFFFFFFF;
    uint32_t code = 0;

    void init() {
        for (int i = 0; i < 4; i++) code = (code << 8) | get_byte();
    }

    uint8_t get_byte() { return (pos < size) ? data[pos++] : 0; }

    uint32_t get_freq(uint32_t total) {
        range /= total;
        uint32_t offset = (code - lo) / range;
        return offset < total ? offset : total - 1;
    }

    void decode(uint32_t cum_lo, uint32_t cum_hi, uint32_t total) {
        lo += cum_lo * range;
        range = (cum_hi - cum_lo) * range;

        while (range < AC_BOT) {
            if ((lo ^ (lo + range)) >= AC_TOP) {
                range = (uint32_t)(-(int32_t)lo) & (AC_BOT - 1);
            }
            code = (code << 8) | get_byte();
            lo <<= 8;
            range <<= 8;
        }
    }
};

// Format:
// [8 bytes] original_size (LE)
// [256 * 2 bytes] freq table (uint16 LE)
// [N bytes] range-coded stream

int cpu_arithmetic_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out) {
    if (!data || !out) return -1;

    ByteHistogram hist = {};
    hist.total = size;
    for (uint64_t i = 0; i < size; i++) hist.histogram[data[i]]++;

    uint32_t freq[256], cum[257];
    ac_quantize(&hist, freq, cum);

    RangeEncoder enc;
    for (uint64_t i = 0; i < size; i++) {
        uint8_t s = data[i];
        enc.encode(cum[s], cum[s + 1], AC_TOTAL);
    }
    enc.finish();

    // Header
    std::vector<uint8_t> header;
    for (int i = 0; i < 8; i++) header.push_back((uint8_t)(size >> (i * 8)));
    for (int i = 0; i < 256; i++) {
        header.push_back((uint8_t)(freq[i] & 0xFF));
        header.push_back((uint8_t)((freq[i] >> 8) & 0xFF));
    }

    uint64_t total = header.size() + enc.out.size();
    out->data = (uint8_t*)malloc(total);
    if (!out->data) return -1;
    memcpy(out->data, header.data(), header.size());
    memcpy(out->data + header.size(), enc.out.data(), enc.out.size());
    out->size = total;
    out->capacity = total;
    out->algorithm = COMP_ARITHMETIC;
    return 0;
}

int cpu_arithmetic_decompress(const uint8_t* src, uint64_t src_size,
                              uint8_t** out_data, uint64_t* out_size) {
    if (!src || !out_data || !out_size) return -1;
    uint64_t hdr = 8 + 256 * 2;
    if (src_size < hdr + 4) return -1;

    uint64_t orig_size = 0;
    for (int i = 0; i < 8; i++) orig_size |= (uint64_t)src[i] << (i * 8);

    uint32_t freq[256], cum[257];
    for (int i = 0; i < 256; i++) {
        freq[i] = (uint32_t)src[8 + i * 2] | ((uint32_t)src[8 + i * 2 + 1] << 8);
    }
    cum[0] = 0;
    for (int i = 0; i < 256; i++) cum[i + 1] = cum[i] + freq[i];

    // Lookup table: cumulative freq → symbol
    uint8_t cum2sym[AC_TOTAL];
    for (int i = 0; i < 256; i++) {
        for (uint32_t j = cum[i]; j < cum[i + 1]; j++) cum2sym[j] = (uint8_t)i;
    }

    RangeDecoder dec;
    dec.data = src + hdr;
    dec.size = src_size - hdr;
    dec.init();

    uint8_t* output = (uint8_t*)malloc(orig_size);
    if (!output) return -1;

    for (uint64_t i = 0; i < orig_size; i++) {
        uint32_t f = dec.get_freq(AC_TOTAL);
        uint8_t s = cum2sym[f];
        output[i] = s;
        dec.decode(cum[s], cum[s + 1], AC_TOTAL);
    }

    *out_data = output;
    *out_size = orig_size;
    return 0;
}
