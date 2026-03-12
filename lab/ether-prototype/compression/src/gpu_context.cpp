#include "gpu_context_internal.h"

#ifdef HAS_WEBGPU
#include "cpu/huffman_cpu.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

GpuContext* gpu_context_create(void) {
    GpuContext* ctx = new GpuContext();

#ifdef HAS_WEBGPU
    ctx->has_gpu = false;
    if (webgpu_backend_init(&ctx->backend) == 0) {
        ctx->has_gpu = true;
        BackendInfo info;
        webgpu_backend_get_info(&ctx->backend, &info);
        fprintf(stderr, "[gpu] WebGPU initialized: %s (%s)\n", info.device_name, info.vendor);
    } else {
        fprintf(stderr, "[gpu] WebGPU not available, using CPU fallback\n");
    }
#else
    fprintf(stderr, "[gpu] Built without WebGPU support, using CPU only\n");
#endif

    return ctx;
}

void gpu_context_destroy(GpuContext* ctx) {
    if (!ctx) return;
#ifdef HAS_WEBGPU
    if (ctx->has_gpu) webgpu_backend_destroy(&ctx->backend);
#endif
    delete ctx;
}

int gpu_context_get_info(const GpuContext* ctx, BackendInfo* info) {
    if (!ctx || !info) return -1;
#ifdef HAS_WEBGPU
    if (ctx->has_gpu) {
        webgpu_backend_get_info(&ctx->backend, info);
        return 0;
    }
#endif
    info->backend = COMP_BACKEND_CPU;
    strncpy(info->device_name, "CPU fallback", sizeof(info->device_name) - 1);
    strncpy(info->vendor, "none", sizeof(info->vendor) - 1);
    info->max_buffer_size = 0;
    info->max_slots = 0;
    return 0;
}

uint32_t gpu_context_max_slots(const GpuContext* ctx, uint64_t data_size) {
    if (!ctx || !ctx->has_gpu || data_size == 0) return 0;
#ifdef HAS_WEBGPU
    BackendInfo info;
    webgpu_backend_get_info(&ctx->backend, &info);
    uint64_t available = info.max_buffer_size;
    if (available <= data_size) return 0;
    return (uint32_t)((available - data_size) / data_size);
#else
    return 0;
#endif
}

int gpu_context_histogram(GpuContext* ctx, const uint8_t* data, uint64_t size, ByteHistogram* out) {
    if (!ctx || !data || !out || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    WGPUBuffer buf = webgpu_upload(&ctx->backend, data, size);
    if (!buf) return -1;

    uint32_t hist32[256] = {};
    int rc = webgpu_histogram(&ctx->backend, buf, size, hist32);
    webgpu_free_buffer(buf);
    if (rc != 0) return -1;

    memset(out, 0, sizeof(*out));
    out->total = size;
    for (int i = 0; i < 256; i++) out->histogram[i] = hist32[i];
    return 0;
#else
    return -1;
#endif
}

int gpu_context_entropy_order0(GpuContext* ctx, const uint8_t* data, uint64_t size, double* out_bits) {
    if (!ctx || !data || !out_bits || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    // Compute histogram on GPU → read back → then compute entropy on GPU
    WGPUBuffer data_buf = webgpu_upload(&ctx->backend, data, size);
    if (!data_buf) return -1;

    uint32_t hist32[256] = {};
    int rc = webgpu_histogram(&ctx->backend, data_buf, size, hist32);
    webgpu_free_buffer(data_buf);
    if (rc != 0) return -1;

    // Upload histogram for entropy shader
    WGPUBuffer hist_buf = webgpu_upload(&ctx->backend, (const uint8_t*)hist32, 256 * sizeof(uint32_t));
    if (!hist_buf) return -1;

    float bits_f = 0.0f;
    rc = webgpu_entropy_order0(&ctx->backend, hist_buf, (uint32_t)size, &bits_f);
    *out_bits = (double)bits_f;

    webgpu_free_buffer(hist_buf);
    return rc;
#else
    return -1;
#endif
}

int gpu_context_entropy_order1(GpuContext* ctx, const uint8_t* data, uint64_t size, double* out_bits) {
    if (!ctx || !data || !out_bits || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    WGPUBuffer data_buf = webgpu_upload(&ctx->backend, data, size);
    if (!data_buf) return -1;

    uint32_t* bigram = (uint32_t*)calloc(256 * 256, sizeof(uint32_t));
    if (!bigram) { webgpu_free_buffer(data_buf); return -1; }

    int rc = webgpu_bigram_histogram(&ctx->backend, data_buf, size, bigram);
    webgpu_free_buffer(data_buf);
    if (rc != 0) { free(bigram); return -1; }

    // Compute order-1 entropy from bigram on CPU
    double total_bits = 0.0;
    for (int prev = 0; prev < 256; prev++) {
        uint64_t row_total = 0;
        for (int cur = 0; cur < 256; cur++) row_total += bigram[prev * 256 + cur];
        if (row_total == 0) continue;
        double rt = (double)row_total;
        for (int cur = 0; cur < 256; cur++) {
            uint32_t c = bigram[prev * 256 + cur];
            if (c == 0) continue;
            total_bits -= (double)c * log2((double)c / rt);
        }
    }

    free(bigram);
    *out_bits = total_bits;
    return 0;
#else
    return -1;
#endif
}

int gpu_context_lz_stats(GpuContext* ctx, const uint8_t* data, uint64_t size, uint32_t out_stats[4]) {
    if (!ctx || !data || !out_stats || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    WGPUBuffer data_buf = webgpu_upload(&ctx->backend, data, size);
    if (!data_buf) return -1;

    int rc = webgpu_lz_match_stats(&ctx->backend, data_buf, size, out_stats);
    webgpu_free_buffer(data_buf);
    return rc;
#else
    return -1;
#endif
}

int gpu_context_huffman_compress(GpuContext* ctx, const uint8_t* data, uint64_t size,
                                  CompressedBuffer* out) {
    if (!ctx || !data || !out || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    // Build Huffman table on CPU from histogram
    ByteHistogram hist = {};
    hist.total = size;
    for (uint64_t i = 0; i < size; i++) hist.histogram[data[i]]++;

    uint32_t lengths[256] = {};
    cpu_huffman_size(&hist, lengths);

    // Build canonical codes
    // Sort by (length, symbol), assign codes
    struct SymLen { int sym; uint32_t len; };
    SymLen syms[256];
    int nsyms = 0;
    for (int i = 0; i < 256; i++) {
        if (lengths[i] > 0) syms[nsyms++] = {i, lengths[i]};
    }
    // Sort
    for (int i = 0; i < nsyms - 1; i++)
        for (int j = i + 1; j < nsyms; j++)
            if (syms[j].len < syms[i].len || (syms[j].len == syms[i].len && syms[j].sym < syms[i].sym))
                { SymLen t = syms[i]; syms[i] = syms[j]; syms[j] = t; }

    uint32_t codes[256] = {};
    if (nsyms > 0) {
        uint32_t code = 0;
        uint32_t prev_len = syms[0].len;
        codes[syms[0].sym] = 0;
        for (int i = 1; i < nsyms; i++) {
            code++;
            if (syms[i].len > prev_len) code <<= (syms[i].len - prev_len);
            codes[syms[i].sym] = code;
            prev_len = syms[i].len;
        }
    }

    // Build code table: (code << 8) | length
    uint32_t code_table[256] = {};
    for (int i = 0; i < 256; i++) {
        code_table[i] = (codes[i] << 8) | lengths[i];
    }

    WGPUBuffer data_buf = webgpu_upload(&ctx->backend, data, size);
    if (!data_buf) return -1;

    int rc = webgpu_huffman_compress(&ctx->backend, data_buf, size, code_table, out);
    webgpu_free_buffer(data_buf);

    // Prepend code table to output for decompression
    if (rc == 0) {
        // Reformat: add code lengths to header so decompressor can rebuild codes
        // We'll store nsyms + (sym, length) pairs before the existing data
        uint64_t table_hdr_size = 4 + nsyms * 2;  // nsyms(4) + nsyms*(sym(1) + len(1))
        uint64_t new_size = table_hdr_size + out->size;
        uint8_t* new_data = (uint8_t*)malloc(new_size);
        if (!new_data) { compressed_buffer_free(out); return -1; }
        uint8_t* p = new_data;
        // nsyms (LE 32)
        for (int i = 0; i < 4; i++) *p++ = (uint8_t)(nsyms >> (i * 8));
        for (int i = 0; i < nsyms; i++) {
            *p++ = (uint8_t)syms[i].sym;
            *p++ = (uint8_t)syms[i].len;
        }
        memcpy(p, out->data, out->size);
        free(out->data);
        out->data = new_data;
        out->size = new_size;
        out->capacity = new_size;
    }

    return rc;
#else
    return -1;
#endif
}

int gpu_context_huffman_size(GpuContext* ctx, const uint8_t* data, uint64_t size,
                              uint64_t* out_bits) {
    if (!ctx || !data || !out_bits || size == 0) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    ByteHistogram hist = {};
    hist.total = size;
    for (uint64_t i = 0; i < size; i++) hist.histogram[data[i]]++;

    uint32_t lengths[256] = {};
    cpu_huffman_size(&hist, lengths);

    // Build code table
    struct SymLen { int sym; uint32_t len; };
    SymLen syms[256];
    int nsyms = 0;
    for (int i = 0; i < 256; i++) {
        if (lengths[i] > 0) syms[nsyms++] = {i, lengths[i]};
    }
    for (int i = 0; i < nsyms - 1; i++)
        for (int j = i + 1; j < nsyms; j++)
            if (syms[j].len < syms[i].len || (syms[j].len == syms[i].len && syms[j].sym < syms[i].sym))
                { SymLen t = syms[i]; syms[i] = syms[j]; syms[j] = t; }

    uint32_t codes[256] = {};
    if (nsyms > 0) {
        uint32_t code = 0, prev_len = syms[0].len;
        for (int i = 1; i < nsyms; i++) {
            code++;
            if (syms[i].len > prev_len) code <<= (syms[i].len - prev_len);
            codes[syms[i].sym] = code;
            prev_len = syms[i].len;
        }
    }

    uint32_t code_table[256] = {};
    for (int i = 0; i < 256; i++) code_table[i] = (codes[i] << 8) | lengths[i];

    WGPUBuffer data_buf = webgpu_upload(&ctx->backend, data, size);
    if (!data_buf) return -1;

    int rc = webgpu_huffman_size(&ctx->backend, data_buf, size, code_table, out_bits);
    webgpu_free_buffer(data_buf);
    return rc;
#else
    return -1;
#endif
}

int gpu_context_huffman_decompress(GpuContext* ctx, const uint8_t* compressed, uint64_t comp_size,
                                    uint64_t original_size, uint8_t** out_data, uint64_t* out_size) {
    if (!ctx || !compressed || !out_data || !out_size) return -1;
#ifdef HAS_WEBGPU
    if (!ctx->has_gpu) return -1;

    // Parse code table header: nsyms(4) + nsyms*(sym(1)+len(1))
    if (comp_size < 4) return -1;
    uint32_t nsyms = 0;
    for (int i = 0; i < 4; i++) nsyms |= (uint32_t)compressed[i] << (i * 8);
    uint64_t table_hdr_size = 4 + nsyms * 2;
    if (comp_size < table_hdr_size) return -1;

    // Rebuild canonical codes and decode table
    uint32_t lengths[256] = {};
    const uint8_t* p = compressed + 4;
    for (uint32_t i = 0; i < nsyms; i++) {
        uint8_t sym = *p++;
        lengths[sym] = *p++;
    }

    // Build canonical codes
    struct SymLen { int sym; uint32_t len; };
    SymLen sorted[256];
    int ns = 0;
    for (int i = 0; i < 256; i++) if (lengths[i] > 0) sorted[ns++] = {i, lengths[i]};
    for (int i = 0; i < ns - 1; i++)
        for (int j = i + 1; j < ns; j++)
            if (sorted[j].len < sorted[i].len || (sorted[j].len == sorted[i].len && sorted[j].sym < sorted[i].sym))
                { SymLen t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

    uint32_t codes[256] = {};
    uint32_t max_code_len = 0;
    if (ns > 0) {
        uint32_t code = 0, prev_len = sorted[0].len;
        max_code_len = sorted[ns - 1].len;
        for (int i = 1; i < ns; i++) {
            code++;
            if (sorted[i].len > prev_len) code <<= (sorted[i].len - prev_len);
            codes[sorted[i].sym] = code;
            prev_len = sorted[i].len;
        }
    }

    // Build decode lookup table (16-bit prefix → (symbol << 8) | code_length)
    const uint32_t LOOKUP_BITS = 16;
    const uint32_t LOOKUP_SIZE = 1u << LOOKUP_BITS;
    uint32_t* decode_table = (uint32_t*)calloc(LOOKUP_SIZE, sizeof(uint32_t));
    for (int i = 0; i < 256; i++) {
        if (lengths[i] == 0) continue;
        uint32_t code = codes[i];
        uint32_t len = lengths[i];
        if (len > LOOKUP_BITS) continue; // skip for simplicity
        // This code with 'len' bits should match any prefix where the top 'len' bits match
        uint32_t prefix = code << (LOOKUP_BITS - len);
        uint32_t count = 1u << (LOOKUP_BITS - len);
        for (uint32_t j = 0; j < count; j++) {
            decode_table[prefix + j] = ((uint32_t)i << 8) | len;
        }
    }

    uint8_t* result = nullptr;
    int rc = webgpu_huffman_decompress(&ctx->backend, compressed + table_hdr_size,
                                        comp_size - table_hdr_size,
                                        decode_table, max_code_len,
                                        original_size, &result);
    free(decode_table);
    if (rc == 0) {
        *out_data = result;
        *out_size = original_size;
    }
    return rc;
#else
    return -1;
#endif
}
