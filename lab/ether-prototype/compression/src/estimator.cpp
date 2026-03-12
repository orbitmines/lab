#include "compression/estimator.h"
#include "compression/gpu_context.h"

#include "cpu/entropy_cpu.h"
#include "cpu/huffman_cpu.h"
#include "cpu/arithmetic_cpu.h"
#include "cpu/ans_cpu.h"
#include "cpu/reference_wrappers.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Estimator {
    GpuContext* gpu;
    std::vector<uint8_t> data;
    ByteHistogram hist;
    bool hist_valid;
};

static void fill_result(EstimateResult* r, CompressionAlgorithm algo, double bits, uint64_t input_size) {
    r->algorithm = algo;
    r->estimated_bits = (uint64_t)ceil(bits);
    r->estimated_bytes = (r->estimated_bits + 7) / 8;
    r->bits_per_byte = (input_size > 0) ? bits / (double)input_size : 0.0;
    r->ratio = (input_size > 0) ? (double)r->estimated_bytes / (double)input_size : 0.0;
}

static void fill_result_bytes(EstimateResult* r, CompressionAlgorithm algo, uint64_t bytes, uint64_t input_size) {
    r->algorithm = algo;
    r->estimated_bytes = bytes;
    r->estimated_bits = bytes * 8;
    r->bits_per_byte = (input_size > 0) ? (double)(bytes * 8) / (double)input_size : 0.0;
    r->ratio = (input_size > 0) ? (double)bytes / (double)input_size : 0.0;
}

Estimator* estimator_create(int use_gpu) {
    Estimator* est = new Estimator();
    est->gpu = use_gpu ? gpu_context_create() : nullptr;
    est->hist_valid = false;
    return est;
}

void estimator_destroy(Estimator* est) {
    if (!est) return;
    if (est->gpu) gpu_context_destroy(est->gpu);
    delete est;
}

int estimator_load(Estimator* est, const uint8_t* data, uint64_t size) {
    if (!est || !data) return -1;
    est->data.assign(data, data + size);
    est->hist_valid = false;
    return 0;
}

int estimator_load_file(Estimator* est, const char* path) {
    if (!est || !path) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    est->data.resize((size_t)sz);
    size_t read = fread(est->data.data(), 1, (size_t)sz, f);
    fclose(f);
    if (read != (size_t)sz) return -1;
    est->hist_valid = false;
    return 0;
}

int estimator_get_backend_info(const Estimator* est, BackendInfo* info) {
    if (!est || !info) return -1;
    if (est->gpu) return gpu_context_get_info(est->gpu, info);
    info->backend = COMP_BACKEND_CPU;
    strncpy(info->device_name, "CPU only", sizeof(info->device_name) - 1);
    strncpy(info->vendor, "none", sizeof(info->vendor) - 1);
    info->max_buffer_size = 0;
    info->max_slots = 0;
    return 0;
}

static void ensure_histogram(Estimator* est) {
    if (!est->hist_valid) {
        // Try GPU histogram first, fall back to CPU
        if (est->gpu &&
            gpu_context_histogram(est->gpu, est->data.data(), est->data.size(), &est->hist) == 0) {
            // GPU histogram succeeded
        } else {
            cpu_histogram(est->data.data(), est->data.size(), &est->hist);
        }
        est->hist_valid = true;
    }
}

int estimator_histogram(const Estimator* est, ByteHistogram* out) {
    if (!est || est->data.empty() || !out) return -1;
    const_cast<Estimator*>(est)->hist_valid = false; // force recompute
    ensure_histogram(const_cast<Estimator*>(est));
    *out = est->hist;
    return 0;
}

int estimator_estimate(const Estimator* est, CompressionAlgorithm algo, EstimateResult* out) {
    if (!est || est->data.empty() || !out) return -1;
    ensure_histogram(const_cast<Estimator*>(est));

    uint64_t sz = est->data.size();
    const ByteHistogram* h = &est->hist;
    const uint8_t* d = est->data.data();

    switch (algo) {
    case COMP_SHANNON_ORDER0: fill_result(out, algo, cpu_shannon_order0(h), sz); break;
    case COMP_SHANNON_ORDER1: fill_result(out, algo, cpu_shannon_order1(d, sz), sz); break;
    case COMP_HUFFMAN:        fill_result(out, algo, cpu_huffman_size(h, nullptr), sz); break;
    case COMP_ARITHMETIC:     fill_result(out, algo, cpu_arithmetic_size(h), sz); break;
    case COMP_ANS:            fill_result(out, algo, cpu_ans_size(h), sz); break;
    case COMP_GZIP_EST:       fill_result_bytes(out, algo, ref_gzip_size(d, sz, 1), sz); break;
    case COMP_ZSTD_EST:       fill_result_bytes(out, algo, ref_zstd_size(d, sz, 1), sz); break;
    case COMP_LZ4_EST:        fill_result_bytes(out, algo, ref_lz4_size(d, sz), sz); break;
    case COMP_GZIP_EXACT:     fill_result_bytes(out, algo, ref_gzip_size(d, sz, 6), sz); break;
    case COMP_ZSTD_EXACT:     fill_result_bytes(out, algo, ref_zstd_size(d, sz, 3), sz); break;
    case COMP_LZ4_EXACT:      fill_result_bytes(out, algo, ref_lz4_size(d, sz), sz); break;
    case COMP_DEFLATE_EXACT:  fill_result_bytes(out, algo, ref_deflate_size(d, sz, 6), sz); break;
    default: return -1;
    }
    return 0;
}

int estimator_estimate_all(const Estimator* est, EstimateAllResult* out) {
    if (!est || est->data.empty() || !out) return -1;
    out->input_size = est->data.size();
    out->count = COMP_ALGORITHM_COUNT;
    for (int i = 0; i < COMP_ALGORITHM_COUNT; i++) {
        estimator_estimate(est, (CompressionAlgorithm)i, &out->results[i]);
    }
    return 0;
}

// ── Compression ─────────────────────────────────────────────────────────────

int estimator_compress(const Estimator* est, CompressionAlgorithm algo, int level, CompressedBuffer* out) {
    if (!est || est->data.empty() || !out) return -1;
    const uint8_t* d = est->data.data();
    uint64_t sz = est->data.size();

    memset(out, 0, sizeof(*out));

    switch (algo) {
    case COMP_HUFFMAN:       return cpu_huffman_compress(d, sz, out);
    case COMP_ARITHMETIC:    return cpu_arithmetic_compress(d, sz, out);
    case COMP_ANS:           return cpu_ans_compress(d, sz, out);
    case COMP_GZIP_EST:
    case COMP_GZIP_EXACT:    return ref_gzip_compress(d, sz, level > 0 ? level : 6, out);
    case COMP_ZSTD_EST:
    case COMP_ZSTD_EXACT:    return ref_zstd_compress(d, sz, level > 0 ? level : 3, out);
    case COMP_LZ4_EST:
    case COMP_LZ4_EXACT:     return ref_lz4_compress(d, sz, out);
    case COMP_DEFLATE_EXACT: return ref_deflate_compress(d, sz, level > 0 ? level : 6, out);
    default: return -1;
    }
}

int estimator_decompress(CompressionAlgorithm algo, const uint8_t* src, uint64_t src_size,
                         uint8_t** out_data, uint64_t* out_size, uint64_t expected_size) {
    if (!src || !out_data || !out_size) return -1;

    switch (algo) {
    case COMP_HUFFMAN:       return cpu_huffman_decompress(src, src_size, out_data, out_size);
    case COMP_ARITHMETIC:    return cpu_arithmetic_decompress(src, src_size, out_data, out_size);
    case COMP_ANS:           return cpu_ans_decompress(src, src_size, out_data, out_size);
    case COMP_GZIP_EST:
    case COMP_GZIP_EXACT:    return ref_gzip_decompress(src, src_size, out_data, out_size);
    case COMP_ZSTD_EST:
    case COMP_ZSTD_EXACT:    return ref_zstd_decompress(src, src_size, out_data, out_size);
    case COMP_LZ4_EST:
    case COMP_LZ4_EXACT:     return ref_lz4_decompress(src, src_size, out_data, out_size, expected_size);
    case COMP_DEFLATE_EXACT: return ref_deflate_decompress(src, src_size, out_data, out_size, expected_size);
    default: return -1;
    }
}

GpuContext* estimator_get_gpu(const Estimator* est) {
    if (!est) return nullptr;
    return est->gpu;
}

const uint8_t* estimator_get_data(const Estimator* est, uint64_t* out_size) {
    if (!est || est->data.empty()) return nullptr;
    if (out_size) *out_size = est->data.size();
    return est->data.data();
}

uint32_t estimator_parallel_slots(const Estimator* est) {
    if (!est || est->data.empty() || !est->gpu) return 0;
    return gpu_context_max_slots(est->gpu, est->data.size());
}

int estimator_verify(const Estimator* est) {
    if (!est || est->data.empty() || !est->gpu) return 0;

    int mismatches = 0;
    const uint8_t* d = est->data.data();
    uint64_t sz = est->data.size();

    // 1. Histogram: must be exact
    ByteHistogram cpu_hist = {};
    cpu_histogram(d, sz, &cpu_hist);

    ByteHistogram gpu_hist = {};
    if (gpu_context_histogram(est->gpu, d, sz, &gpu_hist) == 0) {
        for (int i = 0; i < 256; i++) {
            if (cpu_hist.histogram[i] != gpu_hist.histogram[i]) {
                fprintf(stderr, "[verify] histogram mismatch at bin %d: CPU=%lu GPU=%lu\n",
                        i, (unsigned long)cpu_hist.histogram[i], (unsigned long)gpu_hist.histogram[i]);
                mismatches++;
            }
        }
    } else {
        mismatches++;
    }

    // 2. Entropy order-0: within 100 bits
    double cpu_e0 = cpu_shannon_order0(&cpu_hist);
    double gpu_e0 = 0;
    if (gpu_context_entropy_order0(est->gpu, d, sz, &gpu_e0) == 0) {
        if (fabs(gpu_e0 - cpu_e0) > 100.0) {
            fprintf(stderr, "[verify] entropy_order0 mismatch: CPU=%.1f GPU=%.1f\n", cpu_e0, gpu_e0);
            mismatches++;
        }
    }

    // 3. Huffman size: within 1%
    double cpu_huff = cpu_huffman_size(&cpu_hist, nullptr);
    uint64_t gpu_huff = 0;
    if (gpu_context_huffman_size(est->gpu, d, sz, &gpu_huff) == 0) {
        double diff_pct = fabs((double)gpu_huff - cpu_huff) / cpu_huff * 100.0;
        if (diff_pct > 1.0) {
            fprintf(stderr, "[verify] huffman_size mismatch: CPU=%.0f GPU=%lu (%.2f%%)\n",
                    cpu_huff, (unsigned long)gpu_huff, diff_pct);
            mismatches++;
        }
    }

    return mismatches;
}
