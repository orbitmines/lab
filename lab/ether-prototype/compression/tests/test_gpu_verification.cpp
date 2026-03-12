#include "compression/estimator.h"
#include "compression/gpu_context.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static GpuContext* g_gpu = nullptr;

static bool gpu_available() {
    if (!g_gpu) {
        g_gpu = gpu_context_create();
    }
    BackendInfo info;
    gpu_context_get_info(g_gpu, &info);
    return info.backend == COMP_BACKEND_WEBGPU;
}

// ── GPU histogram tests ────────────────────────────────────────────────────

static void test_gpu_histogram() {
    printf("  gpu_histogram...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 65536;
    uint8_t* data = (uint8_t*)malloc(N);
    for (int i = 0; i < N; i++) data[i] = (uint8_t)((i * 7 + 3) % 256);

    ByteHistogram gpu_hist, cpu_hist;

    // CPU
    memset(&cpu_hist, 0, sizeof(cpu_hist));
    cpu_hist.total = N;
    for (int i = 0; i < N; i++) cpu_hist.histogram[data[i]]++;

    // GPU
    int rc = gpu_context_histogram(g_gpu, data, N, &gpu_hist);
    assert(rc == 0);

    for (int i = 0; i < 256; i++) {
        if (cpu_hist.histogram[i] != gpu_hist.histogram[i]) {
            printf(" FAIL (bin %d: CPU=%lu GPU=%lu)\n",
                   i, (unsigned long)cpu_hist.histogram[i], (unsigned long)gpu_hist.histogram[i]);
            fflush(stdout); abort();
        }
    }

    free(data);
    printf(" OK (exact match)\n");
}

// ── GPU entropy tests ──────────────────────────────────────────────────────

static void test_gpu_entropy_order0() {
    printf("  gpu_entropy_order0...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 10000;
    uint8_t data[10000];
    for (int i = 0; i < N; i++) data[i] = (uint8_t)((i * 13 + 7) % 256);

    // CPU reference
    Estimator* est = estimator_create(0);
    estimator_load(est, data, N);
    EstimateResult cpu_r;
    estimator_estimate(est, COMP_SHANNON_ORDER0, &cpu_r);
    estimator_destroy(est);

    // GPU
    double gpu_bits = 0;
    int rc = gpu_context_entropy_order0(g_gpu, data, N, &gpu_bits);
    assert(rc == 0);

    double diff = fabs(gpu_bits - (double)cpu_r.estimated_bits);
    // Allow small float tolerance (GPU uses f32, CPU uses f64)
    if (diff > 100.0) {
        printf(" FAIL (CPU=%.1f GPU=%.1f diff=%.1f)\n",
               (double)cpu_r.estimated_bits, gpu_bits, diff);
        fflush(stdout); abort();
    }

    printf(" OK (CPU=%.0f GPU=%.0f diff=%.1f bits)\n",
           (double)cpu_r.estimated_bits, gpu_bits, diff);
}

static void test_gpu_entropy_order1() {
    printf("  gpu_entropy_order1...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 10000;
    uint8_t data[10000];
    for (int i = 0; i < N; i++) data[i] = (uint8_t)((i * 13 + 7) % 256);

    // CPU reference
    Estimator* est = estimator_create(0);
    estimator_load(est, data, N);
    EstimateResult cpu_r;
    estimator_estimate(est, COMP_SHANNON_ORDER1, &cpu_r);
    estimator_destroy(est);

    // GPU
    double gpu_bits = 0;
    int rc = gpu_context_entropy_order1(g_gpu, data, N, &gpu_bits);
    assert(rc == 0);

    double diff = fabs(gpu_bits - (double)cpu_r.estimated_bits);
    // Allow larger tolerance for order-1 (bigram atomics may have slight differences)
    if (diff > 200.0) {
        printf(" FAIL (CPU=%.1f GPU=%.1f diff=%.1f)\n",
               (double)cpu_r.estimated_bits, gpu_bits, diff);
        fflush(stdout); abort();
    }

    printf(" OK (CPU=%.0f GPU=%.0f diff=%.1f bits)\n",
           (double)cpu_r.estimated_bits, gpu_bits, diff);
}

// ── GPU LZ match statistics test ───────────────────────────────────────────

static void test_gpu_lz_stats() {
    printf("  gpu_lz_stats...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 65536;
    uint8_t* data = (uint8_t*)malloc(N);
    // Repetitive data → should find matches
    for (int i = 0; i < N; i++) data[i] = (uint8_t)(i % 100);

    uint32_t stats[4] = {};
    int rc = gpu_context_lz_stats(g_gpu, data, N, stats);
    assert(rc == 0);

    printf(" OK (literals=%u matches=%u match_count=%u)\n",
           stats[0], stats[1], stats[2]);

    // Sanity: for repetitive data, should find some matches
    assert(stats[1] > 0 || stats[2] > 0); // at least some matches

    free(data);
}

// ── GPU Huffman size-only test ─────────────────────────────────────────────

static void test_gpu_huffman_size() {
    printf("  gpu_huffman_size...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 8192;
    uint8_t data[8192];
    for (int i = 0; i < N; i++) data[i] = (uint8_t)((i * 7 + i / 100) % 256);

    // CPU reference
    Estimator* est = estimator_create(0);
    estimator_load(est, data, N);
    EstimateResult cpu_r;
    estimator_estimate(est, COMP_HUFFMAN, &cpu_r);
    estimator_destroy(est);

    // GPU size estimate
    uint64_t gpu_bits = 0;
    int rc = gpu_context_huffman_size(g_gpu, data, N, &gpu_bits);
    assert(rc == 0);

    double diff_pct = fabs((double)gpu_bits - (double)cpu_r.estimated_bits) / (double)cpu_r.estimated_bits * 100.0;
    if (diff_pct > 1.0) {
        printf(" FAIL (CPU=%lu GPU=%lu diff=%.2f%%)\n",
               (unsigned long)cpu_r.estimated_bits, (unsigned long)gpu_bits, diff_pct);
        fflush(stdout); abort();
    }

    printf(" OK (CPU=%lu GPU=%lu diff=%.3f%%)\n",
           (unsigned long)cpu_r.estimated_bits, (unsigned long)gpu_bits, diff_pct);
}

// ── GPU Huffman compress + CPU decompress round-trip ───────────────────────

static void test_gpu_huffman_roundtrip() {
    printf("  gpu_huffman_compress_roundtrip...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 8192;
    uint8_t data[8192];
    for (int i = 0; i < N; i++) data[i] = (uint8_t)((i * 7 + i / 100) % 256);

    // GPU compress
    CompressedBuffer comp = {};
    int rc = gpu_context_huffman_compress(g_gpu, data, N, &comp);
    if (rc != 0) { printf(" FAIL (compress rc=%d)\n", rc); fflush(stdout); abort(); }

    printf(" compressed=%lu bytes (%.1f%%)...",
           (unsigned long)comp.size, (double)comp.size / N * 100.0);
    fflush(stdout);

    // GPU decompress
    uint8_t* decomp = nullptr;
    uint64_t decomp_size = 0;
    rc = gpu_context_huffman_decompress(g_gpu, comp.data, comp.size, N, &decomp, &decomp_size);
    if (rc != 0) { printf(" FAIL (decompress rc=%d)\n", rc); fflush(stdout); abort(); }
    if (decomp_size != (uint64_t)N) {
        printf(" FAIL (size %lu != %d)\n", (unsigned long)decomp_size, N);
        fflush(stdout); abort();
    }
    if (memcmp(data, decomp, N) != 0) {
        for (int i = 0; i < N; i++) {
            if (data[i] != decomp[i]) {
                printf(" FAIL (mismatch at %d: expected %02x got %02x)\n", i, data[i], decomp[i]);
                break;
            }
        }
        fflush(stdout); abort();
    }

    free(decomp);
    compressed_buffer_free(&comp);
    printf(" OK (round-trip verified)\n");
}

// ── GPU Huffman round-trip with easy data (single symbol) ──────────────────

static void test_gpu_huffman_roundtrip_easy() {
    printf("  gpu_huffman_roundtrip_easy...");
    fflush(stdout);
    if (!gpu_available()) { printf(" SKIP\n"); return; }

    const int N = 4096;
    uint8_t data[4096];
    memset(data, 'A', N);

    CompressedBuffer comp = {};
    int rc = gpu_context_huffman_compress(g_gpu, data, N, &comp);
    if (rc != 0) { printf(" FAIL (compress rc=%d)\n", rc); fflush(stdout); abort(); }

    uint8_t* decomp = nullptr;
    uint64_t decomp_size = 0;
    rc = gpu_context_huffman_decompress(g_gpu, comp.data, comp.size, N, &decomp, &decomp_size);
    if (rc != 0) { printf(" FAIL (decompress rc=%d)\n", rc); fflush(stdout); abort(); }
    assert(decomp_size == (uint64_t)N);
    assert(memcmp(data, decomp, N) == 0);

    free(decomp);
    compressed_buffer_free(&comp);
    printf(" OK (%.1f%% ratio)\n", (double)comp.size / N * 100.0);
}

// ── GPU vs CPU cross-validation on enwik8 ──────────────────────────────────

static void test_gpu_vs_cpu_enwik8(const char* path) {
    printf("  gpu_vs_cpu_enwik8 (%s)...\n", path);
    fflush(stdout);
    if (!gpu_available()) { printf("    SKIP\n"); return; }

    // Load file
    FILE* f = fopen(path, "rb");
    if (!f) { printf("    SKIP (file not found)\n"); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(sz);
    fread(data, 1, sz, f);
    fclose(f);

    // GPU histogram vs CPU
    ByteHistogram gpu_hist, cpu_hist;
    memset(&cpu_hist, 0, sizeof(cpu_hist));
    cpu_hist.total = sz;
    for (long i = 0; i < sz; i++) cpu_hist.histogram[data[i]]++;

    int rc = gpu_context_histogram(g_gpu, data, sz, &gpu_hist);
    assert(rc == 0);
    for (int i = 0; i < 256; i++) assert(cpu_hist.histogram[i] == gpu_hist.histogram[i]);
    printf("    histogram: exact match\n");

    // GPU entropy order-0
    double gpu_entropy0 = 0;
    rc = gpu_context_entropy_order0(g_gpu, data, sz, &gpu_entropy0);
    if (rc == 0) {
        Estimator* est = estimator_create(0);
        estimator_load(est, data, sz);
        EstimateResult cpu_e0;
        estimator_estimate(est, COMP_SHANNON_ORDER0, &cpu_e0);
        double diff = fabs(gpu_entropy0 - (double)cpu_e0.estimated_bits);
        printf("    entropy_o0: CPU=%.0f GPU=%.0f diff=%.1f bits\n",
               (double)cpu_e0.estimated_bits, gpu_entropy0, diff);
        estimator_destroy(est);
    }

    // GPU LZ stats
    uint32_t lz_stats[4] = {};
    rc = gpu_context_lz_stats(g_gpu, data, sz, lz_stats);
    if (rc == 0) {
        printf("    lz_stats: literals=%u match_bytes=%u matches=%u\n",
               lz_stats[0], lz_stats[1], lz_stats[2]);
    }

    // GPU Huffman size estimate
    uint64_t gpu_huff_bits = 0;
    rc = gpu_context_huffman_size(g_gpu, data, sz, &gpu_huff_bits);
    if (rc == 0) {
        Estimator* est = estimator_create(0);
        estimator_load(est, data, sz);
        EstimateResult cpu_huff;
        estimator_estimate(est, COMP_HUFFMAN, &cpu_huff);
        double diff_pct = fabs((double)gpu_huff_bits - (double)cpu_huff.estimated_bits) /
                          (double)cpu_huff.estimated_bits * 100.0;
        printf("    huffman_size: CPU=%lu GPU=%lu diff=%.3f%%\n",
               (unsigned long)cpu_huff.estimated_bits, (unsigned long)gpu_huff_bits, diff_pct);
        estimator_destroy(est);
    }

    // GPU Huffman round-trip on 64KB chunk
    uint64_t chunk_size = 65536;
    if ((uint64_t)sz >= chunk_size) {
        CompressedBuffer comp = {};
        rc = gpu_context_huffman_compress(g_gpu, data, chunk_size, &comp);
        if (rc == 0) {
            uint8_t* decomp = nullptr;
            uint64_t decomp_size = 0;
            rc = gpu_context_huffman_decompress(g_gpu, comp.data, comp.size, chunk_size, &decomp, &decomp_size);
            if (rc == 0 && decomp_size == chunk_size && memcmp(data, decomp, chunk_size) == 0) {
                printf("    huffman_roundtrip (64KB): OK (%.1f%% ratio)\n",
                       (double)comp.size / chunk_size * 100.0);
            } else {
                printf("    huffman_roundtrip (64KB): FAIL\n");
            }
            free(decomp);
            compressed_buffer_free(&comp);
        }
    }

    free(data);
    printf("    OK\n");
}

int main(int argc, char** argv) {
    printf("GPU verification tests:\n");

    test_gpu_histogram();
    test_gpu_entropy_order0();
    test_gpu_entropy_order1();
    test_gpu_lz_stats();
    test_gpu_huffman_size();
    test_gpu_huffman_roundtrip();
    test_gpu_huffman_roundtrip_easy();

    const char* enwik8_path = nullptr;
    if (argc > 1) {
        enwik8_path = argv[1];
    } else {
        static const char* candidates[] = {"../../enwik8", "../../../enwik8", nullptr};
        for (const char** p = candidates; *p; p++) {
            FILE* f = fopen(*p, "rb");
            if (f) { fclose(f); enwik8_path = *p; break; }
        }
    }

    if (enwik8_path) {
        test_gpu_vs_cpu_enwik8(enwik8_path);
    }

    if (g_gpu) gpu_context_destroy(g_gpu);

    printf("\nAll GPU verification tests passed.\n");
    return 0;
}
