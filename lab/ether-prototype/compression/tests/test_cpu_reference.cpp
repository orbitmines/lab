#include "compression/estimator.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* algo_name(CompressionAlgorithm a) {
    switch (a) {
    case COMP_SHANNON_ORDER0: return "shannon_order0";
    case COMP_SHANNON_ORDER1: return "shannon_order1";
    case COMP_HUFFMAN:        return "huffman";
    case COMP_ARITHMETIC:     return "arithmetic";
    case COMP_ANS:            return "ans";
    case COMP_GZIP_EST:       return "gzip_est";
    case COMP_ZSTD_EST:       return "zstd_est";
    case COMP_LZ4_EST:        return "lz4_est";
    case COMP_GZIP_EXACT:     return "gzip_exact";
    case COMP_ZSTD_EXACT:     return "zstd_exact";
    case COMP_LZ4_EXACT:      return "lz4_exact";
    case COMP_DEFLATE_EXACT:  return "deflate_exact";
    default:                  return "unknown";
    }
}

static void test_histogram() {
    printf("  test_histogram...");
    uint8_t data[] = {0, 0, 0, 1, 1, 2};
    Estimator* est = estimator_create(0);
    estimator_load(est, data, sizeof(data));
    ByteHistogram hist;
    assert(estimator_histogram(est, &hist) == 0);
    assert(hist.total == 6);
    assert(hist.histogram[0] == 3);
    assert(hist.histogram[1] == 2);
    assert(hist.histogram[2] == 1);
    assert(hist.histogram[3] == 0);
    estimator_destroy(est);
    printf(" OK\n");
}

static void test_entropy_known() {
    printf("  test_entropy_known...");
    uint8_t data[1000];
    for (int i = 0; i < 1000; i++) data[i] = i % 2;
    Estimator* est = estimator_create(0);
    estimator_load(est, data, sizeof(data));
    EstimateResult r;
    assert(estimator_estimate(est, COMP_SHANNON_ORDER0, &r) == 0);
    assert(fabs(r.estimated_bits - 1000.0) < 2.0);
    estimator_destroy(est);
    printf(" OK\n");
}

static void test_huffman_vs_entropy() {
    printf("  test_huffman_vs_entropy...");
    uint8_t data[10000];
    for (int i = 0; i < 10000; i++) data[i] = (i * 7 + 3) % 256;
    Estimator* est = estimator_create(0);
    estimator_load(est, data, sizeof(data));
    EstimateResult shannon, huffman;
    estimator_estimate(est, COMP_SHANNON_ORDER0, &shannon);
    estimator_estimate(est, COMP_HUFFMAN, &huffman);
    assert(huffman.estimated_bits >= shannon.estimated_bits - 10);
    assert(huffman.estimated_bits <= shannon.estimated_bits + 10000);
    estimator_destroy(est);
    printf(" OK\n");
}

static void test_arithmetic_vs_shannon() {
    printf("  test_arithmetic_vs_shannon...");
    uint8_t data[5000];
    for (int i = 0; i < 5000; i++) data[i] = (i * 13) % 200;
    Estimator* est = estimator_create(0);
    estimator_load(est, data, sizeof(data));
    EstimateResult shannon, arith;
    estimator_estimate(est, COMP_SHANNON_ORDER0, &shannon);
    estimator_estimate(est, COMP_ARITHMETIC, &arith);
    double diff = fabs((double)arith.estimated_bits - (double)shannon.estimated_bits);
    assert(diff < 10.0);
    estimator_destroy(est);
    printf(" OK\n");
}

static void test_ans_vs_arithmetic() {
    printf("  test_ans_vs_arithmetic...");
    uint8_t data[10000];
    for (int i = 0; i < 10000; i++) data[i] = (i * 31 + 17) % 256;
    Estimator* est = estimator_create(0);
    estimator_load(est, data, sizeof(data));
    EstimateResult arith, ans;
    estimator_estimate(est, COMP_ARITHMETIC, &arith);
    estimator_estimate(est, COMP_ANS, &ans);
    double diff_pct = fabs((double)ans.estimated_bits - (double)arith.estimated_bits) / (double)arith.estimated_bits * 100.0;
    // ANS table overhead (256 symbols * 16 bits = 4096 bits) is significant on small data
    assert(diff_pct < 10.0);
    estimator_destroy(est);
    printf(" OK\n");
}

// ── Round-trip compression tests ────────────────────────────────────────────

static void test_roundtrip(const char* name, CompressionAlgorithm algo, const uint8_t* data, uint64_t size) {
    printf("  test_roundtrip_%s...", name);
    fflush(stdout);

    Estimator* est = estimator_create(0);
    estimator_load(est, data, size);

    CompressedBuffer compressed = {};
    int rc = estimator_compress(est, algo, 0, &compressed);
    if (rc != 0) { printf(" FAIL (compress rc=%d)\n", rc); abort(); }
    if (!compressed.data) { printf(" FAIL (null data)\n"); abort(); }
    if (compressed.size == 0) { printf(" FAIL (zero size)\n"); abort(); }

    uint8_t* decompressed = nullptr;
    uint64_t decom_size = 0;
    rc = estimator_decompress(algo, compressed.data, compressed.size,
                              &decompressed, &decom_size, size);
    if (rc != 0) { printf(" FAIL (decompress rc=%d)\n", rc); fflush(stdout); abort(); }
    if (decom_size != size) { printf(" FAIL (size %lu != %lu)\n", (unsigned long)decom_size, (unsigned long)size); fflush(stdout); abort(); }
    if (memcmp(data, decompressed, size) != 0) {
        for (uint64_t i = 0; i < size; i++) {
            if (data[i] != decompressed[i]) {
                printf(" FAIL (mismatch at byte %lu: expected %02x got %02x)\n", (unsigned long)i, data[i], decompressed[i]);
                break;
            }
        }
        fflush(stdout);
        abort();
    }

    printf(" OK (%.1f%% ratio)\n", (double)compressed.size / size * 100.0);

    free(decompressed);
    compressed_buffer_free(&compressed);
    estimator_destroy(est);
}

static void test_roundtrips() {
    // Test data: mix of patterns
    uint8_t data[8192];
    for (int i = 0; i < 8192; i++) data[i] = (uint8_t)((i * 7 + i / 100) % 256);

    test_roundtrip("huffman", COMP_HUFFMAN, data, sizeof(data));
    test_roundtrip("arithmetic", COMP_ARITHMETIC, data, sizeof(data));
    test_roundtrip("ans", COMP_ANS, data, sizeof(data));
    test_roundtrip("gzip", COMP_GZIP_EXACT, data, sizeof(data));
    test_roundtrip("zstd", COMP_ZSTD_EXACT, data, sizeof(data));
    test_roundtrip("deflate", COMP_DEFLATE_EXACT, data, sizeof(data));
    test_roundtrip("lz4", COMP_LZ4_EXACT, data, sizeof(data));

    // Test with highly compressible data
    uint8_t easy[4096];
    memset(easy, 'A', sizeof(easy));
    test_roundtrip("huffman_easy", COMP_HUFFMAN, easy, sizeof(easy));
    test_roundtrip("ans_easy", COMP_ANS, easy, sizeof(easy));
    test_roundtrip("arithmetic_easy", COMP_ARITHMETIC, easy, sizeof(easy));
    test_roundtrip("zstd_easy", COMP_ZSTD_EXACT, easy, sizeof(easy));
}

static void test_enwik8(const char* path) {
    printf("  test_enwik8 (%s)...\n", path);

    Estimator* est = estimator_create(0);
    if (estimator_load_file(est, path) != 0) {
        printf("    SKIP (file not found)\n");
        estimator_destroy(est);
        return;
    }

    EstimateAllResult all;
    estimator_estimate_all(est, &all);

    printf("    input size: %lu bytes\n", (unsigned long)all.input_size);
    printf("    %-20s %15s %15s %10s %8s\n", "algorithm", "bits", "bytes", "bpb", "ratio");
    printf("    %-20s %15s %15s %10s %8s\n", "─────────", "────", "─────", "───", "─────");

    for (int i = 0; i < all.count; i++) {
        EstimateResult* r = &all.results[i];
        if (r->estimated_bytes == 0) {
            printf("    %-20s %15s (unavailable)\n", algo_name(r->algorithm), "—");
            continue;
        }
        printf("    %-20s %15lu %15lu %10.4f %7.2f%%\n",
               algo_name(r->algorithm),
               (unsigned long)r->estimated_bits,
               (unsigned long)r->estimated_bytes,
               r->bits_per_byte,
               r->ratio * 100.0);
    }

    // Sanity checks
    for (int i = 0; i < all.count; i++) {
        EstimateResult* r = &all.results[i];
        if (r->estimated_bytes == 0) continue;
        assert(r->estimated_bytes < all.input_size);
        assert(r->estimated_bytes > 0);
        assert(r->bits_per_byte > 0.0 && r->bits_per_byte < 8.0);
    }

    assert(all.results[COMP_SHANNON_ORDER1].estimated_bits <
           all.results[COMP_SHANNON_ORDER0].estimated_bits);

    // Test actual compression round-trip on a chunk of enwik8
    printf("    round-trip tests on 1MB chunk...\n");
    CompressedBuffer comp = {};
    uint8_t* decomp = nullptr;
    uint64_t decomp_size = 0;
    uint64_t chunk_size = 1024 * 1024; // 1 MB

    Estimator* chunk_est = estimator_create(0);
    // Load just first 1MB
    FILE* f = fopen(path, "rb");
    if (f) {
        uint8_t* chunk = (uint8_t*)malloc(chunk_size);
        fread(chunk, 1, chunk_size, f);
        fclose(f);
        estimator_load(chunk_est, chunk, chunk_size);

        CompressionAlgorithm algos[] = {COMP_HUFFMAN, COMP_ARITHMETIC, COMP_ANS,
                                        COMP_GZIP_EXACT, COMP_ZSTD_EXACT, COMP_LZ4_EXACT};
        const char* names[] = {"huffman", "arithmetic", "ans", "gzip", "zstd", "lz4"};
        for (int i = 0; i < 6; i++) {
            memset(&comp, 0, sizeof(comp));
            int rc = estimator_compress(chunk_est, algos[i], 0, &comp);
            assert(rc == 0);
            rc = estimator_decompress(algos[i], comp.data, comp.size, &decomp, &decomp_size, chunk_size);
            assert(rc == 0);
            assert(decomp_size == chunk_size);
            assert(memcmp(chunk, decomp, chunk_size) == 0);
            printf("      %-12s compressed: %lu → %lu (%.1f%%)\n",
                   names[i], (unsigned long)chunk_size, (unsigned long)comp.size,
                   (double)comp.size / chunk_size * 100.0);
            free(decomp);
            compressed_buffer_free(&comp);
        }
        free(chunk);
    }
    estimator_destroy(chunk_est);

    estimator_destroy(est);
    printf("    OK\n");
}

int main(int argc, char** argv) {
    printf("CPU reference tests:\n");

    test_histogram();
    test_entropy_known();
    test_huffman_vs_entropy();
    test_arithmetic_vs_shannon();
    test_ans_vs_arithmetic();
    test_roundtrips();

    const char* enwik8_path = nullptr;
    if (argc > 1) {
        enwik8_path = argv[1];
    } else {
        static const char* candidates[] = {"../enwik8", "../../enwik8", "../../../enwik8", nullptr};
        for (const char** p = candidates; *p; p++) {
            FILE* f = fopen(*p, "rb");
            if (f) { fclose(f); enwik8_path = *p; break; }
        }
    }

    if (enwik8_path) {
        test_enwik8(enwik8_path);
    } else {
        printf("  test_enwik8... SKIP (enwik8 not found, pass path as arg)\n");
    }

    printf("\nAll tests passed.\n");
    return 0;
}
