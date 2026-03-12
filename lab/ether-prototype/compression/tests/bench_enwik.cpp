#include "compression/estimator.h"
#include "compression/gpu_context.h"

#include <chrono>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <enwik8_path>\n", argv[0]);
        return 1;
    }

    printf("Loading %s...\n", argv[1]);

    // Load file manually for GPU benchmarks
    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "Failed to open %s\n", argv[1]); return 1; }
    fseek(fp, 0, SEEK_END);
    long file_sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t* raw_data = (uint8_t*)malloc(file_sz);
    fread(raw_data, 1, file_sz, fp);
    fclose(fp);

    // ── CPU estimator ────────────────────────────────────────────────────────

    auto t0 = std::chrono::high_resolution_clock::now();

    Estimator* est = estimator_create(0);
    estimator_load(est, raw_data, file_sz);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Loaded %ld bytes in %.1f ms\n\n", file_sz, load_ms);

    BackendInfo info;
    estimator_get_backend_info(est, &info);
    printf("CPU Backend: %s (%s)\n\n", info.device_name, info.vendor);

    printf("%-20s %12s %15s %10s %8s\n", "algorithm", "time_ms", "bytes", "bpb", "ratio");
    printf("%-20s %12s %15s %10s %8s\n", "─────────", "───────", "─────", "───", "─────");

    for (int i = 0; i < COMP_ALGORITHM_COUNT; i++) {
        auto algo = (CompressionAlgorithm)i;
        EstimateResult r;

        auto start = std::chrono::high_resolution_clock::now();
        int rc = estimator_estimate(est, algo, &r);
        auto end = std::chrono::high_resolution_clock::now();

        if (rc != 0 || r.estimated_bytes == 0) {
            printf("%-20s %12s\n", algo_name(algo), "—");
            continue;
        }

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        printf("%-20s %11.1f %15lu %10.4f %7.2f%%\n",
               algo_name(algo),
               ms,
               (unsigned long)r.estimated_bytes,
               r.bits_per_byte,
               r.ratio * 100.0);
    }

    estimator_destroy(est);

    // ── GPU benchmarks ──────────────────────────────────────────────────────

    printf("\n── GPU benchmarks ──────────────────────────────────────────────\n\n");

    GpuContext* gpu = gpu_context_create();
    gpu_context_get_info(gpu, &info);
    if (info.backend != COMP_BACKEND_WEBGPU) {
        printf("No GPU available, skipping GPU benchmarks.\n");
        gpu_context_destroy(gpu);
        free(raw_data);
        return 0;
    }

    printf("GPU Backend: %s (%s)\n", info.device_name, info.vendor);
    printf("Max buffer size: %lu MB\n", (unsigned long)(info.max_buffer_size / (1024 * 1024)));

    // Parallel slots
    uint32_t slots = gpu_context_max_slots(gpu, file_sz);
    printf("Parallel slots for %ld bytes: %u\n\n", file_sz, slots);

    printf("%-25s %12s %15s\n", "operation", "time_ms", "result");
    printf("%-25s %12s %15s\n", "─────────", "───────", "──────");

    // GPU histogram
    {
        ByteHistogram hist;
        auto start = std::chrono::high_resolution_clock::now();
        int rc = gpu_context_histogram(gpu, raw_data, file_sz, &hist);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (rc == 0) {
            printf("%-25s %11.1f %15s\n", "histogram", ms, "OK");
        }
    }

    // GPU entropy order-0
    {
        double bits = 0;
        auto start = std::chrono::high_resolution_clock::now();
        int rc = gpu_context_entropy_order0(gpu, raw_data, file_sz, &bits);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (rc == 0) {
            printf("%-25s %11.1f %13.0f b\n", "entropy_order0", ms, bits);
        }
    }

    // GPU entropy order-1
    {
        double bits = 0;
        auto start = std::chrono::high_resolution_clock::now();
        int rc = gpu_context_entropy_order1(gpu, raw_data, file_sz, &bits);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (rc == 0) {
            printf("%-25s %11.1f %13.0f b\n", "entropy_order1", ms, bits);
        }
    }

    // GPU LZ stats
    {
        uint32_t stats[4] = {};
        auto start = std::chrono::high_resolution_clock::now();
        int rc = gpu_context_lz_stats(gpu, raw_data, file_sz, stats);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (rc == 0) {
            printf("%-25s %11.1f %8u matches\n", "lz_match_stats", ms, stats[2]);
        }
    }

    // GPU Huffman size
    {
        uint64_t bits = 0;
        auto start = std::chrono::high_resolution_clock::now();
        int rc = gpu_context_huffman_size(gpu, raw_data, file_sz, &bits);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (rc == 0) {
            printf("%-25s %11.1f %13lu b\n", "huffman_size", ms, (unsigned long)bits);
        }
    }

    // GPU Huffman compress (64KB chunk)
    {
        uint64_t chunk = 65536;
        if ((uint64_t)file_sz >= chunk) {
            CompressedBuffer comp = {};
            auto start = std::chrono::high_resolution_clock::now();
            int rc = gpu_context_huffman_compress(gpu, raw_data, chunk, &comp);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (rc == 0) {
                printf("%-25s %11.1f %11lu B (%.1f%%)\n", "huffman_compress_64K", ms,
                       (unsigned long)comp.size, (double)comp.size / chunk * 100.0);
                compressed_buffer_free(&comp);
            }
        }
    }

    // GPU Huffman compress+decompress round-trip (64KB)
    {
        uint64_t chunk = 65536;
        if ((uint64_t)file_sz >= chunk) {
            auto start = std::chrono::high_resolution_clock::now();
            CompressedBuffer comp = {};
            int rc = gpu_context_huffman_compress(gpu, raw_data, chunk, &comp);
            if (rc == 0) {
                uint8_t* decomp = nullptr;
                uint64_t decomp_size = 0;
                rc = gpu_context_huffman_decompress(gpu, comp.data, comp.size, chunk, &decomp, &decomp_size);
                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                if (rc == 0 && decomp_size == chunk && memcmp(raw_data, decomp, chunk) == 0) {
                    printf("%-25s %11.1f %15s\n", "huffman_roundtrip_64K", ms, "OK");
                } else {
                    printf("%-25s %11.1f %15s\n", "huffman_roundtrip_64K", ms, "FAIL");
                }
                free(decomp);
                compressed_buffer_free(&comp);
            }
        }
    }

    gpu_context_destroy(gpu);
    free(raw_data);
    return 0;
}
