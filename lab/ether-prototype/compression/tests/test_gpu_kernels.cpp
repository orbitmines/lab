#include "compression/estimator.h"
#include "compression/gpu_context.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void test_gpu_init() {
    printf("  test_gpu_init...");
    fflush(stdout);
    GpuContext* ctx = gpu_context_create();
    if (!ctx) {
        printf(" SKIP (no GPU available)\n");
        return;
    }
    BackendInfo info;
    gpu_context_get_info(ctx, &info);
    printf(" OK (device: %s, vendor: %s)\n", info.device_name, info.vendor);
    gpu_context_destroy(ctx);
}

static void test_gpu_histogram_small() {
    printf("  test_gpu_histogram_small...");
    fflush(stdout);

    Estimator* est = estimator_create(1); // request GPU
    uint8_t data[] = {0, 0, 0, 1, 1, 2};
    estimator_load(est, data, sizeof(data));

    ByteHistogram hist;
    int rc = estimator_histogram(est, &hist);
    if (rc != 0) {
        printf(" SKIP (GPU histogram not available)\n");
        estimator_destroy(est);
        return;
    }

    assert(hist.total == 6);
    assert(hist.histogram[0] == 3);
    assert(hist.histogram[1] == 2);
    assert(hist.histogram[2] == 1);
    assert(hist.histogram[3] == 0);

    estimator_destroy(est);
    printf(" OK\n");
}

static void test_gpu_vs_cpu_histogram() {
    printf("  test_gpu_vs_cpu_histogram...");
    fflush(stdout);

    // Generate test data with known pattern
    const uint64_t size = 65536;
    uint8_t* data = (uint8_t*)malloc(size);
    for (uint64_t i = 0; i < size; i++) data[i] = (uint8_t)((i * 7 + 3) % 256);

    // CPU histogram
    Estimator* cpu_est = estimator_create(0);
    estimator_load(cpu_est, data, size);
    ByteHistogram cpu_hist;
    estimator_histogram(cpu_est, &cpu_hist);

    // GPU histogram
    Estimator* gpu_est = estimator_create(1);
    estimator_load(gpu_est, data, size);
    ByteHistogram gpu_hist;
    int rc = estimator_histogram(gpu_est, &gpu_hist);
    if (rc != 0) {
        printf(" SKIP (GPU not available)\n");
        estimator_destroy(cpu_est);
        estimator_destroy(gpu_est);
        free(data);
        return;
    }

    // Compare: must be exact match
    int mismatches = 0;
    for (int i = 0; i < 256; i++) {
        if (cpu_hist.histogram[i] != gpu_hist.histogram[i]) {
            if (mismatches < 5)
                printf("\n    bin %d: CPU=%lu GPU=%lu", i,
                       (unsigned long)cpu_hist.histogram[i],
                       (unsigned long)gpu_hist.histogram[i]);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("\n    %d mismatches total\n", mismatches);
        abort();
    }

    estimator_destroy(cpu_est);
    estimator_destroy(gpu_est);
    free(data);
    printf(" OK (exact match, %lu bytes)\n", (unsigned long)size);
}

static void test_gpu_vs_cpu_enwik8(const char* path) {
    printf("  test_gpu_vs_cpu_enwik8 (%s)...", path);
    fflush(stdout);

    Estimator* cpu_est = estimator_create(0);
    if (estimator_load_file(cpu_est, path) != 0) {
        printf(" SKIP (file not found)\n");
        estimator_destroy(cpu_est);
        return;
    }
    ByteHistogram cpu_hist;
    estimator_histogram(cpu_est, &cpu_hist);

    Estimator* gpu_est = estimator_create(1);
    estimator_load_file(gpu_est, path);
    ByteHistogram gpu_hist;
    int rc = estimator_histogram(gpu_est, &gpu_hist);
    if (rc != 0) {
        printf(" SKIP (GPU not available)\n");
        estimator_destroy(cpu_est);
        estimator_destroy(gpu_est);
        return;
    }

    int mismatches = 0;
    for (int i = 0; i < 256; i++) {
        if (cpu_hist.histogram[i] != gpu_hist.histogram[i]) {
            if (mismatches < 5)
                printf("\n    bin %d: CPU=%lu GPU=%lu", i,
                       (unsigned long)cpu_hist.histogram[i],
                       (unsigned long)gpu_hist.histogram[i]);
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("\n    %d mismatches total\n", mismatches);
        abort();
    }

    estimator_destroy(cpu_est);
    estimator_destroy(gpu_est);
    printf(" OK (exact match, %lu bytes)\n", (unsigned long)cpu_hist.total);
}

int main(int argc, char** argv) {
    printf("GPU kernel tests:\n");

    test_gpu_init();
    test_gpu_histogram_small();
    test_gpu_vs_cpu_histogram();

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
        test_gpu_vs_cpu_enwik8(enwik8_path);
    } else {
        printf("  test_gpu_vs_cpu_enwik8... SKIP (enwik8 not found)\n");
    }

    printf("\nAll GPU tests passed.\n");
    return 0;
}
