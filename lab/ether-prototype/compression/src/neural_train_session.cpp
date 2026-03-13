#include "compression/neural_train_session.h"

#ifdef HAS_WEBGPU

#include "neural_compressor_internal.h"
#include "gpu_context_internal.h"
#include "gpu/webgpu_backend.h"
#include <webgpu/wgpu.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

struct NeuralTrainSession {
    WebGpuBackend* backend;

    int ctx, emb, hid, inp, n_params;
    int batch_size;
    float lr;

    // Per-layer weight buffers
    WGPUBuffer buf_embed, buf_w1, buf_b1, buf_w2, buf_b2;

    // Flat weight buffer (for Adam)
    WGPUBuffer buf_weights, buf_grads, buf_adam_m, buf_adam_v;

    // Per-layer gradient accumulators
    WGPUBuffer buf_d_embed; // atomic<u32> for CAS float add
    WGPUBuffer buf_d_w1, buf_d_w2, buf_d_b1, buf_d_b2;

    // Activation buffers
    WGPUBuffer buf_input, buf_hidden, buf_logits, buf_d_hidden, buf_d_input;

    // Data + targets + loss
    WGPUBuffer buf_data, buf_targets, buf_loss;
};

static WGPUBuffer create_storage_buf(WebGpuBackend* b, uint64_t size) {
    return webgpu_create_buffer(b, size,
        (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc));
}

static WGPUBuffer upload_floats(WebGpuBackend* b, const float* data, int count) {
    uint64_t size = (uint64_t)count * sizeof(float);
    WGPUBuffer buf = create_storage_buf(b, size);
    if (buf) wgpuQueueWriteBuffer(b->queue, buf, 0, data, size);
    return buf;
}

static WGPUBuffer upload_uniform_data(WebGpuBackend* b, const void* data, uint64_t size) {
    uint64_t aligned = (size + 15) & ~15ULL;
    WGPUBufferDescriptor desc = {};
    desc.size = aligned;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (!buf) return nullptr;
    void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned);
    memset(ptr, 0, aligned);
    memcpy(ptr, data, size);
    wgpuBufferUnmap(buf);
    return buf;
}

static WGPUBindGroup make_bind_group(WebGpuBackend* b, WGPUBindGroupLayout bgl,
                                      WGPUBindGroupEntry* entries, uint32_t count) {
    WGPUBindGroupDescriptor desc = {};
    desc.layout = bgl;
    desc.entryCount = count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroup(b->device, &desc);
}

static void add_compute_pass(WGPUCommandEncoder enc, WGPUComputePipeline pipeline,
                               WGPUBindGroup bg, uint32_t wg_x, uint32_t wg_y, uint32_t wg_z) {
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, wg_x, wg_y, wg_z);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

static NeuralTrainSession* session_create(WebGpuBackend* backend, NeuralCompressor* nc,
                                            const uint8_t* data, uint64_t data_size) {
    auto* s = new NeuralTrainSession();
    s->backend = backend;
    s->ctx = nc->ctx; s->emb = nc->emb; s->hid = nc->hid;
    s->inp = nc->inp; s->n_params = nc->n_params; s->lr = nc->lr;

    s->batch_size = 131072;
    if (s->batch_size > (int)data_size) s->batch_size = (int)data_size;
    int B = s->batch_size;

    // Upload data
    uint64_t data_aligned = (data_size + 3) & ~3ULL;
    s->buf_data = create_storage_buf(backend, data_aligned);
    wgpuQueueWriteBuffer(backend->queue, s->buf_data, 0, data, data_size);

    // Per-layer weight buffers
    s->buf_embed = upload_floats(backend, nc->weights + nc->off_embed, 256 * nc->emb);
    s->buf_w1    = upload_floats(backend, nc->weights + nc->off_w1, nc->inp * nc->hid);
    s->buf_b1    = upload_floats(backend, nc->weights + nc->off_b1, nc->hid);
    s->buf_w2    = upload_floats(backend, nc->weights + nc->off_w2, nc->hid * 256);
    s->buf_b2    = upload_floats(backend, nc->weights + nc->off_b2, 256);

    // Flat weight buffer (for Adam)
    s->buf_weights = upload_floats(backend, nc->weights, nc->n_params);
    s->buf_grads   = create_storage_buf(backend, (uint64_t)nc->n_params * sizeof(float));
    s->buf_adam_m  = upload_floats(backend, nc->adam_m, nc->n_params);
    s->buf_adam_v  = upload_floats(backend, nc->adam_v, nc->n_params);

    // Per-layer gradient accumulators
    s->buf_d_embed = create_storage_buf(backend, (uint64_t)256 * nc->emb * sizeof(float));
    s->buf_d_w1    = create_storage_buf(backend, (uint64_t)nc->inp * nc->hid * sizeof(float));
    s->buf_d_w2    = create_storage_buf(backend, (uint64_t)nc->hid * 256 * sizeof(float));
    s->buf_d_b1    = create_storage_buf(backend, (uint64_t)nc->hid * sizeof(float));
    s->buf_d_b2    = create_storage_buf(backend, (uint64_t)256 * sizeof(float));

    // Activation buffers
    s->buf_input    = create_storage_buf(backend, (uint64_t)B * nc->inp * sizeof(float));
    s->buf_hidden   = create_storage_buf(backend, (uint64_t)B * nc->hid * sizeof(float));
    s->buf_logits   = create_storage_buf(backend, (uint64_t)B * 256 * sizeof(float));
    s->buf_d_hidden = create_storage_buf(backend, (uint64_t)B * nc->hid * sizeof(float));
    s->buf_d_input  = create_storage_buf(backend, (uint64_t)B * nc->inp * sizeof(float));

    // Targets (packed as u32)
    s->buf_targets = create_storage_buf(backend, ((uint64_t)B + 3) & ~3ULL);

    // Loss accumulator
    s->buf_loss = create_storage_buf(backend, sizeof(uint32_t));

    return s;
}

static void session_destroy(NeuralTrainSession* s) {
    if (!s) return;
    WGPUBuffer* bufs[] = {
        &s->buf_embed, &s->buf_w1, &s->buf_b1, &s->buf_w2, &s->buf_b2,
        &s->buf_weights, &s->buf_grads, &s->buf_adam_m, &s->buf_adam_v,
        &s->buf_d_embed, &s->buf_d_w1, &s->buf_d_w2, &s->buf_d_b1, &s->buf_d_b2,
        &s->buf_input, &s->buf_hidden, &s->buf_logits,
        &s->buf_d_hidden, &s->buf_d_input,
        &s->buf_data, &s->buf_targets, &s->buf_loss,
    };
    for (auto* bp : bufs) {
        if (*bp) { wgpuBufferRelease(*bp); *bp = nullptr; }
    }
    delete s;
}

static void sync_weights_to_flat(NeuralTrainSession* s, NeuralCompressor* nc,
                                   WGPUCommandEncoder enc) {
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_embed, 0, s->buf_weights,
        (uint64_t)nc->off_embed * sizeof(float), (uint64_t)256 * nc->emb * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_w1, 0, s->buf_weights,
        (uint64_t)nc->off_w1 * sizeof(float), (uint64_t)nc->inp * nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_b1, 0, s->buf_weights,
        (uint64_t)nc->off_b1 * sizeof(float), (uint64_t)nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_w2, 0, s->buf_weights,
        (uint64_t)nc->off_w2 * sizeof(float), (uint64_t)nc->hid * 256 * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_b2, 0, s->buf_weights,
        (uint64_t)nc->off_b2 * sizeof(float), (uint64_t)256 * sizeof(float));
}

static void sync_flat_to_weights(NeuralTrainSession* s, NeuralCompressor* nc,
                                   WGPUCommandEncoder enc) {
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_weights,
        (uint64_t)nc->off_embed * sizeof(float), s->buf_embed, 0, (uint64_t)256 * nc->emb * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_weights,
        (uint64_t)nc->off_w1 * sizeof(float), s->buf_w1, 0, (uint64_t)nc->inp * nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_weights,
        (uint64_t)nc->off_b1 * sizeof(float), s->buf_b1, 0, (uint64_t)nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_weights,
        (uint64_t)nc->off_w2 * sizeof(float), s->buf_w2, 0, (uint64_t)nc->hid * 256 * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_weights,
        (uint64_t)nc->off_b2 * sizeof(float), s->buf_b2, 0, (uint64_t)256 * sizeof(float));
}

static void sync_grads_to_flat(NeuralTrainSession* s, NeuralCompressor* nc,
                                 WGPUCommandEncoder enc) {
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_d_embed, 0, s->buf_grads,
        (uint64_t)nc->off_embed * sizeof(float), (uint64_t)256 * nc->emb * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_d_w1, 0, s->buf_grads,
        (uint64_t)nc->off_w1 * sizeof(float), (uint64_t)nc->inp * nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_d_b1, 0, s->buf_grads,
        (uint64_t)nc->off_b1 * sizeof(float), (uint64_t)nc->hid * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_d_w2, 0, s->buf_grads,
        (uint64_t)nc->off_w2 * sizeof(float), (uint64_t)nc->hid * 256 * sizeof(float));
    wgpuCommandEncoderCopyBufferToBuffer(enc, s->buf_d_b2, 0, s->buf_grads,
        (uint64_t)nc->off_b2 * sizeof(float), (uint64_t)256 * sizeof(float));
}

// Helper to create a uniform buffer, dispatch, and release
struct TempBuf {
    WGPUBuffer buf;
    ~TempBuf() { if (buf) wgpuBufferRelease(buf); }
};

float neural_compressor_train_gpu(NeuralCompressor* nc,
                                   const uint8_t* data, uint64_t size,
                                   int seconds, GpuContext* gpu_ctx) {
    if (!nc || !data || size == 0 || !gpu_ctx) return 8.0f;

    if (!gpu_ctx->has_gpu) {
        fprintf(stderr, "[neural_gpu] No GPU available, falling back to CPU\n");
        return neural_compressor_train(nc, data, size, seconds);
    }

    WebGpuBackend* b = &gpu_ctx->backend;
    NeuralTrainSession* s = session_create(b, nc, data, size);
    if (!s) {
        fprintf(stderr, "[neural_gpu] Failed to create training session\n");
        return 8.0f;
    }

    int N = (int)size;
    int B = s->batch_size;
    int n_batches = (N + B - 1) / B;

    printf("  [GPU] Training: %d positions, batch=%d, %d batches/epoch\n", N, B, n_batches);
    printf("  epoch   bpb     total_bits    elapsed\n");
    printf("  ─────   ───     ──────────    ───────\n");

    auto start = std::chrono::steady_clock::now();
    int epoch = 0;
    double last_bpb = 8.0;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= seconds) break;

        double epoch_loss = 0.0;  // accumulate in double on CPU

        for (int batch_idx = 0; batch_idx < n_batches; batch_idx++) {
            int batch_start = batch_idx * B;
            int batch_end = batch_start + B;
            if (batch_end > N) batch_end = N;
            int actual_batch = batch_end - batch_start;

            // Clear gradient accumulators + loss before each batch
            {
                uint32_t zero = 0;
                wgpuQueueWriteBuffer(b->queue, s->buf_loss, 0, &zero, sizeof(zero));
                WGPUCommandEncoder clr = wgpuDeviceCreateCommandEncoder(b->device, nullptr);
                wgpuCommandEncoderClearBuffer(clr, s->buf_d_embed, 0, (uint64_t)256 * s->emb * sizeof(float));
                wgpuCommandEncoderClearBuffer(clr, s->buf_d_w1, 0, (uint64_t)s->inp * s->hid * sizeof(float));
                wgpuCommandEncoderClearBuffer(clr, s->buf_d_b1, 0, (uint64_t)s->hid * sizeof(float));
                wgpuCommandEncoderClearBuffer(clr, s->buf_d_w2, 0, (uint64_t)s->hid * 256 * sizeof(float));
                wgpuCommandEncoderClearBuffer(clr, s->buf_d_b2, 0, (uint64_t)256 * sizeof(float));
                WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(clr, nullptr);
                wgpuQueueSubmit(b->queue, 1, &cmd);
                wgpuCommandBufferRelease(cmd);
                wgpuCommandEncoderRelease(clr);
            }

            // Upload target bytes
            std::vector<uint8_t> targets(((actual_batch + 3) / 4) * 4, 0);
            for (int i = 0; i < actual_batch; i++) targets[i] = data[batch_start + i];
            wgpuQueueWriteBuffer(b->queue, s->buf_targets, 0, targets.data(), targets.size());

            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);

            // Clear activation buffers
            wgpuCommandEncoderClearBuffer(enc, s->buf_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float));
            wgpuCommandEncoderClearBuffer(enc, s->buf_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float));
            wgpuCommandEncoderClearBuffer(enc, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float));

            // ── Forward ──

            // 1. Embedding gather
            {
                uint32_t params[8] = {
                    (uint32_t)actual_batch, (uint32_t)s->ctx, (uint32_t)s->emb,
                    (uint32_t)batch_start, (uint32_t)size, 0, 0, 0
                };
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_data, 0, (size + 3) & ~3ULL, nullptr};
                e[1] = {nullptr, 1, s->buf_embed, 0, (uint64_t)256 * s->emb * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_embed_gather_bgl, e, 4);
                add_compute_pass(enc, b->nn_embed_gather_pipeline, bg,
                    ((uint32_t)actual_batch + 255) / 256, 1, 1);
                wgpuBindGroupRelease(bg);
                wgpuBufferRelease(pbuf);
            }

            // 2. Matmul: input[B,inp] @ W1[inp,hid] → hidden[B,hid]
            {
                float one = 1.0f, zero_f = 0.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)actual_batch; params[1] = (uint32_t)s->inp; params[2] = (uint32_t)s->hid;
                params[3] = 0; params[4] = 0;
                memcpy(&params[5], &one, 4); memcpy(&params[6], &zero_f, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_w1, 0, (uint64_t)s->inp * s->hid * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    ((uint32_t)s->hid + 15) / 16, ((uint32_t)actual_batch + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 3. Bias + ReLU on hidden
            {
                uint32_t params[4] = {(uint32_t)actual_batch, (uint32_t)s->hid, 0, 0};
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[3] = {};
                e[0] = {nullptr, 0, s->buf_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_b1, 0, (uint64_t)s->hid * sizeof(float), nullptr};
                e[2] = {nullptr, 2, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_bias_relu_bgl, e, 3);
                uint32_t total = (uint32_t)(actual_batch * s->hid);
                add_compute_pass(enc, b->nn_bias_relu_pipeline, bg, (total + 255) / 256, 1, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 4. Matmul: hidden[B,hid] @ W2[hid,256] → logits[B,256]
            {
                float one = 1.0f, zero_f = 0.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)actual_batch; params[1] = (uint32_t)s->hid; params[2] = 256u;
                memcpy(&params[5], &one, 4); memcpy(&params[6], &zero_f, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_w2, 0, (uint64_t)s->hid * 256 * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    (256u + 15) / 16, ((uint32_t)actual_batch + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 5. Softmax + CE + d_logits (adds b2 bias internally)
            {
                uint32_t wg_total = (uint32_t)actual_batch;
                uint32_t wg_x = wg_total <= 65535 ? wg_total : 512;
                uint32_t wg_y = (wg_total + wg_x - 1) / wg_x;

                uint32_t params[4] = {(uint32_t)actual_batch, 256u, wg_x, 0};
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[5] = {};
                e[0] = {nullptr, 0, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_targets, 0, ((uint64_t)actual_batch + 3) & ~3ULL, nullptr};
                e[2] = {nullptr, 2, s->buf_loss, 0, sizeof(uint32_t), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                e[4] = {nullptr, 4, s->buf_b2, 0, 256 * sizeof(float), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_softmax_ce_bgl, e, 5);
                add_compute_pass(enc, b->nn_softmax_ce_pipeline, bg, wg_x, wg_y, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // ── Backward ──

            // 6. d_W2 = hidden^T @ d_logits (accumulate β=1)
            {
                float one = 1.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)s->hid; params[1] = (uint32_t)actual_batch; params[2] = 256u;
                params[3] = 1; params[4] = 0; // transpose_a
                memcpy(&params[5], &one, 4); memcpy(&params[6], &one, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_d_w2, 0, (uint64_t)s->hid * 256 * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    (256u + 15) / 16, ((uint32_t)s->hid + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 7. d_b2 = column sum of d_logits
            {
                uint32_t params[4] = {(uint32_t)actual_batch, 256u, 0, 0};
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[3] = {};
                e[0] = {nullptr, 0, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_d_b2, 0, 256 * sizeof(float), nullptr};
                e[2] = {nullptr, 2, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_bias_grad_bgl, e, 3);
                add_compute_pass(enc, b->nn_bias_grad_pipeline, bg, 1, 1, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 8. d_hidden = d_logits @ W2^T
            {
                float one = 1.0f, zero_f = 0.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)actual_batch; params[1] = 256u; params[2] = (uint32_t)s->hid;
                params[3] = 0; params[4] = 1; // transpose_b
                memcpy(&params[5], &one, 4); memcpy(&params[6], &zero_f, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_logits, 0, (uint64_t)actual_batch * 256 * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_w2, 0, (uint64_t)s->hid * 256 * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_d_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    ((uint32_t)s->hid + 15) / 16, ((uint32_t)actual_batch + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 9. ReLU mask on d_hidden
            {
                uint32_t count = (uint32_t)(actual_batch * s->hid);
                uint32_t params[4] = {count, 0, 0, 0};
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[3] = {};
                e[0] = {nullptr, 0, s->buf_d_hidden, 0, count * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_hidden, 0, count * sizeof(float), nullptr};
                e[2] = {nullptr, 2, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_relu_mask_bgl, e, 3);
                add_compute_pass(enc, b->nn_relu_mask_pipeline, bg, (count + 255) / 256, 1, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 10. d_W1 = input^T @ d_hidden (accumulate)
            {
                float one = 1.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)s->inp; params[1] = (uint32_t)actual_batch; params[2] = (uint32_t)s->hid;
                params[3] = 1; params[4] = 0;
                memcpy(&params[5], &one, 4); memcpy(&params[6], &one, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_d_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_d_w1, 0, (uint64_t)s->inp * s->hid * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    ((uint32_t)s->hid + 15) / 16, ((uint32_t)s->inp + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 11. d_b1 = column sum of d_hidden
            {
                uint32_t params[4] = {(uint32_t)actual_batch, (uint32_t)s->hid, 0, 0};
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[3] = {};
                e[0] = {nullptr, 0, s->buf_d_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_d_b1, 0, (uint64_t)s->hid * sizeof(float), nullptr};
                e[2] = {nullptr, 2, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_bias_grad_bgl, e, 3);
                add_compute_pass(enc, b->nn_bias_grad_pipeline, bg,
                    ((uint32_t)s->hid + 255) / 256, 1, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 12. d_input = d_hidden @ W1^T
            {
                float one = 1.0f, zero_f = 0.0f;
                uint32_t params[8] = {};
                params[0] = (uint32_t)actual_batch; params[1] = (uint32_t)s->hid; params[2] = (uint32_t)s->inp;
                params[3] = 0; params[4] = 1;
                memcpy(&params[5], &one, 4); memcpy(&params[6], &zero_f, 4);
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_d_hidden, 0, (uint64_t)actual_batch * s->hid * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_w1, 0, (uint64_t)s->inp * s->hid * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_d_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_matmul_bgl, e, 4);
                add_compute_pass(enc, b->nn_matmul_pipeline, bg,
                    ((uint32_t)s->inp + 15) / 16, ((uint32_t)actual_batch + 15) / 16, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            // 13. Embed scatter: d_input → d_embed
            {
                uint32_t params[8] = {
                    (uint32_t)actual_batch, (uint32_t)s->ctx, (uint32_t)s->emb,
                    (uint32_t)batch_start, (uint32_t)size, 0, 0, 0
                };
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[4] = {};
                e[0] = {nullptr, 0, s->buf_data, 0, (size + 3) & ~3ULL, nullptr};
                e[1] = {nullptr, 1, s->buf_d_input, 0, (uint64_t)actual_batch * s->inp * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_d_embed, 0, (uint64_t)256 * s->emb * sizeof(float), nullptr};
                e[3] = {nullptr, 3, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_embed_scatter_bgl, e, 4);
                add_compute_pass(enc, b->nn_embed_scatter_pipeline, bg,
                    ((uint32_t)actual_batch + 255) / 256, 1, 1);
                wgpuBindGroupRelease(bg); wgpuBufferRelease(pbuf);
            }

            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(b->queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);

            // Adam update after each batch (mini-batch SGD)
            nc->adam_step++;
            {
                WGPUCommandEncoder adam_enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);
                sync_grads_to_flat(s, nc, adam_enc);
                WGPUCommandBuffer cmd2 = wgpuCommandEncoderFinish(adam_enc, nullptr);
                wgpuQueueSubmit(b->queue, 1, &cmd2);
                wgpuCommandBufferRelease(cmd2);
                wgpuCommandEncoderRelease(adam_enc);
            }

            {
                uint32_t params[4] = {(uint32_t)nc->n_params, (uint32_t)nc->adam_step, 0, 0};
                memcpy(&params[2], &nc->lr, sizeof(float));
                WGPUBuffer pbuf = upload_uniform_data(b, params, sizeof(params));
                WGPUBindGroupEntry e[5] = {};
                e[0] = {nullptr, 0, s->buf_weights, 0, (uint64_t)nc->n_params * sizeof(float), nullptr};
                e[1] = {nullptr, 1, s->buf_grads, 0, (uint64_t)nc->n_params * sizeof(float), nullptr};
                e[2] = {nullptr, 2, s->buf_adam_m, 0, (uint64_t)nc->n_params * sizeof(float), nullptr};
                e[3] = {nullptr, 3, s->buf_adam_v, 0, (uint64_t)nc->n_params * sizeof(float), nullptr};
                e[4] = {nullptr, 4, pbuf, 0, sizeof(params), nullptr};
                WGPUBindGroup bg = make_bind_group(b, b->nn_adam_update_bgl, e, 5);

                WGPUCommandEncoder adam_enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);
                add_compute_pass(adam_enc, b->nn_adam_update_pipeline, bg,
                    ((uint32_t)nc->n_params + 255) / 256, 1, 1);
                sync_flat_to_weights(s, nc, adam_enc);
                WGPUCommandBuffer cmd2 = wgpuCommandEncoderFinish(adam_enc, nullptr);
                wgpuQueueSubmit(b->queue, 1, &cmd2);
                wgpuCommandBufferRelease(cmd2);
                wgpuCommandEncoderRelease(adam_enc);
                wgpuBindGroupRelease(bg);
                wgpuBufferRelease(pbuf);
            }

            // Poll + readback loss periodically
            if ((batch_idx & 15) == 15 || batch_idx == n_batches - 1) {
                wgpuDevicePoll(b->device, true, nullptr);
                uint32_t loss_u = 0;
                webgpu_readback(b, s->buf_loss, &loss_u, sizeof(uint32_t));
                float loss_f;
                memcpy(&loss_f, &loss_u, sizeof(float));
                epoch_loss += (double)loss_f;
            }
        }

        wgpuDevicePoll(b->device, true, nullptr);

        // Use CPU-accumulated double-precision loss
        double bits = epoch_loss / log(2.0);
        last_bpb = bits / N;
        epoch++;

        auto now2 = std::chrono::steady_clock::now();
        double elapsed2 = std::chrono::duration<double>(now2 - start).count();
        printf("  %5d  %6.4f  %12.0f  %6.1fs\n", epoch, last_bpb, bits, elapsed2);
    }

    printf("  [GPU] Training complete: %d epochs\n", epoch);

    // Readback final weights
    webgpu_readback(b, s->buf_weights, nc->weights, (uint64_t)nc->n_params * sizeof(float));
    webgpu_readback(b, s->buf_adam_m, nc->adam_m, (uint64_t)nc->n_params * sizeof(float));
    webgpu_readback(b, s->buf_adam_v, nc->adam_v, (uint64_t)nc->n_params * sizeof(float));

    session_destroy(s);
    return (float)last_bpb;
}

#else // !HAS_WEBGPU

float neural_compressor_train_gpu(NeuralCompressor* nc,
                                   const uint8_t* data, uint64_t size,
                                   int seconds, GpuContext* gpu_ctx) {
    (void)gpu_ctx;
    fprintf(stderr, "[neural_gpu] Built without WebGPU, falling back to CPU\n");
    return neural_compressor_train(nc, data, size, seconds);
}

#endif
