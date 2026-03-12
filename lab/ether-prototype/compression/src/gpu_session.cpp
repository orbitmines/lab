#include "compression/gpu_session.h"
#include "gpu_context_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef HAS_WEBGPU

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#include <emscripten/emscripten.h>
#else
#include <webgpu/wgpu.h>
#endif

// ── Platform helpers ────────────────────────────────────────────────────────

struct MapUD { bool done; WGPUMapAsyncStatus status; };

#ifdef __EMSCRIPTEN__
static void on_map(WGPUBufferMapAsyncStatus status, void* ud) {
    auto* d = (MapUD*)ud;
    d->status = (WGPUMapAsyncStatus)status;
    d->done = true;
}
#else
static void on_map(WGPUMapAsyncStatus status, WGPUStringView msg, void* ud1, void* ud2) {
    (void)msg; (void)ud2;
    auto* d = (MapUD*)ud1;
    d->status = status;
    d->done = true;
}
#endif

static void poll_events([[maybe_unused]] WGPUInstance instance) {
#ifdef __EMSCRIPTEN__
    emscripten_sleep(0);
#else
    wgpuInstanceProcessEvents(instance);
#endif
}

// ── Session struct ──────────────────────────────────────────────────────────

struct GpuSession {
    GpuContext* ctx;
    WebGpuBackend* backend;

    WGPUBuffer original_buf;
    uint64_t data_size;
    uint32_t num_slots;

    WGPUBuffer* scratch_bufs;
    WGPUBuffer* hist_bufs;
    WGPUBuffer* transform_params;

    WGPUBuffer hist_params;

    WGPUBindGroup* transform_bgs;
    WGPUBindGroup* histogram_bgs;

    WGPUBuffer readback_buf;
};

// ── Buffer helpers ──────────────────────────────────────────────────────────

static WGPUBuffer create_uniform_buf(WebGpuBackend* b, const void* data, uint64_t size) {
    uint64_t aligned = (size + 15) & ~15ULL;
    WGPUBufferDescriptor desc = {};
    desc.size = aligned;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (!buf) return nullptr;
    void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned);
    memset(ptr, 0, aligned);
    if (data) memcpy(ptr, data, size);
    wgpuBufferUnmap(buf);
    return buf;
}

static WGPUBuffer create_storage_buf(WebGpuBackend* b, uint64_t size, uint32_t extra_usage) {
    uint64_t aligned = (size + 3) & ~3ULL;
    WGPUBufferDescriptor desc = {};
    desc.size = aligned;
    desc.usage = WGPUBufferUsage_Storage | extra_usage;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (buf) {
        void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned);
        memset(ptr, 0, aligned);
        wgpuBufferUnmap(buf);
    }
    return buf;
}

static WGPUBindGroup make_bind_group(WGPUDevice device, WGPUBindGroupLayout bgl,
                                      WGPUBindGroupEntry* entries, uint32_t count) {
    WGPUBindGroupDescriptor desc = {};
    desc.layout = bgl;
    desc.entryCount = count;
    desc.entries = entries;
    return wgpuDeviceCreateBindGroup(device, &desc);
}

// ── Session lifecycle ───────────────────────────────────────────────────────

GpuSession* gpu_session_create(GpuContext* ctx, const uint8_t* data, uint64_t size) {
    if (!ctx || !data || size == 0 || !ctx->has_gpu) return nullptr;

    WebGpuBackend* b = &ctx->backend;

    for (int i = 0; i < 5; i++) {
        if (!b->transform_pipelines[i]) {
            fprintf(stderr, "[gpu_session] Transform pipeline %d not available\n", i);
            return nullptr;
        }
    }

    BackendInfo info;
    webgpu_backend_get_info(b, &info);
    uint64_t max_buf = info.max_buffer_size;

    uint64_t per_slot = size + 256 * sizeof(uint32_t) + 16;
    if (max_buf <= size) {
        fprintf(stderr, "[gpu_session] Data too large for GPU buffer (%lu > %lu)\n",
                (unsigned long)size, (unsigned long)max_buf);
        return nullptr;
    }
    uint32_t num_slots = (uint32_t)((max_buf - size) / per_slot);
    if (num_slots == 0) num_slots = 1;
    if (num_slots > 64) num_slots = 64;

    GpuSession* s = new GpuSession();
    memset(s, 0, sizeof(*s));
    s->ctx = ctx;
    s->backend = b;
    s->data_size = size;
    s->num_slots = num_slots;

    // Upload original data (persistent, read-only on GPU)
    uint64_t aligned_size = (size + 3) & ~3ULL;
    {
        WGPUBufferDescriptor odesc = {};
        odesc.size = aligned_size;
        odesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        odesc.mappedAtCreation = true;
        s->original_buf = wgpuDeviceCreateBuffer(b->device, &odesc);
        if (!s->original_buf) { delete s; return nullptr; }
        void* optr = wgpuBufferGetMappedRange(s->original_buf, 0, aligned_size);
        memset(optr, 0, aligned_size);
        memcpy(optr, data, size);
        wgpuBufferUnmap(s->original_buf);
    }

    s->scratch_bufs = new WGPUBuffer[num_slots]();
    s->hist_bufs = new WGPUBuffer[num_slots]();
    s->transform_params = new WGPUBuffer[num_slots]();
    s->transform_bgs = new WGPUBindGroup[num_slots]();
    s->histogram_bgs = new WGPUBindGroup[num_slots]();

    uint64_t hist_size = 256 * sizeof(uint32_t);

    uint32_t hp[4] = {(uint32_t)size, 0, 0, 0};
    s->hist_params = create_uniform_buf(b, hp, sizeof(hp));
    if (!s->hist_params) { gpu_session_destroy(s); return nullptr; }

    for (uint32_t i = 0; i < num_slots; i++) {
        s->scratch_bufs[i] = create_storage_buf(b, aligned_size,
            WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
        s->hist_bufs[i] = create_storage_buf(b, hist_size,
            WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
        s->transform_params[i] = create_uniform_buf(b, nullptr, 16);

        if (!s->scratch_bufs[i] || !s->hist_bufs[i] || !s->transform_params[i]) {
            gpu_session_destroy(s);
            return nullptr;
        }

        // Transform bind group: original(0), scratch[i](1), transform_params[i](2)
        {
            WGPUBindGroupEntry e[3] = {};
            e[0].binding = 0; e[0].buffer = s->original_buf;        e[0].size = aligned_size;
            e[1].binding = 1; e[1].buffer = s->scratch_bufs[i];     e[1].size = aligned_size;
            e[2].binding = 2; e[2].buffer = s->transform_params[i]; e[2].size = 16;
            s->transform_bgs[i] = make_bind_group(b->device, b->transform_bgl, e, 3);
        }

        // Histogram bind group: scratch[i](0), hist_buf[i](1), hist_params(2)
        {
            WGPUBindGroupEntry e[3] = {};
            e[0].binding = 0; e[0].buffer = s->scratch_bufs[i]; e[0].size = aligned_size;
            e[1].binding = 1; e[1].buffer = s->hist_bufs[i];    e[1].size = hist_size;
            e[2].binding = 2; e[2].buffer = s->hist_params;     e[2].size = 16;
            s->histogram_bgs[i] = make_bind_group(b->device, b->histogram_bgl, e, 3);
        }

        if (!s->transform_bgs[i] || !s->histogram_bgs[i]) {
            gpu_session_destroy(s);
            return nullptr;
        }
    }

    // Readback buffer for N histograms
    {
        uint64_t rb_size = (uint64_t)num_slots * hist_size;
        WGPUBufferDescriptor rbdesc = {};
        rbdesc.size = rb_size;
        rbdesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        s->readback_buf = wgpuDeviceCreateBuffer(b->device, &rbdesc);
        if (!s->readback_buf) { gpu_session_destroy(s); return nullptr; }
    }

    fprintf(stderr, "[gpu_session] Created: %lu bytes, %u slots\n",
            (unsigned long)size, num_slots);
    return s;
}

void gpu_session_destroy(GpuSession* session) {
    if (!session) return;
    if (session->readback_buf) wgpuBufferRelease(session->readback_buf);
    if (session->hist_params) wgpuBufferRelease(session->hist_params);
    if (session->original_buf) wgpuBufferRelease(session->original_buf);
    for (uint32_t i = 0; i < session->num_slots; i++) {
        if (session->transform_bgs && session->transform_bgs[i])
            wgpuBindGroupRelease(session->transform_bgs[i]);
        if (session->histogram_bgs && session->histogram_bgs[i])
            wgpuBindGroupRelease(session->histogram_bgs[i]);
        if (session->scratch_bufs && session->scratch_bufs[i])
            wgpuBufferRelease(session->scratch_bufs[i]);
        if (session->hist_bufs && session->hist_bufs[i])
            wgpuBufferRelease(session->hist_bufs[i]);
        if (session->transform_params && session->transform_params[i])
            wgpuBufferRelease(session->transform_params[i]);
    }
    delete[] session->scratch_bufs;
    delete[] session->hist_bufs;
    delete[] session->transform_params;
    delete[] session->transform_bgs;
    delete[] session->histogram_bgs;
    delete session;
}

uint32_t gpu_session_num_slots(const GpuSession* session) {
    return session ? session->num_slots : 0;
}

uint64_t gpu_session_data_size(const GpuSession* session) {
    return session ? session->data_size : 0;
}

// ── Batch evaluation ────────────────────────────────────────────────────────

int gpu_session_evaluate_batch(GpuSession* session,
                                const TransformDesc* transforms, uint32_t count,
                                SlotScore* out_scores) {
    if (!session || !transforms || !out_scores || count == 0) return -1;

    WebGpuBackend* b = session->backend;
    uint64_t data_size = session->data_size;
    uint64_t aligned_size = (data_size + 3) & ~3ULL;
    uint32_t hist_size = 256 * sizeof(uint32_t);
    // Word-level dispatch: each thread processes 4 bytes (one u32), grid-stride for overflow
    uint64_t word_count = (data_size + 3) / 4;
    uint32_t transform_wg = (uint32_t)std::min((word_count + 255) / 256, (uint64_t)65535);

    for (uint32_t batch_start = 0; batch_start < count; batch_start += session->num_slots) {
        uint32_t N = count - batch_start;
        if (N > session->num_slots) N = session->num_slots;

        // Write transform params
        for (uint32_t i = 0; i < N; i++) {
            const TransformDesc* t = &transforms[batch_start + i];
            uint32_t params[4] = {(uint32_t)data_size, t->param0, t->param1, t->param2};
            wgpuQueueWriteBuffer(b->queue, session->transform_params[i], 0, params, sizeof(params));
        }

        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);

        // Pass 1: Transform
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            for (uint32_t i = 0; i < N; i++) {
                const TransformDesc* t = &transforms[batch_start + i];
                if (t->type >= TRANSFORM_COUNT) continue;
                wgpuComputePassEncoderSetPipeline(pass, b->transform_pipelines[t->type]);
                wgpuComputePassEncoderSetBindGroup(pass, 0, session->transform_bgs[i], 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, transform_wg, 1, 1);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // Clear histogram buffers
        for (uint32_t i = 0; i < N; i++) {
            wgpuCommandEncoderClearBuffer(enc, session->hist_bufs[i], 0, hist_size);
        }

        // Pass 2: Histogram
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            for (uint32_t i = 0; i < N; i++) {
                wgpuComputePassEncoderSetPipeline(pass, b->histogram_pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, session->histogram_bgs[i], 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, 256, 1, 1);
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // Copy histograms to readback
        for (uint32_t i = 0; i < N; i++) {
            wgpuCommandEncoderCopyBufferToBuffer(enc, session->hist_bufs[i], 0,
                session->readback_buf, (uint64_t)i * hist_size, hist_size);
        }

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(b->queue, 1, &cmd);

        // Map readback
        uint64_t rb_total = (uint64_t)N * hist_size;
        MapUD mud = {};
#ifdef __EMSCRIPTEN__
        wgpuBufferMapAsync(session->readback_buf, WGPUMapMode_Read, 0, rb_total, on_map, &mud);
#else
        WGPUBufferMapCallbackInfo mcb = {};
        mcb.mode = WGPUCallbackMode_AllowSpontaneous;
        mcb.callback = on_map;
        mcb.userdata1 = &mud;
        wgpuBufferMapAsync(session->readback_buf, WGPUMapMode_Read, 0, rb_total, mcb);
#endif
        while (!mud.done) poll_events(b->instance);

        if (mud.status != WGPUMapAsyncStatus_Success) {
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
            fprintf(stderr, "[gpu_session] Readback map failed\n");
            return -1;
        }

        const uint32_t* mapped = (const uint32_t*)wgpuBufferGetConstMappedRange(
            session->readback_buf, 0, rb_total);

        // CPU: compute entropy from histograms
        for (uint32_t i = 0; i < N; i++) {
            const uint32_t* hist = mapped + i * 256;
            const TransformDesc* t = &transforms[batch_start + i];
            SlotScore* score = &out_scores[batch_start + i];

            score->transform = t->type;
            score->params[0] = t->param0;
            score->params[1] = t->param1;
            score->params[2] = t->param2;

            double total_bits = 0.0;
            double total = (double)data_size;
            for (int j = 0; j < 256; j++) {
                if (hist[j] == 0) continue;
                double p = (double)hist[j] / total;
                total_bits -= (double)hist[j] * log2(p);
            }

            score->entropy_o0_bpb = (float)(total_bits / (double)data_size);
            score->entropy_o0_total = total_bits;
        }

        wgpuBufferUnmap(session->readback_buf);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    return 0;
}

#else // !HAS_WEBGPU

struct GpuSession {};

GpuSession* gpu_session_create(GpuContext*, const uint8_t*, uint64_t) { return nullptr; }
void gpu_session_destroy(GpuSession*) {}
uint32_t gpu_session_num_slots(const GpuSession*) { return 0; }
uint64_t gpu_session_data_size(const GpuSession*) { return 0; }
int gpu_session_evaluate_batch(GpuSession*, const TransformDesc*, uint32_t, SlotScore*) { return -1; }

#endif // HAS_WEBGPU
