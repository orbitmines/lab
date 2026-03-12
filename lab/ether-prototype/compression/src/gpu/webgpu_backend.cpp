#include "webgpu_backend.h"
#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#include <emscripten/emscripten.h>
#else
#include <webgpu/wgpu.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// Embedded WGSL shader sources
static const char HISTOGRAM_WGSL_SRC[] =
#include "shaders/histogram.wgsl.inc"
;
static const char ENTROPY_WGSL_SRC[] =
#include "shaders/entropy.wgsl.inc"
;
static const char BIGRAM_WGSL_SRC[] =
#include "shaders/bigram.wgsl.inc"
;
static const char LZ_MATCH_WGSL_SRC[] =
#include "shaders/lz_match.wgsl.inc"
;
static const char HUFFMAN_ENC_WGSL_SRC[] =
#include "shaders/huffman_encode.wgsl.inc"
;
static const char HUFFMAN_DEC_WGSL_SRC[] =
#include "shaders/huffman_decode.wgsl.inc"
;
static const char TRANSFORM_DELTA_WGSL_SRC[] =
#include "shaders/transform_delta.wgsl.inc"
;
static const char TRANSFORM_XOR_WGSL_SRC[] =
#include "shaders/transform_xor.wgsl.inc"
;
static const char TRANSFORM_ROTATE_WGSL_SRC[] =
#include "shaders/transform_rotate.wgsl.inc"
;
static const char TRANSFORM_SUB_MEAN_WGSL_SRC[] =
#include "shaders/transform_sub_mean.wgsl.inc"
;
static const char TRANSFORM_BYTE_SWAP_WGSL_SRC[] =
#include "shaders/transform_byte_swap.wgsl.inc"
;

// ── Helpers ─────────────────────────────────────────────────────────────────

#ifndef __EMSCRIPTEN__
static WGPUStringView sv(const char* s) {
    WGPUStringView v; v.data = s; v.length = strlen(s); return v;
}
static WGPUStringView sv(const char* s, size_t len) {
    WGPUStringView v; v.data = s; v.length = len; return v;
}
#endif

static WGPUShaderModule create_shader(WGPUDevice device, const char* src, size_t len, const char* label) {
#ifdef __EMSCRIPTEN__
    // Emscripten uses WGPUShaderModuleWGSLDescriptor with plain C string
    WGPUShaderModuleWGSLDescriptor wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    // Need null-terminated string
    std::string code(src, len);
    wgsl.code = code.c_str();
    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = (WGPUChainedStruct*)&wgsl;
#else
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(src, len);
    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = (WGPUChainedStruct*)&wgsl;
#endif
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(device, &desc);
    if (!mod) fprintf(stderr, "[webgpu] Shader compile failed: %s\n", label);
    return mod;
}

struct PipelineKit {
    WGPUShaderModule shader;
    WGPUBindGroupLayout bgl;
    WGPUComputePipeline pipeline;
};

static PipelineKit create_pipeline(WGPUDevice device, const char* src, size_t src_len,
                                    const char* label,
                                    const WGPUBindGroupLayoutEntry* entries, uint32_t entry_count) {
    PipelineKit kit = {};
    kit.shader = create_shader(device, src, src_len, label);
    if (!kit.shader) return kit;

    WGPUBindGroupLayoutDescriptor bgldesc = {};
    bgldesc.entryCount = entry_count;
    bgldesc.entries = entries;
    kit.bgl = wgpuDeviceCreateBindGroupLayout(device, &bgldesc);

    WGPUPipelineLayoutDescriptor pldesc = {};
    pldesc.bindGroupLayoutCount = 1;
    pldesc.bindGroupLayouts = &kit.bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pldesc);

    WGPUComputePipelineDescriptor cpdesc = {};
    cpdesc.layout = pl;
    cpdesc.compute.module = kit.shader;
#ifdef __EMSCRIPTEN__
    cpdesc.compute.entryPoint = "main";
#else
    cpdesc.compute.entryPoint = sv("main");
#endif
    kit.pipeline = wgpuDeviceCreateComputePipeline(device, &cpdesc);
    wgpuPipelineLayoutRelease(pl);

    if (!kit.pipeline) fprintf(stderr, "[webgpu] Pipeline creation failed: %s\n", label);
    return kit;
}

static void release_pipeline_kit(PipelineKit* k) {
    if (k->pipeline) wgpuComputePipelineRelease(k->pipeline);
    if (k->bgl) wgpuBindGroupLayoutRelease(k->bgl);
    if (k->shader) wgpuShaderModuleRelease(k->shader);
    *k = {};
}

static WGPUBindGroupLayoutEntry bgle_storage_ro(uint32_t binding) {
    WGPUBindGroupLayoutEntry e = {};
    e.binding = binding;
    e.visibility = WGPUShaderStage_Compute;
    e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    return e;
}
static WGPUBindGroupLayoutEntry bgle_storage_rw(uint32_t binding) {
    WGPUBindGroupLayoutEntry e = {};
    e.binding = binding;
    e.visibility = WGPUShaderStage_Compute;
    e.buffer.type = WGPUBufferBindingType_Storage;
    return e;
}
static WGPUBindGroupLayoutEntry bgle_uniform(uint32_t binding) {
    WGPUBindGroupLayoutEntry e = {};
    e.binding = binding;
    e.visibility = WGPUShaderStage_Compute;
    e.buffer.type = WGPUBufferBindingType_Uniform;
    return e;
}

// ── Platform-specific polling ─────────────────────────────────────────────

static void poll_events([[maybe_unused]] WGPUInstance instance) {
#ifdef __EMSCRIPTEN__
    emscripten_sleep(0);
#else
    wgpuInstanceProcessEvents(instance);
#endif
}

// ── Callbacks ──────────────────────────────────────────────────────────────

struct AdapterUD { WGPUAdapter adapter; bool done; };
struct DeviceUD { WGPUDevice device; bool done; };
struct MapUD { bool done; WGPUMapAsyncStatus status; };

#ifdef __EMSCRIPTEN__
// Emscripten uses simpler callback signatures
static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       const char* msg, void* ud) {
    (void)msg;
    auto* d = (AdapterUD*)ud;
    if (status == WGPURequestAdapterStatus_Success) d->adapter = adapter;
    d->done = true;
}
static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      const char* msg, void* ud) {
    (void)msg;
    auto* d = (DeviceUD*)ud;
    if (status == WGPURequestDeviceStatus_Success) d->device = device;
    d->done = true;
}
static void on_map(WGPUBufferMapAsyncStatus status, void* ud) {
    auto* d = (MapUD*)ud;
    d->status = (WGPUMapAsyncStatus)status;
    d->done = true;
}
#else
// wgpu-native / Dawn use WGPUStringView and two userdata pointers
static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                       WGPUStringView msg, void* ud1, void* ud2) {
    (void)msg; (void)ud2;
    auto* d = (AdapterUD*)ud1;
    if (status == WGPURequestAdapterStatus_Success) d->adapter = adapter;
    d->done = true;
}
static void on_device(WGPURequestDeviceStatus status, WGPUDevice device,
                      WGPUStringView msg, void* ud1, void* ud2) {
    (void)msg; (void)ud2;
    auto* d = (DeviceUD*)ud1;
    if (status == WGPURequestDeviceStatus_Success) d->device = device;
    d->done = true;
}
static void on_map(WGPUMapAsyncStatus status, WGPUStringView msg, void* ud1, void* ud2) {
    (void)msg; (void)ud2;
    auto* d = (MapUD*)ud1;
    d->status = status;
    d->done = true;
}
#endif

// ── Initialization ──────────────────────────────────────────────────────────

int webgpu_backend_init(WebGpuBackend* b) {
    memset(b, 0, sizeof(*b));

    WGPUInstanceDescriptor idesc = {};
    b->instance = wgpuCreateInstance(&idesc);
    if (!b->instance) { fprintf(stderr, "[webgpu] Failed to create instance\n"); return -1; }

    AdapterUD aud = {};
    WGPURequestAdapterOptions opts = {};
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
#ifdef __EMSCRIPTEN__
    wgpuInstanceRequestAdapter(b->instance, &opts, on_adapter, &aud);
#else
    WGPURequestAdapterCallbackInfo acb = {};
    acb.mode = WGPUCallbackMode_AllowSpontaneous;
    acb.callback = on_adapter;
    acb.userdata1 = &aud;
    wgpuInstanceRequestAdapter(b->instance, &opts, acb);
#endif
    while (!aud.done) poll_events(b->instance);
    if (!aud.adapter) { fprintf(stderr, "[webgpu] No adapter\n"); wgpuInstanceRelease(b->instance); return -1; }
    b->adapter = aud.adapter;

    DeviceUD dud = {};
    WGPUDeviceDescriptor ddesc = {};
#ifdef __EMSCRIPTEN__
    wgpuAdapterRequestDevice(b->adapter, &ddesc, on_device, &dud);
#else
    WGPURequestDeviceCallbackInfo dcb = {};
    dcb.mode = WGPUCallbackMode_AllowSpontaneous;
    dcb.callback = on_device;
    dcb.userdata1 = &dud;
    wgpuAdapterRequestDevice(b->adapter, &ddesc, dcb);
#endif
    while (!dud.done) poll_events(b->instance);
    if (!dud.device) { fprintf(stderr, "[webgpu] No device\n"); wgpuAdapterRelease(b->adapter); wgpuInstanceRelease(b->instance); return -1; }
    b->device = dud.device;
    b->queue = wgpuDeviceGetQueue(b->device);

    // ── Create all pipelines ────────────────────────────────────────────────

    // Histogram: (data_ro, histogram_rw, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = { bgle_storage_ro(0), bgle_storage_rw(1), bgle_uniform(2) };
        PipelineKit k = create_pipeline(b->device, HISTOGRAM_WGSL_SRC, sizeof(HISTOGRAM_WGSL_SRC)-1,
                                         "histogram", e, 3);
        b->histogram_shader = k.shader; b->histogram_bgl = k.bgl; b->histogram_pipeline = k.pipeline;
    }

    // Entropy: (histogram_ro, output_rw, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = { bgle_storage_ro(0), bgle_storage_rw(1), bgle_uniform(2) };
        PipelineKit k = create_pipeline(b->device, ENTROPY_WGSL_SRC, sizeof(ENTROPY_WGSL_SRC)-1,
                                         "entropy", e, 3);
        b->entropy_shader = k.shader; b->entropy_bgl = k.bgl; b->entropy_pipeline = k.pipeline;
    }

    // Bigram: (data_ro, bigram_rw, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = { bgle_storage_ro(0), bgle_storage_rw(1), bgle_uniform(2) };
        PipelineKit k = create_pipeline(b->device, BIGRAM_WGSL_SRC, sizeof(BIGRAM_WGSL_SRC)-1,
                                         "bigram", e, 3);
        b->bigram_shader = k.shader; b->bigram_bgl = k.bgl; b->bigram_pipeline = k.pipeline;
    }

    // LZ match: (data_ro, hash_table_rw, stats_rw, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = { bgle_storage_ro(0), bgle_storage_rw(1), bgle_storage_rw(2), bgle_uniform(3) };
        PipelineKit k = create_pipeline(b->device, LZ_MATCH_WGSL_SRC, sizeof(LZ_MATCH_WGSL_SRC)-1,
                                         "lz_match", e, 4);
        b->lz_shader = k.shader; b->lz_bgl = k.bgl; b->lz_pipeline = k.pipeline;
    }

    // Huffman encode: (data_ro, code_table_ro, compressed_rw, block_sizes_rw, block_offsets_ro, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = {
            bgle_storage_ro(0), bgle_storage_ro(1), bgle_storage_rw(2),
            bgle_storage_rw(3), bgle_storage_ro(4), bgle_uniform(5)
        };
        PipelineKit k = create_pipeline(b->device, HUFFMAN_ENC_WGSL_SRC, sizeof(HUFFMAN_ENC_WGSL_SRC)-1,
                                         "huffman_enc", e, 6);
        b->huffman_enc_shader = k.shader; b->huffman_enc_bgl = k.bgl; b->huffman_enc_pipeline = k.pipeline;
    }

    // Huffman decode: (compressed_ro, block_offsets_ro, block_sizes_ro, decode_table_ro, output_rw, params_uniform)
    {
        WGPUBindGroupLayoutEntry e[] = {
            bgle_storage_ro(0), bgle_storage_ro(1), bgle_storage_ro(2),
            bgle_storage_ro(3), bgle_storage_rw(4), bgle_uniform(5)
        };
        PipelineKit k = create_pipeline(b->device, HUFFMAN_DEC_WGSL_SRC, sizeof(HUFFMAN_DEC_WGSL_SRC)-1,
                                         "huffman_dec", e, 6);
        b->huffman_dec_shader = k.shader; b->huffman_dec_bgl = k.bgl; b->huffman_dec_pipeline = k.pipeline;
    }

    // Transform pipelines: all share the same BGL (storage_ro, storage_rw, uniform)
    {
        WGPUBindGroupLayoutEntry te[] = { bgle_storage_ro(0), bgle_storage_rw(1), bgle_uniform(2) };
        WGPUBindGroupLayoutDescriptor tbgld = {};
        tbgld.entryCount = 3;
        tbgld.entries = te;
        b->transform_bgl = wgpuDeviceCreateBindGroupLayout(b->device, &tbgld);

        struct { const char* src; size_t len; const char* label; } tshaders[] = {
            {TRANSFORM_DELTA_WGSL_SRC,     sizeof(TRANSFORM_DELTA_WGSL_SRC)-1,     "transform_delta"},
            {TRANSFORM_XOR_WGSL_SRC,       sizeof(TRANSFORM_XOR_WGSL_SRC)-1,       "transform_xor"},
            {TRANSFORM_ROTATE_WGSL_SRC,    sizeof(TRANSFORM_ROTATE_WGSL_SRC)-1,    "transform_rotate"},
            {TRANSFORM_SUB_MEAN_WGSL_SRC,  sizeof(TRANSFORM_SUB_MEAN_WGSL_SRC)-1,  "transform_sub_mean"},
            {TRANSFORM_BYTE_SWAP_WGSL_SRC, sizeof(TRANSFORM_BYTE_SWAP_WGSL_SRC)-1, "transform_byte_swap"},
        };

        WGPUPipelineLayoutDescriptor tpld = {};
        tpld.bindGroupLayoutCount = 1;
        tpld.bindGroupLayouts = &b->transform_bgl;
        WGPUPipelineLayout tpl = wgpuDeviceCreatePipelineLayout(b->device, &tpld);

        for (int i = 0; i < 5; i++) {
            b->transform_shaders[i] = create_shader(b->device, tshaders[i].src, tshaders[i].len, tshaders[i].label);
            WGPUComputePipelineDescriptor cpd = {};
            cpd.layout = tpl;
            cpd.compute.module = b->transform_shaders[i];
#ifdef __EMSCRIPTEN__
            cpd.compute.entryPoint = "main";
#else
            cpd.compute.entryPoint = sv("main");
#endif
            b->transform_pipelines[i] = wgpuDeviceCreateComputePipeline(b->device, &cpd);
            if (!b->transform_pipelines[i])
                fprintf(stderr, "[webgpu] Transform pipeline creation failed: %s\n", tshaders[i].label);
        }
        wgpuPipelineLayoutRelease(tpl);
    }

    if (!b->histogram_pipeline || !b->entropy_pipeline || !b->bigram_pipeline ||
        !b->lz_pipeline || !b->huffman_enc_pipeline || !b->huffman_dec_pipeline) {
        fprintf(stderr, "[webgpu] One or more pipelines failed to create\n");
        webgpu_backend_destroy(b);
        return -1;
    }

    b->valid = true;
    return 0;
}

void webgpu_backend_destroy(WebGpuBackend* b) {
    PipelineKit kits[] = {
        {b->histogram_shader, b->histogram_bgl, b->histogram_pipeline},
        {b->entropy_shader, b->entropy_bgl, b->entropy_pipeline},
        {b->bigram_shader, b->bigram_bgl, b->bigram_pipeline},
        {b->lz_shader, b->lz_bgl, b->lz_pipeline},
        {b->huffman_enc_shader, b->huffman_enc_bgl, b->huffman_enc_pipeline},
        {b->huffman_dec_shader, b->huffman_dec_bgl, b->huffman_dec_pipeline},
    };
    for (auto& k : kits) release_pipeline_kit(&k);
    // Transform pipelines
    for (int i = 0; i < 5; i++) {
        if (b->transform_pipelines[i]) wgpuComputePipelineRelease(b->transform_pipelines[i]);
        if (b->transform_shaders[i]) wgpuShaderModuleRelease(b->transform_shaders[i]);
    }
    if (b->transform_bgl) wgpuBindGroupLayoutRelease(b->transform_bgl);
    if (b->queue) wgpuQueueRelease(b->queue);
    if (b->device) wgpuDeviceRelease(b->device);
    if (b->adapter) wgpuAdapterRelease(b->adapter);
    if (b->instance) wgpuInstanceRelease(b->instance);
    memset(b, 0, sizeof(*b));
}

void webgpu_backend_get_info(const WebGpuBackend* b, BackendInfo* info) {
    info->backend = COMP_BACKEND_WEBGPU;
#ifdef __EMSCRIPTEN__
    // Emscripten doesn't have wgpuAdapterGetInfo
    strncpy(info->device_name, "WebGPU (browser)", sizeof(info->device_name) - 1);
    strncpy(info->vendor, "browser", sizeof(info->vendor) - 1);
    WGPUSupportedLimits supported = {};
    wgpuDeviceGetLimits(b->device, &supported);
    info->max_buffer_size = supported.limits.maxBufferSize;
#else
    WGPUAdapterInfo ai = {};
    wgpuAdapterGetInfo(b->adapter, &ai);
    if (ai.device.data && ai.device.length > 0) {
        size_t len = ai.device.length < sizeof(info->device_name) - 1 ? ai.device.length : sizeof(info->device_name) - 1;
        memcpy(info->device_name, ai.device.data, len);
        info->device_name[len] = '\0';
    } else {
        strncpy(info->device_name, "WebGPU device", sizeof(info->device_name) - 1);
    }
    if (ai.vendor.data && ai.vendor.length > 0) {
        size_t len = ai.vendor.length < sizeof(info->vendor) - 1 ? ai.vendor.length : sizeof(info->vendor) - 1;
        memcpy(info->vendor, ai.vendor.data, len);
        info->vendor[len] = '\0';
    } else {
        strncpy(info->vendor, "unknown", sizeof(info->vendor) - 1);
    }
    wgpuAdapterInfoFreeMembers(ai);
    WGPULimits limits = {};
    wgpuDeviceGetLimits(b->device, &limits);
    info->max_buffer_size = limits.maxBufferSize;
#endif
    info->max_slots = (uint32_t)(info->max_buffer_size / (128ULL * 1024 * 1024));
}

// ── Buffer operations ──────────────────────────────────────────────────────

static WGPUBuffer upload_uniform(WebGpuBackend* b, const void* data, uint64_t size) {
    uint64_t aligned_size = (size + 15) & ~15ULL; // uniforms need 16-byte alignment
    WGPUBufferDescriptor desc = {};
    desc.size = aligned_size;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (!buf) return nullptr;
    void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned_size);
    memset(ptr, 0, aligned_size);
    memcpy(ptr, data, size);
    wgpuBufferUnmap(buf);
    return buf;
}

WGPUBuffer webgpu_upload(WebGpuBackend* b, const uint8_t* data, uint64_t size) {
    uint64_t aligned_size = (size + 3) & ~3ULL;
    WGPUBufferDescriptor desc = {};
    desc.size = aligned_size;
    desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (!buf) return nullptr;
    void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned_size);
    memset(ptr, 0, aligned_size);
    memcpy(ptr, data, size);
    wgpuBufferUnmap(buf);
    return buf;
}

WGPUBuffer webgpu_create_buffer(WebGpuBackend* b, uint64_t size, WGPUBufferUsage usage) {
    uint64_t aligned = (size + 3) & ~3ULL;
    WGPUBufferDescriptor desc = {};
    desc.size = aligned;
    desc.usage = usage;
    desc.mappedAtCreation = true;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(b->device, &desc);
    if (buf) {
        void* ptr = wgpuBufferGetMappedRange(buf, 0, aligned);
        memset(ptr, 0, aligned);
        wgpuBufferUnmap(buf);
    }
    return buf;
}

void webgpu_free_buffer(WGPUBuffer buf) {
    if (buf) wgpuBufferRelease(buf);
}

int webgpu_readback(WebGpuBackend* b, WGPUBuffer src, void* dst, uint64_t size) {
    uint64_t aligned = (size + 3) & ~3ULL;
    WGPUBufferDescriptor rb_desc = {};
    rb_desc.size = aligned;
    rb_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer rb = wgpuDeviceCreateBuffer(b->device, &rb_desc);
    if (!rb) return -1;

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, rb, 0, aligned);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(b->queue, 1, &cmd);

    MapUD mud = {};
#ifdef __EMSCRIPTEN__
    wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, aligned, on_map, &mud);
#else
    WGPUBufferMapCallbackInfo mcb = {};
    mcb.mode = WGPUCallbackMode_AllowSpontaneous;
    mcb.callback = on_map;
    mcb.userdata1 = &mud;
    wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, aligned, mcb);
#endif
    while (!mud.done) poll_events(b->instance);

    int ret = -1;
    if (mud.status == WGPUMapAsyncStatus_Success) {
        const void* mapped = wgpuBufferGetMappedRange(rb, 0, aligned);
        memcpy(dst, mapped, size);
        wgpuBufferUnmap(rb);
        ret = 0;
    }

    wgpuBufferRelease(rb);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    return ret;
}

// ── Generic dispatch helper ────────────────────────────────────────────────

struct DispatchDesc {
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout bgl;
    WGPUBindGroupEntry* entries;
    uint32_t entry_count;
    uint32_t workgroups_x, workgroups_y, workgroups_z;
    // Optional copy-to-readback
    WGPUBuffer copy_src;
    uint64_t copy_size;
};

static int dispatch_and_readback(WebGpuBackend* b, const DispatchDesc& dd, void* out, uint64_t out_size) {
    WGPUBindGroupDescriptor bgdesc = {};
    bgdesc.layout = dd.bgl;
    bgdesc.entryCount = dd.entry_count;
    bgdesc.entries = dd.entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(b->device, &bgdesc);
    if (!bg) return -1;

    uint64_t rb_size = (out_size + 3) & ~3ULL;
    WGPUBufferDescriptor rb_desc = {};
    rb_desc.size = rb_size;
    rb_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    WGPUBuffer rb = wgpuDeviceCreateBuffer(b->device, &rb_desc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(b->device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, dd.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, dd.workgroups_x, dd.workgroups_y, dd.workgroups_z);
    wgpuComputePassEncoderEnd(pass);

    if (dd.copy_src && out) {
        wgpuCommandEncoderCopyBufferToBuffer(enc, dd.copy_src, 0, rb, 0, rb_size);
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(b->queue, 1, &cmd);

    int ret = 0;
    if (dd.copy_src && out) {
        MapUD mud = {};
#ifdef __EMSCRIPTEN__
        wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, rb_size, on_map, &mud);
#else
        WGPUBufferMapCallbackInfo mcb = {};
        mcb.mode = WGPUCallbackMode_AllowSpontaneous;
        mcb.callback = on_map;
        mcb.userdata1 = &mud;
        wgpuBufferMapAsync(rb, WGPUMapMode_Read, 0, rb_size, mcb);
#endif
        while (!mud.done) poll_events(b->instance);

        if (mud.status == WGPUMapAsyncStatus_Success) {
            const void* mapped = wgpuBufferGetMappedRange(rb, 0, rb_size);
            memcpy(out, mapped, out_size);
            wgpuBufferUnmap(rb);
        } else {
            ret = -1;
        }
    }

    wgpuBufferRelease(rb);
    wgpuBindGroupRelease(bg);
    wgpuCommandBufferRelease(cmd);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(enc);
    return ret;
}

// ── Histogram ──────────────────────────────────────────────────────────────

int webgpu_histogram(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                     uint32_t* out_histogram) {
    if (!b->valid || !data_buf) return -1;

    WGPUBuffer hist_buf = webgpu_create_buffer(b, 256 * sizeof(uint32_t),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params[4] = {(uint32_t)data_size, 0, 0, 0};
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = data_buf; entries[0].size = (data_size + 3) & ~3ULL;
    entries[1].binding = 1; entries[1].buffer = hist_buf;  entries[1].size = 256 * sizeof(uint32_t);
    entries[2].binding = 2; entries[2].buffer = params_buf; entries[2].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->histogram_pipeline;
    dd.bgl = b->histogram_bgl;
    dd.entries = entries;
    dd.entry_count = 3;
    dd.workgroups_x = 256; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = hist_buf;
    dd.copy_size = 256 * sizeof(uint32_t);

    int ret = dispatch_and_readback(b, dd, out_histogram, 256 * sizeof(uint32_t));

    webgpu_free_buffer(hist_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}

// ── Entropy order-0 ────────────────────────────────────────────────────────

int webgpu_entropy_order0(WebGpuBackend* b, WGPUBuffer hist_buf, uint32_t total,
                          float* out_bits) {
    if (!b->valid || !hist_buf) return -1;

    WGPUBuffer out_buf = webgpu_create_buffer(b, sizeof(float),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params[4] = {total, 0, 0, 0};
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = hist_buf;   entries[0].size = 256 * sizeof(uint32_t);
    entries[1].binding = 1; entries[1].buffer = out_buf;    entries[1].size = sizeof(float);
    entries[2].binding = 2; entries[2].buffer = params_buf; entries[2].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->entropy_pipeline;
    dd.bgl = b->entropy_bgl;
    dd.entries = entries;
    dd.entry_count = 3;
    dd.workgroups_x = 1; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = out_buf;

    int ret = dispatch_and_readback(b, dd, out_bits, sizeof(float));

    webgpu_free_buffer(out_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}

// ── Bigram histogram ───────────────────────────────────────────────────────

int webgpu_bigram_histogram(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                            uint32_t* out_bigram) {
    if (!b->valid || !data_buf) return -1;

    uint64_t bigram_size = 256 * 256 * sizeof(uint32_t);
    WGPUBuffer bigram_buf = webgpu_create_buffer(b, bigram_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params[4] = {(uint32_t)data_size, 0, 0, 0};
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0; entries[0].buffer = data_buf;   entries[0].size = (data_size + 3) & ~3ULL;
    entries[1].binding = 1; entries[1].buffer = bigram_buf; entries[1].size = bigram_size;
    entries[2].binding = 2; entries[2].buffer = params_buf; entries[2].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->bigram_pipeline;
    dd.bgl = b->bigram_bgl;
    dd.entries = entries;
    dd.entry_count = 3;
    dd.workgroups_x = 256; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = bigram_buf;

    int ret = dispatch_and_readback(b, dd, out_bigram, bigram_size);

    webgpu_free_buffer(bigram_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}

// ── LZ match statistics ────────────────────────────────────────────────────

int webgpu_lz_match_stats(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                          uint32_t out_stats[4]) {
    if (!b->valid || !data_buf) return -1;

    uint32_t num_wg = 256; // number of workgroups
    uint64_t hash_size = 262144 * sizeof(uint32_t); // HASH_SIZE from shader
    uint64_t stats_size = num_wg * 4 * sizeof(uint32_t);

    WGPUBuffer hash_buf = webgpu_create_buffer(b, hash_size,
        WGPUBufferUsage_Storage);
    WGPUBuffer stats_buf = webgpu_create_buffer(b, stats_size,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params[4] = {(uint32_t)data_size, num_wg, 0, 0};
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[4] = {};
    entries[0].binding = 0; entries[0].buffer = data_buf;   entries[0].size = (data_size + 3) & ~3ULL;
    entries[1].binding = 1; entries[1].buffer = hash_buf;   entries[1].size = hash_size;
    entries[2].binding = 2; entries[2].buffer = stats_buf;  entries[2].size = stats_size;
    entries[3].binding = 3; entries[3].buffer = params_buf; entries[3].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->lz_pipeline;
    dd.bgl = b->lz_bgl;
    dd.entries = entries;
    dd.entry_count = 4;
    dd.workgroups_x = num_wg; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = stats_buf;

    // Read back per-workgroup stats and aggregate
    std::vector<uint32_t> wg_stats(num_wg * 4);
    int ret = dispatch_and_readback(b, dd, wg_stats.data(), stats_size);

    if (ret == 0) {
        out_stats[0] = out_stats[1] = out_stats[2] = out_stats[3] = 0;
        for (uint32_t i = 0; i < num_wg; i++) {
            out_stats[0] += wg_stats[i * 4 + 0]; // literal_bytes
            out_stats[1] += wg_stats[i * 4 + 1]; // match_bytes
            out_stats[2] += wg_stats[i * 4 + 2]; // match_count
        }
    }

    webgpu_free_buffer(hash_buf);
    webgpu_free_buffer(stats_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}

// ── GPU Huffman compression ────────────────────────────────────────────────

static const uint32_t GPU_HUFFMAN_BLOCK_SIZE = 4096;

int webgpu_huffman_size(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                        const uint32_t* code_table, uint64_t* out_bits) {
    if (!b->valid || !data_buf) return -1;

    uint32_t num_blocks = (uint32_t)((data_size + GPU_HUFFMAN_BLOCK_SIZE - 1) / GPU_HUFFMAN_BLOCK_SIZE);

    WGPUBuffer ct_buf = webgpu_upload(b, (const uint8_t*)code_table, 256 * sizeof(uint32_t));
    WGPUBuffer comp_buf = webgpu_create_buffer(b, 4, WGPUBufferUsage_Storage); // dummy
    WGPUBuffer sizes_buf = webgpu_create_buffer(b, num_blocks * sizeof(uint32_t),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer offsets_buf = webgpu_create_buffer(b, num_blocks * sizeof(uint32_t),
        WGPUBufferUsage_Storage); // dummy

    uint32_t params[4] = {GPU_HUFFMAN_BLOCK_SIZE, num_blocks, (uint32_t)data_size, 0}; // mode=0 (size-only)
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[6] = {};
    entries[0].binding = 0; entries[0].buffer = data_buf;    entries[0].size = (data_size + 3) & ~3ULL;
    entries[1].binding = 1; entries[1].buffer = ct_buf;      entries[1].size = 256 * sizeof(uint32_t);
    entries[2].binding = 2; entries[2].buffer = comp_buf;    entries[2].size = 4;
    entries[3].binding = 3; entries[3].buffer = sizes_buf;   entries[3].size = num_blocks * sizeof(uint32_t);
    entries[4].binding = 4; entries[4].buffer = offsets_buf; entries[4].size = num_blocks * sizeof(uint32_t);
    entries[5].binding = 5; entries[5].buffer = params_buf;  entries[5].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->huffman_enc_pipeline;
    dd.bgl = b->huffman_enc_bgl;
    dd.entries = entries;
    dd.entry_count = 6;
    dd.workgroups_x = num_blocks; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = sizes_buf;

    std::vector<uint32_t> block_sizes(num_blocks);
    int ret = dispatch_and_readback(b, dd, block_sizes.data(), num_blocks * sizeof(uint32_t));

    if (ret == 0) {
        uint64_t total = 0;
        for (uint32_t i = 0; i < num_blocks; i++) total += block_sizes[i];
        *out_bits = total;
    }

    webgpu_free_buffer(ct_buf);
    webgpu_free_buffer(comp_buf);
    webgpu_free_buffer(sizes_buf);
    webgpu_free_buffer(offsets_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}

int webgpu_huffman_compress(WebGpuBackend* b, WGPUBuffer data_buf, uint64_t data_size,
                            const uint32_t* code_table, CompressedBuffer* out) {
    if (!b->valid || !data_buf || !out) return -1;

    uint32_t num_blocks = (uint32_t)((data_size + GPU_HUFFMAN_BLOCK_SIZE - 1) / GPU_HUFFMAN_BLOCK_SIZE);

    // Phase 1: compute block sizes (size-only mode)
    WGPUBuffer ct_buf = webgpu_upload(b, (const uint8_t*)code_table, 256 * sizeof(uint32_t));
    WGPUBuffer dummy_comp = webgpu_create_buffer(b, 4, WGPUBufferUsage_Storage);
    WGPUBuffer sizes_buf = webgpu_create_buffer(b, num_blocks * sizeof(uint32_t),
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer dummy_offsets = webgpu_create_buffer(b, num_blocks * sizeof(uint32_t),
        WGPUBufferUsage_Storage);

    uint32_t params_size[4] = {GPU_HUFFMAN_BLOCK_SIZE, num_blocks, (uint32_t)data_size, 0};
    WGPUBuffer params_buf_size = upload_uniform(b, params_size, sizeof(params_size));

    WGPUBindGroupEntry e1[6] = {};
    e1[0].binding = 0; e1[0].buffer = data_buf;       e1[0].size = (data_size + 3) & ~3ULL;
    e1[1].binding = 1; e1[1].buffer = ct_buf;         e1[1].size = 256 * sizeof(uint32_t);
    e1[2].binding = 2; e1[2].buffer = dummy_comp;     e1[2].size = 4;
    e1[3].binding = 3; e1[3].buffer = sizes_buf;      e1[3].size = num_blocks * sizeof(uint32_t);
    e1[4].binding = 4; e1[4].buffer = dummy_offsets;   e1[4].size = num_blocks * sizeof(uint32_t);
    e1[5].binding = 5; e1[5].buffer = params_buf_size; e1[5].size = sizeof(params_size);

    DispatchDesc dd1 = {};
    dd1.pipeline = b->huffman_enc_pipeline;
    dd1.bgl = b->huffman_enc_bgl;
    dd1.entries = e1;
    dd1.entry_count = 6;
    dd1.workgroups_x = num_blocks; dd1.workgroups_y = 1; dd1.workgroups_z = 1;
    dd1.copy_src = sizes_buf;

    std::vector<uint32_t> block_sizes(num_blocks);
    int ret = dispatch_and_readback(b, dd1, block_sizes.data(), num_blocks * sizeof(uint32_t));
    webgpu_free_buffer(dummy_comp);
    webgpu_free_buffer(dummy_offsets);
    webgpu_free_buffer(params_buf_size);
    if (ret != 0) { webgpu_free_buffer(ct_buf); webgpu_free_buffer(sizes_buf); return -1; }

    // CPU: compute prefix sum of block sizes to get bit offsets
    std::vector<uint32_t> block_offsets(num_blocks);
    uint64_t total_bits = 0;
    for (uint32_t i = 0; i < num_blocks; i++) {
        block_offsets[i] = (uint32_t)total_bits;
        total_bits += block_sizes[i];
    }
    uint64_t total_bytes = (total_bits + 31) / 32 * 4; // u32-aligned

    // Phase 2: encode
    WGPUBuffer offsets_buf = webgpu_upload(b, (const uint8_t*)block_offsets.data(),
                                            num_blocks * sizeof(uint32_t));
    WGPUBuffer comp_buf = webgpu_create_buffer(b, total_bytes > 0 ? total_bytes : 4,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params_enc[4] = {GPU_HUFFMAN_BLOCK_SIZE, num_blocks, (uint32_t)data_size, 1}; // mode=1
    WGPUBuffer params_buf_enc = upload_uniform(b, params_enc, sizeof(params_enc));

    WGPUBindGroupEntry e2[6] = {};
    e2[0].binding = 0; e2[0].buffer = data_buf;       e2[0].size = (data_size + 3) & ~3ULL;
    e2[1].binding = 1; e2[1].buffer = ct_buf;         e2[1].size = 256 * sizeof(uint32_t);
    e2[2].binding = 2; e2[2].buffer = comp_buf;       e2[2].size = total_bytes > 0 ? total_bytes : 4;
    e2[3].binding = 3; e2[3].buffer = sizes_buf;      e2[3].size = num_blocks * sizeof(uint32_t);
    e2[4].binding = 4; e2[4].buffer = offsets_buf;    e2[4].size = num_blocks * sizeof(uint32_t);
    e2[5].binding = 5; e2[5].buffer = params_buf_enc; e2[5].size = sizeof(params_enc);

    DispatchDesc dd2 = {};
    dd2.pipeline = b->huffman_enc_pipeline;
    dd2.bgl = b->huffman_enc_bgl;
    dd2.entries = e2;
    dd2.entry_count = 6;
    dd2.workgroups_x = num_blocks; dd2.workgroups_y = 1; dd2.workgroups_z = 1;
    dd2.copy_src = comp_buf;

    // Build output: header + block_offsets + compressed data
    // Header: [8 bytes orig_size LE] [4 bytes num_blocks LE] [4 bytes block_size LE]
    //         [num_blocks * 4 bytes block bit offsets] [num_blocks * 4 bytes block bit sizes]
    //         [compressed bitstream]
    uint64_t header_size = 16 + num_blocks * 8;
    uint64_t output_size = header_size + total_bytes;

    std::vector<uint8_t> comp_data(total_bytes > 0 ? total_bytes : 4);
    ret = dispatch_and_readback(b, dd2, comp_data.data(), total_bytes > 0 ? total_bytes : 4);

    if (ret == 0) {
        out->data = (uint8_t*)malloc(output_size);
        if (!out->data) { ret = -1; }
        else {
            uint8_t* p = out->data;
            // Original size (LE)
            for (int i = 0; i < 8; i++) { *p++ = (uint8_t)(data_size >> (i * 8)); }
            // Num blocks (LE)
            for (int i = 0; i < 4; i++) { *p++ = (uint8_t)(num_blocks >> (i * 8)); }
            // Block size (LE)
            uint32_t bs = GPU_HUFFMAN_BLOCK_SIZE;
            for (int i = 0; i < 4; i++) { *p++ = (uint8_t)(bs >> (i * 8)); }
            // Block offsets
            memcpy(p, block_offsets.data(), num_blocks * 4); p += num_blocks * 4;
            // Block sizes
            memcpy(p, block_sizes.data(), num_blocks * 4); p += num_blocks * 4;
            // Compressed bitstream
            memcpy(p, comp_data.data(), total_bytes);

            out->size = output_size;
            out->capacity = output_size;
            out->algorithm = COMP_HUFFMAN;
        }
    }

    webgpu_free_buffer(ct_buf);
    webgpu_free_buffer(sizes_buf);
    webgpu_free_buffer(offsets_buf);
    webgpu_free_buffer(comp_buf);
    webgpu_free_buffer(params_buf_enc);
    return ret;
}

int webgpu_huffman_decompress(WebGpuBackend* b, const uint8_t* compressed, uint64_t comp_size,
                              const uint32_t* decode_table, uint32_t max_code_len,
                              uint64_t original_size, uint8_t** out_data) {
    if (!b->valid || !compressed || !out_data) return -1;

    // Parse header
    if (comp_size < 16) return -1;
    uint64_t orig_size = 0;
    for (int i = 0; i < 8; i++) orig_size |= (uint64_t)compressed[i] << (i * 8);
    uint32_t num_blocks = 0;
    for (int i = 0; i < 4; i++) num_blocks |= (uint32_t)compressed[8 + i] << (i * 8);

    uint64_t header_size = 16 + num_blocks * 8;
    if (comp_size < header_size) return -1;

    const uint8_t* offsets_p = compressed + 16;
    const uint8_t* sizes_p = compressed + 16 + num_blocks * 4;
    const uint8_t* bitstream = compressed + header_size;
    uint64_t bitstream_size = comp_size - header_size;

    // Build block uncompressed sizes
    std::vector<uint32_t> block_uncomp(num_blocks);
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t start = (uint64_t)i * GPU_HUFFMAN_BLOCK_SIZE;
        block_uncomp[i] = (uint32_t)(start + GPU_HUFFMAN_BLOCK_SIZE <= orig_size
                          ? GPU_HUFFMAN_BLOCK_SIZE : orig_size - start);
    }

    // Upload to GPU
    WGPUBuffer comp_buf = webgpu_upload(b, bitstream, bitstream_size);
    WGPUBuffer offsets_buf = webgpu_upload(b, offsets_p, num_blocks * 4);
    WGPUBuffer uncomp_buf = webgpu_upload(b, (const uint8_t*)block_uncomp.data(), num_blocks * 4);
    WGPUBuffer dt_buf = webgpu_upload(b, (const uint8_t*)decode_table, 65536 * sizeof(uint32_t));
    uint64_t out_aligned = (orig_size + 3) & ~3ULL;
    WGPUBuffer out_buf = webgpu_create_buffer(b, out_aligned,
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);

    uint32_t params[4] = {num_blocks, (uint32_t)orig_size, max_code_len, 0};
    WGPUBuffer params_buf = upload_uniform(b, params, sizeof(params));

    WGPUBindGroupEntry entries[6] = {};
    entries[0].binding = 0; entries[0].buffer = comp_buf;    entries[0].size = (bitstream_size + 3) & ~3ULL;
    entries[1].binding = 1; entries[1].buffer = offsets_buf; entries[1].size = num_blocks * 4;
    entries[2].binding = 2; entries[2].buffer = uncomp_buf;  entries[2].size = num_blocks * 4;
    entries[3].binding = 3; entries[3].buffer = dt_buf;      entries[3].size = 65536 * sizeof(uint32_t);
    entries[4].binding = 4; entries[4].buffer = out_buf;     entries[4].size = out_aligned;
    entries[5].binding = 5; entries[5].buffer = params_buf;  entries[5].size = sizeof(params);

    DispatchDesc dd = {};
    dd.pipeline = b->huffman_dec_pipeline;
    dd.bgl = b->huffman_dec_bgl;
    dd.entries = entries;
    dd.entry_count = 6;
    dd.workgroups_x = num_blocks; dd.workgroups_y = 1; dd.workgroups_z = 1;
    dd.copy_src = out_buf;

    uint8_t* result = (uint8_t*)malloc(orig_size);
    if (!result) {
        webgpu_free_buffer(comp_buf); webgpu_free_buffer(offsets_buf);
        webgpu_free_buffer(uncomp_buf); webgpu_free_buffer(dt_buf);
        webgpu_free_buffer(out_buf); webgpu_free_buffer(params_buf);
        return -1;
    }

    int ret = dispatch_and_readback(b, dd, result, orig_size);

    if (ret == 0) {
        *out_data = result;
    } else {
        free(result);
    }

    webgpu_free_buffer(comp_buf);
    webgpu_free_buffer(offsets_buf);
    webgpu_free_buffer(uncomp_buf);
    webgpu_free_buffer(dt_buf);
    webgpu_free_buffer(out_buf);
    webgpu_free_buffer(params_buf);
    return ret;
}
