#ifndef GPU_CONTEXT_INTERNAL_H
#define GPU_CONTEXT_INTERNAL_H

#include "compression/gpu_context.h"

#ifdef HAS_WEBGPU
#include "gpu/webgpu_backend.h"
#endif

struct GpuContext {
#ifdef HAS_WEBGPU
    WebGpuBackend backend;
    bool has_gpu;
#else
    bool has_gpu = false;
#endif
};

#endif
