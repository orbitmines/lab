#ifndef COMPRESSION_GPU_SESSION_H
#define COMPRESSION_GPU_SESSION_H

#include "compression/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuSession GpuSession;
typedef struct GpuContext GpuContext;

// Create a GPU session: uploads data once, allocates scratch slots.
// Data stays GPU-resident. Returns NULL on failure.
GpuSession* gpu_session_create(GpuContext* ctx, const uint8_t* data, uint64_t size);

void gpu_session_destroy(GpuSession* session);

// Number of parallel slots available in this session.
uint32_t gpu_session_num_slots(const GpuSession* session);

// Data size loaded into this session.
uint64_t gpu_session_data_size(const GpuSession* session);

// Evaluate a batch of transforms. All work happens on GPU; only small
// per-slot scores are read back. Internally batches by available slots.
// out_scores must have room for count entries.
// Returns 0 on success.
int gpu_session_evaluate_batch(GpuSession* session,
                                const TransformDesc* transforms, uint32_t count,
                                SlotScore* out_scores);

#ifdef __cplusplus
}
#endif

#endif
