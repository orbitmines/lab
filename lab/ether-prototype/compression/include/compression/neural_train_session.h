#ifndef COMPRESSION_NEURAL_TRAIN_SESSION_H
#define COMPRESSION_NEURAL_TRAIN_SESSION_H

#include "compression/neural_compressor.h"
#include "compression/gpu_context.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Train neural compressor on GPU for the specified number of seconds.
// Weights are copied to GPU before training and back to nc after training.
// Returns final cross-entropy in bits per byte.
float neural_compressor_train_gpu(NeuralCompressor* nc,
                                   const uint8_t* data, uint64_t size,
                                   int seconds, GpuContext* gpu_ctx);

#ifdef __cplusplus
}
#endif

#endif
