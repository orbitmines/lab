#ifndef COMPRESSION_NEURAL_COMPRESSOR_H
#define COMPRESSION_NEURAL_COMPRESSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NeuralCompressor NeuralCompressor;

typedef struct {
    int context_size;     // Previous bytes as context (default: 8)
    int embed_dim;        // Embedding dimension per byte (default: 8)
    int hidden_dim;       // Hidden layer size (default: 64)
    float learning_rate;  // Adam learning rate (default: 0.001)
    int batch_size;       // Mini-batch size for training (default: 4096)
} NeuralCompressorConfig;

// Create with config (NULL for defaults)
NeuralCompressor* neural_compressor_create(const NeuralCompressorConfig* config);
void neural_compressor_destroy(NeuralCompressor* nc);

// Train for the specified number of seconds, printing progress per epoch.
// Returns final cross-entropy in bits per byte.
float neural_compressor_train(NeuralCompressor* nc, const uint8_t* data, uint64_t size, int seconds);

// Number of trainable parameters
uint32_t neural_compressor_param_count(const NeuralCompressor* nc);

// Compress data using the trained model + arithmetic coding.
// Returns compressed data (caller frees with free()). Sets out_size.
uint8_t* neural_compressor_compress(NeuralCompressor* nc,
                                     const uint8_t* data, uint64_t size,
                                     uint64_t* out_size);

// Decompress. Model weights are stored in the compressed stream.
// Returns decompressed data (caller frees with free()). Sets out_size.
uint8_t* neural_compressor_decompress(const uint8_t* compressed, uint64_t comp_size,
                                       uint64_t* out_size);

#ifdef __cplusplus
}
#endif

#endif
