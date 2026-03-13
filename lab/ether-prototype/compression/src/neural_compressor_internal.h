#ifndef NEURAL_COMPRESSOR_INTERNAL_H
#define NEURAL_COMPRESSOR_INTERNAL_H

#include "compression/neural_compressor.h"
#include <cstdint>

struct NeuralCompressor {
    int ctx, emb, hid, inp;
    int batch_size;
    float lr;

    // Weight layout offsets (all in one flat array)
    int off_embed, off_w1, off_b1, off_w2, off_b2;
    int n_params;

    float* weights;
    float* grads;
    float* adam_m;
    float* adam_v;
    int adam_step;

    // Activation buffers (batch_size × dim)
    float* act_input;   // [batch, inp]
    float* act_hidden;  // [batch, hid]
    float* act_logits;  // [batch, 256]
    float* act_probs;   // [batch, 256]

    // Gradient buffers
    float* g_logits;    // [batch, 256]
    float* g_hidden;    // [batch, hid]
    float* g_input;     // [batch, inp]

    // Context/target buffers
    uint8_t* batch_ctx;    // [batch, ctx]
    uint8_t* batch_target; // [batch]
};

#endif
