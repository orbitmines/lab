#include "neural_compressor_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ── Constants ────────────────────────────────────────────────────────────────

static const uint32_t FREQ_BITS = 14;
static const uint32_t FREQ_TOTAL = 1u << FREQ_BITS;  // 16384

// Arithmetic coder precision (31-bit)
static const uint32_t CODE_BITS = 31;
static const uint32_t CODE_HALF = 1u << (CODE_BITS - 1);     // 2^30
static const uint32_t CODE_QUARTER = 1u << (CODE_BITS - 2);  // 2^29
static const uint32_t CODE_MAX = (1u << CODE_BITS) - 1;      // 2^31 - 1

// File format magic
static const uint32_t NCMP_MAGIC = 0x504D434E;  // "NCMP" little-endian
static const uint32_t NCMP_VERSION = 2;  // v2: 16-bit weight quantization

// ── Matrix operations ────────────────────────────────────────────────────────

// C[M,N] = A[M,K] @ B[K,N]
static void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        memset(c_row, 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            float a_val = a_row[k];
            const float* b_row = B + k * N;
            for (int j = 0; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// C[M,N] += A[K,M]^T @ B[K,N]  (A transposed)
static void matmul_at_add(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int k = 0; k < K; k++) {
        const float* a_col = A + k * M;
        const float* b_row = B + k * N;
        for (int i = 0; i < M; i++) {
            float a_val = a_col[i];
            float* c_row = C + i * N;
            for (int j = 0; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

// C[M,N] = A[M,K] @ B[N,K]^T  (B transposed)
static void matmul_bt(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        for (int j = 0; j < N; j++) {
            const float* b_row = B + j * K;
            float sum = 0;
            for (int k = 0; k < K; k++) sum += a_row[k] * b_row[k];
            c_row[j] = sum;
        }
    }
}

// ── Create / destroy ─────────────────────────────────────────────────────────

NeuralCompressor* neural_compressor_create(const NeuralCompressorConfig* cfg) {
    NeuralCompressorConfig defaults = {8, 8, 64, 0.001f, 4096};
    if (!cfg) cfg = &defaults;

    auto* nc = new NeuralCompressor();
    nc->ctx = cfg->context_size > 0 ? cfg->context_size : 8;
    nc->emb = cfg->embed_dim > 0 ? cfg->embed_dim : 8;
    nc->hid = cfg->hidden_dim > 0 ? cfg->hidden_dim : 64;
    nc->inp = nc->ctx * nc->emb;
    nc->batch_size = cfg->batch_size > 0 ? cfg->batch_size : 4096;
    nc->lr = cfg->learning_rate > 0 ? cfg->learning_rate : 0.001f;

    // Weight layout
    nc->off_embed = 0;
    nc->off_w1 = 256 * nc->emb;
    nc->off_b1 = nc->off_w1 + nc->inp * nc->hid;
    nc->off_w2 = nc->off_b1 + nc->hid;
    nc->off_b2 = nc->off_w2 + nc->hid * 256;
    nc->n_params = nc->off_b2 + 256;

    nc->weights = new float[nc->n_params]();
    nc->grads   = new float[nc->n_params]();
    nc->adam_m  = new float[nc->n_params]();
    nc->adam_v  = new float[nc->n_params]();
    nc->adam_step = 0;

    // Xavier initialization
    std::mt19937 rng(42);
    auto xavier = [&](float* w, int fan_in, int fan_out, int count) {
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0, scale);
        for (int i = 0; i < count; i++) w[i] = dist(rng);
    };

    xavier(nc->weights + nc->off_embed, 1, nc->emb, 256 * nc->emb);
    xavier(nc->weights + nc->off_w1, nc->inp, nc->hid, nc->inp * nc->hid);
    // b1 stays zero
    xavier(nc->weights + nc->off_w2, nc->hid, 256, nc->hid * 256);
    // b2 stays zero

    // Activation buffers
    int B = nc->batch_size;
    nc->act_input  = new float[B * nc->inp]();
    nc->act_hidden = new float[B * nc->hid]();
    nc->act_logits = new float[B * 256]();
    nc->act_probs  = new float[B * 256]();
    nc->g_logits   = new float[B * 256]();
    nc->g_hidden   = new float[B * nc->hid]();
    nc->g_input    = new float[B * nc->inp]();
    nc->batch_ctx    = new uint8_t[B * nc->ctx]();
    nc->batch_target = new uint8_t[B]();

    return nc;
}

void neural_compressor_destroy(NeuralCompressor* nc) {
    if (!nc) return;
    delete[] nc->weights;
    delete[] nc->grads;
    delete[] nc->adam_m;
    delete[] nc->adam_v;
    delete[] nc->act_input;
    delete[] nc->act_hidden;
    delete[] nc->act_logits;
    delete[] nc->act_probs;
    delete[] nc->g_logits;
    delete[] nc->g_hidden;
    delete[] nc->g_input;
    delete[] nc->batch_ctx;
    delete[] nc->batch_target;
    delete nc;
}

uint32_t neural_compressor_param_count(const NeuralCompressor* nc) {
    return nc ? nc->n_params : 0;
}

// ── Forward pass (single position) ──────────────────────────────────────────

static void forward_single(const NeuralCompressor* nc, const uint8_t* context,
                            float* out_probs) {
    const float* embed = nc->weights + nc->off_embed;
    const float* W1 = nc->weights + nc->off_w1;
    const float* b1 = nc->weights + nc->off_b1;
    const float* W2 = nc->weights + nc->off_w2;
    const float* b2 = nc->weights + nc->off_b2;

    // Embedding lookup → input vector [inp]
    float input[512];  // max inp = ctx*emb, should be <= 512
    for (int c = 0; c < nc->ctx; c++) {
        memcpy(input + c * nc->emb, embed + context[c] * nc->emb, nc->emb * sizeof(float));
    }

    // Hidden = ReLU(input @ W1 + b1)
    float hidden[256];  // max hid
    for (int j = 0; j < nc->hid; j++) {
        float sum = b1[j];
        for (int k = 0; k < nc->inp; k++) {
            sum += input[k] * W1[k * nc->hid + j];
        }
        hidden[j] = sum > 0 ? sum : 0;  // ReLU
    }

    // Logits = hidden @ W2 + b2, then softmax
    float logits[256];
    float max_logit = -1e30f;
    for (int j = 0; j < 256; j++) {
        float sum = b2[j];
        for (int k = 0; k < nc->hid; k++) {
            sum += hidden[k] * W2[k * 256 + j];
        }
        logits[j] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    float sum_exp = 0;
    for (int j = 0; j < 256; j++) {
        out_probs[j] = expf(logits[j] - max_logit);
        sum_exp += out_probs[j];
    }
    float inv = 1.0f / sum_exp;
    for (int j = 0; j < 256; j++) {
        out_probs[j] *= inv;
    }
}

// ── Forward + backward (batched) ────────────────────────────────────────────

static double forward_backward_batch(NeuralCompressor* nc, int actual_batch) {
    const int B = actual_batch;
    const int inp = nc->inp;
    const int hid = nc->hid;

    float* embed = nc->weights + nc->off_embed;
    float* W1 = nc->weights + nc->off_w1;
    float* b1 = nc->weights + nc->off_b1;
    float* W2 = nc->weights + nc->off_w2;
    float* b2 = nc->weights + nc->off_b2;

    // ── Forward ──

    // Embedding lookup
    for (int b = 0; b < B; b++) {
        float* row = nc->act_input + b * inp;
        for (int c = 0; c < nc->ctx; c++) {
            uint8_t byte = nc->batch_ctx[b * nc->ctx + c];
            memcpy(row + c * nc->emb, embed + byte * nc->emb, nc->emb * sizeof(float));
        }
    }

    // Hidden = input @ W1 + b1
    matmul(nc->act_input, W1, nc->act_hidden, B, inp, hid);
    for (int b = 0; b < B; b++) {
        float* h = nc->act_hidden + b * hid;
        for (int j = 0; j < hid; j++) {
            h[j] += b1[j];
            h[j] = h[j] > 0 ? h[j] : 0;  // ReLU
        }
    }

    // Logits = hidden @ W2 + b2
    matmul(nc->act_hidden, W2, nc->act_logits, B, hid, 256);

    // Softmax + cross-entropy loss
    double total_loss = 0;
    for (int b = 0; b < B; b++) {
        float* logit = nc->act_logits + b * 256;
        float* prob = nc->act_probs + b * 256;

        // Add bias
        for (int j = 0; j < 256; j++) logit[j] += b2[j];

        // Softmax
        float mx = *std::max_element(logit, logit + 256);
        float sum = 0;
        for (int j = 0; j < 256; j++) {
            prob[j] = expf(logit[j] - mx);
            sum += prob[j];
        }
        float inv = 1.0f / sum;
        for (int j = 0; j < 256; j++) prob[j] *= inv;

        // Cross-entropy (nats)
        uint8_t target = nc->batch_target[b];
        float p = prob[target];
        if (p < 1e-10f) p = 1e-10f;
        total_loss -= logf(p);
    }

    // ── Backward ──

    // d_logits = (probs - one_hot) / B
    float inv_b = 1.0f / B;
    for (int b = 0; b < B; b++) {
        float* dl = nc->g_logits + b * 256;
        const float* prob = nc->act_probs + b * 256;
        uint8_t target = nc->batch_target[b];
        for (int j = 0; j < 256; j++) {
            dl[j] = (prob[j] - (j == target ? 1.0f : 0.0f)) * inv_b;
        }
    }

    // Zero gradients
    memset(nc->grads, 0, nc->n_params * sizeof(float));

    float* d_embed = nc->grads + nc->off_embed;
    float* d_W1 = nc->grads + nc->off_w1;
    float* d_b1 = nc->grads + nc->off_b1;
    float* d_W2 = nc->grads + nc->off_w2;
    float* d_b2 = nc->grads + nc->off_b2;

    // d_b2 = sum(d_logits, axis=0)
    for (int b = 0; b < B; b++) {
        const float* dl = nc->g_logits + b * 256;
        for (int j = 0; j < 256; j++) d_b2[j] += dl[j];
    }

    // d_W2 = hidden^T @ d_logits
    matmul_at_add(nc->act_hidden, nc->g_logits, d_W2, hid, B, 256);

    // d_hidden = d_logits @ W2^T, masked by ReLU
    matmul_bt(nc->g_logits, W2, nc->g_hidden, B, 256, hid);
    for (int b = 0; b < B; b++) {
        float* dh = nc->g_hidden + b * hid;
        const float* h = nc->act_hidden + b * hid;
        for (int j = 0; j < hid; j++) {
            if (h[j] <= 0) dh[j] = 0;  // ReLU mask
        }
    }

    // d_b1 = sum(d_hidden, axis=0)
    for (int b = 0; b < B; b++) {
        const float* dh = nc->g_hidden + b * hid;
        for (int j = 0; j < hid; j++) d_b1[j] += dh[j];
    }

    // d_W1 = input^T @ d_hidden
    matmul_at_add(nc->act_input, nc->g_hidden, d_W1, inp, B, hid);

    // d_input = d_hidden @ W1^T → scatter back to d_embed
    matmul_bt(nc->g_hidden, W1, nc->g_input, B, hid, inp);
    for (int b = 0; b < B; b++) {
        const float* di = nc->g_input + b * inp;
        for (int c = 0; c < nc->ctx; c++) {
            uint8_t byte = nc->batch_ctx[b * nc->ctx + c];
            float* de = d_embed + byte * nc->emb;
            const float* di_c = di + c * nc->emb;
            for (int e = 0; e < nc->emb; e++) de[e] += di_c[e];
        }
    }

    return total_loss;
}

// ── Adam optimizer step ──────────────────────────────────────────────────────

static void adam_step(NeuralCompressor* nc) {
    nc->adam_step++;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(beta1, nc->adam_step);
    float bc2 = 1.0f - powf(beta2, nc->adam_step);

    for (int i = 0; i < nc->n_params; i++) {
        float g = nc->grads[i];
        nc->adam_m[i] = beta1 * nc->adam_m[i] + (1 - beta1) * g;
        nc->adam_v[i] = beta2 * nc->adam_v[i] + (1 - beta2) * g * g;
        float m_hat = nc->adam_m[i] / bc1;
        float v_hat = nc->adam_v[i] / bc2;
        nc->weights[i] -= nc->lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ── Training ─────────────────────────────────────────────────────────────────

float neural_compressor_train(NeuralCompressor* nc, const uint8_t* data, uint64_t size, int seconds) {
    if (!nc || !data || size == 0) return 8.0f;

    int N = (int)size;
    auto start = std::chrono::steady_clock::now();
    int epoch = 0;
    double last_bpb = 8.0;

    printf("  epoch   bpb     total_bits    elapsed\n");
    printf("  ─────   ───     ──────────    ───────\n");

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= seconds) break;

        double epoch_loss = 0;

        for (int batch_start = 0; batch_start < N; batch_start += nc->batch_size) {
            int batch_end = std::min(batch_start + nc->batch_size, N);
            int B = batch_end - batch_start;

            // Fill context and target buffers
            for (int b = 0; b < B; b++) {
                int pos = batch_start + b;
                for (int c = 0; c < nc->ctx; c++) {
                    int src = pos - nc->ctx + c;
                    nc->batch_ctx[b * nc->ctx + c] = src >= 0 ? data[src] : 0;
                }
                nc->batch_target[b] = data[pos];
            }

            double batch_loss = forward_backward_batch(nc, B);
            epoch_loss += batch_loss;
            adam_step(nc);
        }

        double bits = epoch_loss / log(2.0);
        last_bpb = bits / N;
        epoch++;

        now = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double>(now - start).count();
        printf("  %5d  %6.4f  %12.0f  %6.1fs\n", epoch, last_bpb, bits, elapsed);
    }

    printf("  Training complete: %d epochs\n", epoch);
    return (float)last_bpb;
}

// ── Arithmetic coder (bit-level, Witten-Neal-Cleary style) ──────────────────

struct BitWriter {
    std::vector<uint8_t> bytes;
    uint8_t current = 0;
    int bit_pos = 7;

    void put_bit(int b) {
        if (b) current |= (1 << bit_pos);
        if (--bit_pos < 0) {
            bytes.push_back(current);
            current = 0;
            bit_pos = 7;
        }
    }

    void flush() {
        if (bit_pos < 7) bytes.push_back(current);
    }
};

struct BitReader {
    const uint8_t* data;
    size_t size;
    size_t byte_pos = 0;
    int bit_pos = 7;

    int get_bit() {
        if (byte_pos >= size) return 0;
        int b = (data[byte_pos] >> bit_pos) & 1;
        if (--bit_pos < 0) { bit_pos = 7; byte_pos++; }
        return b;
    }
};

struct ArithEncoder {
    uint32_t low = 0;
    uint32_t high = CODE_MAX;
    int pending = 0;
    BitWriter bits;

    void output_bit(int b) {
        bits.put_bit(b);
        while (pending > 0) {
            bits.put_bit(1 - b);
            pending--;
        }
    }

    void encode(const uint32_t* cum, int symbol) {
        uint64_t range = (uint64_t)high - low + 1;
        high = (uint32_t)(low + range * cum[symbol + 1] / FREQ_TOTAL - 1);
        low  = (uint32_t)(low + range * cum[symbol]     / FREQ_TOTAL);

        for (;;) {
            if (high < CODE_HALF) {
                output_bit(0);
            } else if (low >= CODE_HALF) {
                output_bit(1);
                low -= CODE_HALF;
                high -= CODE_HALF;
            } else if (low >= CODE_QUARTER && high < 3 * CODE_QUARTER) {
                pending++;
                low -= CODE_QUARTER;
                high -= CODE_QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
        }
    }

    void finish() {
        pending++;
        if (low < CODE_QUARTER) output_bit(0);
        else output_bit(1);
        // Pad with enough bits so decoder has full lookahead
        for (uint32_t i = 0; i < CODE_BITS; i++) bits.put_bit(0);
        bits.flush();
    }
};

struct ArithDecoder {
    uint32_t low = 0;
    uint32_t high = CODE_MAX;
    uint32_t code = 0;
    BitReader bits;

    void init(const uint8_t* data, size_t size) {
        bits.data = data;
        bits.size = size;
        code = 0;
        for (int i = 0; i < (int)CODE_BITS; i++) {
            code = (code << 1) | bits.get_bit();
        }
    }

    int decode(const uint32_t* cum) {
        uint64_t range = (uint64_t)high - low + 1;
        uint64_t temp = ((uint64_t)(code - low) + 1) * FREQ_TOTAL - 1;
        uint32_t scaled = (uint32_t)(temp / range);
        if (scaled >= FREQ_TOTAL) scaled = FREQ_TOTAL - 1;

        // Find symbol (linear search — 256 symbols is fine)
        int sym = 0;
        while (sym < 255 && cum[sym + 1] <= scaled) sym++;

        high = (uint32_t)(low + range * cum[sym + 1] / FREQ_TOTAL - 1);
        low  = (uint32_t)(low + range * cum[sym]     / FREQ_TOTAL);

        for (;;) {
            if (high < CODE_HALF) {
                // nothing
            } else if (low >= CODE_HALF) {
                code -= CODE_HALF;
                low -= CODE_HALF;
                high -= CODE_HALF;
            } else if (low >= CODE_QUARTER && high < 3 * CODE_QUARTER) {
                code -= CODE_QUARTER;
                low -= CODE_QUARTER;
                high -= CODE_QUARTER;
            } else {
                break;
            }
            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | bits.get_bit();
        }

        return sym;
    }
};

// ── Probability → frequency conversion ──────────────────────────────────────

static void probs_to_cum_freq(const float* probs, uint32_t* cum) {
    uint32_t freqs[256];
    uint32_t sum = 0;

    // Initial assignment — ensure minimum of 1 per symbol
    for (int i = 0; i < 256; i++) {
        freqs[i] = std::max(1u, (uint32_t)(probs[i] * FREQ_TOTAL));
        sum += freqs[i];
    }

    // Adjust to match FREQ_TOTAL exactly — modify the largest frequency
    while (sum != FREQ_TOTAL) {
        if (sum > FREQ_TOTAL) {
            // Find largest frequency > 1
            int max_i = 0;
            for (int i = 1; i < 256; i++)
                if (freqs[i] > freqs[max_i]) max_i = i;
            if (freqs[max_i] <= 1) break;
            uint32_t delta = std::min(sum - FREQ_TOTAL, freqs[max_i] - 1);
            freqs[max_i] -= delta;
            sum -= delta;
        } else {
            int max_i = 0;
            for (int i = 1; i < 256; i++)
                if (freqs[i] > freqs[max_i]) max_i = i;
            uint32_t delta = FREQ_TOTAL - sum;
            freqs[max_i] += delta;
            sum += delta;
        }
    }

    // Build cumulative
    cum[0] = 0;
    for (int i = 0; i < 256; i++) {
        cum[i + 1] = cum[i] + freqs[i];
    }
}

// ── Weight quantization ─────────────────────────────────────────────────────

struct QuantLayer {
    float min_val;
    float scale;  // (max - min) / 65535
};

static void quantize_weights(const float* weights, int count,
                              uint16_t* qweights, QuantLayer* ql) {
    float mn = weights[0], mx = weights[0];
    for (int i = 1; i < count; i++) {
        if (weights[i] < mn) mn = weights[i];
        if (weights[i] > mx) mx = weights[i];
    }

    float range = mx - mn;
    if (range < 1e-10f) range = 1e-10f;
    ql->min_val = mn;
    ql->scale = range / 65535.0f;

    for (int i = 0; i < count; i++) {
        float normalized = (weights[i] - mn) / range * 65535.0f;
        qweights[i] = (uint16_t)std::clamp((int)(normalized + 0.5f), 0, 65535);
    }
}

static void dequantize_weights(const uint16_t* qweights, int count,
                                const QuantLayer* ql, float* weights) {
    for (int i = 0; i < count; i++) {
        weights[i] = ql->min_val + qweights[i] * ql->scale;
    }
}

// ── File format ──────────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct NcmpHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t original_size;
    uint16_t context_size;
    uint16_t embed_dim;
    uint16_t hidden_dim;
    uint16_t num_layers;     // number of quantization layers (5)
    uint32_t param_count;
    uint32_t stream_size;    // arithmetic stream size in bytes
    uint32_t padding;
};
#pragma pack(pop)

// ── Compress ─────────────────────────────────────────────────────────────────

uint8_t* neural_compressor_compress(NeuralCompressor* nc,
                                     const uint8_t* data, uint64_t size,
                                     uint64_t* out_size) {
    if (!nc || !data || size == 0 || !out_size) return nullptr;

    // Layer definitions for quantization
    struct { int offset; int count; } layers[5] = {
        {nc->off_embed, 256 * nc->emb},
        {nc->off_w1, nc->inp * nc->hid},
        {nc->off_b1, nc->hid},
        {nc->off_w2, nc->hid * 256},
        {nc->off_b2, 256},
    };

    // Quantize weights (16-bit)
    std::vector<uint16_t> qweights(nc->n_params);
    QuantLayer qlayers[5];
    for (int l = 0; l < 5; l++) {
        quantize_weights(nc->weights + layers[l].offset, layers[l].count,
                          qweights.data() + layers[l].offset, &qlayers[l]);
    }

    // Create a temporary model with dequantized weights for compression
    // (so encoder and decoder use identical weights)
    std::vector<float> saved_weights(nc->weights, nc->weights + nc->n_params);
    for (int l = 0; l < 5; l++) {
        dequantize_weights(qweights.data() + layers[l].offset, layers[l].count,
                            &qlayers[l], nc->weights + layers[l].offset);
    }

    // Arithmetic coding using model predictions
    ArithEncoder enc;
    uint8_t context[64] = {};  // zero-initialized
    float probs[256];
    uint32_t cum[257];

    for (uint64_t i = 0; i < size; i++) {
        // Build context
        for (int c = 0; c < nc->ctx; c++) {
            int src = (int)i - nc->ctx + c;
            context[c] = src >= 0 ? data[src] : 0;
        }

        forward_single(nc, context, probs);
        probs_to_cum_freq(probs, cum);
        enc.encode(cum, data[i]);
    }
    enc.finish();

    // Restore original weights
    memcpy(nc->weights, saved_weights.data(), nc->n_params * sizeof(float));

    // Build output: header + quant tables + quantized weights + stream
    uint64_t header_size = sizeof(NcmpHeader);
    uint64_t quant_size = 5 * sizeof(QuantLayer);
    uint64_t weight_size = nc->n_params * sizeof(uint16_t);
    uint64_t stream_size = enc.bits.bytes.size();
    uint64_t total = header_size + quant_size + weight_size + stream_size;

    uint8_t* output = (uint8_t*)malloc(total);
    if (!output) return nullptr;

    NcmpHeader hdr = {};
    hdr.magic = NCMP_MAGIC;
    hdr.version = NCMP_VERSION;
    hdr.original_size = (uint32_t)size;
    hdr.context_size = (uint16_t)nc->ctx;
    hdr.embed_dim = (uint16_t)nc->emb;
    hdr.hidden_dim = (uint16_t)nc->hid;
    hdr.num_layers = 5;
    hdr.param_count = (uint32_t)nc->n_params;
    hdr.stream_size = (uint32_t)stream_size;

    uint8_t* p = output;
    memcpy(p, &hdr, header_size); p += header_size;
    memcpy(p, qlayers, quant_size); p += quant_size;
    memcpy(p, qweights.data(), weight_size); p += weight_size;
    memcpy(p, enc.bits.bytes.data(), stream_size);

    *out_size = total;

    printf("  Model weights: %u bytes (quantized 16-bit)\n", (uint32_t)weight_size);
    printf("  Arithmetic stream: %u bytes\n", (uint32_t)stream_size);
    printf("  Header + quant tables: %u bytes\n", (uint32_t)(header_size + quant_size));

    return output;
}

// ── Decompress ───────────────────────────────────────────────────────────────

uint8_t* neural_compressor_decompress(const uint8_t* compressed, uint64_t comp_size,
                                       uint64_t* out_size) {
    if (!compressed || comp_size < sizeof(NcmpHeader) || !out_size) return nullptr;

    // Read header
    NcmpHeader hdr;
    memcpy(&hdr, compressed, sizeof(hdr));

    if (hdr.magic != NCMP_MAGIC || hdr.version != NCMP_VERSION) {
        fprintf(stderr, "Invalid neural compression header\n");
        return nullptr;
    }

    // Create model with same architecture
    NeuralCompressorConfig cfg = {};
    cfg.context_size = hdr.context_size;
    cfg.embed_dim = hdr.embed_dim;
    cfg.hidden_dim = hdr.hidden_dim;
    cfg.batch_size = 1;  // not training
    NeuralCompressor* nc = neural_compressor_create(&cfg);
    if (!nc || nc->n_params != (int)hdr.param_count) {
        fprintf(stderr, "Model architecture mismatch (expected %d params, header says %u)\n",
                nc ? nc->n_params : 0, hdr.param_count);
        neural_compressor_destroy(nc);
        return nullptr;
    }

    // Parse quant tables + weights + stream
    const uint8_t* p = compressed + sizeof(NcmpHeader);
    uint64_t remaining = comp_size - sizeof(NcmpHeader);

    uint64_t quant_size = hdr.num_layers * sizeof(QuantLayer);
    if (remaining < quant_size) { neural_compressor_destroy(nc); return nullptr; }
    QuantLayer qlayers[5];
    memcpy(qlayers, p, quant_size);
    p += quant_size;
    remaining -= quant_size;

    uint64_t weight_bytes = (uint64_t)hdr.param_count * sizeof(uint16_t);
    if (remaining < weight_bytes) { neural_compressor_destroy(nc); return nullptr; }
    const uint16_t* qweights = (const uint16_t*)p;
    p += weight_bytes;
    remaining -= weight_bytes;

    // Dequantize weights (16-bit)
    struct { int offset; int count; } layers[5] = {
        {nc->off_embed, 256 * nc->emb},
        {nc->off_w1, nc->inp * nc->hid},
        {nc->off_b1, nc->hid},
        {nc->off_w2, nc->hid * 256},
        {nc->off_b2, 256},
    };
    for (int l = 0; l < 5; l++) {
        dequantize_weights(qweights + layers[l].offset, layers[l].count,
                            &qlayers[l], nc->weights + layers[l].offset);
    }

    // Arithmetic decoding
    const uint8_t* stream = p;
    uint64_t stream_size = remaining;

    ArithDecoder dec;
    dec.init(stream, stream_size);

    uint32_t orig_size = hdr.original_size;
    uint8_t* output = (uint8_t*)malloc(orig_size);
    if (!output) { neural_compressor_destroy(nc); return nullptr; }

    uint8_t context[64] = {};
    float probs[256];
    uint32_t cum[257];

    for (uint32_t i = 0; i < orig_size; i++) {
        for (int c = 0; c < nc->ctx; c++) {
            int src = (int)i - nc->ctx + c;
            context[c] = src >= 0 ? output[src] : 0;
        }

        forward_single(nc, context, probs);
        probs_to_cum_freq(probs, cum);
        output[i] = (uint8_t)dec.decode(cum);
    }

    neural_compressor_destroy(nc);
    *out_size = orig_size;
    return output;
}
