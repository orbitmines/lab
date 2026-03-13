#ifndef COMPRESSION_EXTENSION_H
#define COMPRESSION_EXTENSION_H

#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include "compression/estimator.h"
#include "compression/gpu_context.h"
#include "compression/gpu_session.h"
#include "compression/neural_compressor.h"
#include "compression/neural_train_session.h"

using namespace godot;

class CompressionEstimator : public RefCounted {
    GDCLASS(CompressionEstimator, RefCounted);

    Estimator* est = nullptr;
    GpuSession* session = nullptr;
    NeuralCompressor* neural = nullptr;

protected:
    static void _bind_methods();

public:
    CompressionEstimator();
    ~CompressionEstimator();

    // Load data from a PackedByteArray
    Error load_data(const PackedByteArray& data);

    // Load data from a file path
    Error load_file(const String& path);

    // Get byte histogram as Dictionary {byte_value: count}
    Dictionary get_histogram();

    // Estimate compressed size for a single algorithm
    // Returns Dictionary with: bits, bytes, bits_per_byte, ratio
    Dictionary estimate(const StringName& algorithm);

    // Estimate all algorithms
    // Returns Dictionary {algorithm_name: {bits, bytes, bits_per_byte, ratio}}
    Dictionary estimate_all();

    // Actually compress data with a given algorithm
    // Returns PackedByteArray of compressed data (empty on error)
    PackedByteArray compress(const StringName& algorithm, int level = 0);

    // Decompress data
    // Returns PackedByteArray of decompressed data (empty on error)
    static PackedByteArray decompress(const StringName& algorithm,
                                       const PackedByteArray& compressed_data,
                                       int64_t expected_size = 0);

    // Backend info
    Dictionary get_backend_info();

    // Number of parallel transformation slots for current data size
    int get_parallel_slots();

    // GPU-resident session: create session from loaded data, evaluate transforms on GPU
    Error create_session();
    void destroy_session();
    int get_session_slots();
    Array evaluate_batch(const Array& transforms);

    // Neural compression: the model IS the compression algorithm
    Error neural_create(int context_size, int embed_dim, int hidden_dim);
    void neural_destroy();
    float neural_train(int seconds);
    float neural_train_gpu(int seconds);
    Dictionary neural_compress();
    static PackedByteArray neural_decompress(const PackedByteArray& compressed_data);
};

#endif
