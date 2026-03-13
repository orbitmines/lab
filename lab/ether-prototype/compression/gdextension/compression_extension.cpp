#include "compression_extension.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

// Algorithm name <-> enum mapping

static const struct { const char* name; CompressionAlgorithm algo; } ALGO_MAP[] = {
    {"shannon_order0", COMP_SHANNON_ORDER0},
    {"shannon_order1", COMP_SHANNON_ORDER1},
    {"huffman",        COMP_HUFFMAN},
    {"arithmetic",     COMP_ARITHMETIC},
    {"ans",            COMP_ANS},
    {"gzip_est",       COMP_GZIP_EST},
    {"zstd_est",       COMP_ZSTD_EST},
    {"lz4_est",        COMP_LZ4_EST},
    {"gzip",           COMP_GZIP_EXACT},
    {"zstd",           COMP_ZSTD_EXACT},
    {"lz4",            COMP_LZ4_EXACT},
    {"deflate",        COMP_DEFLATE_EXACT},
};
static const int ALGO_MAP_COUNT = sizeof(ALGO_MAP) / sizeof(ALGO_MAP[0]);

static bool name_to_algo(const StringName& name, CompressionAlgorithm* out) {
    String s = String(name).to_lower();
    for (int i = 0; i < ALGO_MAP_COUNT; i++) {
        if (s == ALGO_MAP[i].name) {
            *out = ALGO_MAP[i].algo;
            return true;
        }
    }
    return false;
}

static const char* algo_to_name(CompressionAlgorithm a) {
    for (int i = 0; i < ALGO_MAP_COUNT; i++) {
        if (ALGO_MAP[i].algo == a) return ALGO_MAP[i].name;
    }
    return "unknown";
}

static Dictionary result_to_dict(const EstimateResult& r) {
    Dictionary d;
    d["algorithm"] = algo_to_name(r.algorithm);
    d["bits"] = (int64_t)r.estimated_bits;
    d["bytes"] = (int64_t)r.estimated_bytes;
    d["bits_per_byte"] = r.bits_per_byte;
    d["ratio"] = r.ratio;
    return d;
}

// ── CompressionEstimator implementation ─────────────────────────────────────

CompressionEstimator::CompressionEstimator() {
    est = estimator_create(1); // try GPU
}

CompressionEstimator::~CompressionEstimator() {
    if (neural) neural_compressor_destroy(neural);
    if (session) gpu_session_destroy(session);
    if (est) estimator_destroy(est);
}

void CompressionEstimator::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_data", "data"), &CompressionEstimator::load_data);
    ClassDB::bind_method(D_METHOD("load_file", "path"), &CompressionEstimator::load_file);
    ClassDB::bind_method(D_METHOD("get_histogram"), &CompressionEstimator::get_histogram);
    ClassDB::bind_method(D_METHOD("estimate", "algorithm"), &CompressionEstimator::estimate);
    ClassDB::bind_method(D_METHOD("estimate_all"), &CompressionEstimator::estimate_all);
    ClassDB::bind_method(D_METHOD("compress", "algorithm", "level"), &CompressionEstimator::compress, DEFVAL(0));
    ClassDB::bind_static_method("CompressionEstimator",
        D_METHOD("decompress", "algorithm", "compressed_data", "expected_size"),
        &CompressionEstimator::decompress, DEFVAL(0));
    ClassDB::bind_method(D_METHOD("get_backend_info"), &CompressionEstimator::get_backend_info);
    ClassDB::bind_method(D_METHOD("get_parallel_slots"), &CompressionEstimator::get_parallel_slots);
    ClassDB::bind_method(D_METHOD("create_session"), &CompressionEstimator::create_session);
    ClassDB::bind_method(D_METHOD("destroy_session"), &CompressionEstimator::destroy_session);
    ClassDB::bind_method(D_METHOD("get_session_slots"), &CompressionEstimator::get_session_slots);
    ClassDB::bind_method(D_METHOD("evaluate_batch", "transforms"), &CompressionEstimator::evaluate_batch);
    ClassDB::bind_method(D_METHOD("neural_create", "context_size", "embed_dim", "hidden_dim"),
                         &CompressionEstimator::neural_create, DEFVAL(8), DEFVAL(8), DEFVAL(64));
    ClassDB::bind_method(D_METHOD("neural_destroy"), &CompressionEstimator::neural_destroy);
    ClassDB::bind_method(D_METHOD("neural_train", "seconds"), &CompressionEstimator::neural_train, DEFVAL(60));
    ClassDB::bind_method(D_METHOD("neural_train_gpu", "seconds"), &CompressionEstimator::neural_train_gpu, DEFVAL(60));
    ClassDB::bind_method(D_METHOD("neural_compress"), &CompressionEstimator::neural_compress);
    ClassDB::bind_static_method("CompressionEstimator",
        D_METHOD("neural_decompress", "compressed_data"),
        &CompressionEstimator::neural_decompress);
}

Error CompressionEstimator::load_data(const PackedByteArray& data) {
    if (data.is_empty()) return ERR_INVALID_PARAMETER;
    int rc = estimator_load(est, data.ptr(), data.size());
    return rc == 0 ? OK : FAILED;
}

Error CompressionEstimator::load_file(const String& path) {
    CharString utf8 = path.utf8();
    int rc = estimator_load_file(est, utf8.get_data());
    return rc == 0 ? OK : FAILED;
}

Dictionary CompressionEstimator::get_histogram() {
    ByteHistogram hist;
    if (estimator_histogram(est, &hist) != 0) return Dictionary();
    Dictionary d;
    d["total"] = (int64_t)hist.total;
    Dictionary bins;
    for (int i = 0; i < 256; i++) {
        if (hist.histogram[i] > 0)
            bins[i] = (int64_t)hist.histogram[i];
    }
    d["bins"] = bins;
    return d;
}

Dictionary CompressionEstimator::estimate(const StringName& algorithm) {
    CompressionAlgorithm algo;
    if (!name_to_algo(algorithm, &algo)) {
        UtilityFunctions::push_error("Unknown algorithm: ", algorithm);
        return Dictionary();
    }
    EstimateResult r;
    if (estimator_estimate(est, algo, &r) != 0) return Dictionary();
    return result_to_dict(r);
}

Dictionary CompressionEstimator::estimate_all() {
    EstimateAllResult all;
    if (estimator_estimate_all(est, &all) != 0) return Dictionary();
    Dictionary d;
    d["input_size"] = (int64_t)all.input_size;
    Dictionary results;
    for (int i = 0; i < all.count; i++) {
        results[algo_to_name(all.results[i].algorithm)] = result_to_dict(all.results[i]);
    }
    d["results"] = results;
    return d;
}

PackedByteArray CompressionEstimator::compress(const StringName& algorithm, int level) {
    CompressionAlgorithm algo;
    if (!name_to_algo(algorithm, &algo)) {
        UtilityFunctions::push_error("Unknown algorithm: ", algorithm);
        return PackedByteArray();
    }
    CompressedBuffer buf = {};
    if (estimator_compress(est, algo, level, &buf) != 0) return PackedByteArray();
    PackedByteArray result;
    result.resize(buf.size);
    memcpy(result.ptrw(), buf.data, buf.size);
    compressed_buffer_free(&buf);
    return result;
}

PackedByteArray CompressionEstimator::decompress(const StringName& algorithm,
                                                   const PackedByteArray& compressed_data,
                                                   int64_t expected_size) {
    CompressionAlgorithm algo;
    if (!name_to_algo(algorithm, &algo)) {
        UtilityFunctions::push_error("Unknown algorithm: ", algorithm);
        return PackedByteArray();
    }
    uint8_t* out = nullptr;
    uint64_t out_size = 0;
    int rc = estimator_decompress(algo, compressed_data.ptr(), compressed_data.size(),
                                   &out, &out_size, (uint64_t)expected_size);
    if (rc != 0 || !out) return PackedByteArray();
    PackedByteArray result;
    result.resize(out_size);
    memcpy(result.ptrw(), out, out_size);
    ::free(out);
    return result;
}

Dictionary CompressionEstimator::get_backend_info() {
    BackendInfo info;
    if (estimator_get_backend_info(est, &info) != 0) return Dictionary();
    Dictionary d;
    d["backend"] = info.backend == COMP_BACKEND_WEBGPU ? "webgpu" : "cpu";
    d["device_name"] = info.device_name;
    d["vendor"] = info.vendor;
    d["max_buffer_size"] = (int64_t)info.max_buffer_size;
    d["max_slots"] = (int)info.max_slots;
    return d;
}

int CompressionEstimator::get_parallel_slots() {
    if (!est) return 0;
    return (int)estimator_parallel_slots(est);
}

// ── GPU session methods ─────────────────────────────────────────────────────

static const struct { const char* name; TransformType type; } TRANSFORM_MAP[] = {
    {"delta",       TRANSFORM_DELTA},
    {"xor",         TRANSFORM_XOR_PREV},
    {"rotate",      TRANSFORM_ROTATE_BITS},
    {"sub_mean",    TRANSFORM_SUB_MEAN},
    {"byte_swap",   TRANSFORM_BYTE_SWAP},
};
static const int TRANSFORM_MAP_COUNT = sizeof(TRANSFORM_MAP) / sizeof(TRANSFORM_MAP[0]);

static const char* transform_to_name(TransformType t) {
    for (int i = 0; i < TRANSFORM_MAP_COUNT; i++) {
        if (TRANSFORM_MAP[i].type == t) return TRANSFORM_MAP[i].name;
    }
    return "unknown";
}

static bool name_to_transform(const String& name, TransformType* out) {
    String s = name.to_lower();
    for (int i = 0; i < TRANSFORM_MAP_COUNT; i++) {
        if (s == TRANSFORM_MAP[i].name) {
            *out = TRANSFORM_MAP[i].type;
            return true;
        }
    }
    return false;
}

Error CompressionEstimator::create_session() {
    if (!est) return ERR_UNCONFIGURED;
    if (session) { gpu_session_destroy(session); session = nullptr; }

    // Get data from estimator — we need to re-read the loaded data
    // The estimator stores a copy internally, but we access it via estimator_load
    // For now, get the GPU context and data from the estimator internals
    // We need the estimator to expose its data... let's use a workaround:
    // Re-load the data by getting histogram first (ensures data is loaded)
    ByteHistogram hist;
    if (estimator_histogram(est, &hist) != 0) return FAILED;

    GpuContext* gpu = estimator_get_gpu(est);
    uint64_t data_size = 0;
    const uint8_t* data = estimator_get_data(est, &data_size);

    if (!gpu || !data || data_size == 0) return FAILED;

    session = gpu_session_create(gpu, data, data_size);
    return session ? OK : FAILED;
}

void CompressionEstimator::destroy_session() {
    if (session) { gpu_session_destroy(session); session = nullptr; }
}

int CompressionEstimator::get_session_slots() {
    return session ? (int)gpu_session_num_slots(session) : 0;
}

Array CompressionEstimator::evaluate_batch(const Array& transforms) {
    Array results;
    if (!session || transforms.is_empty()) return results;

    int count = transforms.size();
    TransformDesc* descs = new TransformDesc[count];
    int valid_count = 0;

    for (int i = 0; i < count; i++) {
        Dictionary td = transforms[i];
        TransformType type;
        String type_name = td.get("type", "");
        if (!name_to_transform(type_name, &type)) {
            UtilityFunctions::push_error("Unknown transform type: ", type_name);
            delete[] descs;
            return results;
        }
        descs[valid_count].type = type;
        descs[valid_count].param0 = (uint32_t)(int)td.get("param0", 0);
        descs[valid_count].param1 = (uint32_t)(int)td.get("param1", 0);
        descs[valid_count].param2 = (uint32_t)(int)td.get("param2", 0);
        valid_count++;
    }

    SlotScore* scores = new SlotScore[valid_count];
    int rc = gpu_session_evaluate_batch(session, descs, valid_count, scores);

    if (rc == 0) {
        for (int i = 0; i < valid_count; i++) {
            Dictionary d;
            d["transform"] = transform_to_name(scores[i].transform);
            Dictionary params;
            params["param0"] = (int)scores[i].params[0];
            params["param1"] = (int)scores[i].params[1];
            params["param2"] = (int)scores[i].params[2];
            d["params"] = params;
            d["entropy_o0_bpb"] = (double)scores[i].entropy_o0_bpb;
            d["entropy_o0_total"] = scores[i].entropy_o0_total;
            results.append(d);
        }
    }

    delete[] descs;
    delete[] scores;
    return results;
}

// ── Neural compression methods ──────────────────────────────────────────────

Error CompressionEstimator::neural_create(int context_size, int embed_dim, int hidden_dim) {
    if (neural) { neural_compressor_destroy(neural); neural = nullptr; }
    NeuralCompressorConfig cfg = {};
    cfg.context_size = context_size;
    cfg.embed_dim = embed_dim;
    cfg.hidden_dim = hidden_dim;
    cfg.learning_rate = 0.001f;
    cfg.batch_size = 4096;
    neural = neural_compressor_create(&cfg);
    return neural ? OK : FAILED;
}

void CompressionEstimator::neural_destroy() {
    if (neural) { neural_compressor_destroy(neural); neural = nullptr; }
}

float CompressionEstimator::neural_train(int seconds) {
    if (!neural || !est) return 8.0f;
    uint64_t data_size = 0;
    const uint8_t* data = estimator_get_data(est, &data_size);
    if (!data || data_size == 0) return 8.0f;
    return neural_compressor_train(neural, data, data_size, seconds);
}

float CompressionEstimator::neural_train_gpu(int seconds) {
    if (!neural || !est) return 8.0f;
    uint64_t data_size = 0;
    const uint8_t* data = estimator_get_data(est, &data_size);
    if (!data || data_size == 0) return 8.0f;
    GpuContext* gpu = estimator_get_gpu(est);
    if (!gpu) return neural_compressor_train(neural, data, data_size, seconds);
    return neural_compressor_train_gpu(neural, data, data_size, seconds, gpu);
}

Dictionary CompressionEstimator::neural_compress() {
    Dictionary result;
    if (!neural || !est) return result;

    uint64_t data_size = 0;
    const uint8_t* data = estimator_get_data(est, &data_size);
    if (!data || data_size == 0) return result;

    uint64_t comp_size = 0;
    uint8_t* compressed = neural_compressor_compress(neural, data, data_size, &comp_size);
    if (!compressed) return result;

    PackedByteArray comp_data;
    comp_data.resize(comp_size);
    memcpy(comp_data.ptrw(), compressed, comp_size);
    ::free(compressed);

    result["compressed"] = comp_data;
    result["original_size"] = (int64_t)data_size;
    result["compressed_size"] = (int64_t)comp_size;
    result["ratio"] = (double)comp_size / data_size;
    result["param_count"] = (int)neural_compressor_param_count(neural);
    return result;
}

PackedByteArray CompressionEstimator::neural_decompress(const PackedByteArray& compressed_data) {
    if (compressed_data.is_empty()) return PackedByteArray();
    uint64_t out_size = 0;
    uint8_t* out = neural_compressor_decompress(compressed_data.ptr(), compressed_data.size(), &out_size);
    if (!out) return PackedByteArray();
    PackedByteArray result;
    result.resize(out_size);
    memcpy(result.ptrw(), out, out_size);
    ::free(out);
    return result;
}
