#include "reference_wrappers.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include <zlib.h>
#include <zstd.h>
#include <libdeflate.h>

#ifdef HAS_LZ4
#include <lz4.h>
#endif

// ── gzip ────────────────────────────────────────────────────────────────────

uint64_t ref_gzip_size(const uint8_t* data, uint64_t size, int level) {
    if (size == 0) return 0;
    z_stream strm = {};
    if (deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) return 0;
    strm.next_in = (Bytef*)data;
    strm.avail_in = (uInt)size;
    uint64_t total_out = 0;
    std::vector<uint8_t> buf(1 << 18);
    int ret;
    do {
        strm.next_out = buf.data();
        strm.avail_out = (uInt)buf.size();
        ret = deflate(&strm, Z_FINISH);
        total_out += buf.size() - strm.avail_out;
    } while (ret == Z_OK);
    deflateEnd(&strm);
    return (ret == Z_STREAM_END) ? total_out : 0;
}

int ref_gzip_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out) {
    if (!data || !out) return -1;
    z_stream strm = {};
    if (deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) return -1;
    strm.next_in = (Bytef*)data;
    strm.avail_in = (uInt)size;

    std::vector<uint8_t> result;
    std::vector<uint8_t> buf(1 << 18);
    int ret;
    do {
        strm.next_out = buf.data();
        strm.avail_out = (uInt)buf.size();
        ret = deflate(&strm, Z_FINISH);
        size_t have = buf.size() - strm.avail_out;
        result.insert(result.end(), buf.data(), buf.data() + have);
    } while (ret == Z_OK);
    deflateEnd(&strm);
    if (ret != Z_STREAM_END) return -1;

    out->data = (uint8_t*)malloc(result.size());
    if (!out->data) return -1;
    memcpy(out->data, result.data(), result.size());
    out->size = result.size();
    out->capacity = result.size();
    out->algorithm = COMP_GZIP_EXACT;
    return 0;
}

int ref_gzip_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out_p, uint64_t* out_size) {
    if (!src || !out_p || !out_size) return -1;
    z_stream strm = {};
    if (inflateInit2(&strm, 15 + 16) != Z_OK) return -1;
    strm.next_in = (Bytef*)src;
    strm.avail_in = (uInt)src_size;

    std::vector<uint8_t> result;
    std::vector<uint8_t> buf(1 << 18);
    int ret;
    do {
        strm.next_out = buf.data();
        strm.avail_out = (uInt)buf.size();
        ret = inflate(&strm, Z_NO_FLUSH);
        size_t have = buf.size() - strm.avail_out;
        result.insert(result.end(), buf.data(), buf.data() + have);
    } while (ret == Z_OK);
    inflateEnd(&strm);
    if (ret != Z_STREAM_END) return -1;

    *out_p = (uint8_t*)malloc(result.size());
    if (!*out_p) return -1;
    memcpy(*out_p, result.data(), result.size());
    *out_size = result.size();
    return 0;
}

// ── deflate (libdeflate) ────────────────────────────────────────────────────

uint64_t ref_deflate_size(const uint8_t* data, uint64_t size, int level) {
    if (size == 0) return 0;
    struct libdeflate_compressor* c = libdeflate_alloc_compressor(level);
    if (!c) return 0;
    size_t bound = libdeflate_deflate_compress_bound(c, size);
    std::vector<uint8_t> tmp(bound);
    size_t compressed = libdeflate_deflate_compress(c, data, size, tmp.data(), tmp.size());
    libdeflate_free_compressor(c);
    return compressed;
}

int ref_deflate_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out) {
    if (!data || !out) return -1;
    struct libdeflate_compressor* c = libdeflate_alloc_compressor(level);
    if (!c) return -1;
    size_t bound = libdeflate_deflate_compress_bound(c, size);
    out->data = (uint8_t*)malloc(bound);
    if (!out->data) { libdeflate_free_compressor(c); return -1; }
    size_t compressed = libdeflate_deflate_compress(c, data, size, out->data, bound);
    libdeflate_free_compressor(c);
    if (compressed == 0) { free(out->data); out->data = nullptr; return -1; }
    out->size = compressed;
    out->capacity = bound;
    out->algorithm = COMP_DEFLATE_EXACT;
    return 0;
}

int ref_deflate_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out_p, uint64_t* out_size, uint64_t expected_size) {
    if (!src || !out_p || !out_size || expected_size == 0) return -1;
    struct libdeflate_decompressor* d = libdeflate_alloc_decompressor();
    if (!d) return -1;
    *out_p = (uint8_t*)malloc(expected_size);
    if (!*out_p) { libdeflate_free_decompressor(d); return -1; }
    size_t actual = 0;
    enum libdeflate_result r = libdeflate_deflate_decompress(d, src, src_size, *out_p, expected_size, &actual);
    libdeflate_free_decompressor(d);
    if (r != LIBDEFLATE_SUCCESS) { free(*out_p); *out_p = nullptr; return -1; }
    *out_size = actual;
    return 0;
}

// ── zstd ────────────────────────────────────────────────────────────────────

uint64_t ref_zstd_size(const uint8_t* data, uint64_t size, int level) {
    if (size == 0) return 0;
    size_t bound = ZSTD_compressBound(size);
    std::vector<uint8_t> tmp(bound);
    size_t compressed = ZSTD_compress(tmp.data(), tmp.size(), data, size, level);
    if (ZSTD_isError(compressed)) return 0;
    return compressed;
}

int ref_zstd_compress(const uint8_t* data, uint64_t size, int level, CompressedBuffer* out) {
    if (!data || !out) return -1;
    size_t bound = ZSTD_compressBound(size);
    out->data = (uint8_t*)malloc(bound);
    if (!out->data) return -1;
    size_t compressed = ZSTD_compress(out->data, bound, data, size, level);
    if (ZSTD_isError(compressed)) { free(out->data); out->data = nullptr; return -1; }
    out->size = compressed;
    out->capacity = bound;
    out->algorithm = COMP_ZSTD_EXACT;
    return 0;
}

int ref_zstd_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out_p, uint64_t* out_size) {
    if (!src || !out_p || !out_size) return -1;
    unsigned long long decom_size = ZSTD_getFrameContentSize(src, src_size);
    if (decom_size == ZSTD_CONTENTSIZE_ERROR || decom_size == ZSTD_CONTENTSIZE_UNKNOWN) return -1;
    *out_p = (uint8_t*)malloc((size_t)decom_size);
    if (!*out_p) return -1;
    size_t result = ZSTD_decompress(*out_p, (size_t)decom_size, src, src_size);
    if (ZSTD_isError(result)) { free(*out_p); *out_p = nullptr; return -1; }
    *out_size = result;
    return 0;
}

// ── lz4 ─────────────────────────────────────────────────────────────────────

uint64_t ref_lz4_size(const uint8_t* data, uint64_t size) {
#ifdef HAS_LZ4
    if (size == 0) return 0;
    int bound = LZ4_compressBound((int)size);
    if (bound <= 0) return 0;
    std::vector<char> tmp(bound);
    int compressed = LZ4_compress_default((const char*)data, tmp.data(), (int)size, bound);
    return (compressed > 0) ? (uint64_t)compressed : 0;
#else
    (void)data; (void)size; return 0;
#endif
}

int ref_lz4_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out) {
#ifdef HAS_LZ4
    if (!data || !out) return -1;
    int bound = LZ4_compressBound((int)size);
    if (bound <= 0) return -1;
    out->data = (uint8_t*)malloc(bound);
    if (!out->data) return -1;
    int compressed = LZ4_compress_default((const char*)data, (char*)out->data, (int)size, bound);
    if (compressed <= 0) { free(out->data); out->data = nullptr; return -1; }
    out->size = (uint64_t)compressed;
    out->capacity = (uint64_t)bound;
    out->algorithm = COMP_LZ4_EXACT;
    return 0;
#else
    (void)data; (void)size; (void)out; return -1;
#endif
}

int ref_lz4_decompress(const uint8_t* src, uint64_t src_size, uint8_t** out_p, uint64_t* out_size, uint64_t expected_size) {
#ifdef HAS_LZ4
    if (!src || !out_p || !out_size || expected_size == 0) return -1;
    *out_p = (uint8_t*)malloc(expected_size);
    if (!*out_p) return -1;
    int result = LZ4_decompress_safe((const char*)src, (char*)*out_p, (int)src_size, (int)expected_size);
    if (result < 0) { free(*out_p); *out_p = nullptr; return -1; }
    *out_size = (uint64_t)result;
    return 0;
#else
    (void)src; (void)src_size; (void)out_p; (void)out_size; (void)expected_size; return -1;
#endif
}
