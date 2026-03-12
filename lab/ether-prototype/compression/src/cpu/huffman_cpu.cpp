#include "huffman_cpu.h"
#include <algorithm>
#include <cstring>
#include <queue>
#include <vector>

// ── Huffman tree building ───────────────────────────────────────────────────

struct HuffNode {
    uint64_t freq;
    int symbol; // -1 for internal
    int left, right;
};

static void compute_lengths(const std::vector<HuffNode>& tree, int node, uint32_t depth,
                            uint32_t* lengths) {
    if (tree[node].symbol >= 0) {
        lengths[tree[node].symbol] = depth > 0 ? depth : 1;
        return;
    }
    compute_lengths(tree, tree[node].left, depth + 1, lengths);
    compute_lengths(tree, tree[node].right, depth + 1, lengths);
}

static int build_lengths(const ByteHistogram* hist, uint32_t* lengths) {
    struct Entry { uint64_t freq; int idx; };
    auto cmp = [](const Entry& a, const Entry& b) { return a.freq > b.freq; };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> pq(cmp);
    std::vector<HuffNode> tree;

    memset(lengths, 0, 256 * sizeof(uint32_t));

    for (int i = 0; i < 256; i++) {
        if (hist->histogram[i] == 0) continue;
        int idx = (int)tree.size();
        tree.push_back({hist->histogram[i], i, -1, -1});
        pq.push({hist->histogram[i], idx});
    }

    int num_symbols = (int)pq.size();
    if (num_symbols == 0) return 0;
    if (num_symbols == 1) {
        lengths[tree[pq.top().idx].symbol] = 1;
        return 1;
    }

    while (pq.size() > 1) {
        auto a = pq.top(); pq.pop();
        auto b = pq.top(); pq.pop();
        int idx = (int)tree.size();
        tree.push_back({a.freq + b.freq, -1, a.idx, b.idx});
        pq.push({a.freq + b.freq, idx});
    }

    compute_lengths(tree, pq.top().idx, 0, lengths);
    return num_symbols;
}

// ── Canonical Huffman codes from lengths ─────────────────────────────────────

struct CanonCode {
    uint32_t code;
    uint32_t length;
};

static void build_canonical_codes(const uint32_t* lengths, CanonCode* codes) {
    // Sort symbols by (length, symbol)
    struct SymLen { int sym; uint32_t len; };
    std::vector<SymLen> syms;
    for (int i = 0; i < 256; i++) {
        if (lengths[i] > 0) syms.push_back({i, lengths[i]});
    }
    std::sort(syms.begin(), syms.end(), [](const SymLen& a, const SymLen& b) {
        return a.len < b.len || (a.len == b.len && a.sym < b.sym);
    });

    memset(codes, 0, 256 * sizeof(CanonCode));
    if (syms.empty()) return;

    uint32_t code = 0;
    uint32_t prev_len = syms[0].len;
    codes[syms[0].sym] = {0, syms[0].len};

    for (size_t i = 1; i < syms.size(); i++) {
        code++;
        if (syms[i].len > prev_len) {
            code <<= (syms[i].len - prev_len);
        }
        codes[syms[i].sym] = {code, syms[i].len};
        prev_len = syms[i].len;
    }
}

// ── Size estimation ─────────────────────────────────────────────────────────

double cpu_huffman_size(const ByteHistogram* hist, uint32_t* codelengths_out) {
    uint32_t lengths[256] = {};
    int n = build_lengths(hist, lengths);
    if (codelengths_out) memcpy(codelengths_out, lengths, sizeof(lengths));

    if (n == 0) return 0.0;
    if (n == 1) return (double)hist->total;

    double bits = 0.0;
    for (int i = 0; i < 256; i++) {
        bits += (double)hist->histogram[i] * lengths[i];
    }
    return bits;
}

// ── Bitstream writer/reader ─────────────────────────────────────────────────

struct BitWriter {
    std::vector<uint8_t> buf;
    uint64_t accum = 0;
    int bits = 0;

    void write(uint32_t code, uint32_t nbits) {
        accum = (accum << nbits) | code;
        bits += nbits;
        while (bits >= 8) {
            bits -= 8;
            buf.push_back((uint8_t)(accum >> bits));
            accum &= (1ULL << bits) - 1;
        }
    }

    void flush() {
        if (bits > 0) {
            buf.push_back((uint8_t)(accum << (8 - bits)));
            bits = 0;
            accum = 0;
        }
    }
};

struct BitReader {
    const uint8_t* data;
    uint64_t size;
    uint64_t pos = 0;
    uint64_t accum = 0;
    int bits = 0;

    int read_bit() {
        if (bits == 0) {
            if (pos >= size) return -1;
            accum = data[pos++];
            bits = 8;
        }
        bits--;
        return (accum >> bits) & 1;
    }
};

// ── Compression ─────────────────────────────────────────────────────────────
// Format:
// [1 byte]  num_symbols (256 encoded as 0 with flag)
// [1 byte]  flags: bit0 = has_256_symbols
// [8 bytes] original_size (LE uint64)
// For each symbol present (sorted by symbol):
//   [1 byte] symbol
//   [1 byte] code length
// [N bytes] canonical Huffman bitstream

int cpu_huffman_compress(const uint8_t* data, uint64_t size, CompressedBuffer* out) {
    if (!data || !out) return -1;

    ByteHistogram hist;
    memset(&hist, 0, sizeof(hist));
    hist.total = size;
    for (uint64_t i = 0; i < size; i++) hist.histogram[data[i]]++;

    uint32_t lengths[256] = {};
    int num_symbols = build_lengths(&hist, lengths);

    CanonCode codes[256] = {};
    build_canonical_codes(lengths, codes);

    // Header
    std::vector<uint8_t> header;
    uint8_t flags = 0;
    uint8_t nsym_byte = (uint8_t)(num_symbols & 0xFF);
    if (num_symbols == 256) flags |= 1;
    header.push_back(nsym_byte);
    header.push_back(flags);
    for (int i = 0; i < 8; i++) header.push_back((uint8_t)(size >> (i * 8)));

    for (int i = 0; i < 256; i++) {
        if (lengths[i] == 0) continue;
        header.push_back((uint8_t)i);
        header.push_back((uint8_t)lengths[i]);
    }

    // Encode using canonical codes
    BitWriter bw;
    for (uint64_t i = 0; i < size; i++) {
        bw.write(codes[data[i]].code, codes[data[i]].length);
    }
    bw.flush();

    uint64_t total = header.size() + bw.buf.size();
    out->data = (uint8_t*)malloc(total);
    if (!out->data) return -1;
    memcpy(out->data, header.data(), header.size());
    memcpy(out->data + header.size(), bw.buf.data(), bw.buf.size());
    out->size = total;
    out->capacity = total;
    out->algorithm = COMP_HUFFMAN;
    return 0;
}

int cpu_huffman_decompress(const uint8_t* src, uint64_t src_size,
                           uint8_t** out_data, uint64_t* out_size) {
    if (!src || src_size < 10 || !out_data || !out_size) return -1;

    uint64_t pos = 0;
    int nsym = src[pos++];
    uint8_t flags = src[pos++];
    if (flags & 1) nsym = 256;

    uint64_t orig_size = 0;
    for (int i = 0; i < 8; i++) orig_size |= (uint64_t)src[pos++] << (i * 8);

    if (nsym == 0 || orig_size == 0) {
        *out_data = nullptr; *out_size = 0; return 0;
    }

    // Read code lengths
    uint32_t lengths[256] = {};
    for (int i = 0; i < nsym; i++) {
        if (pos + 2 > src_size) return -1;
        uint8_t sym = src[pos++];
        lengths[sym] = src[pos++];
    }

    // Rebuild canonical codes
    CanonCode codes[256] = {};
    build_canonical_codes(lengths, codes);

    // Build decode tree
    struct DecNode { int children[2]; int symbol; };
    std::vector<DecNode> dtree(1, {{-1, -1}, -1});

    for (int i = 0; i < 256; i++) {
        if (codes[i].length == 0) continue;
        int node = 0;
        for (int b = (int)codes[i].length - 1; b >= 0; b--) {
            int bit = (codes[i].code >> b) & 1;
            if (dtree[node].children[bit] < 0) {
                dtree[node].children[bit] = (int)dtree.size();
                dtree.push_back({{-1, -1}, -1});
            }
            node = dtree[node].children[bit];
        }
        dtree[node].symbol = i;
    }

    // Decode
    uint8_t* output = (uint8_t*)malloc(orig_size);
    if (!output) return -1;

    BitReader br{src + pos, src_size - pos};
    for (uint64_t i = 0; i < orig_size; i++) {
        int node = 0;
        while (dtree[node].symbol < 0) {
            int bit = br.read_bit();
            if (bit < 0) { free(output); return -1; }
            node = dtree[node].children[bit];
            if (node < 0) { free(output); return -1; }
        }
        output[i] = (uint8_t)dtree[node].symbol;
    }

    *out_data = output;
    *out_size = orig_size;
    return 0;
}
