// Block-parallel Huffman decoder
// Each workgroup (size 1) decodes one block

const BLOCK_SIZE: u32 = 4096u;
const LOOKUP_BITS: u32 = 16u;

@group(0) @binding(0) var<storage, read> compressed: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> block_uncompressed_sizes: array<u32>;
@group(0) @binding(3) var<storage, read> decode_table: array<u32, 65536>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> params: HuffDecParams;

struct HuffDecParams {
    num_blocks: u32,
    data_size: u32,
    max_code_length: u32,
    _pad: u32,
}

fn read_bits(bit_offset: u32, count: u32) -> u32 {
    let word_idx = bit_offset / 32u;
    let bit_pos = bit_offset % 32u;

    var result: u32;
    if (bit_pos + count <= 32u) {
        result = (compressed[word_idx] >> (32u - bit_pos - count)) & ((1u << count) - 1u);
    } else {
        let from_first = 32u - bit_pos;
        let from_second = count - from_first;
        let hi = compressed[word_idx] & ((1u << from_first) - 1u);
        let lo = compressed[word_idx + 1u] >> (32u - from_second);
        result = (hi << from_second) | lo;
    }
    return result;
}

fn write_byte(pos: u32, val: u32) {
    let word_idx = pos / 4u;
    let byte_off = pos % 4u;
    let shift = byte_off * 8u;
    let current = output[word_idx];
    output[word_idx] = current | (val << shift);
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wgid: vec3u) {
    let block_idx = wgid.x;
    if (block_idx >= params.num_blocks) { return; }

    let block_start = block_idx * BLOCK_SIZE;
    let block_len = block_uncompressed_sizes[block_idx];
    var bit_pos = block_offsets[block_idx];

    // Determine how many bits to read for the lookup
    let lookup_bits = min(LOOKUP_BITS, params.max_code_length);

    for (var i = 0u; i < block_len; i++) {
        // Always read LOOKUP_BITS bits for table lookup (pad with zeros if near end)
        let prefix = read_bits(bit_pos, lookup_bits);
        // Left-justify: shift to fill LOOKUP_BITS positions
        let table_idx = prefix << (LOOKUP_BITS - lookup_bits);
        let entry = decode_table[table_idx];
        let symbol = entry >> 8u;
        let code_len = entry & 0xFFu;

        write_byte(block_start + i, symbol);
        bit_pos += code_len;
    }
}
