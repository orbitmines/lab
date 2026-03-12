// Block-parallel Huffman encoder
// Each workgroup encodes one block of data
//
// Input:
//   data: raw bytes (as u32 words)
//   code_table: 256 entries, each packed as (code << 8) | length
//   block_offsets: output bit offset for each block (computed by CPU from block sizes)
//   params: block_size, num_blocks, data_size
//
// Output:
//   compressed: packed bit output buffer (as u32 words)
//   block_sizes: compressed size in bits for each block (written in size-only mode)
//
// Two modes controlled by params.mode:
//   mode 0: size-only — just compute block_sizes
//   mode 1: encode — write compressed data at block_offsets

const BLOCK_SIZE: u32 = 4096u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> code_table: array<u32, 256>;
@group(0) @binding(2) var<storage, read_write> compressed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> block_sizes: array<u32>;
@group(0) @binding(4) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(5) var<uniform> params: HuffEncParams;

struct HuffEncParams {
    block_size: u32,
    num_blocks: u32,
    data_size: u32,
    mode: u32,       // 0 = size-only, 1 = encode
}

var<workgroup> partial_sizes: array<u32, 256>;

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_off = pos % 4u;
    return (data[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

fn write_bits(bit_offset: u32, code: u32, length: u32) {
    // Write 'length' bits of 'code' at 'bit_offset' in compressed buffer
    // Handle word boundaries
    var remaining = length;
    var bits = code;
    var offset = bit_offset;

    while (remaining > 0u) {
        let word_idx = offset / 32u;
        let bit_pos = offset % 32u;
        let can_write = min(remaining, 32u - bit_pos);

        // Extract the top 'can_write' bits from our remaining bits
        let shift = remaining - can_write;
        let mask = ((1u << can_write) - 1u);
        let value = (bits >> shift) & mask;

        // OR into the output word at the correct position
        atomicOr(&compressed[word_idx], value << (32u - bit_pos - can_write));

        remaining -= can_write;
        offset += can_write;
        bits &= (1u << shift) - 1u;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wgid: vec3u) {

    let block_idx = wgid.x;
    if (block_idx >= params.num_blocks) { return; }

    let block_start = block_idx * params.block_size;
    let block_end = min(block_start + params.block_size, params.data_size);
    let block_len = block_end - block_start;

    // Phase 1: Each thread computes total bits for its portion of the block
    var my_bits: u32 = 0u;
    var pos = lid.x;
    while (pos < block_len) {
        let byte_val = read_byte(block_start + pos);
        let entry = code_table[byte_val];
        let length = entry & 0xFFu;
        my_bits += length;
        pos += 256u;
    }

    partial_sizes[lid.x] = my_bits;
    workgroupBarrier();

    // Parallel reduction for total block size
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            partial_sizes[lid.x] += partial_sizes[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        block_sizes[block_idx] = partial_sizes[0];
    }

    // If size-only mode, we're done
    if (params.mode == 0u) { return; }

    // Phase 2: Encode — thread 0 does sequential encoding for correctness
    // (bit packing is inherently sequential within a block)
    if (lid.x == 0u) {
        let base_offset = block_offsets[block_idx];
        var bit_pos = base_offset;

        for (var i = 0u; i < block_len; i++) {
            let byte_val = read_byte(block_start + i);
            let entry = code_table[byte_val];
            let code = entry >> 8u;
            let length = entry & 0xFFu;
            write_bits(bit_pos, code, length);
            bit_pos += length;
        }
    }
}
