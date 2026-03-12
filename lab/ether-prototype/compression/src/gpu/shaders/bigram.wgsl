// Order-1 bigram co-occurrence histogram
// Input: data buffer (as u32 words), params
// Output: 256x256 = 65536 u32 counts (row-major: bigram[prev][cur])
// Uses global atomics — no workgroup shared mem (too large for 256x256)

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> bigram: array<atomic<u32>, 65536>;
@group(0) @binding(2) var<uniform> params: BigramParams;

struct BigramParams {
    data_size: u32,   // total bytes
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_off = pos % 4u;
    return (data[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let stride = 256u * 256u; // total threads
    var idx = gid.x;

    while (idx + 1u < params.data_size) {
        let prev = read_byte(idx);
        let cur = read_byte(idx + 1u);
        atomicAdd(&bigram[prev * 256u + cur], 1u);
        idx += stride;
    }
}
