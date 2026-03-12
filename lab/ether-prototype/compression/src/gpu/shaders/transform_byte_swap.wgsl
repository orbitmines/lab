// Swap adjacent byte pairs: ABCD -> BADC
// Self-inverse (applying twice gives back original).
// Each thread processes one u32 word. Grid-stride loop for large data.

@group(0) @binding(0) var<storage, read> original: array<u32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(2) var<uniform> params: TransformParams;

struct TransformParams {
    data_size: u32,
    param0: u32,
    param1: u32,
    param2: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
    let stride = nwg.x * 256u;
    let word_count = (params.data_size + 3u) / 4u;
    var word_idx = gid.x;

    while (word_idx < word_count) {
        let byte_base = word_idx * 4u;
        let w = original[word_idx];
        let b0 = w & 0xFFu;
        let b1 = (w >> 8u) & 0xFFu;
        let b2 = (w >> 16u) & 0xFFu;
        let b3 = (w >> 24u) & 0xFFu;

        var out_word = (b1) | (b0 << 8u) | (b3 << 16u) | (b2 << 24u);

        // Handle trailing bytes
        if (byte_base + 1u >= params.data_size) {
            out_word = w;
        } else if (byte_base + 3u >= params.data_size) {
            out_word = (b1) | (b0 << 8u) | (b2 << 16u) | (b3 << 24u);
        }

        scratch[word_idx] = out_word;
        word_idx += stride;
    }
}
