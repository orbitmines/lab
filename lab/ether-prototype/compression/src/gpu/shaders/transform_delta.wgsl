// Delta transform: out[i] = in[i] - in[i-1], first byte preserved.
// Each thread processes one u32 word (4 bytes). Grid-stride loop for large data.

@group(0) @binding(0) var<storage, read> original: array<u32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(2) var<uniform> params: TransformParams;

struct TransformParams {
    data_size: u32,
    param0: u32,
    param1: u32,
    param2: u32,
}

fn orig_byte(idx: u32) -> u32 {
    return (original[idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
    let stride = nwg.x * 256u;
    let word_count = (params.data_size + 3u) / 4u;
    var word_idx = gid.x;

    while (word_idx < word_count) {
        let base = word_idx * 4u;
        var out_word = 0u;
        for (var k = 0u; k < 4u; k++) {
            let idx = base + k;
            if (idx >= params.data_size) { break; }
            let cur = orig_byte(idx);
            var val: u32;
            if (idx == 0u) {
                val = cur;
            } else {
                val = (cur - orig_byte(idx - 1u)) & 0xFFu;
            }
            out_word |= val << (k * 8u);
        }
        scratch[word_idx] = out_word;
        word_idx += stride;
    }
}
