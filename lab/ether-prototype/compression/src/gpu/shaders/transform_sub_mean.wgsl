// Subtract running mean: out[i] = in[i] - mean(in[i-W..i-1])
// param0 = window size W (1-256)
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
    let w = max(params.param0, 1u);
    var word_idx = gid.x;

    while (word_idx < word_count) {
        let base = word_idx * 4u;
        var out_word = 0u;
        for (var k = 0u; k < 4u; k++) {
            let idx = base + k;
            if (idx >= params.data_size) { break; }

            let start = select(0u, idx - w, idx >= w);
            let count = idx - start;

            var sum = 0u;
            for (var j = start; j < idx; j++) {
                sum += orig_byte(j);
            }

            let mean_val = select(0u, sum / count, count > 0u);
            let cur = orig_byte(idx);
            let val = (cur - mean_val) & 0xFFu;
            out_word |= val << (k * 8u);
        }
        scratch[word_idx] = out_word;
        word_idx += stride;
    }
}
