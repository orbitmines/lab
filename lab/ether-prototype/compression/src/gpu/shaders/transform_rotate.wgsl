// Rotate bits: out[i] = rotl(in[i], param0)
// param0 = rotation amount (0-7)
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(num_workgroups) nwg: vec3u) {
    let stride = nwg.x * 256u;
    let word_count = (params.data_size + 3u) / 4u;
    let rot = params.param0 & 7u;
    var word_idx = gid.x;

    while (word_idx < word_count) {
        let w = original[word_idx];
        // Rotate each byte independently
        let b0 = w & 0xFFu;
        let b1 = (w >> 8u) & 0xFFu;
        let b2 = (w >> 16u) & 0xFFu;
        let b3 = (w >> 24u) & 0xFFu;

        let r0 = ((b0 << rot) | (b0 >> (8u - rot))) & 0xFFu;
        let r1 = ((b1 << rot) | (b1 >> (8u - rot))) & 0xFFu;
        let r2 = ((b2 << rot) | (b2 >> (8u - rot))) & 0xFFu;
        let r3 = ((b3 << rot) | (b3 >> (8u - rot))) & 0xFFu;

        scratch[word_idx] = r0 | (r1 << 8u) | (r2 << 16u) | (r3 << 24u);
        word_idx += stride;
    }
}
