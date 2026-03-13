// Scatter d_input back to d_embed: d_embed[byte_val, e] += d_input[pos, c*emb_dim + e]
// Uses inline atomic CAS loop for float addition since multiple positions may map to the same byte

struct Params {
    n_positions: u32,
    ctx_size: u32,
    emb_dim: u32,
    data_offset: u32,
    data_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read> d_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_embed: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

fn read_byte(idx: u32) -> u32 {
    if (idx >= params.data_size) { return 0u; }
    let word = data[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if (pos >= params.n_positions) { return; }

    let inp_dim = params.ctx_size * params.emb_dim;
    let data_pos = params.data_offset + pos;

    for (var c = 0u; c < params.ctx_size; c++) {
        var byte_val = 0u;
        let src_pos = i32(data_pos) - i32(params.ctx_size) + i32(c);
        if (src_pos >= 0) {
            byte_val = read_byte(u32(src_pos));
        }

        let d_input_offset = pos * inp_dim + c * params.emb_dim;
        let d_embed_offset = byte_val * params.emb_dim;

        for (var e = 0u; e < params.emb_dim; e++) {
            let grad_val = d_input[d_input_offset + e];
            let idx = d_embed_offset + e;
            // Inline atomic float add via CAS loop
            var old_val = atomicLoad(&d_embed[idx]);
            loop {
                let old_f = bitcast<f32>(old_val);
                let new_f = old_f + grad_val;
                let new_val = bitcast<u32>(new_f);
                let cas = atomicCompareExchangeWeak(&d_embed[idx], old_val, new_val);
                if (cas.exchanged) { break; }
                old_val = cas.old_value;
            }
        }
    }
}
