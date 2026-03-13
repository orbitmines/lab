// Embedding gather: for each position, lookup 8 embeddings and concatenate into input vector
// input[pos, c*emb_dim .. (c+1)*emb_dim] = embed[data[pos - ctx + c], :]

struct Params {
    n_positions: u32,  // number of positions in this mini-batch
    ctx_size: u32,     // context window (8)
    emb_dim: u32,      // embedding dimension (8)
    data_offset: u32,  // offset into data buffer for this mini-batch
    data_size: u32,    // total data size
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> data: array<u32>;       // byte data (packed as u32)
@group(0) @binding(1) var<storage, read> embed: array<f32>;      // [256, emb_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [n_positions, ctx*emb_dim]
@group(0) @binding(3) var<uniform> params: Params;

fn read_byte(idx: u32) -> u32 {
    if (idx >= params.data_size) {
        return 0u;
    }
    let word = data[idx >> 2u];
    let shift = (idx & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if (pos >= params.n_positions) {
        return;
    }

    let inp_dim = params.ctx_size * params.emb_dim;
    let data_pos = params.data_offset + pos;

    for (var c = 0u; c < params.ctx_size; c++) {
        // Context byte position: data_pos - ctx_size + c
        var byte_val = 0u;
        let src_pos = i32(data_pos) - i32(params.ctx_size) + i32(c);
        if (src_pos >= 0) {
            byte_val = read_byte(u32(src_pos));
        }

        let embed_offset = byte_val * params.emb_dim;
        let out_offset = pos * inp_dim + c * params.emb_dim;

        for (var e = 0u; e < params.emb_dim; e++) {
            output[out_offset + e] = embed[embed_offset + e];
        }
    }
}
