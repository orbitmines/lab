// Column-wise sum for bias gradients: d_bias[j] += sum_i(d_data[i, j])

struct Params {
    rows: u32,
    cols: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> d_data: array<f32>;       // [rows, cols]
@group(0) @binding(1) var<storage, read_write> d_bias: array<f32>; // [cols]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.cols) {
        return;
    }

    var sum = 0.0f;
    for (var row = 0u; row < params.rows; row++) {
        sum += d_data[row * params.cols + col];
    }
    d_bias[col] += sum;
}
