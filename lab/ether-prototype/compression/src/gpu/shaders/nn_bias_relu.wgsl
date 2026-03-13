// Add bias and apply ReLU in-place: data[i,j] = max(0, data[i,j] + bias[j])

struct Params {
    rows: u32,
    cols: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;  // [rows, cols]
@group(0) @binding(1) var<storage, read> bias: array<f32>;         // [cols]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;
    if (idx >= total) {
        return;
    }

    let col = idx % params.cols;
    let val = data[idx] + bias[col];
    data[idx] = max(0.0, val);
}
