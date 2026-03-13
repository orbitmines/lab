// Zero gradient where hidden activation <= 0 (ReLU backward)

struct Params {
    count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> grad: array<f32>;     // gradient to mask
@group(0) @binding(1) var<storage, read> activations: array<f32>;    // hidden activations (post-ReLU)
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    if (activations[idx] <= 0.0) {
        grad[idx] = 0.0;
    }
}
