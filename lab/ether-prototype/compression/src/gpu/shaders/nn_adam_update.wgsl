// Per-parameter Adam optimizer step
// w -= lr * m_hat / (sqrt(v_hat) + eps)

struct Params {
    n_params: u32,
    step: u32,         // adam step count (for bias correction)
    lr_bits: u32,      // learning rate as bitcast u32
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;  // [n_params]
@group(0) @binding(1) var<storage, read_write> grads: array<f32>;    // [n_params]
@group(0) @binding(2) var<storage, read_write> adam_m: array<f32>;   // [n_params]
@group(0) @binding(3) var<storage, read_write> adam_v: array<f32>;   // [n_params]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_params) {
        return;
    }

    let lr = bitcast<f32>(params.lr_bits);
    let beta1 = 0.9f;
    let beta2 = 0.999f;
    let eps = 1e-8f;

    let g = grads[idx];

    // Update moments
    let m = beta1 * adam_m[idx] + (1.0 - beta1) * g;
    let v = beta2 * adam_v[idx] + (1.0 - beta2) * g * g;

    adam_m[idx] = m;
    adam_v[idx] = v;

    // Bias correction
    let step_f = f32(params.step);
    let bc1 = 1.0 - pow(beta1, step_f);
    let bc2 = 1.0 - pow(beta2, step_f);

    let m_hat = m / bc1;
    let v_hat = v / bc2;

    weights[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
}
