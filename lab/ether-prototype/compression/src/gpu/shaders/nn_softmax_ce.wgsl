// Fused softmax + cross-entropy loss + d_logits computation
// Adds bias to logits before softmax.
// One workgroup per row (position). 256 threads = 1 per class.
// Uses 2D dispatch to handle > 65535 positions: row = wg_id.y * num_wg_x + wg_id.x

struct Params {
    n_positions: u32,
    n_classes: u32,
    num_wg_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> loss_acc: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

var<workgroup> reduction_buf: array<f32, 256>;

fn get_target_byte(pos: u32) -> u32 {
    let word = targets[pos >> 2u];
    let shift = (pos & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let row = wg_id.y * params.num_wg_x + wg_id.x;
    if (row >= params.n_positions) {
        return;
    }

    let j = lid;
    let base = row * 256u;

    let logit_val = logits[base + j] + bias[j];

    // Max reduction
    reduction_buf[lid] = logit_val;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            reduction_buf[lid] = max(reduction_buf[lid], reduction_buf[lid + stride]);
        }
        workgroupBarrier();
    }
    let max_val = reduction_buf[0];
    workgroupBarrier();

    // Exp + sum reduction
    let exp_val = exp(logit_val - max_val);
    reduction_buf[lid] = exp_val;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            reduction_buf[lid] = reduction_buf[lid] + reduction_buf[lid + stride];
        }
        workgroupBarrier();
    }
    let sum_exp = reduction_buf[0];
    workgroupBarrier();

    let prob = exp_val / sum_exp;
    let tgt = get_target_byte(row);

    // Accumulate loss via inline CAS loop
    if (j == tgt) {
        let p_clamped = max(prob, 1e-10);
        let loss_val = -log(p_clamped);
        var old_val = atomicLoad(&loss_acc[0]);
        loop {
            let old_f = bitcast<f32>(old_val);
            let new_f = old_f + loss_val;
            let new_val = bitcast<u32>(new_f);
            let cas = atomicCompareExchangeWeak(&loss_acc[0], old_val, new_val);
            if (cas.exchanged) { break; }
            old_val = cas.old_value;
        }
    }

    // d_logits = (prob - one_hot) / N
    var one_hot = 0.0f;
    if (j == tgt) { one_hot = 1.0; }
    let inv_n = 1.0 / f32(params.n_positions);
    logits[base + j] = (prob - one_hot) * inv_n;
}
