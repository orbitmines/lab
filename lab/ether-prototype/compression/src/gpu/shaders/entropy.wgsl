// Order-0 Shannon entropy from histogram
// Input: histogram (256 x u32 counts), params (total count)
// Output: single f32 total_bits
// Dispatch: 1 workgroup of 256 threads

@group(0) @binding(0) var<storage, read> histogram: array<u32, 256>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 1>;
@group(0) @binding(2) var<uniform> params: EntropyParams;

struct EntropyParams {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let count = histogram[lid.x];
    let total = f32(params.total);

    var bits: f32 = 0.0;
    if (count > 0u && params.total > 0u) {
        let p = f32(count) / total;
        bits = -f32(count) * log2(p);
    }

    partial_sums[lid.x] = bits;
    workgroupBarrier();

    // Parallel reduction (256 → 1)
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            partial_sums[lid.x] += partial_sums[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[0] = partial_sums[0];
    }
}
