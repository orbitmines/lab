// Byte histogram shader: counts frequency of each byte value (0-255)
// Input: storage buffer of raw bytes (as u32 words, 4 bytes each)
// Output: storage buffer of 256 x u32 counts

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    data_size: u32,  // total bytes
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u) {
    // Clear local histogram
    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();

    // Each thread processes multiple u32 words (4 bytes each)
    let words_count = (params.data_size + 3u) / 4u;
    let stride = 256u * 256u; // total threads across all workgroups
    var idx = gid.x;

    while (idx < words_count) {
        let word = data[idx];
        let base_byte = idx * 4u;

        // Extract 4 bytes from the u32 word
        if (base_byte < params.data_size) {
            atomicAdd(&local_hist[word & 0xFFu], 1u);
        }
        if (base_byte + 1u < params.data_size) {
            atomicAdd(&local_hist[(word >> 8u) & 0xFFu], 1u);
        }
        if (base_byte + 2u < params.data_size) {
            atomicAdd(&local_hist[(word >> 16u) & 0xFFu], 1u);
        }
        if (base_byte + 3u < params.data_size) {
            atomicAdd(&local_hist[(word >> 24u) & 0xFFu], 1u);
        }

        idx += stride;
    }

    workgroupBarrier();

    // Merge local histogram into global
    atomicAdd(&histogram[lid.x], atomicLoad(&local_hist[lid.x]));
}
