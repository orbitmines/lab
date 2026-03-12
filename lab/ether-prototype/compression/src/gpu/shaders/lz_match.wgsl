// LZ77 hash-based match finding for compression size estimation
// Produces per-workgroup match statistics (not full LZ77 output)
//
// Input: data buffer, hash table (for match finding), params
// Output: per-workgroup stats: (literal_bytes, match_bytes, match_count, 0)
//
// Algorithm: greedy hash-based matching with 4-byte minimum match
// Hash table stores most recent position for each hash (atomicExchange)
// Two-phase: 1) populate hash table, 2) find matches and aggregate stats

const HASH_BITS: u32 = 18u;
const HASH_SIZE: u32 = 1u << 18u; // 262144 entries
const HASH_MASK: u32 = HASH_SIZE - 1u;
const MIN_MATCH: u32 = 4u;
const MAX_MATCH: u32 = 258u;
const MAX_DISTANCE: u32 = 32768u;

@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> hash_table: array<atomic<u32>, 262144>;
@group(0) @binding(2) var<storage, read_write> stats: array<u32>;
@group(0) @binding(3) var<uniform> params: LzParams;

struct LzParams {
    data_size: u32,
    num_workgroups: u32,
    _pad1: u32,
    _pad2: u32,
}

var<workgroup> local_stats: array<atomic<u32>, 4>;

fn read_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_off = pos % 4u;
    return (data[word_idx] >> (byte_off * 8u)) & 0xFFu;
}

fn hash4(pos: u32) -> u32 {
    if (pos + 3u >= params.data_size) { return 0u; }
    let b0 = read_byte(pos);
    let b1 = read_byte(pos + 1u);
    let b2 = read_byte(pos + 2u);
    let b3 = read_byte(pos + 3u);
    let val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    return ((val * 0x1e35a7bdu) >> (32u - HASH_BITS)) & HASH_MASK;
}

fn match_length(pos1: u32, pos2: u32) -> u32 {
    var len: u32 = 0u;
    let max_len = min(MAX_MATCH, min(params.data_size - pos1, params.data_size - pos2));
    while (len < max_len && read_byte(pos1 + len) == read_byte(pos2 + len)) {
        len += 1u;
    }
    return len;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wgid: vec3u) {

    // Clear local stats
    if (lid.x < 4u) {
        atomicStore(&local_stats[lid.x], 0u);
    }
    workgroupBarrier();

    // Each workgroup processes a chunk of the data
    let chunk_size = (params.data_size + params.num_workgroups - 1u) / params.num_workgroups;
    let chunk_start = wgid.x * chunk_size;
    let chunk_end = min(chunk_start + chunk_size, params.data_size);

    // Each thread in workgroup processes a stride of positions within the chunk
    var pos = chunk_start + lid.x;
    var skip: u32 = 0u;

    while (pos < chunk_end) {
        if (skip > 0u) {
            // Still inside a previous match — update hash but don't count
            let h = hash4(pos);
            if (pos + 3u < params.data_size) {
                atomicStore(&hash_table[h], pos);
            }
            skip -= 1u;
            pos += 256u;
            continue;
        }

        if (pos + 3u < params.data_size) {
            let h = hash4(pos);
            let prev = atomicExchange(&hash_table[h], pos);

            if (prev != 0u && prev < pos && (pos - prev) <= MAX_DISTANCE) {
                let mlen = match_length(prev, pos);
                if (mlen >= MIN_MATCH) {
                    atomicAdd(&local_stats[1], mlen); // match_bytes
                    atomicAdd(&local_stats[2], 1u);   // match_count
                    // skip would be mlen-1 but since we stride by 256, just record
                } else {
                    atomicAdd(&local_stats[0], 1u);   // literal_bytes
                }
            } else {
                atomicAdd(&local_stats[0], 1u);       // literal_bytes
            }
        } else {
            atomicAdd(&local_stats[0], 1u);           // literal_bytes (tail)
        }

        pos += 256u;
    }

    workgroupBarrier();

    // Write workgroup stats to global buffer
    if (lid.x < 4u) {
        stats[wgid.x * 4u + lid.x] = atomicLoad(&local_stats[lid.x]);
    }
}
