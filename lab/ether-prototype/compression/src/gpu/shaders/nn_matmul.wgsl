// Tiled GEMM: C = alpha * op(A) @ op(B) + beta * C
// op(X) = X if not transposed, X^T if transposed
// 16x16 tiles with shared memory

struct Params {
    M: u32,           // rows of op(A) and C
    K: u32,           // cols of op(A) = rows of op(B)
    N: u32,           // cols of op(B) and C
    transpose_a: u32, // 0 = no transpose, 1 = transpose
    transpose_b: u32,
    alpha_bits: u32,   // f32 reinterpreted as u32
    beta_bits: u32,    // f32 reinterpreted as u32
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE = 16u;

var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

fn read_a(row: u32, col: u32) -> f32 {
    if (params.transpose_a == 0u) {
        // A is [M, K], access A[row, col]
        if (row >= params.M || col >= params.K) { return 0.0; }
        return A[row * params.K + col];
    } else {
        // A is [K, M], access A[col, row] (transposed)
        if (row >= params.M || col >= params.K) { return 0.0; }
        return A[col * params.M + row];
    }
}

fn read_b(row: u32, col: u32) -> f32 {
    if (params.transpose_b == 0u) {
        // B is [K, N], access B[row, col]
        if (row >= params.K || col >= params.N) { return 0.0; }
        return B[row * params.N + col];
    } else {
        // B is [N, K], access B[col, row] (transposed)
        if (row >= params.K || col >= params.N) { return 0.0; }
        return B[col * params.K + row];
    }
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let alpha = bitcast<f32>(params.alpha_bits);
    let beta = bitcast<f32>(params.beta_bits);

    let row = wg_id.y * TILE + lid.y;
    let col = wg_id.x * TILE + lid.x;

    var acc = 0.0f;

    let n_tiles = (params.K + TILE - 1u) / TILE;

    for (var t = 0u; t < n_tiles; t++) {
        // Load tile of A
        let a_col = t * TILE + lid.x;
        tile_a[lid.y * TILE + lid.x] = read_a(row, a_col);

        // Load tile of B
        let b_row = t * TILE + lid.y;
        tile_b[lid.y * TILE + lid.x] = read_b(b_row, col);

        workgroupBarrier();

        // Accumulate
        for (var k = 0u; k < TILE; k++) {
            acc += tile_a[lid.y * TILE + k] * tile_b[k * TILE + lid.x];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        let idx = row * params.N + col;
        C[idx] = alpha * acc + beta * C[idx];
    }
}
