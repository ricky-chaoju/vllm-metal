// Fused MoE decode gate/up/SwiGLU kernel (body-only; helpers come from
// moe_q4_dot.metal via the mx.fast.metal_kernel header).
//
// Template params: T (fp16/bf16), K (in dim), N (out dim), TOPK,
//   ROWS_PER_SG, SIMDGROUPS
// Inputs: x [T_tokens, K], gate_w/up_w [E, N, K/8] uint32,
//   gate_scales/gate_biases/up_scales/up_biases [E, N, K/64] T,
//   expert_ids [T_tokens * TOPK] int32 (row-major flattened top-k)
// Output: h [T_tokens * TOPK, N] T, h = silu(gate_e(x)) * up_e(x)
// Grid: (32, N / ROWS_PER_SG, T_tokens * TOPK) threads,
//   threadgroup (32, SIMDGROUPS, 1). One simdgroup owns ROWS_PER_SG rows
//   of one (token, expert) pair; lanes tile the contraction dim in
//   1024-value blocks (one uint4 = 32 nibbles per lane per row per block).

constexpr int K_WORDS = K / 8;
constexpr int K_GROUPS = K / 64;
constexpr int BLOCKS = K / 1024;

uint pair = threadgroup_position_in_grid.z;
uint tok = pair / TOPK;
uint e = (uint)expert_ids[pair];
uint row0 = threadgroup_position_in_grid.y * (SIMDGROUPS * ROWS_PER_SG)
          + simdgroup_index_in_threadgroup * ROWS_PER_SG;
uint lane = thread_index_in_simdgroup;

const device uint32_t* gw_base = gate_w + ((size_t)e * N + row0) * K_WORDS;
const device uint32_t* uw_base = up_w + ((size_t)e * N + row0) * K_WORDS;
const device T* gs_base = gate_scales + ((size_t)e * N + row0) * K_GROUPS;
const device T* gb_base = gate_biases + ((size_t)e * N + row0) * K_GROUPS;
const device T* us_base = up_scales + ((size_t)e * N + row0) * K_GROUPS;
const device T* ub_base = up_biases + ((size_t)e * N + row0) * K_GROUPS;
const device T* x_base = x + (size_t)tok * K;

float g_acc[ROWS_PER_SG];
float u_acc[ROWS_PER_SG];
for (int r = 0; r < ROWS_PER_SG; r++) {
  g_acc[r] = 0.0f;
  u_acc[r] = 0.0f;
}
thread float x_th[32];

for (int blk = 0; blk < BLOCKS; blk++) {
  int xoff = blk * 1024 + lane * 32;
  float x_sum = load_x32_q4(x_base + xoff, x_th);
  // A 32-value lane tile spans half a 64-value quant group, so the group
  // index derives from the absolute offset.
  int gidx = (blk * 1024 + lane * 32) / 64;
  int woff = blk * 128 + lane * 4;
  for (int r = 0; r < ROWS_PER_SG; r++) {
    uint4 gw4 = *((const device uint4*)(gw_base + (size_t)r * K_WORDS + woff));
    uint4 uw4 = *((const device uint4*)(uw_base + (size_t)r * K_WORDS + woff));
    size_t so = (size_t)r * K_GROUPS + gidx;
    g_acc[r] += (float)gs_base[so] * qdot32_q4(gw4, x_th)
              + x_sum * (float)gb_base[so];
    u_acc[r] += (float)us_base[so] * qdot32_q4(uw4, x_th)
              + x_sum * (float)ub_base[so];
  }
}

for (int r = 0; r < ROWS_PER_SG; r++) {
  float g = simd_sum(g_acc[r]);
  float u = simd_sum(u_acc[r]);
  if (lane == 0) {
    float sg = g / (1.0f + metal::exp(-g));
    h[(size_t)pair * N + row0 + r] = (T)(sg * u);
  }
}
