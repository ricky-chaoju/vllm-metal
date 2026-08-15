// Header helpers for the fused MoE decode kernel (MLX affine 4-bit,
// group_size 64). Adapted from MLX's quantized.h qmv access pattern:
// x is pre-scaled by 16^-k so nibble dots need masks only, and the affine
// bias term folds in via the raw x sum. Scales may be negative — applied
// verbatim.
//
// Layout contract (mx.quantize, bits=4, group_size=64): 8 nibbles per
// uint32, LSB-first along the contraction dim; scales/biases are one fp16
// value per 64-element group.
//
// The lane tile is 32 values wide (one uint4 = 4 words = 32 nibbles per
// row per pass): the wide 16-byte weight loads clearly beat the earlier
// 2-word/uint16 tile in the geometry sweep at the Qwen3.6 decode shape
// (tools/fused_moe_microbench.py phase "shipped" re-runs the sweep on the
// pinned MLX; r16 regresses on register pressure).

template <typename T>
inline float load_x32_q4(const device T* x, thread float* x_th) {
  float s = 0.0f;
  for (int j = 0; j < 32; j += 4) {
    float a = x[j];
    float b = x[j + 1];
    float c = x[j + 2];
    float d = x[j + 3];
    s += a + b + c + d;
    x_th[j] = a;
    x_th[j + 1] = b / 16.0f;
    x_th[j + 2] = c / 256.0f;
    x_th[j + 3] = d / 4096.0f;
  }
  return s;
}

inline float qdot32_q4(uint4 w4, const thread float* x_th) {
  float acc = 0.0f;
  uint ws[4] = {w4.x, w4.y, w4.z, w4.w};
  for (int wi = 0; wi < 4; wi++) {
    uint wv = ws[wi];
    const thread float* xx = x_th + wi * 8;
    acc += xx[0] * (wv & 0x0000000fu) + xx[1] * (wv & 0x000000f0u)
         + xx[2] * (wv & 0x00000f00u) + xx[3] * (wv & 0x0000f000u)
         + xx[4] * ((wv >> 16) & 0x0000000fu) + xx[5] * ((wv >> 16) & 0x000000f0u)
         + xx[6] * ((wv >> 16) & 0x00000f00u) + xx[7] * ((wv >> 16) & 0x0000f000u);
  }
  return acc;
}
