# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode kernel bench harness.

Phases (run: python tools/fused_moe_microbench.py <phase>):
baseline | gateup | down | serial | serial2 | compile | shipped

- baseline: dispatch truth at REAL dims — gather_qmm M=1 top-8 (unsorted +
  sorted) vs a Python loop of 8x quantized_matmul on contiguous expert
  slices. Decides how much of the win is custom-matvec vs fusion.
- shipped: the SHIPPED FusedMoEDecodeKernels path (uint4 lane tile through
  its compiled step) against stock SwitchGLU at the serve decode shape,
  plus a ROWS_PER_SG/SIMDGROUPS sweep of the shipped kernel. Serial
  per-call timing understates in-graph behavior — treat the e2e serve A/B
  as the value case and this phase as the kernel-level reference.

Real dims: E=256 experts, gate/up [256, 512, 2048], down [256, 2048, 512],
fp16, MLX affine 4-bit group_size=64.
"""

from __future__ import annotations

import os
import sys
import time

import mlx.core as mx

E = 256
TOP_K = 8
HIDDEN = 2048
INTER = 512
GROUP = 64
BITS = 4
ITERS = 100
WARMUP = 20


def _timeit(fn) -> float:
    for _ in range(WARMUP):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(ITERS)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / ITERS


def _quant_stacked(rows: int, cols: int):
    w = mx.random.normal((E * rows, cols), dtype=mx.float16)
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    return (
        wq.reshape(E, rows, -1),
        s.reshape(E, rows, -1),
        b.reshape(E, rows, -1),
    )


def _qbytes(rows: int, cols: int) -> float:
    return rows * cols * BITS / 8 + rows * (cols // GROUP) * 4


def _load_router_trace():
    """Router indices for trace-replayed phases.

    Set MOE_TRACE_NPY to a ``(steps, top_k)`` int array captured from a real
    serve (log ``inds`` in SwitchGLU.__call__ for a decode run) to replay real
    router behavior; without it the phases use uniform-random unsorted top-8
    draws, which match the real traces' dispatch pattern (unsorted, high
    entropy) but not their expert-reuse distribution.
    """
    import numpy as np

    trace_path = os.environ.get("MOE_TRACE_NPY")
    if trace_path:
        return np.load(trace_path)
    rng = np.random.default_rng(0)
    steps = 512
    return np.stack(
        [rng.choice(E, size=TOP_K, replace=False) for _ in range(steps)]
    ).astype(np.uint32)


def phase_baseline() -> None:
    gw, gs, gb = _quant_stacked(INTER, HIDDEN)
    x = mx.random.normal((1, 1, 1, 1, HIDDEN), dtype=mx.float16)
    idx = mx.array([[[3, 17, 42, 91, 130, 175, 201, 250]]], dtype=mx.uint32)
    mx.eval(gw, gs, gb, x, idx)
    per_call_bytes = TOP_K * _qbytes(INTER, HIDDEN)

    dt = _timeit(
        lambda: mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
    )
    print(
        f"(a) gather_qmm unsorted M=1 top8: {dt * 1e6:7.1f} us  "
        f"{per_call_bytes / dt / 1e9:6.1f} GB/s"
    )

    dt = _timeit(
        lambda: mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            sorted_indices=True,
        )
    )
    print(
        f"(c) gather_qmm sorted   M=1 top8: {dt * 1e6:7.1f} us  "
        f"{per_call_bytes / dt / 1e9:6.1f} GB/s"
    )

    x2 = mx.random.normal((1, HIDDEN), dtype=mx.float16)
    expert_list = [3, 17, 42, 91, 130, 175, 201, 250]
    mx.eval(x2)

    def loop_qmv():
        outs = []
        for e in expert_list:
            outs.append(
                mx.quantized_matmul(
                    x2,
                    gw[e],
                    gs[e],
                    gb[e],
                    transpose=True,
                    group_size=GROUP,
                    bits=BITS,
                )
            )
        return mx.concatenate(outs, axis=0)

    dt = _timeit(loop_qmv)
    print(
        f"(b) 8x quantized_matmul slices  : {dt * 1e6:7.1f} us  "
        f"{per_call_bytes / dt / 1e9:6.1f} GB/s  (8 dispatches)"
    )

    # Reference: the FULL current per-layer chain (gate+up+swiglu+down).
    dw, ds, db = _quant_stacked_down()
    mx.eval(dw, ds, db)

    def full_chain():
        g = mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        u = mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        h = mx.multiply(g * mx.sigmoid(g), u)
        return mx.gather_qmm(
            h,
            dw,
            ds,
            db,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )

    dt = _timeit(full_chain)
    layer_bytes = 2 * per_call_bytes + TOP_K * _qbytes(HIDDEN, INTER)
    print(
        f"(d) full layer chain (3 qmm+act): {dt * 1e6:7.1f} us  "
        f"{layer_bytes / dt / 1e9:6.1f} GB/s  -> x40 layers = "
        f"{dt * 40 * 1e3:.2f} ms/step"
    )

    # (e) down gather alone + (f) full chain with SORTED indices.
    dt = _timeit(
        lambda: mx.gather_qmm(
            mx.random.normal((1, 1, TOP_K, 1, INTER), dtype=mx.float16),
            dw,
            ds,
            db,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
    )
    print(f"(e) down gather alone (K=512)   : {dt * 1e6:7.1f} us")

    def full_chain_sorted():
        g = mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            sorted_indices=True,
        )
        u = mx.gather_qmm(
            x,
            gw,
            gs,
            gb,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            sorted_indices=True,
        )
        h = mx.multiply(g * mx.sigmoid(g), u)
        return mx.gather_qmm(
            h,
            dw,
            ds,
            db,
            rhs_indices=idx,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
            sorted_indices=True,
        )

    dt = _timeit(full_chain_sorted)
    print(
        f"(f) full chain SORTED           : {dt * 1e6:7.1f} us  -> x40 = "
        f"{dt * 40 * 1e3:.2f} ms/step"
    )


def _quant_stacked_down():
    w = mx.random.normal((E * HIDDEN, INTER), dtype=mx.float16)
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    return (
        wq.reshape(E, HIDDEN, -1),
        s.reshape(E, HIDDEN, -1),
        b.reshape(E, HIDDEN, -1),
    )


_KERNEL_A_HEADER = """
template <typename T>
inline float load_vector_q4(const device T* x, thread float* x_th) {
  float s = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    float a = x[i];
    float b = x[i + 1];
    float c = x[i + 2];
    float d = x[i + 3];
    s += a + b + c + d;
    x_th[i] = a;
    x_th[i + 1] = b / 16.0f;
    x_th[i + 2] = c / 256.0f;
    x_th[i + 3] = d / 4096.0f;
  }
  return s;
}

inline float qdot_q4(const device uint16_t* w, const thread float* x_th,
                     float scale, float bias, float x_sum) {
  float accum = 0.0f;
  for (int i = 0; i < 4; i++) {
    accum += x_th[4 * i] * (w[i] & 0x000f)
           + x_th[4 * i + 1] * (w[i] & 0x00f0)
           + x_th[4 * i + 2] * (w[i] & 0x0f00)
           + x_th[4 * i + 3] * (w[i] & 0xf000);
  }
  return scale * accum + x_sum * bias;
}
"""

_KERNEL_A_SOURCE = """
  constexpr int K_WORDS = K / 8;
  constexpr int K_GROUPS = K / 64;
  constexpr int BLOCKS = K / 512;

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
  thread float x_th[16];

  for (int blk = 0; blk < BLOCKS; blk++) {
    int xoff = blk * 512 + lane * 16;
    float x_sum = load_vector_q4(x_base + xoff, x_th);
    int gidx = blk * 8 + lane / 4;
    int woff = blk * 64 + lane * 2;
    for (int r = 0; r < ROWS_PER_SG; r++) {
      const device uint16_t* gwp =
          (const device uint16_t*)(gw_base + (size_t)r * K_WORDS + woff);
      const device uint16_t* uwp =
          (const device uint16_t*)(uw_base + (size_t)r * K_WORDS + woff);
      float gsv = gs_base[(size_t)r * K_GROUPS + gidx];
      float gbv = gb_base[(size_t)r * K_GROUPS + gidx];
      float usv = us_base[(size_t)r * K_GROUPS + gidx];
      float ubv = ub_base[(size_t)r * K_GROUPS + gidx];
      g_acc[r] += qdot_q4(gwp, x_th, gsv, gbv, x_sum);
      u_acc[r] += qdot_q4(uwp, x_th, usv, ubv, x_sum);
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
"""


def phase_gateup() -> None:
    import numpy as np

    traces = _load_router_trace()
    gw, gs, gb = _quant_stacked(INTER, HIDDEN)
    uw, us, ub = _quant_stacked(INTER, HIDDEN)
    x2 = mx.random.normal((1, HIDDEN), dtype=mx.float16)
    mx.eval(gw, gs, gb, uw, us, ub, x2)

    kernel = mx.fast.metal_kernel(
        name="moe_gateup_swiglu_decode_bench",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "expert_ids",
        ],
        output_names=["h"],
        header=_KERNEL_A_HEADER,
        source=_KERNEL_A_SOURCE,
    )

    def run_kernel(ids_arr, rows_per_sg, simdgroups):
        p = int(ids_arr.size)
        return kernel(
            inputs=[x2, gw, gs, gb, uw, us, ub, ids_arr],
            template=[
                ("T", mx.float16),
                ("K", HIDDEN),
                ("N", INTER),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", rows_per_sg),
                ("SIMDGROUPS", simdgroups),
            ],
            grid=(32, INTER // rows_per_sg, p),
            threadgroup=(32, simdgroups, 1),
            output_shapes=[(p, INTER)],
            output_dtypes=[mx.float16],
        )[0]

    # ---- parity vs reference on 5 real traces ----
    x5 = x2.reshape(1, 1, 1, 1, HIDDEN)
    worst_cos, worst_abs = 1.0, 0.0
    for i in range(5):
        ids = mx.array(traces[i].astype(np.int32))
        idx5 = mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32))
        g = mx.gather_qmm(
            x5,
            gw,
            gs,
            gb,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        u = mx.gather_qmm(
            x5,
            uw,
            us,
            ub,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        ref = (mx.multiply(g * mx.sigmoid(g), u)).reshape(TOP_K, INTER)
        out = run_kernel(ids, 4, 2)
        mx.eval(ref, out)
        r = np.array(ref, dtype=np.float32).ravel()
        o = np.array(out, dtype=np.float32).ravel()
        cos = float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12))
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, float(np.max(np.abs(r - o))))
    print(f"parity: worst cosine={worst_cos:.6f}  worst |diff|={worst_abs:.4f}")

    # ---- bench: kernel A vs 2x gather_qmm + swiglu, real indices ----
    ids_list = [mx.array(traces[i].astype(np.int32)) for i in range(200)]
    idx5_list = [
        mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32)) for i in range(200)
    ]
    mx.eval(*ids_list, *idx5_list)

    def ref_group(i):
        g = mx.gather_qmm(
            x5,
            gw,
            gs,
            gb,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        u = mx.gather_qmm(
            x5,
            uw,
            us,
            ub,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        return mx.multiply(g * mx.sigmoid(g), u)

    dt_ref = _timeit_idx(lambda i: ref_group(i))
    print(f"reference gate+up+swiglu (real idx): {dt_ref * 1e6:7.1f} us")
    for rps in (2, 4):
        for sgs in (2, 4):
            dt = _timeit_idx(lambda i, r=rps, s=sgs: run_kernel(ids_list[i], r, s))
            print(f"kernel A ROWS_PER_SG={rps} SIMDGROUPS={sgs}: {dt * 1e6:7.1f} us")


def _timeit_idx(fn, iters=300, warmup=50):
    for i in range(warmup):
        mx.eval(fn(i % 200))
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn(i % 200) for i in range(iters)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


_KERNEL_B_SOURCE = """
  constexpr int K_WORDS = K / 8;
  constexpr int K_GROUPS = K / 64;
  constexpr int BLOCKS = K / 512;

  uint pair = threadgroup_position_in_grid.z;
  uint e = (uint)expert_ids[pair];
  uint row0 = threadgroup_position_in_grid.y * (SIMDGROUPS * ROWS_PER_SG)
            + simdgroup_index_in_threadgroup * ROWS_PER_SG;
  uint lane = thread_index_in_simdgroup;

  const device uint32_t* w_base = down_w + ((size_t)e * N + row0) * K_WORDS;
  const device T* s_base = down_scales + ((size_t)e * N + row0) * K_GROUPS;
  const device T* b_base = down_biases + ((size_t)e * N + row0) * K_GROUPS;
  const device T* x_base = h + (size_t)pair * K;

  float acc[ROWS_PER_SG];
  for (int r = 0; r < ROWS_PER_SG; r++) {
    acc[r] = 0.0f;
  }
  thread float x_th[16];

  for (int blk = 0; blk < BLOCKS; blk++) {
    int xoff = blk * 512 + lane * 16;
    float x_sum = load_vector_q4(x_base + xoff, x_th);
    int gidx = blk * 8 + lane / 4;
    int woff = blk * 64 + lane * 2;
    for (int r = 0; r < ROWS_PER_SG; r++) {
      const device uint16_t* wp =
          (const device uint16_t*)(w_base + (size_t)r * K_WORDS + woff);
      acc[r] += qdot_q4(wp, x_th,
                        (float)s_base[(size_t)r * K_GROUPS + gidx],
                        (float)b_base[(size_t)r * K_GROUPS + gidx], x_sum);
    }
  }

  for (int r = 0; r < ROWS_PER_SG; r++) {
    float v = simd_sum(acc[r]);
    if (lane == 0) {
      y[(size_t)pair * N + row0 + r] = (T)v;
    }
  }
"""


def phase_down() -> None:
    import numpy as np

    traces = _load_router_trace()
    dw, ds, db = _quant_stacked_down()
    hin = mx.random.normal((TOP_K, INTER), dtype=mx.float16)
    mx.eval(dw, ds, db, hin)

    kernel = mx.fast.metal_kernel(
        name="moe_down_decode_bench",
        input_names=["h", "down_w", "down_scales", "down_biases", "expert_ids"],
        output_names=["y"],
        header=_KERNEL_A_HEADER,
        source=_KERNEL_B_SOURCE,
    )

    def run_kernel(ids_arr, rows_per_sg, simdgroups):
        p = int(ids_arr.size)
        return kernel(
            inputs=[hin, dw, ds, db, ids_arr],
            template=[
                ("T", mx.float16),
                ("K", INTER),
                ("N", HIDDEN),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", rows_per_sg),
                ("SIMDGROUPS", simdgroups),
            ],
            grid=(32, HIDDEN // rows_per_sg, p),
            threadgroup=(32, simdgroups, 1),
            output_shapes=[(p, HIDDEN)],
            output_dtypes=[mx.float16],
        )[0]

    h5 = hin.reshape(1, 1, TOP_K, 1, INTER)
    worst_cos, worst_rel = 1.0, 0.0
    for i in range(5):
        ids = mx.array(traces[i].astype(np.int32))
        idx5 = mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32))
        ref = mx.gather_qmm(
            h5,
            dw,
            ds,
            db,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        ).reshape(TOP_K, HIDDEN)
        out = run_kernel(ids, 4, 2)
        mx.eval(ref, out)
        r = np.array(ref, dtype=np.float32).ravel()
        o = np.array(out, dtype=np.float32).ravel()
        cos = float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12))
        worst_cos = min(worst_cos, cos)
        rel = float(np.max(np.abs(r - o)) / (np.max(np.abs(r)) + 1e-9))
        worst_rel = max(worst_rel, rel)
    print(f"B parity: worst cosine={worst_cos:.6f}  worst rel-diff={worst_rel:.5f}")

    ids_list = [mx.array(traces[i].astype(np.int32)) for i in range(200)]
    idx5_list = [
        mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32)) for i in range(200)
    ]
    mx.eval(*ids_list, *idx5_list)

    dt_ref = _timeit_idx(
        lambda i: mx.gather_qmm(
            h5,
            dw,
            ds,
            db,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
    )
    print(f"reference down gather (real idx): {dt_ref * 1e6:7.1f} us")
    for rps in (2, 4, 8):
        for sgs in (1, 2, 4):
            dt = _timeit_idx(lambda i, r=rps, s=sgs: run_kernel(ids_list[i], r, s))
            print(f"kernel B ROWS_PER_SG={rps} SIMDGROUPS={sgs}: {dt * 1e6:7.1f} us")


def phase_serial() -> None:
    """Serial-latency bench: dependency-chained calls, the real decode metric."""
    import numpy as np

    traces = _load_router_trace()
    gw, gs, gb = _quant_stacked(INTER, HIDDEN)
    uw, us, ub = _quant_stacked(INTER, HIDDEN)
    dw, ds, db = _quant_stacked_down()
    x0 = mx.random.normal((1, HIDDEN), dtype=mx.float16)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db, x0)

    ka = mx.fast.metal_kernel(
        name="moe_gateup_swiglu_decode_bench",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "expert_ids",
        ],
        output_names=["h"],
        header=_KERNEL_A_HEADER,
        source=_KERNEL_A_SOURCE,
    )
    kb = mx.fast.metal_kernel(
        name="moe_down_decode_bench",
        input_names=["h", "down_w", "down_scales", "down_biases", "expert_ids"],
        output_names=["y"],
        header=_KERNEL_A_HEADER,
        source=_KERNEL_B_SOURCE,
    )

    ids_list = [mx.array(traces[i].astype(np.int32)) for i in range(200)]
    idx5_list = [
        mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32)) for i in range(200)
    ]
    mx.eval(*ids_list, *idx5_list)

    def run_ka(x_flat, i, rps=4, sgs=2):
        return ka(
            inputs=[x_flat, gw, gs, gb, uw, us, ub, ids_list[i]],
            template=[
                ("T", mx.float16),
                ("K", HIDDEN),
                ("N", INTER),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", rps),
                ("SIMDGROUPS", sgs),
            ],
            grid=(32, INTER // rps, TOP_K),
            threadgroup=(32, sgs, 1),
            output_shapes=[(TOP_K, INTER)],
            output_dtypes=[mx.float16],
        )[0]

    def run_kb(h_flat, i, rps=4, sgs=2):
        return kb(
            inputs=[h_flat, dw, ds, db, ids_list[i]],
            template=[
                ("T", mx.float16),
                ("K", INTER),
                ("N", HIDDEN),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", rps),
                ("SIMDGROUPS", sgs),
            ],
            grid=(32, HIDDEN // rps, TOP_K),
            threadgroup=(32, sgs, 1),
            output_shapes=[(TOP_K, HIDDEN)],
            output_dtypes=[mx.float16],
        )[0]

    def ref_gateup(x5, i):
        g = mx.gather_qmm(
            x5,
            gw,
            gs,
            gb,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        u = mx.gather_qmm(
            x5,
            uw,
            us,
            ub,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        return mx.multiply(g * mx.sigmoid(g), u)

    def ref_down(h5, i):
        return mx.gather_qmm(
            h5,
            dw,
            ds,
            db,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )

    def bench_serial(step, iters=200, warmup=30):
        x = x0
        for i in range(warmup):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        t0 = time.perf_counter()
        x = x0
        for i in range(iters):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    def dep(x_flat, out):
        # cheap scalar dependency: forces serial ordering across iterations
        return x_flat + out.astype(mx.float16).reshape(-1)[:1] * mx.array(
            0.0, dtype=mx.float16
        )

    def step_ref(x_flat, i):
        x5 = x_flat.reshape(1, 1, 1, 1, HIDDEN)
        h = ref_gateup(x5, i)
        y = ref_down(h, i)
        return dep(x_flat, y)

    def step_a_refdown(x_flat, i):
        h = run_ka(x_flat, i)
        y = ref_down(h.reshape(1, 1, TOP_K, 1, INTER), i)
        return dep(x_flat, y)

    def step_refgateup_b(x_flat, i):
        x5 = x_flat.reshape(1, 1, 1, 1, HIDDEN)
        h = ref_gateup(x5, i)
        y = run_kb(h.reshape(TOP_K, INTER), i)
        return dep(x_flat, y)

    def step_ab(x_flat, i):
        h = run_ka(x_flat, i)
        y = run_kb(h, i)
        return dep(x_flat, y)

    for name, fn in (
        ("ref chain (3 gathers + act)", step_ref),
        ("kernelA + ref down        ", step_a_refdown),
        ("ref gate/up + kernelB     ", step_refgateup_b),
        ("kernelA + kernelB         ", step_ab),
    ):
        dt = bench_serial(fn)
        print(
            f"SERIAL {name}: {dt * 1e6:7.1f} us/layer -> x40 = {dt * 40 * 1e3:.2f} ms/step"
        )


_KERNEL_A2_HEADER = (
    _KERNEL_A_HEADER
    + """
inline float qdot_q4_u2(uint2 w2, const thread float* x_th,
                        float scale, float bias, float x_sum) {
  float a = 0.0f;
  uint w0 = w2.x & 0xffffu;
  uint w1 = w2.x >> 16;
  uint wq = w2.y & 0xffffu;
  uint w3 = w2.y >> 16;
  a += x_th[0] * (w0 & 0x000fu) + x_th[1] * (w0 & 0x00f0u)
     + x_th[2] * (w0 & 0x0f00u) + x_th[3] * (w0 & 0xf000u);
  a += x_th[4] * (w1 & 0x000fu) + x_th[5] * (w1 & 0x00f0u)
     + x_th[6] * (w1 & 0x0f00u) + x_th[7] * (w1 & 0xf000u);
  a += x_th[8] * (wq & 0x000fu) + x_th[9] * (wq & 0x00f0u)
     + x_th[10] * (wq & 0x0f00u) + x_th[11] * (wq & 0xf000u);
  a += x_th[12] * (w3 & 0x000fu) + x_th[13] * (w3 & 0x00f0u)
     + x_th[14] * (w3 & 0x0f00u) + x_th[15] * (w3 & 0xf000u);
  return scale * a + x_sum * bias;
}

template <typename T>
inline float load_vector_q4_v2(const device T* x, thread float* x_th) {
  const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
  float s = 0.0f;
  for (int i = 0; i < 4; i++) {
    vec<T, 4> v = x4[i];
    float a = v.x;
    float b = v.y;
    float c = v.z;
    float d = v.w;
    s += a + b + c + d;
    x_th[4 * i] = a;
    x_th[4 * i + 1] = b / 16.0f;
    x_th[4 * i + 2] = c / 256.0f;
    x_th[4 * i + 3] = d / 4096.0f;
  }
  return s;
}
"""
)

_KERNEL_A2_SOURCE = """
  constexpr int K_WORDS = K / 8;
  constexpr int K_GROUPS = K / 64;
  constexpr int BLOCKS = K / 512;

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
  thread float x_th[16];

  for (int blk = 0; blk < BLOCKS; blk++) {
    int xoff = blk * 512 + lane * 16;
    float x_sum = load_vector_q4_v2(x_base + xoff, x_th);
    int gidx = blk * 8 + lane / 4;
    int w2off = (blk * 64 + lane * 2) / 2;
    uint2 gvals[ROWS_PER_SG];
    uint2 uvals[ROWS_PER_SG];
    for (int r = 0; r < ROWS_PER_SG; r++) {
      gvals[r] = ((const device uint2*)(gw_base + (size_t)r * K_WORDS))[w2off];
      uvals[r] = ((const device uint2*)(uw_base + (size_t)r * K_WORDS))[w2off];
    }
    for (int r = 0; r < ROWS_PER_SG; r++) {
      size_t so = (size_t)r * K_GROUPS + gidx;
      g_acc[r] += qdot_q4_u2(gvals[r], x_th, (float)gs_base[so],
                             (float)gb_base[so], x_sum);
      u_acc[r] += qdot_q4_u2(uvals[r], x_th, (float)us_base[so],
                             (float)ub_base[so], x_sum);
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
"""


def phase_serial2() -> None:
    """A v2 (vectorized loads) serial-latency vs v1 and ref."""
    import numpy as np

    traces = _load_router_trace()
    gw, gs, gb = _quant_stacked(INTER, HIDDEN)
    uw, us, ub = _quant_stacked(INTER, HIDDEN)
    dw, ds, db = _quant_stacked_down()
    x0 = mx.random.normal((1, HIDDEN), dtype=mx.float16)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db, x0)

    ka2 = mx.fast.metal_kernel(
        name="moe_gateup_swiglu_decode_bench_v2",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "expert_ids",
        ],
        output_names=["h"],
        header=_KERNEL_A2_HEADER,
        source=_KERNEL_A2_SOURCE,
    )

    ids_list = [mx.array(traces[i].astype(np.int32)) for i in range(200)]
    idx5_list = [
        mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32)) for i in range(200)
    ]
    mx.eval(*ids_list, *idx5_list)

    def run_ka2(x_flat, i, rps, sgs):
        return ka2(
            inputs=[x_flat, gw, gs, gb, uw, us, ub, ids_list[i]],
            template=[
                ("T", mx.float16),
                ("K", HIDDEN),
                ("N", INTER),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", rps),
                ("SIMDGROUPS", sgs),
            ],
            grid=(32, INTER // rps, TOP_K),
            threadgroup=(32, sgs, 1),
            output_shapes=[(TOP_K, INTER)],
            output_dtypes=[mx.float16],
        )[0]

    def ref_down(h5, i):
        return mx.gather_qmm(
            h5,
            dw,
            ds,
            db,
            rhs_indices=idx5_list[i],
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )

    # parity spot check vs v1 reference math
    x5 = x0.reshape(1, 1, 1, 1, HIDDEN)
    g = mx.gather_qmm(
        x5,
        gw,
        gs,
        gb,
        rhs_indices=idx5_list[0],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
    )
    u = mx.gather_qmm(
        x5,
        uw,
        us,
        ub,
        rhs_indices=idx5_list[0],
        transpose=True,
        group_size=GROUP,
        bits=BITS,
    )
    ref = (mx.multiply(g * mx.sigmoid(g), u)).reshape(TOP_K, INTER)
    out = run_ka2(x0, 0, 4, 2)
    mx.eval(ref, out)
    r = np.array(ref, dtype=np.float32).ravel()
    o = np.array(out, dtype=np.float32).ravel()
    cos = float(np.dot(r, o) / (np.linalg.norm(r) * np.linalg.norm(o) + 1e-12))
    print(f"A2 parity cosine={cos:.6f}")

    def bench_serial(step, iters=200, warmup=30):
        x = x0
        for i in range(warmup):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        t0 = time.perf_counter()
        x = x0
        for i in range(iters):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    def dep(x_flat, out):
        return x_flat + out.astype(mx.float16).reshape(-1)[:1] * mx.array(
            0.0, dtype=mx.float16
        )

    for rps, sgs in ((4, 2), (4, 1), (8, 1), (8, 2), (2, 2)):

        def step(x_flat, i, r=rps, s=sgs):
            h = run_ka2(x_flat, i, r, s)
            y = ref_down(h.reshape(1, 1, TOP_K, 1, INTER), i)
            return dep(x_flat, y)

        dt = bench_serial(step)
        print(
            f"SERIAL A2({rps},{sgs}) + ref down: {dt * 1e6:7.1f} us/layer "
            f"-> x40 = {dt * 40 * 1e3:.2f} ms/step"
        )


def phase_compile() -> None:
    """mx.compile feasibility: does compiling the layer step cut serial latency?"""
    import numpy as np

    traces = _load_router_trace()
    gw, gs, gb = _quant_stacked(INTER, HIDDEN)
    uw, us, ub = _quant_stacked(INTER, HIDDEN)
    dw, ds, db = _quant_stacked_down()
    x0 = mx.random.normal((1, HIDDEN), dtype=mx.float16)
    mx.eval(gw, gs, gb, uw, us, ub, dw, ds, db, x0)

    ka = mx.fast.metal_kernel(
        name="moe_gateup_swiglu_decode_bench",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
            "expert_ids",
        ],
        output_names=["h"],
        header=_KERNEL_A_HEADER,
        source=_KERNEL_A_SOURCE,
    )

    def run_ka(x_flat, ids):
        return ka(
            inputs=[x_flat, gw, gs, gb, uw, us, ub, ids],
            template=[
                ("T", mx.float16),
                ("K", HIDDEN),
                ("N", INTER),
                ("TOPK", TOP_K),
                ("ROWS_PER_SG", 4),
                ("SIMDGROUPS", 2),
            ],
            grid=(32, INTER // 4, TOP_K),
            threadgroup=(32, 2, 1),
            output_shapes=[(TOP_K, INTER)],
            output_dtypes=[mx.float16],
        )[0]

    def layer_ref(x_flat, idx5):
        x5 = x_flat.reshape(1, 1, 1, 1, HIDDEN)
        g = mx.gather_qmm(
            x5,
            gw,
            gs,
            gb,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        u = mx.gather_qmm(
            x5,
            uw,
            us,
            ub,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        h = mx.multiply(g * mx.sigmoid(g), u)
        y = mx.gather_qmm(
            h, dw, ds, db, rhs_indices=idx5, transpose=True, group_size=GROUP, bits=BITS
        )
        return x_flat + y.astype(mx.float16).reshape(-1)[:1] * mx.array(
            0.0, dtype=mx.float16
        )

    def layer_fused(x_flat, ids, idx5):
        h = run_ka(x_flat, ids)
        y = mx.gather_qmm(
            h.reshape(1, 1, TOP_K, 1, INTER),
            dw,
            ds,
            db,
            rhs_indices=idx5,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        return x_flat + y.astype(mx.float16).reshape(-1)[:1] * mx.array(
            0.0, dtype=mx.float16
        )

    ids_list = [mx.array(traces[i].astype(np.int32)) for i in range(200)]
    idx5_list = [
        mx.array(traces[i].reshape(1, 1, 8).astype(np.uint32)) for i in range(200)
    ]
    mx.eval(*ids_list, *idx5_list)

    def bench(step, iters=200, warmup=30):
        x = x0
        for i in range(warmup):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        t0 = time.perf_counter()
        x = x0
        for i in range(iters):
            x = step(x, i % 200)
        mx.eval(x)
        mx.synchronize()
        return (time.perf_counter() - t0) / iters

    dt = bench(lambda x, i: layer_ref(x, idx5_list[i]))
    print(f"eager  ref layer   : {dt * 1e6:7.1f} us")
    cref = mx.compile(layer_ref)
    dt = bench(lambda x, i: cref(x, idx5_list[i]))
    print(f"COMPILED ref layer : {dt * 1e6:7.1f} us")
    dt = bench(lambda x, i: layer_fused(x, ids_list[i], idx5_list[i]))
    print(f"eager  fused layer : {dt * 1e6:7.1f} us")
    cfused = mx.compile(layer_fused)
    dt = bench(lambda x, i: cfused(x, ids_list[i], idx5_list[i]))
    print(f"COMPILED fused layer: {dt * 1e6:7.1f} us")


def phase_shipped() -> None:
    """Shipped FusedMoEDecodeKernels vs stock SwitchGLU at the serve shape."""
    import mlx.nn as nn
    import numpy as np
    from mlx_lm.models.switch_layers import SwitchGLU

    from vllm_metal.fused_moe import FusedMoEDecodeKernels as F

    glu = SwitchGLU(HIDDEN, INTER, E)
    nn.quantize(glu, group_size=GROUP, bits=BITS)
    glu.eval()
    x = mx.random.normal((1, 1, HIDDEN), dtype=mx.float16)
    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        proj.scales = proj.scales.astype(mx.float16)
        proj.biases = proj.biases.astype(mx.float16)
    rng = np.random.default_rng(0)
    idx_steps = [
        mx.array(rng.choice(E, size=(1, 1, TOP_K), replace=False).astype(np.uint32))
        for _ in range(64)
    ]
    mx.eval(glu.parameters(), x, *idx_steps)

    def timed(fn):
        for i in range(WARMUP):
            mx.eval(fn(idx_steps[i % 64]))
        mx.synchronize()
        t0 = time.perf_counter()
        outs = [fn(idx_steps[i % 64]) for i in range(ITERS)]
        mx.eval(*outs)
        mx.synchronize()
        return (time.perf_counter() - t0) / ITERS * 1e6

    print(f"stock SwitchGLU               : {timed(lambda idx: glu(x, idx)):7.1f} us")
    ref = glu(x, idx_steps[0])
    out = F._run(glu, x, idx_steps[0])
    a = np.array(ref.astype(mx.float32), copy=False).ravel()
    b = np.array(out.astype(mx.float32), copy=False).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"parity: cosine={cos:.6f} max|diff|={float(np.max(np.abs(a - b))):.6f}")
    shipped_rows, shipped_sgs = F.ROWS_PER_SG, F.SIMDGROUPS
    for rows in (4, 8, 16):
        for sgs in (2, 4):
            if INTER % (rows * sgs):
                continue
            F.ROWS_PER_SG, F.SIMDGROUPS = rows, sgs
            F._compiled.clear()
            marker = " (shipped)" if (rows, sgs) == (shipped_rows, shipped_sgs) else ""
            t = timed(lambda idx: F._run(glu, x, idx))
            print(f"shipped kernel r{rows:<2d} x s{sgs}      : {t:7.1f} us{marker}")
    F.ROWS_PER_SG, F.SIMDGROUPS = shipped_rows, shipped_sgs
    F._compiled.clear()


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"mlx={mx.__version__} phase={phase}")
    if phase == "baseline":
        phase_baseline()
    elif phase == "gateup":
        phase_gateup()
    elif phase == "down":
        phase_down()
    elif phase == "serial":
        phase_serial()
    elif phase == "serial2":
        phase_serial2()
    elif phase == "compile":
        phase_compile()
    elif phase == "shipped":
        phase_shipped()
    else:
        raise SystemExit(f"phase {phase!r} not implemented yet")


if __name__ == "__main__":
    main()
