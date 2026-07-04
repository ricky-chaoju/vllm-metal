# SPDX-License-Identifier: Apache-2.0
"""Bitwise parity gate for the spec-verify window mode (issue #465).

For each cell, runs the SAME query/cache through:

  A. expanded metadata (K+1 one-token segments, window_seqlen_q=1)
     -> the per-token kernel path, bit-for-bit today's behavior
  B. merged metadata (one segment of q_len=K+1, window_seqlen_q=K+1)
     -> the window-mode kernel

and compares rows with ``mx.array_equal``.  Bitwise equality is expected
whenever both shapes take the same split-KV decision; cells where the
occupancy gate diverges are checked with fp tolerances instead.  Dense
cells are also checked against the pure-MLX reference; TurboQuant cells
compare A/B only (both read the same packed caches).

Run from the repo root:

    PYTHONPATH=$PWD VLLM_METAL_BUILD_FROM_SOURCE=1 \
        python tools/spec_window_parity.py
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
from attention_bench_utils import ref_paged_attn

from vllm_metal.attention.caches.turboquant import (
    get_v_centroids,
    turbo_quant_encode,
)
from vllm_metal.metal import get_ops

BLOCK_SIZE = 16
HEAD_SIZE = 128
V_BITS = 3
TOLS = {
    "float16": (1.5e-2, 2e-2),
    "bfloat16": (3e-2, 2e-2),
    "float32": (1e-3, 1e-3),
}

# (name, num_q_heads, num_kv_heads, dtype, ctx_list, window_sets, sliding)
DENSE_CELLS = [
    (
        "fp16 16q/8kv",
        16,
        8,
        mx.float16,
        [8, 63, 512, 1024, 8192],
        [[2], [4], [6], [8]],
        -1,
    ),
    ("fp16 32q/8kv", 32, 8, mx.float16, [512, 8192], [[6]], -1),
    ("bf16 16q/8kv", 16, 8, mx.bfloat16, [512, 8192], [[4], [6]], -1),
    ("fp32 16q/8kv", 16, 8, mx.float32, [512, 8192], [[6]], -1),
    ("fp16 mixed", 16, 8, mx.float16, [512, 8192], [[1, 4, 6]], -1),
    ("fp16 SWA512", 16, 8, mx.float16, [8192], [[2], [6]], 512),
]
TQ_CELLS = [
    (quant, ctx, w) for quant in ("q8_0", "q4_0") for ctx in (512, 8192) for w in (4, 6)
]


def _run_dense(
    ops,
    query,
    kc,
    vc,
    nkv,
    tables,
    seq_lens,
    cu,
    max_kv,
    window_seqlen_q,
    sliding_window,
):
    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        kc,
        vc,
        nkv,
        HEAD_SIZE**-0.5,
        0.0,
        mx.array(tables, dtype=mx.int32),
        mx.array(seq_lens, dtype=mx.int32),
        mx.array(cu, dtype=mx.int32),
        BLOCK_SIZE,
        max_kv,
        sliding_window,
        out,
        window_seqlen_q=window_seqlen_q,
    )
    mx.eval(out)
    return out


def _run_tq(ops, query, caches, tables, seq_lens, cu, max_kv, window_seqlen_q, quant):
    k_cache, v_cache, k_scale, v_scale, k_zero = caches
    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        k_cache,
        v_cache,
        8,
        HEAD_SIZE**-0.5,
        0.0,
        mx.array(tables, dtype=mx.int32),
        mx.array(seq_lens, dtype=mx.int32),
        mx.array(cu, dtype=mx.int32),
        BLOCK_SIZE,
        max_kv,
        -1,
        out,
        key_scale_cache=k_scale,
        value_scale_cache=v_scale,
        key_zero_cache=k_zero,
        v_centroids=get_v_centroids(V_BITS),
        use_turboquant=True,
        quant_type=quant,
        v_bits=V_BITS,
        window_seqlen_q=window_seqlen_q,
    )
    mx.eval(out)
    return out


def _dense_pass(ops, min_grid) -> int:
    failures = 0
    for name, nq, nkv, dtype, ctxs, window_sets, sw in DENSE_CELLS:
        for ctx in ctxs:
            for windows in window_sets:
                total_q = sum(windows)
                lens = [ctx + w for w in windows]
                max_kv = max(lens)
                bps = (max_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
                kc = mx.random.normal(
                    shape=(bps * len(windows), BLOCK_SIZE, nkv, HEAD_SIZE)
                ).astype(dtype)
                vc = mx.random.normal(
                    shape=(bps * len(windows), BLOCK_SIZE, nkv, HEAD_SIZE)
                ).astype(dtype)
                query = mx.random.normal(shape=(total_q, nq, HEAD_SIZE)).astype(dtype)
                mx.eval(kc, vc, query)
                tables = [
                    list(range(s * bps, (s + 1) * bps)) for s in range(len(windows))
                ]

                e_tables, e_seq, e_cu = [], [], [0]
                for s, w in enumerate(windows):
                    for j in range(w):
                        e_tables.append(tables[s])
                        e_seq.append(ctx + j + 1)
                        e_cu.append(e_cu[-1] + 1)
                out_a = _run_dense(
                    ops, query, kc, vc, nkv, e_tables, e_seq, e_cu, max_kv, 1, sw
                )

                m_cu = [0]
                for w in windows:
                    m_cu.append(m_cu[-1] + w)
                out_b = _run_dense(
                    ops,
                    query,
                    kc,
                    vc,
                    nkv,
                    tables,
                    lens,
                    m_cu,
                    max_kv,
                    max(windows),
                    sw,
                )

                split_a = nq * total_q < min_grid and max_kv > 512
                split_b = nq * len(windows) < min_grid and max_kv > 512
                same_gate = split_a == split_b
                bitwise = bool(mx.array_equal(out_a, out_b))
                atol, rtol = TOLS[str(dtype).split(".")[-1]]
                close = np.allclose(
                    np.array(out_a.astype(mx.float32)),
                    np.array(out_b.astype(mx.float32)),
                    atol=atol,
                    rtol=rtol,
                )
                ref_ok = "-"
                if sw < 0:
                    ref = ref_paged_attn(
                        query=query,
                        key_cache=kc,
                        value_cache=vc,
                        query_lens=windows,
                        kv_lens=lens,
                        block_tables=np.array(tables),
                        scale=HEAD_SIZE**-0.5,
                    )
                    mx.eval(ref)
                    ref_ok = str(
                        np.allclose(
                            np.array(out_b.astype(mx.float32)),
                            np.array(ref.astype(mx.float32)),
                            atol=atol,
                            rtol=rtol,
                        )
                    )

                status = "OK"
                if same_gate and not bitwise:
                    status, failures = "FAIL(bitwise)", failures + 1
                elif not close:
                    status, failures = "FAIL(allclose)", failures + 1
                elif ref_ok == "False":
                    status, failures = "FAIL(ref)", failures + 1
                print(
                    f"{name:14s} ctx={ctx:>5} win={windows} "
                    f"split=({int(split_a)},{int(split_b)}) "
                    f"bitwise={int(bitwise)} close={int(close)} "
                    f"ref={ref_ok:5s} {status}"
                )
    return failures


def _tq_pass(ops, min_grid) -> int:
    failures = 0
    for quant, ctx, w in TQ_CELLS:
        total = ctx + w
        nblocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
        kv_len = nblocks * BLOCK_SIZE
        k = mx.random.normal(shape=(kv_len, 8, HEAD_SIZE)).astype(mx.float16)
        v = mx.random.normal(shape=(kv_len, 8, HEAD_SIZE)).astype(mx.float16)
        query = mx.random.normal(shape=(w, 16, HEAD_SIZE)).astype(mx.float16)
        mx.eval(k, v, query)
        (k_packed, k_scale, k_zero), (v_packed, v_scale) = turbo_quant_encode(
            k, v, quant
        )
        sg = k_scale.shape[-1]
        caches = (
            k_packed.reshape(nblocks, BLOCK_SIZE, 8, -1),
            v_packed.reshape(nblocks, BLOCK_SIZE, 8, -1),
            k_scale.reshape(nblocks, BLOCK_SIZE, 8, sg),
            v_scale.reshape(nblocks, BLOCK_SIZE, 8, sg),
            k_zero.reshape(nblocks, BLOCK_SIZE, 8, sg),
        )
        mx.eval(*caches)
        blocks = list(range(nblocks))

        out_a = _run_tq(
            ops,
            query,
            caches,
            [blocks] * w,
            [ctx + j + 1 for j in range(w)],
            list(range(w + 1)),
            total,
            1,
            quant,
        )
        out_b = _run_tq(ops, query, caches, [blocks], [total], [0, w], total, w, quant)

        split_a = 16 * w < min_grid and total > 512
        split_b = 16 < min_grid and total > 512
        same_gate = split_a == split_b
        bitwise = bool(mx.array_equal(out_a, out_b))
        status = "OK"
        if same_gate and not bitwise:
            status, failures = "FAIL(bitwise)", failures + 1
        print(
            f"tq={quant:5s} ctx={ctx:>5} w={w} "
            f"split=({int(split_a)},{int(split_b)}) "
            f"bitwise={int(bitwise)} {status}"
        )
    return failures


def main() -> None:
    ops = get_ops()
    min_grid = ops.min_decode_grid()
    print(f"min_decode_grid={min_grid}")
    mx.random.seed(0)
    failures = _dense_pass(ops, min_grid) + _tq_pass(ops, min_grid)
    print("PARITY_RESULT:", "PASS" if failures == 0 else f"{failures} FAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
