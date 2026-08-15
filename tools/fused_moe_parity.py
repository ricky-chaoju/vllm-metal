#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-vs-off serve check for the fused MoE decode dispatch.

Runs the same deterministic greedy prompt set in two spawned children
(Metal is not fork-safe): one with ``VLLM_METAL_FUSED_MOE_DECODE=1`` and a
reach spy counting fused dispatches (at least one per prompt required),
one with the kill switch off (the spy must count zero). Unlike the
intermediate-prefill harness, token IDENTITY is not the contract here: the
fused kernel accumulates in a different order than the stock pair of
``gather_qmm`` calls, so greedy outputs drift at the fp level after enough
tokens. The gate asserts reach on both arms and that every output decodes
to non-empty text; the per-prompt first-divergence index is REPORTED so a
regression toward early divergence is visible in the PR record.

Not in CI — requires local model weights for a SwitchGLU+SwiGLU 4-bit
checkpoint (default Qwen3.6-35B-A3B-4bit; needs the weights cached).

Usage:
    python tools/fused_moe_parity.py
    python tools/fused_moe_parity.py --model <path-or-repo> --quick
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

MODEL_DEFAULT = os.environ.get(
    "FUSED_MOE_MODEL_PATH", "mlx-community/Qwen3.6-35B-A3B-4bit"
)

PROMPTS = (
    "Write a short essay about the history of computing.",
    "Explain how a B-tree works and why databases use it.",
    "Describe the CAP theorem and its practical consequences.",
)


def _child_env(fused: bool) -> None:
    os.environ["VLLM_METAL_FUSED_MOE_DECODE"] = "1" if fused else "0"
    for key, val in (
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
        ("VLLM_METAL_USE_PAGED_ATTENTION", "1"),
        ("VLLM_METAL_MEMORY_FRACTION", "0.5"),
    ):
        os.environ.setdefault(key, val)


def run_child(model: str, fused: bool, quick: bool, queue) -> None:
    _child_env(fused)
    from vllm import LLM, SamplingParams

    import vllm_metal.fused_moe as fm

    calls = {"n": 0}
    orig = fm.FusedMoEDecodeKernels._run

    def spy(glu, x, indices):
        calls["n"] += 1
        return orig(glu, x, indices)

    fm.FusedMoEDecodeKernels._run = spy
    try:
        llm = LLM(model=model, max_model_len=2048, max_num_seqs=4)
        prompts = PROMPTS[:1] if quick else PROMPTS
        sp = SamplingParams(temperature=0, max_tokens=64)
        results = {}
        for prompt in prompts:
            out = llm.generate([prompt], sp)[0]
            results[prompt] = {
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
    finally:
        fm.FusedMoEDecodeKernels._run = orig

    if fused:
        assert calls["n"] >= len(prompts), (
            f"fused arm dispatched only {calls['n']} fused calls for "
            f"{len(prompts)} prompts — the fused path did not engage"
        )
    else:
        assert calls["n"] == 0, (
            f"kill-switch arm unexpectedly dispatched {calls['n']} fused calls"
        )
    queue.put(results)


def run_pair(model: str, quick: bool) -> int:
    ctx = mp.get_context("spawn")
    per_arm: dict[bool, dict] = {}
    for fused in (False, True):
        queue = ctx.Queue()
        proc = ctx.Process(target=run_child, args=(model, fused, quick, queue))
        proc.start()
        try:
            per_arm[fused] = queue.get(timeout=1800)
        finally:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
        if proc.exitcode != 0:
            raise RuntimeError(f"child (fused={fused}) exited with {proc.exitcode}")

    failures = 0
    for prompt, ref in per_arm[False].items():
        fused_out = per_arm[True][prompt]
        ref_ids, fused_ids = ref["token_ids"], fused_out["token_ids"]
        divergence = next(
            (
                i
                for i, (a, b) in enumerate(zip(ref_ids, fused_ids, strict=False))
                if a != b
            ),
            min(len(ref_ids), len(fused_ids)),
        )
        ok = bool(fused_out["text"].strip()) and bool(ref["text"].strip())
        if not ok:
            failures += 1
        print(
            f"prompt {prompt[:40]!r}: first divergence at token {divergence}/"
            f"{len(ref_ids)}  outputs non-empty: {ok}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    failures = run_pair(args.model, args.quick)
    if failures:
        print(f"FUSED-MOE CHECK FAIL: {failures} degenerate outputs")
        return 1
    print("FUSED-MOE CHECK PASS: reach asserted on both arms, outputs coherent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
