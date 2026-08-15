# SPDX-License-Identifier: Apache-2.0
"""Fused MoE decode dispatch: gate/up/SwiGLU in one Metal kernel.

At decode, mlx_lm's ``SwitchGLU`` issues two unsorted ``gather_qmm`` calls
plus three elementwise ops before the down projection; the fused kernel
runs that group as one dispatch. The value case is end-to-end: serving is
measurably faster with the kernel than with either the stock dispatch or
the stock ops inside the same compiled packaging (so the win is the
kernel, not the packaging; per-PR A/B numbers live in the PR body, and
``tools/fused_moe_microbench.py`` phase ``shipped`` re-measures the
shipped kernel against stock on the pinned MLX). The down projection
stays on MLX's gather. Prefill-sized calls keep the stock path.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger

from vllm_metal import envs as metal_envs
from vllm_metal.metal import _read_v2_metal_source

logger = init_logger(__name__)


class FusedMoEDecodeKernels:
    """Owner for the fused MoE decode gate/up/SwiGLU fast path."""

    # Decode-shaped calls only: at top-8 this admits batches up to 7 tokens,
    # mirroring mlx_lm's own sort threshold (>=64 indices switches its path,
    # so exactly 64 stays on the stock dispatch too).
    MAX_DECODE_INDICES: ClassVar[int] = 64
    # Kernel tile geometry: best (ROWS_PER_SG, SIMDGROUPS) from the
    # marginal-cost sweep at the Qwen3.6 decode shape on M1 Ultra — r8
    # clearly beats r4, r16 regresses on register pressure, and the two
    # simdgroup widths sit within serial-timing noise of each other.
    ROWS_PER_SG: ClassVar[int] = 8
    SIMDGROUPS: ClassVar[int] = 4
    # The lane tiling walks the contraction dim in 1024-value blocks
    # (one uint4 = 32 nibbles per lane per row per block).
    K_BLOCK: ClassVar[int] = 1024
    GROUP_SIZE: ClassVar[int] = 64
    BITS: ClassVar[int] = 4

    _kernel: ClassVar[Any | None] = None
    # Compiled per (num_tokens, top_k, hidden, inter, dtype) shape key; all
    # 40 layers share entries because weights are passed as arguments, and
    # mx.compile removes the per-call lazy-graph build/traversal cost.
    # Bounded LRU: token counts under the decode cap and dtype variants give
    # at most a few dozen legitimate keys per served model; the cap only
    # guards pathological churn (e.g. repeated reloads of different models).
    MAX_COMPILED_SHAPES: ClassVar[int] = 64
    _compiled: ClassVar[OrderedDict[tuple[Any, ...], Callable[..., mx.array]]] = (
        OrderedDict()
    )

    @classmethod
    def install(cls, model: Any) -> int:
        """Wrap every ``SwitchGLU`` in *model* with the fused decode dispatch.

        Walks the module tree and replaces each exact ``SwitchGLU`` instance
        with a :class:`FusedSwitchGLU` wrapper (the GDN/SDPA wrapper-install
        precedent — no process-wide monkeypatch). Gated by
        ``VLLM_METAL_FUSED_MOE_DECODE``; idempotent per module. Returns the
        number of wrapped modules.
        """
        if not metal_envs.VLLM_METAL_FUSED_MOE_DECODE:
            return 0
        if cls._make_kernel() is None:
            logger.info(
                "Metal: fused MoE decode kernel unavailable (no Metal device); "
                "keeping stock SwitchGLU dispatch."
            )
            return 0

        from mlx_lm.models.switch_layers import SwitchGLU

        if not isinstance(model, nn.Module):
            raise TypeError(
                "FusedMoEDecodeKernels.install expects an mlx nn.Module, got "
                f"{type(model).__name__}"
            )
        wrapped = 0

        def walk_container(owner: nn.Module, sub: Any) -> None:
            # children() rebuilds dict/list containers, so writes to them
            # never reach the model — a bare SwitchGLU here would be an
            # un-wrappable target and must not be skipped silently.
            if type(sub) is SwitchGLU:
                raise RuntimeError(
                    "FusedMoEDecodeKernels.install found a SwitchGLU inside "
                    f"a raw container of {type(owner).__name__}; it can only "
                    "wrap module attributes."
                )
            walk(sub)

        def walk(mod: Any) -> None:
            nonlocal wrapped
            if not isinstance(mod, nn.Module):
                return
            for name, child in mod.children().items():
                if type(child) is SwitchGLU:
                    setattr(mod, name, FusedSwitchGLU(child))
                    wrapped += 1
                elif isinstance(child, FusedSwitchGLU):
                    continue  # already wrapped (idempotency)
                elif isinstance(child, nn.Module):
                    walk(child)
                elif isinstance(child, dict):
                    for sub in child.values():
                        walk_container(mod, sub)
                elif isinstance(child, list):
                    for sub in child:
                        walk_container(mod, sub)

        walk(model)
        if wrapped:
            logger.info(
                "Metal: fused MoE decode dispatch wrapped %d SwitchGLU "
                "modules (gate/up/SwiGLU in one kernel on eligible decode "
                "calls).",
                wrapped,
            )
        return wrapped

    @classmethod
    def _quant_linear_ok(cls, module: Any, allowed_bits: tuple[int, ...]) -> bool:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear

        return (
            type(module) is QuantizedSwitchLinear
            and getattr(module, "bits", None) in allowed_bits
            and getattr(module, "group_size", None) == cls.GROUP_SIZE
            and getattr(module, "mode", "affine") == "affine"
            and "bias" not in module
            and module["weight"].dtype == mx.uint32
        )

    @classmethod
    def _eligible_glu_only_probe(cls, glu: Any) -> bool:
        """Load-final SwitchGLU quant/shape checks, resolved once at install."""
        from mlx_lm.models.switch_layers import SwiGLU

        # The kernel hardcodes silu(gate) * up; any other gating activation
        # (e.g. a GeGLU variant) must keep the stock dispatch.
        if type(getattr(glu, "activation", None)) is not SwiGLU:
            return False
        gate = getattr(glu, "gate_proj", None)
        up = getattr(glu, "up_proj", None)
        down = getattr(glu, "down_proj", None)
        for proj in (gate, up, down):
            if proj is None or not cls._quant_linear_ok(proj, (cls.BITS,)):
                return False
        if not (gate["scales"].dtype == up["scales"].dtype == down["scales"].dtype):
            return False
        hidden = gate["weight"].shape[-1] * 8
        if hidden % cls.K_BLOCK != 0:
            return False
        if up["weight"].shape[-1] != hidden // 8:
            return False
        inter = gate["weight"].shape[1]
        if down["weight"].shape[-1] != inter // 8 or inter % cls.GROUP_SIZE != 0:
            return False
        return inter % (cls.SIMDGROUPS * cls.ROWS_PER_SG) == 0

    @classmethod
    def _make_kernel(cls) -> Any | None:
        if cls._kernel is not None:
            return cls._kernel
        try:
            if not mx.metal.is_available():
                return None
        except AttributeError:
            return None
        cls._kernel = mx.fast.metal_kernel(
            name="moe_gateup_swiglu_decode",
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
            header=_read_v2_metal_source("moe_q4_dot.metal"),
            source=_read_v2_metal_source("moe_gateup_swiglu_decode.metal"),
        )
        return cls._kernel

    @classmethod
    def _call_eligible(
        cls, wrapper: FusedSwitchGLU, x: mx.array, indices: mx.array
    ) -> bool:
        """Per-call facts for a statically eligible wrapper (else stock).

        The wrapper delegates ineligible calls to the stock dispatch
        silently instead of failing fast — the stock path is the working
        default, not an error condition.
        """
        if indices.size >= cls.MAX_DECODE_INDICES or indices.size == 0:
            return False
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False
        if wrapper.training:
            return False
        gate = wrapper.inner.gate_proj
        return (
            x.shape[-1] == gate["weight"].shape[-1] * 8
            and gate["scales"].dtype == x.dtype
        )

    @classmethod
    def _run(cls, glu: Any, x: mx.array, indices: mx.array) -> mx.array:
        gate = glu.gate_proj
        up = glu.up_proj
        down = glu.down_proj
        lead = x.shape[:-1]
        hidden = x.shape[-1]
        top_k = indices.shape[-1]
        num_tokens = math.prod(lead) if lead else 1
        inter = gate["weight"].shape[1]

        step = cls._compiled_step(num_tokens, top_k, hidden, inter, x.dtype)
        y = step(
            x.reshape(num_tokens, hidden),
            indices.reshape(-1).astype(mx.int32),
            indices.reshape(num_tokens, 1, top_k),
            gate["weight"],
            gate["scales"],
            gate["biases"],
            up["weight"],
            up["scales"],
            up["biases"],
            down["weight"],
            down["scales"],
            down["biases"],
        )
        # y: (num_tokens, top_k, hidden) — reshape to SwitchGLU's contract.
        return y.reshape(*lead, top_k, hidden)

    @classmethod
    def _compiled_step(
        cls,
        num_tokens: int,
        top_k: int,
        hidden: int,
        inter: int,
        dtype: mx.Dtype,
    ) -> Callable[..., mx.array]:
        key = (num_tokens, top_k, hidden, inter, dtype)
        cached = cls._compiled.get(key)
        if cached is not None:
            cls._compiled.move_to_end(key)
            return cached

        kernel = cls._make_kernel()
        assert kernel is not None  # install() verified availability
        group_size = cls.GROUP_SIZE
        bits = cls.BITS
        template = [
            ("T", dtype),
            ("K", hidden),
            ("N", inter),
            ("TOPK", top_k),
            ("ROWS_PER_SG", cls.ROWS_PER_SG),
            ("SIMDGROUPS", cls.SIMDGROUPS),
        ]
        grid = (32, inter // cls.ROWS_PER_SG, num_tokens * top_k)
        threadgroup = (32, cls.SIMDGROUPS, 1)

        def step(
            x2d: mx.array,
            ids_flat: mx.array,
            idx3: mx.array,
            gate_w: mx.array,
            gate_s: mx.array,
            gate_b: mx.array,
            up_w: mx.array,
            up_s: mx.array,
            up_b: mx.array,
            down_w: mx.array,
            down_s: mx.array,
            down_b: mx.array,
        ) -> mx.array:
            h = kernel(
                inputs=[x2d, gate_w, gate_s, gate_b, up_w, up_s, up_b, ids_flat],
                template=template,
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[(num_tokens * top_k, inter)],
                output_dtypes=[dtype],
            )[0]
            y = mx.gather_qmm(
                h.reshape(num_tokens, 1, top_k, 1, inter),
                down_w,
                down_s,
                down_b,
                rhs_indices=idx3.reshape(num_tokens, 1, top_k),
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            return y.reshape(num_tokens, top_k, hidden)

        compiled = mx.compile(step)
        cls._compiled[key] = compiled
        if len(cls._compiled) > cls.MAX_COMPILED_SHAPES:
            cls._compiled.popitem(last=False)
        return compiled


class FusedSwitchGLU(nn.Module):
    """Per-module wrapper routing eligible decode calls to the fused kernel.

    The inner ``SwitchGLU`` stays a normally registered child, so the
    expert weights remain visible to ``model.parameters()``/``tree_flatten``
    and ``train()``/``eval()`` propagate as usual. LoRA target discovery is
    unaffected on wrapped hosts: expert projection weights are 3-D stacked
    tensors, which the LoRA layer wrappers reject by construction
    (``can_wrap`` requires 2-D weights), and the router lives outside
    ``SwitchGLU`` — so hiding or exposing the experts never changes what
    LoRA can wrap. Module-level eligibility (activation type, projection
    quantization and shapes) is resolved once at install into
    ``_static_ok``; the call path re-checks only per-call facts. Ineligible
    calls delegate to the stock dispatch unchanged.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner
        # Inherit the inner module's mode: install runs on an already
        # eval'd model, and a fresh nn.Module defaults to training=True.
        self.train(inner.training)
        # Load-final module facts (quantization, shapes, activation type):
        # nothing mutates them after install (LoRA cannot wrap 3-D expert
        # weights), so the per-call gate only re-checks per-call facts.
        self._static_ok = FusedMoEDecodeKernels._eligible_glu_only_probe(inner)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        inner = self.inner
        if self._static_ok and FusedMoEDecodeKernels._call_eligible(self, x, indices):
            return FusedMoEDecodeKernels._run(inner, x, indices)
        return inner(x, indices)
