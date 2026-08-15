# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fused MoE decode dispatch."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models import switch_layers

import vllm_metal.envs as envs
from vllm_metal.fused_moe import FusedMoEDecodeKernels, FusedSwitchGLU
from vllm_metal.v1.lora.layers import can_wrap, can_wrap_qlora
from vllm_metal.v1.model_lifecycle import ModelLifecycle

E = 4
HIDDEN = 2048
INTER = 64
TOP_K = 2


def _require_metal() -> None:
    try:
        available = mx.metal.is_available()
    except AttributeError:
        available = False
    if not available:
        pytest.skip("MLX Metal is not available")


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("VLLM_METAL_FUSED_MOE_DECODE", raising=False)
    yield


class _MoEHost(nn.Module):
    """Minimal module tree hosting one SwitchGLU (the install target)."""

    def __init__(self, glu: nn.Module) -> None:
        super().__init__()
        self.switch_mlp = glu


def _quantized_glu(dtype: mx.Dtype = mx.float16) -> switch_layers.SwitchGLU:
    mx.random.seed(7)
    glu = switch_layers.SwitchGLU(HIDDEN, INTER, num_experts=E)
    # Cast BEFORE quantize so scales land in the activation dtype —
    # otherwise eligibility rejects and the parity test silently compares
    # stock against stock.
    glu.set_dtype(dtype)
    nn.quantize(glu, group_size=64, bits=4)
    glu.eval()  # mirror serve state: mlx_lm load puts models in eval mode
    mx.eval(glu.parameters())
    return glu


def _install(monkeypatch, model: nn.Module) -> int:
    monkeypatch.setenv("VLLM_METAL_FUSED_MOE_DECODE", "1")
    return FusedMoEDecodeKernels.install(model)


def _spy_run(monkeypatch) -> dict[str, int]:
    """Reach spy: count fused-path dispatches through the wrapper."""
    calls = {"n": 0}
    orig = FusedMoEDecodeKernels._run

    def spy(glu, x, indices):
        calls["n"] += 1
        return orig(glu, x, indices)

    monkeypatch.setattr(FusedMoEDecodeKernels, "_run", spy)
    return calls


class TestInstall:
    def test_flag_defaults_off(self):
        # Assert — opt-in while the numerical-parity mileage grows.
        assert envs.VLLM_METAL_FUSED_MOE_DECODE is False

    def test_kill_switch_leaves_modules_untouched(self, monkeypatch):
        # Arrange
        _require_metal()
        glu = _quantized_glu()
        host = _MoEHost(glu)
        monkeypatch.setenv("VLLM_METAL_FUSED_MOE_DECODE", "0")

        # Act
        wrapped = FusedMoEDecodeKernels.install(host)

        # Assert
        assert wrapped == 0
        assert host.switch_mlp is glu

    def test_install_wraps_exact_switch_glu_only(self, monkeypatch):
        # Arrange
        _require_metal()

        class NotQuiteSwitchGLU(switch_layers.SwitchGLU):
            pass

        subclassed = NotQuiteSwitchGLU(HIDDEN, INTER, num_experts=E)
        host = _MoEHost(_quantized_glu())
        sub_host = _MoEHost(subclassed)

        # Act
        wrapped = _install(monkeypatch, host)
        wrapped_sub = _install(monkeypatch, sub_host)

        # Assert: exact type wraps; a subclass is not silently claimed.
        assert wrapped == 1
        assert isinstance(host.switch_mlp, FusedSwitchGLU)
        assert wrapped_sub == 0
        assert sub_host.switch_mlp is subclassed

    def test_install_counts_every_target_in_a_nested_tree(self, monkeypatch):
        # Arrange — two hosts at different depths.
        _require_metal()

        class _Deep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = [_MoEHost(_quantized_glu()) for _ in range(2)]
                self.head = _MoEHost(_quantized_glu())

        deep = _Deep()

        # Act
        wrapped = _install(monkeypatch, deep)

        # Assert
        assert wrapped == 3
        assert all(isinstance(h.switch_mlp, FusedSwitchGLU) for h in deep.layers)
        assert isinstance(deep.head.switch_mlp, FusedSwitchGLU)

    def test_install_is_idempotent(self, monkeypatch):
        # Arrange
        _require_metal()
        host = _MoEHost(_quantized_glu())

        # Act
        first = _install(monkeypatch, host)
        wrapper = host.switch_mlp
        second = _install(monkeypatch, host)

        # Assert
        assert first == 1
        assert second == 0
        assert host.switch_mlp is wrapper

    def test_bare_switch_glu_in_raw_container_fails_fast(self, monkeypatch):
        # Arrange — children() rebuilds raw containers, so a bare SwitchGLU
        # inside one would be un-wrappable; install must refuse loudly.
        _require_metal()

        class _RawListHost(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blobs = [(_quantized_glu(),)]

        class _Tupled(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pair = [_quantized_glu()]

        # a list child whose element IS the bare module
        host = _Tupled()

        # Act / Assert
        with pytest.raises(RuntimeError, match="raw container"):
            _install(monkeypatch, host)

    def test_expert_weights_stay_in_the_parameter_tree(self, monkeypatch):
        # Arrange — the wrapper registers the inner module normally, so
        # dtype casts, serialization and mode changes keep working.
        _require_metal()
        host = _MoEHost(_quantized_glu())
        from mlx.utils import tree_flatten

        before = {k for k, _ in tree_flatten(host.parameters())}

        # Act
        _install(monkeypatch, host)
        after = {k for k, _ in tree_flatten(host.parameters())}

        # Assert — same parameter set, re-rooted under the wrapper.
        assert {k.replace("switch_mlp.", "switch_mlp.inner.") for k in before} == (
            after
        )

    def test_lora_target_discovery_is_unchanged_by_install(self, monkeypatch):
        # Arrange — expert projections hold 3-D stacked weights, which the
        # LoRA wrappers reject by construction; install must not change the
        # set of wrappable modules.
        _require_metal()
        host = _MoEHost(_quantized_glu())

        def wrappable(model: nn.Module) -> list[str]:
            return [
                name
                for name, module in model.named_modules()
                if can_wrap(module) or can_wrap_qlora(module)
            ]

        before = wrappable(host)

        # Act
        _install(monkeypatch, host)

        # Assert
        assert wrappable(host) == before == []


class TestDispatch:
    # bf16 keeps 8 mantissa bits, so its normalized bound is wider than
    # fp16's; both bounds sit ~2x above the measured max diff.
    @pytest.mark.parametrize(
        ("dtype", "atol"), [(mx.float16, 4e-3), (mx.bfloat16, 2e-2)]
    )
    def test_decode_shaped_call_engages_and_matches_stock(
        self, monkeypatch, dtype, atol
    ):
        # Arrange
        _require_metal()
        glu = _quantized_glu(dtype)
        host = _MoEHost(glu)
        x = mx.random.normal((3, HIDDEN)).astype(dtype)
        indices = mx.array([[0, 2], [3, 1], [1, 0]], dtype=mx.uint32)
        reference = glu(x, indices)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)

        # Act
        fused = host.switch_mlp(x, indices)

        # Assert — the fused path really ran, and matches stock numerically.
        assert calls["n"] == 1
        mx.eval(reference, fused)
        assert fused.shape == reference.shape
        ref = np.array(reference.astype(mx.float32)).ravel()
        out = np.array(fused.astype(mx.float32)).ravel()
        cosine = float(
            np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-12)
        )
        assert cosine >= 0.9999
        scale = float(np.max(np.abs(ref))) + 1e-6
        np.testing.assert_allclose(out / scale, ref / scale, atol=atol)

    def test_index_cap_boundary_mirrors_mlx_lm(self, monkeypatch):
        # Arrange — mlx_lm switches its own dispatch at >=64 indices; the
        # fused path takes strictly fewer and leaves 64 to stock.
        _require_metal()
        glu = _quantized_glu()
        host = _MoEHost(glu)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)
        under = mx.random.randint(0, E, (31, TOP_K), dtype=mx.uint32)  # 62
        at_cap = mx.random.randint(0, E, (32, TOP_K), dtype=mx.uint32)  # 64

        # Act
        host.switch_mlp(mx.random.normal((31, HIDDEN)).astype(mx.float16), under)
        engaged_under = calls["n"]
        host.switch_mlp(mx.random.normal((32, HIDDEN)).astype(mx.float16), at_cap)

        # Assert
        assert engaged_under == 1
        assert calls["n"] == 1  # 64 indices stayed on stock

    def test_prefill_sized_call_keeps_stock_path(self, monkeypatch):
        # Arrange
        _require_metal()
        glu = _quantized_glu()
        host = _MoEHost(glu)
        tokens = FusedMoEDecodeKernels.MAX_DECODE_INDICES  # 64 tokens x top2
        x = mx.random.normal((tokens, HIDDEN)).astype(mx.float16)
        indices = mx.random.randint(0, E, (tokens, TOP_K), dtype=mx.uint32)
        reference = glu(x, indices)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)

        # Act
        fused = host.switch_mlp(x, indices)

        # Assert: delegation to the inner module — bitwise stock output.
        assert calls["n"] == 0
        mx.eval(reference, fused)
        np.testing.assert_array_equal(
            np.array(reference, dtype=np.float32),
            np.array(fused, dtype=np.float32),
        )

    def test_train_mode_falls_back_to_stock(self, monkeypatch):
        # Arrange — train()/eval() propagate through the registered inner,
        # and the per-call gate rejects training-mode calls.
        _require_metal()
        host = _MoEHost(_quantized_glu())
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)
        x = mx.random.normal((1, HIDDEN)).astype(mx.float16)
        indices = mx.array([[0, 1]], dtype=mx.uint32)

        # Act
        host.train()
        host.switch_mlp(x, indices)
        host.eval()
        host.switch_mlp(x, indices)

        # Assert
        assert calls["n"] == 1  # only the eval-mode call engaged

    def test_non_swiglu_activation_falls_back(self, monkeypatch):
        # The kernel hardcodes silu(gate) * up; a GLU with any other gating
        # activation (e.g. Gemma4's GeGLU) must keep the stock dispatch
        # bitwise — a silent silu substitution would corrupt outputs.
        # Arrange
        _require_metal()

        class _GeGLULike(nn.Module):
            def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
                return nn.gelu_approx(gate) * up

        glu = _quantized_glu()
        glu.activation = _GeGLULike()
        host = _MoEHost(glu)
        x = mx.random.normal((1, HIDDEN)).astype(mx.float16)
        indices = mx.array([[0, 1]], dtype=mx.uint32)
        reference = glu(x, indices)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)

        # Act
        fused = host.switch_mlp(x, indices)

        # Assert
        assert calls["n"] == 0
        assert host.switch_mlp._static_ok is False
        mx.eval(reference, fused)
        np.testing.assert_array_equal(
            np.array(reference, dtype=np.float32),
            np.array(fused, dtype=np.float32),
        )

    def test_non_quantized_glu_falls_back(self, monkeypatch):
        # Arrange
        _require_metal()
        mx.random.seed(3)
        glu = switch_layers.SwitchGLU(HIDDEN, INTER, num_experts=E)
        glu.eval()
        mx.eval(glu.parameters())
        host = _MoEHost(glu)
        x = mx.random.normal((1, HIDDEN)).astype(mx.float16)
        indices = mx.array([[0, 1]], dtype=mx.uint32)
        reference = glu(x, indices)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)

        # Act
        fused = host.switch_mlp(x, indices)

        # Assert
        assert calls["n"] == 0
        assert host.switch_mlp._static_ok is False
        mx.eval(reference, fused)
        np.testing.assert_array_equal(
            np.array(reference, dtype=np.float32),
            np.array(fused, dtype=np.float32),
        )

    def test_unaligned_hidden_falls_back(self, monkeypatch):
        # Arrange — the kernel walks the contraction dim in 1024-value
        # blocks; a hidden size off that grid keeps the stock dispatch.
        _require_metal()
        mx.random.seed(9)
        glu = switch_layers.SwitchGLU(HIDDEN + 512, INTER, num_experts=E)
        glu.set_dtype(mx.float16)
        nn.quantize(glu, group_size=64, bits=4)
        glu.eval()
        mx.eval(glu.parameters())
        host = _MoEHost(glu)
        assert _install(monkeypatch, host) == 1
        calls = _spy_run(monkeypatch)
        x = mx.random.normal((1, HIDDEN + 512)).astype(mx.float16)
        indices = mx.array([[0, 1]], dtype=mx.uint32)

        # Act
        host.switch_mlp(x, indices)

        # Assert
        assert calls["n"] == 0
        assert host.switch_mlp._static_ok is False


class TestLifecycleInstall:
    def test_install_decode_dispatch_threads_the_forward_model(self, monkeypatch):
        # Arrange — the lifecycle hook must hand the runner's forward model
        # (not some other object) to the installer.
        from types import SimpleNamespace

        from tests.stub_runner import make_stub_runner

        received: list[object] = []
        monkeypatch.setattr(
            FusedMoEDecodeKernels, "install", lambda model: received.append(model)
        )
        runner = make_stub_runner(model=SimpleNamespace())
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        # Act
        lifecycle.install_decode_dispatch()

        # Assert
        assert received == [runner._forward_model]

    def test_pooling_runner_skips_decode_dispatch_install(self, monkeypatch):
        # Arrange — pooling runners never decode, and their backends may
        # wrap the model in non-nn.Module shims install must not walk.
        from types import SimpleNamespace

        from tests.stub_runner import make_stub_runner

        def unexpected_install(model):
            raise AssertionError("pooling runner must not install decode dispatch")

        monkeypatch.setattr(FusedMoEDecodeKernels, "install", unexpected_install)
        runner = make_stub_runner(
            model=SimpleNamespace(),
            model_config=SimpleNamespace(
                runner_type="pooling", get_head_size=lambda: 128, max_model_len=64
            ),
            _is_pooling=True,
            _pooling_backend=None,
        )
        lifecycle = ModelLifecycle(runner, runner._model_adapter)

        # Act / Assert (no raise)
        lifecycle.install_decode_dispatch()


class TestCompiledCache:
    def test_compiled_cache_is_bounded_lru(self, monkeypatch):
        # Arrange — shape keys beyond the cap must evict oldest-first
        # instead of growing without bound.
        monkeypatch.setattr(FusedMoEDecodeKernels, "_kernel", object())
        monkeypatch.setattr(
            FusedMoEDecodeKernels, "_compiled", type(FusedMoEDecodeKernels._compiled)()
        )
        cap = FusedMoEDecodeKernels.MAX_COMPILED_SHAPES

        # Act
        for tokens in range(1, cap + 8):
            FusedMoEDecodeKernels._compiled_step(
                tokens, TOP_K, HIDDEN, INTER, mx.float16
            )

        # Assert
        cache = FusedMoEDecodeKernels._compiled
        assert len(cache) == cap
        first_key = next(iter(cache))
        assert first_key[0] == 8  # keys 1..7 were evicted oldest-first
