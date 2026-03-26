# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid model dimension extraction, cache specs, and backend allocation.

Verifies that Qwen3.5-style hybrid models are correctly detected and that
cache allocation uses the right dimensions.

Config fixtures ``qwen35_4b_args`` and ``llama_args`` are defined in conftest.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from vllm_metal.v1.model_runner import MetalModelRunner


def _make_runner(args: dict) -> MagicMock:
    """Create a mock runner with real property/method wiring."""
    runner = MagicMock(spec=MetalModelRunner)
    runner.model_args = dict(args)
    runner.is_mla = MetalModelRunner.is_mla.fget(runner)
    runner.is_hybrid = MetalModelRunner.is_hybrid.fget(runner)
    runner._is_stt = False
    runner.kv_cache_dtype = mx.float16
    runner.metal_config = MagicMock(block_size=16)
    MetalModelRunner._resolve_model_dims(runner)
    return runner


def _make_hybrid_backend(args: dict, **overrides):
    """Create a HybridPagedAttentionBackend from model args."""
    from vllm_metal.paged_attention_backend.hybrid import (
        HybridPagedAttentionBackend,
    )

    defaults = {
        "num_layers": args["num_hidden_layers"],
        "full_attention_interval": args["full_attention_interval"],
        "max_num_seqs": 8,
        "num_kv_heads": args["num_key_value_heads"],
        "head_dim": args["head_dim"],
        "linear_num_k_heads": args["linear_num_key_heads"],
        "linear_num_v_heads": args["linear_num_value_heads"],
        "linear_key_head_dim": args["linear_key_head_dim"],
        "linear_value_head_dim": args["linear_value_head_dim"],
        "linear_conv_kernel_dim": args["linear_conv_kernel_dim"],
        "linear_conv_dim": (
            args["linear_num_key_heads"] * args["linear_key_head_dim"] * 2
            + args["linear_num_value_heads"] * args["linear_value_head_dim"]
        ),
        "block_size": 16,
        "dtype": mx.float16,
    }
    defaults.update(overrides)
    return HybridPagedAttentionBackend(**defaults)


# === _extract_model_args ===


class TestExtractModelArgs:
    @staticmethod
    def _call_extract(model_args_dict: dict) -> dict:
        runner = MagicMock(spec=MetalModelRunner)
        runner.metal_config = MagicMock(debug=False)
        runner.model = SimpleNamespace(args=SimpleNamespace(**model_args_dict))
        runner._is_vlm = False
        MetalModelRunner._extract_model_args(runner)
        return runner.model_args

    def test_qwen35_text_config_unwrapped(self, qwen35_4b_args: dict) -> None:
        args = self._call_extract(
            {"model_type": "qwen3_5_text", "text_config": qwen35_4b_args}
        )
        assert args["num_hidden_layers"] == qwen35_4b_args["num_hidden_layers"]
        assert (
            args["full_attention_interval"] == qwen35_4b_args["full_attention_interval"]
        )

    def test_flat_args_unchanged(self, llama_args: dict) -> None:
        args = self._call_extract(llama_args)
        assert args["num_hidden_layers"] == llama_args["num_hidden_layers"]

    def test_top_level_takes_precedence(self) -> None:
        args = self._call_extract(
            {"num_hidden_layers": 99, "text_config": {"num_hidden_layers": 32}}
        )
        assert args["num_hidden_layers"] == 99


# === is_hybrid ===


class TestIsHybrid:
    def test_hybrid(self, qwen35_4b_args: dict) -> None:
        assert _make_runner(qwen35_4b_args).is_hybrid is True

    def test_not_hybrid(self, llama_args: dict) -> None:
        assert _make_runner(llama_args).is_hybrid is False


# === _resolve_model_dims ===


class TestResolveModelDims:
    def test_hybrid_layer_counts(self, qwen35_4b_args: dict) -> None:
        runner = _make_runner(qwen35_4b_args)
        n = qwen35_4b_args["num_hidden_layers"]
        fai = qwen35_4b_args["full_attention_interval"]
        expected_sdpa = sum(1 for i in range(n) if (i + 1) % fai == 0)
        assert runner.num_layers == n
        assert runner.num_sdpa_layers == expected_sdpa
        assert runner.num_linear_layers == n - expected_sdpa

    def test_hybrid_conv_dim(self, qwen35_4b_args: dict) -> None:
        runner = _make_runner(qwen35_4b_args)
        expected = (
            qwen35_4b_args["linear_num_key_heads"]
            * qwen35_4b_args["linear_key_head_dim"]
            * 2
            + qwen35_4b_args["linear_num_value_heads"]
            * qwen35_4b_args["linear_value_head_dim"]
        )
        assert runner.linear_conv_dim == expected


# === get_cache_block_size_bytes ===


class TestCacheBlockSizeBytes:
    def test_hybrid_counts_sdpa_only(self, qwen35_4b_args: dict) -> None:
        runner = _make_runner(qwen35_4b_args)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        # Removing full_attention_interval makes all layers SDPA → larger block.
        non_hybrid_args = dict(qwen35_4b_args)
        del non_hybrid_args["full_attention_interval"]
        non_hybrid = MetalModelRunner.get_cache_block_size_bytes(
            _make_runner(non_hybrid_args)
        )
        assert 0 < result < non_hybrid

    def test_non_hybrid_all_layers(self, llama_args: dict) -> None:
        runner = _make_runner(llama_args)
        result = MetalModelRunner.get_cache_block_size_bytes(runner)
        block_size = runner.metal_config.block_size
        expected = (
            2
            * llama_args["num_hidden_layers"]
            * block_size
            * llama_args["num_key_value_heads"]
            * llama_args["head_dim"]
            * runner.kv_cache_dtype.size
        )
        assert result == expected

    def test_linear_cache_bytes_positive(self, qwen35_4b_args: dict) -> None:
        runner = _make_runner(qwen35_4b_args)
        assert MetalModelRunner.linear_cache_bytes_per_slot(runner) > 0


# === get_kv_cache_spec ===


class TestKVCacheSpec:
    def test_hybrid_emits_mixed_specs(self, qwen35_4b_args: dict) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        runner = _make_runner(qwen35_4b_args)
        specs = MetalModelRunner.get_kv_cache_spec(runner)
        n = qwen35_4b_args["num_hidden_layers"]
        fai = qwen35_4b_args["full_attention_interval"]
        expected_sdpa = sum(1 for i in range(n) if (i + 1) % fai == 0)

        assert len(specs) == n
        assert (
            sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
            == expected_sdpa
        )
        assert (
            sum(1 for s in specs.values() if isinstance(s, MambaSpec))
            == n - expected_sdpa
        )

    def test_non_hybrid_all_full(self, llama_args: dict) -> None:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        specs = MetalModelRunner.get_kv_cache_spec(_make_runner(llama_args))
        assert all(isinstance(s, FullAttentionSpec) for s in specs.values())


# === HybridPagedAttentionBackend ===


class TestHybridBackend:
    def test_allocates_separate_caches(self, qwen35_4b_args: dict) -> None:
        backend = _make_hybrid_backend(qwen35_4b_args)
        backend.initialize(num_blocks=100)

        n = qwen35_4b_args["num_hidden_layers"]
        fai = qwen35_4b_args["full_attention_interval"]
        expected_sdpa = sum(1 for i in range(n) if (i + 1) % fai == 0)

        assert backend.kv_cache.num_layers == expected_sdpa
        assert backend.kv_cache.num_blocks == 100
        assert backend.linear_cache.num_layers == n - expected_sdpa
        assert backend.linear_cache.num_blocks == 8  # max_num_seqs

    def test_num_blocks_before_init_raises(self, qwen35_4b_args: dict) -> None:
        with pytest.raises(RuntimeError):
            _make_hybrid_backend(qwen35_4b_args).num_blocks()

    def test_kv_budget_subtracts_linear(self, qwen35_4b_args: dict) -> None:
        from vllm_metal.v1.worker import MetalWorker

        budget = MetalWorker._kv_budget_bytes(
            metal_limit=16_000_000_000, model_memory=4_000_000_000, fraction=0.5
        )
        runner = _make_runner(qwen35_4b_args)
        linear_fixed = MetalModelRunner.linear_cache_bytes_per_slot(runner) * 8
        sdpa_per_block = MetalModelRunner.get_cache_block_size_bytes(runner)

        blocks_without = budget // sdpa_per_block
        blocks_with = (budget - linear_fixed) // sdpa_per_block

        assert 0 < blocks_with < blocks_without
