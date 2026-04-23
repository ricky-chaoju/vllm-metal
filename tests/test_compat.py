# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime compatibility patches."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import numpy as np

import vllm_metal.compat as compat


def _install_fake_qwen35_modules(monkeypatch, *, include_moe: bool):
    mlx_pkg = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.bfloat16 = np.float32
    mlx_core.from_fp8 = lambda weight, dtype=None: np.asarray(weight, dtype=np.float32)
    mlx_core.pad = lambda weight, pad_width: np.pad(weight, pad_width)
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    mlx_lm_pkg = ModuleType("mlx_lm")
    mlx_lm_models = ModuleType("mlx_lm.models")
    mlx_lm_pkg.models = mlx_lm_models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", mlx_lm_models)

    dense_module = ModuleType("mlx_lm.models.qwen3_5")

    class DenseModel:
        def sanitize(self, weights):
            return dict(weights)

    dense_module.Model = DenseModel
    monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5", dense_module)
    mlx_lm_models.qwen3_5 = dense_module

    moe_module = None
    if include_moe:
        moe_module = ModuleType("mlx_lm.models.qwen3_5_moe")

        class MoeModel:
            def sanitize(self, weights):
                return dict(weights)

        moe_module.Model = MoeModel
        monkeypatch.setitem(sys.modules, "mlx_lm.models.qwen3_5_moe", moe_module)
        mlx_lm_models.qwen3_5_moe = moe_module

    def _fake_find_spec(name: str):
        if name == "mlx_lm.models.qwen3_5":
            return object()
        if name == "mlx_lm.models.qwen3_5_moe":
            return object() if include_moe else None
        return None

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    return dense_module, moe_module


class TestQwen35Fp8CompatPatch:
    def test_patches_dense_qwen35_even_when_moe_module_is_missing(
        self, monkeypatch
    ) -> None:
        dense_module, _ = _install_fake_qwen35_modules(monkeypatch, include_moe=False)

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        sanitized = dense_module.Model().sanitize(
            {
                "language_model.layers.0.linear.weight": np.ones((128, 128)),
                "language_model.layers.0.linear.weight_scale_inv": np.ones((1, 1)),
            }
        )

        assert "language_model.layers.0.linear.weight_scale_inv" not in sanitized
        assert sanitized["language_model.layers.0.linear.weight"].shape == (128, 128)

    def test_patches_higher_rank_weights_for_moe(
        self, monkeypatch
    ) -> None:
        _, moe_module = _install_fake_qwen35_modules(monkeypatch, include_moe=True)
        gate_up_proj_prefix = "language_model.layers.0.mlp.experts.gate_up_proj"

        compat._patch_mlx_lm_qwen35_fp8_sanitize()

        sanitized = moe_module.Model().sanitize(
            {
                f"{gate_up_proj_prefix}.weight": np.ones((2, 256, 128)),
                f"{gate_up_proj_prefix}.weight_scale_inv": np.ones((2, 2, 1)),
                f"{gate_up_proj_prefix}.activation_scale": np.ones((2, 2, 1)),
            }
        )

        assert f"{gate_up_proj_prefix}.weight_scale_inv" not in sanitized
        assert f"{gate_up_proj_prefix}.activation_scale" not in sanitized
        assert sanitized[f"{gate_up_proj_prefix}.weight"].shape == (2, 256, 128)
