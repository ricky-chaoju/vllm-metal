# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for vLLM + transformers version mismatches.

Applied once at platform registration time.  Each patch is guarded by
try/except so it degrades silently if the target module changes.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_compat_patches() -> None:
    """Apply all known compatibility patches (idempotent)."""
    global _APPLIED  # noqa: PLW0603
    if _APPLIED:
        return
    _APPLIED = True
    _patch_qwen35_rope_validation()
    _patch_mlx_lm_qwen35_fp8_sanitize()


def _patch_qwen35_rope_validation() -> None:
    """Fix vLLM 0.17.1 Qwen3.5 config vs transformers >=5.4 rope validation.

    vLLM's ``Qwen3_5TextConfig.__init__`` hardcodes
    ``kwargs["ignore_keys_at_rope_validation"] = [...]`` (a list), but
    transformers 5.4+ does ``received_keys -= ignore_keys`` which requires
    a set.

    Upstream: vllm-project/vllm#34604 fixed this but was reverted in #34610.
    Remove this patch when vllm-metal upgrades to a vLLM version with the fix.
    """
    from importlib.util import find_spec

    if find_spec("vllm.transformers_utils.configs.qwen3_5") is None:
        return

    try:
        from transformers.modeling_rope_utils import RopeConfigBase

        rope_config_base = RopeConfigBase
    except ImportError:
        rope_config_base = None

    if rope_config_base is None:
        # Try the direct path
        try:
            import transformers.modeling_rope_utils as _rope

            _orig_check = _rope._check_received_keys

            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            _rope._check_received_keys = _safe_check
            logger.debug("Patched _check_received_keys for rope validation compat")
            return
        except (ImportError, AttributeError):
            pass

    # Fallback: patch the static method on PreTrainedConfig if available
    try:
        from transformers import PreTrainedConfig

        if hasattr(PreTrainedConfig, "_check_received_keys"):
            _orig_check = PreTrainedConfig._check_received_keys

            @staticmethod
            def _safe_check(
                rope_type,
                received_keys,
                required_keys,
                optional_keys=None,
                ignore_keys=None,
            ):
                if ignore_keys is not None and isinstance(ignore_keys, list):
                    ignore_keys = set(ignore_keys)
                return _orig_check(
                    rope_type, received_keys, required_keys, optional_keys, ignore_keys
                )

            PreTrainedConfig._check_received_keys = _safe_check
            logger.debug(
                "Patched PreTrainedConfig._check_received_keys for rope compat"
            )
    except (ImportError, AttributeError):
        pass


def _patch_mlx_lm_qwen35_fp8_sanitize() -> None:
    """Teach mlx_lm's Qwen3.5 loaders to consume local FP8 ``weight_scale_inv``.

    Some Qwen3.5/Qwen3.6 local checkpoints store FP8 weights plus
    ``*_weight_scale_inv`` tensors in HuggingFace-style shards. The installed
    mlx_lm ``qwen3_5`` loaders do not currently dequantize those tensors during
    ``sanitize()``, so ``model.load_weights()`` aborts with hundreds of
    unexpected ``weight_scale_inv`` parameters.

    Patch the top-level model ``sanitize()`` methods to dequantize those FP8
    tensors before the upstream remapping logic runs. This keeps the workaround
    narrow to the affected architectures and leaves upstream control flow intact.
    """
    from importlib import import_module
    from importlib.util import find_spec

    try:
        import mlx.core as mx
    except Exception:
        return

    model_modules = []
    for module_name in ("mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_5_moe"):
        if find_spec(module_name) is None:
            continue
        try:
            model_modules.append(import_module(module_name))
        except Exception:
            continue
    if not model_modules:
        return

    def _dequantize_qwen35_fp8_weights(weights):
        if not any("weight_scale_inv" in key for key in weights):
            return weights

        def _dequant(weight, scale_inv):
            block_size = 128
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            if weight.ndim < 2:
                return weight.astype(mx.bfloat16)

            leading_shape = weight.shape[:-2]
            rows, cols = weight.shape[-2:]
            pad_rows = (-rows) % block_size
            pad_cols = (-cols) % block_size
            pad_width = [(0, 0)] * len(leading_shape)
            pad_width.extend(((0, pad_rows), (0, pad_cols)))
            weight = mx.pad(weight, pad_width)
            block_rows = (rows + pad_rows) // block_size
            block_cols = (cols + pad_cols) // block_size
            weight = weight.reshape(
                (*leading_shape, block_rows, block_size, block_cols, block_size)
            )
            weight = (weight * scale_inv[..., :, None, :, None]).reshape(
                *leading_shape,
                rows + pad_rows,
                cols + pad_cols,
            )
            return weight[..., :rows, :cols].astype(mx.bfloat16)

        new_weights = {}
        for key, value in weights.items():
            if "weight_scale_inv" in key:
                weight_key = key.replace("_scale_inv", "")
                weight = weights[weight_key]
                new_weights[weight_key] = _dequant(weight, value)
            elif "activation_scale" in key:
                continue
            elif key not in new_weights:
                new_weights[key] = value
        return new_weights

    def _patch_model_sanitize(model_cls) -> bool:
        sanitize = getattr(model_cls, "sanitize", None)
        if sanitize is None or getattr(sanitize, "_vllm_metal_qwen35_fp8_patch", False):
            return False

        original_sanitize = sanitize

        def _patched_sanitize(self, weights):
            return original_sanitize(self, _dequantize_qwen35_fp8_weights(weights))

        _patched_sanitize._vllm_metal_qwen35_fp8_patch = True
        model_cls.sanitize = _patched_sanitize
        return True

    patched_modules = []
    for module in model_modules:
        model_cls = getattr(module, "Model", None)
        if model_cls is not None and _patch_model_sanitize(model_cls):
            patched_modules.append(module.__name__.rsplit(".", maxsplit=1)[-1])
    if patched_modules:
        logger.debug(
            "Patched mlx_lm %s FP8 sanitize compatibility",
            ", ".join(sorted(patched_modules)),
        )
