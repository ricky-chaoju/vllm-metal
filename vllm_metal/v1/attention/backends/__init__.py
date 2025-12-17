# SPDX-License-Identifier: Apache-2.0
"""vLLM Metal attention backends."""

from vllm_metal.v1.attention.backends.mps_attn import MPSAttentionBackend

__all__ = ["MPSAttentionBackend"]
