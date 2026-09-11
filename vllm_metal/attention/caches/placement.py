# SPDX-License-Identifier: Apache-2.0
"""vLLM's KV cache placement contract as Metal consumes it.

vLLM resolves one physical layout per model from the layouts each worker
reports, then overlays every cache group on one backing allocation and
describes each group's layers with a ``KVCacheTensor`` that places layer
``l`` at ``offset + l * layer_stride``. Layers whose regions start at the
same address alias the same bytes (their groups own disjoint block ids);
Metal's per-runtime layouts derive their physical slots and state pools
from those addresses.
"""

from __future__ import annotations

from vllm.v1.kv_cache_interface import KVCacheLayout, KVCacheTensor

# vLLM's name for the page order Metal stores: [block, token, head, dim].
KV_CACHE_LAYOUT = KVCacheLayout.LBNHC.name


def layer_addresses(tensor: KVCacheTensor) -> list[tuple[str, int]]:
    """Return each layer's region start in the KV allocation."""
    return [
        (layer_name, tensor.offset + position * tensor.layer_stride)
        for position, layer_name in enumerate(tensor.layers)
    ]
