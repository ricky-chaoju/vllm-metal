# SPDX-License-Identifier: Apache-2.0
"""Immutable Metal layout translated from vLLM's standard attention cache DTOs.

vLLM owns KV-cache grouping and capacity planning. This module only validates
and translates the resulting ``KVCacheConfig`` for the standard mixed
full/sliding-window attention path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    SlidingWindowSpec,
)

from vllm_metal.attention.caches.placement import layer_addresses

NO_SLIDING_WINDOW = -1
StandardAttentionSpec: TypeAlias = FullAttentionSpec | SlidingWindowSpec


@dataclass(frozen=True, slots=True)
class AttentionGroupLayout:
    """vLLM cache-group specs and layer-to-group mapping."""

    specs: tuple[StandardAttentionSpec, ...]
    layer_indices: dict[str, int]


@dataclass(frozen=True, slots=True)
class SlotLayout:
    """vLLM physical slots (distinct region addresses) and layer-to-slot mapping."""

    layer_indices: dict[str, int]
    slot_layers: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class AttentionLayerKVLayout:
    """KV-cache shape and vLLM mapping for one model layer."""

    slot_index: int
    group_index: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int

    def cache_shape(self, num_blocks: int) -> tuple[int, int, int, int]:
        """Return the key or value cache shape for ``num_blocks`` pages."""
        return (num_blocks, self.block_size, self.num_kv_heads, self.head_dim)


@dataclass(frozen=True, slots=True)
class AttentionKVCacheLayout:
    """Immutable standard attention cache layout derived from vLLM's DTOs."""

    num_blocks: int
    allocation_bytes: int
    layers: tuple[AttentionLayerKVLayout, ...]
    group_block_sizes: tuple[int, ...]
    slot_layers: tuple[tuple[int, ...], ...]

    @property
    def total_bytes(self) -> int:
        """Return the backing allocation vLLM planned for the KV tensors."""
        return self.allocation_bytes

    @classmethod
    def from_config(
        cls, config: KVCacheConfig, model_layer_names: tuple[str, ...]
    ) -> AttentionKVCacheLayout:
        """Translate a standard mixed-attention ``KVCacheConfig`` without regrouping.

        ``model_layer_names`` is the runner's ordered attention-layer sequence.
        Each layer must occur exactly once in vLLM's group and tensor mappings.
        """
        return AttentionKVCacheLayoutTranslator(config, model_layer_names).translate()


@dataclass(frozen=True, slots=True)
class AttentionKVCacheLayoutTranslator:
    """Translate vLLM's standard attention KV cache config into Metal's layout DTO."""

    config: KVCacheConfig
    model_layer_names: tuple[str, ...]

    def translate(self) -> AttentionKVCacheLayout:
        """Translate without changing vLLM's grouping."""
        group_layout = self._group_layout()
        self._require_model_layers(group_layout.layer_indices, "group")

        slot_layout = self._slot_layout(group_layout)
        self._require_model_layers(slot_layout.layer_indices, "tensor")

        return AttentionKVCacheLayout(
            num_blocks=self.config.num_blocks,
            allocation_bytes=self.config.kv_cache_tensors[0].size,
            layers=self._layer_layouts(group_layout, slot_layout),
            group_block_sizes=tuple(spec.block_size for spec in group_layout.specs),
            slot_layers=slot_layout.slot_layers,
        )

    @property
    def _model_layer_indices(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.model_layer_names)}

    def _group_layout(self) -> AttentionGroupLayout:
        specs: list[StandardAttentionSpec] = []
        layer_indices: dict[str, int] = {}
        for group_index, group in enumerate(self.config.kv_cache_groups):
            spec = group.kv_cache_spec
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                raise NotImplementedError(
                    "standard attention layout requires FullAttentionSpec or "
                    "SlidingWindowSpec groups"
                )
            if spec.head_size_v != spec.head_size:
                raise NotImplementedError(
                    "standard attention layout requires matching key and value "
                    "head sizes"
                )

            specs.append(spec)
            for layer_name in group.layer_names:
                layer_indices[layer_name] = group_index
        return AttentionGroupLayout(specs=tuple(specs), layer_indices=layer_indices)

    def _slot_layout(self, group_layout: AttentionGroupLayout) -> SlotLayout:
        layer_indices: dict[str, int] = {}
        slot_layers: list[list[int]] = []
        slot_by_address: dict[int, int] = {}
        model_layer_indices = self._model_layer_indices

        for tensor_index, tensor in enumerate(self.config.kv_cache_tensors):
            spec = group_layout.specs[group_layout.layer_indices[tensor.layers[0]]]
            self._require_layer_outermost(tensor, tensor_index, spec)
            for layer_name, address in layer_addresses(tensor):
                slot = slot_by_address.setdefault(address, len(slot_layers))
                if slot == len(slot_layers):
                    slot_layers.append([])
                layer_indices[layer_name] = slot
                slot_layers[slot].append(model_layer_indices[layer_name])

        return SlotLayout(
            layer_indices=layer_indices,
            slot_layers=tuple(tuple(layers) for layers in slot_layers),
        )

    def _layer_layouts(
        self,
        group_layout: AttentionGroupLayout,
        slot_layout: SlotLayout,
    ) -> tuple[AttentionLayerKVLayout, ...]:
        return tuple(
            self._layer_layout(layer_name, group_layout, slot_layout)
            for layer_name in self.model_layer_names
        )

    def _layer_layout(
        self,
        layer_name: str,
        group_layout: AttentionGroupLayout,
        slot_layout: SlotLayout,
    ) -> AttentionLayerKVLayout:
        group_index = group_layout.layer_indices[layer_name]
        spec = group_layout.specs[group_index]
        return AttentionLayerKVLayout(
            slot_index=slot_layout.layer_indices[layer_name],
            group_index=group_index,
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_size,
            sliding_window=(
                spec.sliding_window
                if isinstance(spec, SlidingWindowSpec)
                else NO_SLIDING_WINDOW
            ),
        )

    def _require_model_layers(self, layer_mapping: dict[str, int], source: str) -> None:
        if set(layer_mapping) != set(self.model_layer_names):
            raise ValueError(
                f"{source} layer mapping must contain the same layers as "
                "model_layer_names"
            )

    def _require_layer_outermost(
        self, tensor: KVCacheTensor, tensor_index: int, spec: StandardAttentionSpec
    ) -> None:
        region_bytes = self.config.num_blocks * spec.page_size_bytes
        if (
            tensor.block_stride != spec.page_size_bytes
            or tensor.layer_stride != region_bytes
        ):
            raise NotImplementedError(
                "standard attention layout requires layer-outermost KV tensors "
                f"(block_stride {spec.page_size_bytes}, layer_stride "
                f"{region_bytes}); tensor {tensor_index} has block_stride "
                f"{tensor.block_stride}, layer_stride {tensor.layer_stride}"
            )
