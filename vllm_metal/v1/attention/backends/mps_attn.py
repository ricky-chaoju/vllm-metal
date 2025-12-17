# SPDX-License-Identifier: Apache-2.0
"""MPS Attention Backend for vLLM on Apple Silicon.

This backend uses PyTorch's scaled_dot_product_attention which is
natively supported on MPS devices.
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Debug mode - set MPS_ATTN_DEBUG=1 to enable tracing
DEBUG = os.environ.get("MPS_ATTN_DEBUG", "0") == "1"
_debug_decode_count = 0


class MPSAttentionBackend(AttentionBackend):
    """Attention backend using PyTorch SDPA on MPS devices."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """MPS attention supports decoder and encoder-only attention."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionImpl"]:
        return MPSAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        # This matches flash_attn and flex_attention layout
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MPSAttentionMetadata:
    """Metadata for MPS attention."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    """Builder for MPS attention metadata."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config

        parallel_config = vllm_config.parallel_config
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(parallel_config)
        self.num_heads = vllm_config.model_config.get_num_attention_heads(
            parallel_config
        )
        self.head_dim = kv_cache_spec.head_size
        self.dtype = vllm_config.model_config.dtype
        self.window_size = getattr(kv_cache_spec, "sliding_window", -1)
        if self.window_size is None:
            self.window_size = -1
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        if DEBUG:
            # Sync before reading tensor values for debug
            torch.mps.synchronize()
            print("\n[MPS_ATTN DEBUG] build() called:")
            print(f"  num_actual_tokens={num_actual_tokens}")
            print(f"  max_query_len={max_query_len}")
            print(f"  max_seq_len={max_seq_len}")
            print(
                f"  slot_mapping[:min(10,len)]={slot_mapping[: min(10, len(slot_mapping))].tolist()}"
            )
            print(f"  seq_lens={seq_lens.tolist()}")

        # NOTE: We do NOT clone tensors here. The model runner updates these
        # tensors in-place between forward passes, and we need to see the
        # updated values. FlashAttention backend also does not clone.
        # Previous cloning caused stale metadata (seq_lens stayed constant,
        # slot_mapping didn't update between decode steps).

        # CRITICAL: Synchronize MPS to ensure all pending async copies from
        # CPU to MPS are complete. The base GPU model runner uses non_blocking=True
        # when copying tensors (like slot_mapping) from CPU to GPU, but MPS
        # doesn't handle async copies the same way CUDA does. Without this sync,
        # we may read stale data from slot_mapping during decode.
        torch.mps.synchronize()

        return MPSAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=causal,
        )


class MPSAttentionImpl(AttentionImpl):
    """MPS attention implementation using PyTorch SDPA."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.logits_soft_cap = logits_soft_cap if logits_soft_cap else 0.0

        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if is_quantized_kv_cache(kv_cache_dtype):
            raise NotImplementedError("FP8 KV cache is unsupported in MPS_ATTN")
        self.attn_type = attn_type

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for MPS attention backend.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported for MPSAttentionImpl"
            )

        # For warming-up
        if attn_metadata is None:
            return output

        # CRITICAL: Use slot_mapping.shape[0] to determine the actual number of tokens
        # for this forward pass. The num_actual_tokens field in attn_metadata is a
        # Python int that was set when build() was called, and may be stale.
        # The slot_mapping tensor, however, is sliced to the correct size for each
        # forward pass by the model runner (see CommonAttentionMetadata.unpadded()).
        # FlashAttention uses the same approach (see reshape_and_cache_flash).
        slot_mapping = attn_metadata.slot_mapping
        num_tokens = slot_mapping.shape[0]

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._run_sdpa_forward(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                output[:num_tokens],
                attn_metadata,
                self.attn_type,
            )

        # For decoder attention, use KV cache
        key_cache, value_cache = kv_cache.unbind(0)

        # Update KV cache
        # The slot_mapping tensor determines how many tokens to cache.
        # FlashAttention's reshape_and_cache_flash op uses slot_mapping's shape
        # to determine the number of actual tokens, so we follow the same pattern.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            self._update_kv_cache(
                key[:num_tokens],
                value[:num_tokens],
                key_cache,
                value_cache,
                slot_mapping,
            )

        # Run attention with KV cache
        return self._run_paged_attention(
            query[:num_tokens],
            key[:num_tokens],
            value[:num_tokens],
            key_cache,
            value_cache,
            output[:num_tokens],
            attn_metadata,
        )

    def _update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update the KV cache with new key/value tensors.

        Args:
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            key_cache: [num_blocks, block_size, num_kv_heads, head_size]
            value_cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [num_tokens]
        """
        num_tokens = key.shape[0]
        block_size = key_cache.shape[1]

        if DEBUG:
            block_size_debug = key_cache.shape[1]
            print("\n[MPS_ATTN DEBUG] _update_kv_cache:")
            print(f"  num_tokens={num_tokens}")
            print(f"  slot_mapping={slot_mapping.tolist()}")
            # Compute which blocks/offsets the slots map to
            for i in range(min(num_tokens, 5)):
                slot = slot_mapping[i].item()
                blk = slot // block_size_debug
                off = slot % block_size_debug
                print(f"  slot[{i}]={slot} -> block={blk}, offset={off}")

        # Compute block indices and offsets
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Update cache for each token
        # key[i] has shape [num_kv_heads, head_size]
        # key_cache[block_idx] has shape [block_size, num_kv_heads, head_size]
        for i in range(num_tokens):
            block_idx = block_indices[i].item()
            block_offset = block_offsets[i].item()
            if DEBUG and i < 3:
                print(
                    f"  Writing token {i} -> block={block_idx}, offset={block_offset}"
                )
            # key[i] is [num_kv_heads, head_size], we want to put it at position block_offset
            key_cache[block_idx, block_offset, :, :] = key[i]
            value_cache[block_idx, block_offset, :, :] = value[i]

        # Synchronize to ensure writes are committed before any reads
        torch.mps.synchronize()

    def _run_paged_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> torch.Tensor:
        """Run paged attention using SDPA on MPS.

        For now, this uses a simple per-sequence loop. Future optimization
        could batch multiple sequences together.
        """
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        slot_mapping = attn_metadata.slot_mapping
        # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        block_size = key_cache.shape[1]
        causal = attn_metadata.causal

        num_seqs = len(seq_lens)

        for seq_idx in range(num_seqs):
            # Get query range for this sequence
            q_start = query_start_loc[seq_idx].item()
            q_end = query_start_loc[seq_idx + 1].item()
            query_len = q_end - q_start

            # Get sequence length (total number of tokens including cached)
            seq_len = seq_lens[seq_idx].item()

            # Get query for this sequence
            seq_query = query[q_start:q_end]  # [query_len, num_heads, head_size]

            # Determine if this is prefill (query_len == seq_len) or decode (query_len < seq_len)
            is_prefill = query_len == seq_len

            if is_prefill:
                # During prefill, use the key/value directly (already in correct order)
                # This avoids any cache read/write ordering issues
                gathered_key = key[q_start:q_end]  # [seq_len, num_kv_heads, head_size]
                gathered_value = value[q_start:q_end]
            else:
                # During decode, gather from KV cache using block_table
                # block_table[i] gives the physical block index for logical block i
                num_blocks = (seq_len + block_size - 1) // block_size
                seq_block_table = block_table[seq_idx, :num_blocks]

                # Gather KV from cache for historical tokens
                # NOTE: We read seq_len-query_len tokens from cache (the historical K/V)
                # and use the current K/V directly for the new token(s) to avoid
                # cache read-after-write timing issues on MPS.
                historical_len = seq_len - query_len
                if historical_len > 0:
                    hist_num_blocks = (historical_len + block_size - 1) // block_size
                    hist_block_table = block_table[seq_idx, :hist_num_blocks]

                    if DEBUG:
                        print(f"\n[DECODE DEBUG] First decode for seq {seq_idx}:")
                        print(
                            f"  seq_len={seq_len}, query_len={query_len}, historical_len={historical_len}"
                        )
                        print(f"  hist_num_blocks={hist_num_blocks}")
                        print(f"  hist_block_table={hist_block_table.tolist()}")
                        print(f"  full block_table={seq_block_table.tolist()}")
                        # Check if cache at first block is non-zero
                        first_block = hist_block_table[0].item()
                        cache_sample = key_cache[first_block, 0, 0, :3]
                        print(
                            f"  key_cache[{first_block}, 0, 0, :3]={cache_sample.tolist()}"
                        )
                        # Check the current token's K/V from input
                        print(f"  key[q_start:q_end] shape: {key[q_start:q_end].shape}")
                        print(
                            f"  key[q_start:q_end] abs max: {key[q_start:q_end].abs().max().item():.6f}"
                        )
                        # Check slot_mapping to see where this token was written
                        seq_slots = slot_mapping[q_start:q_end]
                        print(f"  slot_mapping[q_start:q_end]={seq_slots.tolist()}")

                    gathered_hist_key = self._gather_kv_from_cache(
                        key_cache, hist_block_table, historical_len, block_size
                    )
                    gathered_hist_value = self._gather_kv_from_cache(
                        value_cache, hist_block_table, historical_len, block_size
                    )

                    if DEBUG:
                        gathered_max = gathered_hist_key.abs().max().item()
                        print(f"  gathered_hist_key abs max: {gathered_max:.6f}")
                        print(
                            f"  gathered_key (after concat) shape: {torch.cat([gathered_hist_key, key[q_start:q_end]], dim=0).shape}"
                        )

                    # Concatenate historical from cache with current from input
                    gathered_key = torch.cat(
                        [gathered_hist_key, key[q_start:q_end]], dim=0
                    )
                    gathered_value = torch.cat(
                        [gathered_hist_value, value[q_start:q_end]], dim=0
                    )
                else:
                    # No historical tokens, just use current
                    gathered_key = key[q_start:q_end]
                    gathered_value = value[q_start:q_end]

            # Run SDPA for this sequence
            # seq_output is [query_len, num_heads, head_size]
            seq_output = self._compute_attention(
                seq_query,
                gathered_key,
                gathered_value,
                causal and self.attn_type == AttentionType.DECODER,
            )

            # Write output - output tensor is [num_tokens, num_heads, head_size] (view of 2D)
            output[q_start:q_end] = seq_output

        # CRITICAL: Synchronize to ensure all attention outputs are written before
        # they are read for hidden states extraction. MPS operations are asynchronous
        # and without this sync, subsequent reads (e.g., hidden_states[logits_indices])
        # may see uninitialized/zero values at the last position.
        torch.mps.synchronize()

        return output

    def _gather_kv_from_cache_using_slots(
        self,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Gather KV from paged cache using slot_mapping directly.

        This is the more direct approach - using the exact same slot indices
        that were used to write to the cache.

        Args:
            cache: [num_blocks, block_size, num_kv_heads, head_size]
            slot_mapping: [seq_len] - slot indices for each position
            block_size: Block size

        Returns:
            [seq_len, num_kv_heads, head_size]
        """
        num_kv_heads = cache.shape[2]
        head_size = cache.shape[3]
        seq_len = slot_mapping.shape[0]

        # Synchronize before reading to ensure writes are complete
        torch.mps.synchronize()

        # Compute block indices and offsets from slot_mapping
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Allocate output tensor
        gathered = torch.zeros(
            seq_len, num_kv_heads, head_size, dtype=cache.dtype, device=cache.device
        )

        # Gather from cache using the same indexing as writes
        for i in range(seq_len):
            block_idx = block_indices[i].item()
            block_offset = block_offsets[i].item()
            gathered[i] = cache[block_idx, block_offset].clone()

        # Synchronize after reading to ensure data is materialized
        torch.mps.synchronize()

        return gathered

    def _gather_kv_from_cache(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        block_size: int,
    ) -> torch.Tensor:
        """Gather KV from paged cache.

        Args:
            cache: [num_blocks, block_size, num_kv_heads, head_size]
            block_table: [num_blocks_for_seq]
            seq_len: Total sequence length
            block_size: Block size

        Returns:
            [seq_len, num_kv_heads, head_size]
        """
        num_kv_heads = cache.shape[2]
        head_size = cache.shape[3]

        # Synchronize before reading to ensure writes are complete
        torch.mps.synchronize()

        # Allocate output tensor
        gathered = torch.zeros(
            seq_len, num_kv_heads, head_size, dtype=cache.dtype, device=cache.device
        )

        # Gather from each block using explicit loop and copy
        # cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        pos = 0
        for i in range(len(block_table)):
            block_idx = block_table[i].item()
            tokens_in_block = min(block_size, seq_len - pos)
            if tokens_in_block <= 0:
                break
            # Copy entire block slice at once for efficiency
            gathered[pos : pos + tokens_in_block] = cache[
                block_idx, :tokens_in_block
            ].clone()
            pos += tokens_in_block

        # Synchronize after reading to ensure data is materialized
        torch.mps.synchronize()

        return gathered

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Compute attention using SDPA.

        Args:
            query: [query_len, num_heads, head_size]
            key: [seq_len, num_kv_heads, head_size]
            value: [seq_len, num_kv_heads, head_size]
            causal: Whether to use causal attention

        Returns:
            [query_len, num_heads, head_size]
        """
        query_len = query.shape[0]
        seq_len = key.shape[0]

        # Expand KV heads if using grouped-query attention
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Reshape for SDPA: [batch, heads, seq_len, head_size]
        # Add batch dimension and transpose
        q = query.unsqueeze(0).transpose(1, 2)  # [1, num_heads, query_len, head_size]
        k = key.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]
        v = value.unsqueeze(0).transpose(1, 2)  # [1, num_heads, seq_len, head_size]

        # Handle causal mask for decode (single token query)
        is_causal = causal and query_len == seq_len

        # For decode with single query token, we need a proper mask
        attn_mask = None
        if causal and query_len != seq_len:
            # Decode mode: query sees all past tokens but not future
            # No mask needed for single query against all keys
            pass

        # Run scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

        # Reshape output: [1, num_heads, query_len, head_size] -> [query_len, num_heads, head_size]
        attn_output = attn_output.squeeze(0).transpose(0, 1)

        return attn_output

    def _run_sdpa_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
        attn_type: str,
    ) -> torch.Tensor:
        """Run SDPA for encoder attention (no KV cache)."""
        query_start_loc = attn_metadata.query_start_loc.cpu()
        causal = attn_type == AttentionType.DECODER

        num_seqs = query_start_loc.shape[0] - 1

        for i in range(num_seqs):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()

            seq_output = self._compute_attention(
                query[start:end],
                key[start:end],
                value[start:end],
                causal,
            )
            output[start:end] = seq_output

        return output
