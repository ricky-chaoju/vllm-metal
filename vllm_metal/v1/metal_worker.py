# SPDX-License-Identifier: Apache-2.0
"""Metal Worker for vLLM v1 API on Apple Silicon."""

import gc
import os
from typing import Any

import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class MetalWorker(Worker):
    """Worker implementation for Apple Metal/MPS backend.

    This worker handles inference on Apple Silicon devices using the
    Metal Performance Shaders (MPS) backend.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # MPS doesn't support custom all-reduce
        self.parallel_config.disable_custom_all_reduce = True

        # Torch profiler setup (CPU-only profiling for MPS)
        self.profiler: Any | None = None
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, worker_name=worker_name, use_gzip=False
                ),
            )

    def init_device(self):
        """Initialize the MPS device for inference."""
        from vllm_metal.v1.mps_model_runner import MPSModelRunner

        # Verify MPS is available
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise RuntimeError(
                    "MPS not available because PyTorch was not built with MPS support. "
                    "Please install a PyTorch version with MPS support."
                )
            else:
                raise RuntimeError(
                    "MPS not available. This may be because the macOS version is too old "
                    "or the hardware doesn't support Metal."
                )

        # Set device to MPS
        self.device = torch.device("mps")
        logger.info("Initializing Metal/MPS device")

        # Set environment variable for distributed identification
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(":")[-1]

        # Initialize distributed environment
        # MPS only supports single-device operation, but we still need to initialize
        # the distributed environment for compatibility
        _init_mps_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        # Set random seed
        set_random_seed(self.model_config.seed)

        # Get memory info
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Get initial memory state
        # MPS uses unified memory, so we get system memory info
        try:
            import psutil

            mem_info = psutil.virtual_memory()
            total_memory = mem_info.total
            available_memory = mem_info.available
        except ImportError:
            # Fallback if psutil not available
            total_memory = 16 * GiB_bytes  # Assume 16GB
            available_memory = total_memory // 2

        self.init_free_memory = available_memory
        self.init_total_memory = total_memory
        self.requested_memory = int(
            total_memory * self.cache_config.gpu_memory_utilization
        )

        logger.info(
            "MPS device initialized: total_memory=%.2f GiB, available=%.2f GiB",
            total_memory / GiB_bytes,
            available_memory / GiB_bytes,
        )

        # Construct the model runner
        self.model_runner = MPSModelRunner(self.vllm_config, self.device)

    def sleep(self, level: int = 1) -> None:
        """Sleep mode is not supported on MPS."""
        logger.warning("Sleep mode is not supported on MPS, ignoring.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up mode is not supported on MPS."""
        logger.warning("Sleep mode is not supported on MPS, ignoring.")

    def determine_available_memory(self) -> int:
        """Determine available memory for KV cache.

        MPS uses unified memory shared with the system, so we need to be
        conservative with memory allocation.
        """
        # If explicitly configured, use that value
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # Still run profile to compile model
            self.model_runner.profile_run()
            logger.info(
                "Using configured KV cache memory: %.2f GiB",
                kv_cache_memory_bytes / GiB_bytes,
            )
            return kv_cache_memory_bytes

        # Run profiling
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.model_runner.profile_run()

        # Synchronize MPS operations
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Get current memory usage
        try:
            import psutil

            mem_info = psutil.virtual_memory()
            current_available = mem_info.available
        except ImportError:
            current_available = self.init_free_memory // 2

        # Calculate model memory usage (estimated)
        model_memory = getattr(self.model_runner, "model_memory_usage", 0)
        if model_memory == 0:
            # Estimate based on model parameters
            model_memory = sum(
                p.numel() * p.element_size()
                for p in self.model_runner.model.parameters()
            )

        # Calculate available KV cache memory
        # Be conservative - use only a fraction of remaining memory
        # to leave room for system and other processes
        memory_fraction = self.cache_config.gpu_memory_utilization
        available_for_kv = int(current_available * memory_fraction * 0.8)

        # Ensure we have at least some memory for KV cache
        min_kv_cache = 512 * 1024 * 1024  # 512 MB minimum
        available_for_kv = max(available_for_kv, min_kv_cache)

        logger.info(
            "Available KV cache memory: %.2f GiB (model: %.2f GiB, available: %.2f GiB)",
            available_for_kv / GiB_bytes,
            model_memory / GiB_bytes,
            current_available / GiB_bytes,
        )

        self.available_kv_cache_memory_bytes = available_for_kv
        return available_for_kv

    def compile_or_warm_up_model(self) -> None:
        """Compile and warm up the model for MPS."""
        # Reset seed for reproducibility
        set_random_seed(self.model_config.seed)

        # Warm up the model
        self.model_runner.warming_up_model()

    def profile(self, is_start: bool = True):
        """Control profiling."""
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            if self.local_rank == 0:
                logger.info(
                    self.profiler.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=50
                    )
                )


def _init_mps_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
) -> None:
    """Initialize distributed environment for MPS.

    MPS only supports single-device operation, but we initialize the
    distributed environment for API compatibility.
    """
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    parallel_config = vllm_config.parallel_config

    init_batch_invariance()
    set_custom_all_reduce(False)  # MPS doesn't support custom all-reduce

    init_method = distributed_init_method or "env://"

    # Use gloo backend for MPS (nccl is CUDA-only)
    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        "gloo",  # Use gloo backend for MPS
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )
