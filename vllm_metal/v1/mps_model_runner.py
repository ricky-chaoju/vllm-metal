# SPDX-License-Identifier: Apache-2.0
"""MPS Model Runner for vLLM v1 API on Apple Silicon."""

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class MPSModelRunner(GPUModelRunner):
    """Model runner for Apple MPS (Metal Performance Shaders) backend.

    This inherits from GPUModelRunner but adapts it for MPS devices,
    similar to how CPUModelRunner adapts it for CPU.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        assert device.type == "mps", f"Expected MPS device, got {device}"
        assert self.speculative_config is None, (
            "Speculative decoding is not supported on MPS."
        )

        # MPS doesn't support CUDA graphs
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        self._postprocess_tensors()

    def _postprocess_tensors(self) -> None:
        """Replace CUDA-specific tensors with MPS-compatible ones.

        MPS uses unified memory, so we can use the same tensors for both
        CPU and GPU operations in many cases.
        """

        def replace_tensor(obj: Any, cpu_attr_name: str, device_attr_name: str) -> None:
            cpu_tensor = getattr(obj, cpu_attr_name, None)
            device_tensor = getattr(obj, device_attr_name, None)
            if cpu_tensor is not None and device_tensor is not None:
                assert isinstance(cpu_tensor, torch.Tensor)
                assert isinstance(device_tensor, torch.Tensor)
                # Move CPU tensor to MPS device
                setattr(obj, device_attr_name, cpu_tensor.to(self.device))

        # Handle CpuGpuBuffer objects - MPS can use unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                # For MPS, move the cpu buffer to MPS device
                v.gpu = v.cpu.to(self.device)

        # Handle input batch tensors
        for k, v in vars(self.input_batch).items():
            if k.endswith("_cpu_tensor") and isinstance(v, torch.Tensor):
                replace_tensor(self.input_batch, k, k[:-11])

        # Handle block tables
        for block_table in self.input_batch.block_table.block_tables:
            for v in vars(block_table).values():
                if isinstance(v, CpuGpuBuffer):
                    v.gpu = v.cpu.to(self.device)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """Load model onto MPS device."""
        logger.info("Starting to load model %s on MPS...", self.model_config.model)
        self.model = get_model(vllm_config=self.vllm_config)

        # Ensure model is on MPS device
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        if self.lora_config:
            self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

    def get_model(self) -> nn.Module:
        return self.model

    def warming_up_model(self) -> None:
        """Warm up the model for MPS compilation."""
        logger.info("Warming up model for MPS...")

        # Run a dummy forward pass to compile any lazy operations
        with _set_mps_compilation_settings(self.vllm_config):
            self._dummy_run(
                min(
                    max(16, self.max_num_reqs),
                    self.scheduler_config.max_num_batched_tokens,
                )
            )

        # Synchronize MPS to ensure compilation is complete
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        logger.info("MPS warming up done.")

    def _init_device_properties(self) -> None:
        """Initialize MPS device properties.

        MPS doesn't have compute capability like CUDA, so this is mostly a no-op.
        """
        pass

    def _sync_device(self) -> None:
        """Synchronize MPS device."""
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """Get data parallel padding.

        MPS doesn't support distributed training, so no padding is needed.
        """
        return 0, None

    def capture_model(self) -> int:
        """Capture model for graph execution.

        MPS doesn't support CUDA graphs, so this returns 0.
        """
        logger.debug("MPS does not support CUDA graph capture, skipping.")
        return 0


@contextmanager
def _torch_cuda_wrapper():
    """Context manager to mock CUDA operations for MPS.

    This allows us to reuse GPU-focused code paths that reference
    CUDA-specific constructs like Events and Streams.
    """

    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def _set_mps_compilation_settings(config: VllmConfig):
    """Set up compilation settings for MPS.

    MPS uses different compilation paths than CUDA.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
