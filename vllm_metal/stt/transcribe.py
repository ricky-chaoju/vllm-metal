# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text model orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from vllm_metal.stt.config import SpeechToTextConfig
from vllm_metal.stt.loader import load_model as _load_model
from vllm_metal.stt.protocol import TranscriptionResult
from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber  # noqa: F401
from vllm_metal.stt.whisper import WhisperModel, WhisperTranscriber

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path, dtype: mx.Dtype = mx.float16):
    """Load an STT model from a local directory or HuggingFace repo."""
    return _load_model(model_path, dtype)


# ===========================================================================
# Convenience entrypoint
# ===========================================================================


def transcribe(
    model: WhisperModel,
    audio: str | np.ndarray | mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    with_timestamps: bool = False,
    model_path: str | None = None,
    stt_config: SpeechToTextConfig | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text (convenience wrapper).

    Creates a :class:`WhisperTranscriber` and delegates. For repeated
    calls, prefer constructing a transcriber directly to reuse the
    tokenizer.

    Args:
        model: Loaded :class:`WhisperModel`.
        audio: File path, numpy array, or MLX array.
        language: Language code.
        task: ``"transcribe"`` or ``"translate"``.
        prompt: Optional context prompt.
        with_timestamps: Emit timestamped segments.
        model_path: Path to model directory (for tokenizer loading).
        stt_config: Optional config overrides.

    Returns:
        :class:`TranscriptionResult`.
    """
    transcriber = WhisperTranscriber(model, model_path=model_path, config=stt_config)
    return transcriber.transcribe(audio, language, task, prompt, with_timestamps)
