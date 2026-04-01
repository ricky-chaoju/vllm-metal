# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text support for vLLM Metal.

Heavy submodules (loader, whisper, qwen3_asr) are lazy-imported to avoid a
circular import chain during platform detection:
  platform.py → stt.detection → stt/__init__ → whisper → vllm.config → current_platform
"""

__all__ = [
    "Qwen3ASRTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "load_model",
]


def __getattr__(name: str):
    if name == "load_model":
        from vllm_metal.stt.loader import load_model

        return load_model
    if name in ("TranscriptionResult", "TranscriptionSegment"):
        from vllm_metal.stt import protocol

        return getattr(protocol, name)
    if name == "Qwen3ASRTranscriber":
        from vllm_metal.stt.qwen3_asr.transcriber import Qwen3ASRTranscriber

        return Qwen3ASRTranscriber
    if name == "WhisperTranscriber":
        from vllm_metal.stt.whisper import WhisperTranscriber

        return WhisperTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
