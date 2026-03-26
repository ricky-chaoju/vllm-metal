"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


def _get_test_seed() -> int:
    """Return the deterministic seed used across tests.

    Override via `VLLM_METAL_TEST_SEED` for debugging.
    """

    raw_seed = os.environ.get("VLLM_METAL_TEST_SEED", "0")
    try:
        return int(raw_seed)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("VLLM_METAL_TEST_SEED must be an integer") from exc


@pytest.fixture(autouse=True)
def _seed_random_generators() -> None:
    """Seed common RNGs to keep tests deterministic."""

    seed = _get_test_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        import mlx.core as mx
    except ImportError:
        return

    mlx_seed = getattr(mx.random, "seed", None)
    if mlx_seed is None:
        return
    mlx_seed(seed)


# === Model config fixtures ===


def _load_hf_text_config(repo_id: str) -> dict:
    """Load config.json from HuggingFace and return flattened text_config.

    Downloads only the config file (~1 KB), not model weights.
    """
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return dict(tc)


@pytest.fixture(scope="session")
def qwen35_4b_args() -> dict:
    """Qwen3.5-4B model args from HuggingFace config.json → text_config."""
    return _load_hf_text_config("Qwen/Qwen3.5-4B")


@pytest.fixture(scope="session")
def llama_args() -> dict:
    """Llama-3.2-1B model args from HuggingFace config.json."""
    return _load_hf_text_config("meta-llama/Llama-3.2-1B")
