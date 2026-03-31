# vllm-metal Project Instructions

## General
- 使用繁體中文回覆
- Code 和 commit message 用英文

## Git & Commits
- **DCO required**: Always use `git commit -s` to add `Signed-off-by: RickyChen / 陳昭儒 <ricky.chen@infinirc.com>`
- **Commit messages**: Short and direct, one line. No `Co-Authored-By` lines, no Claude attribution
- **Remote**: `origin` = `ricky-chaoju/vllm-metal`, upstream = `vllm-project/vllm-metal`, PR target = `vllm-project:main`
- **Lint before commit**: `ruff check --fix && ruff format`

## Code Style
- License header: `# SPDX-License-Identifier: Apache-2.0`
- `from __future__ import annotations`
- Google-style docstrings (Args / Returns / Raises)
- Section separators: `# ===` (match model_runner.py)

## Design Principles (from reviewer feedback)
- **Design before coding**: Think about abstraction boundaries, where logic should live, maintainability
- **DRY**: Never duplicate logic. If two functions differ by one line, parameterize. If a class holds data, operate on it as methods — don't pass to free functions
- **No magic numbers**: Extract to named constants with source/rationale comments
- **No broad `except Exception`**: Use specific types (`ImportError`, `OSError`, `ValueError`)
- **No silent failures**: Always log at least debug-level with exception detail in except blocks
- **No global mutable state**: Don't use module-level `@lru_cache` if instances may need different configs
- **Explicit contracts**: Validate inputs at boundaries. Reject unsupported params with clear errors rather than silently ignoring
- **Derive from canonical sources**: Don't hardcode data that can be derived from config/upstream (e.g., language maps from HF config, not inline dicts)
- **Align with upstream vLLM**: Match naming conventions and interfaces (e.g., `cu_seqlens` not `cu_seq_lens`)
- **Keep PRs small and focused**: Split large features into incremental PRs (config → pipeline → model → orchestration). Each PR independently testable
- **Cap/clamp risky values**: Validate env vars (try/except + clamp + fallback). Guard against negative/overflow (e.g., `max(0, n_text_ctx - prompt_len)`)
- **Avoid stringly-typed values**: Use enums or typed constants instead of raw strings for tasks/modes

## Testing
- **AAA style**: Arrange / Act / Assert — clearly separated
- **Deterministic**: Monkeypatch hardware-dependent calls (e.g., `mx.metal.device_info()`), assert exact values
- **Env-based model paths**: Use env vars for local model paths, `pytest.skip` if unset
- **Test real behavior**: Don't just assert env presence — test the actual logic function. Avoid MagicMock + method rebinding that bypasses class invariants
- **Top-level imports** in test files, not function-local

```bash
source ~/.venv-vllm-metal/bin/activate
pytest tests/ -v -m "not slow"
```

## Architecture Notes
- venv: `~/.venv-vllm-metal`
- STT deps are optional (`[stt]` extra), not hard dependencies
- Prefix caching: KVCache-only (disabled for Mamba/hybrid)
- Scaffolding code: Mark with `# SCAFFOLDING: remove when <condition>`
