# SPDX-License-Identifier: Apache-2.0
"""Patch vLLM's platform resolution to support VLLM_PLATFORM_CLASS env var.

vLLM v0.17.1 spawns APIServer and EngineCore as separate processes using
multiprocessing spawn mode.  In spawned subprocesses on macOS, the
entry-point plugin system (importlib.metadata) may fail to discover OOT
platform plugins, causing CpuPlatform to be selected instead of
MetalPlatform.

This script patches vllm/platforms/__init__.py to check a
VLLM_PLATFORM_CLASS environment variable AFTER normal platform
resolution.  If the env var is set and the override import succeeds,
the platform is replaced.

Usage:
    python scripts/patch_vllm_platform.py          # apply patch
    python scripts/patch_vllm_platform.py --check  # check if patched
    python scripts/patch_vllm_platform.py --revert # revert patch
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

MARKER = "# VLLM_PLATFORM_CLASS override (vllm-metal patch)"

PATCH_LINES = """\
            # VLLM_PLATFORM_CLASS override (vllm-metal patch)
            # OOT plugins can set this env var to force a platform in
            # spawned subprocesses where entry-point discovery fails.
            import os as _os
            _override = _os.environ.get("VLLM_PLATFORM_CLASS")
            if _override:
                try:
                    _current_platform = resolve_obj_by_qualname(_override)()
                except Exception:
                    pass  # Fall back to normally-resolved platform.
"""

# The line after which to insert the patch (unique anchor).
ANCHOR = '            _init_trace = "".join(traceback.format_stack())'


def _find_platforms_init() -> Path:
    """Locate vllm/platforms/__init__.py in the current environment."""
    spec = importlib.util.find_spec("vllm.platforms")
    if spec is None or spec.origin is None:
        print("ERROR: vllm.platforms not found. Is vllm installed?", file=sys.stderr)
        sys.exit(1)
    return Path(spec.origin)


def is_patched(path: Path) -> bool:
    return MARKER in path.read_text()


def _ensure_os_import(lines: list[str]) -> list[str]:
    """Add 'import os' if not already present."""
    for line in lines:
        stripped = line.strip()
        if stripped == "import os" or stripped.startswith("import os "):
            return lines
    # Insert after the first import line
    for i, line in enumerate(lines):
        if line.startswith("import "):
            lines.insert(i + 1, "import os\n")
            return lines
    return lines


def apply_patch(path: Path) -> None:
    if is_patched(path):
        print(f"Already patched: {path}")
        return

    text = path.read_text()
    if ANCHOR not in text:
        print(f"ERROR: Cannot find anchor line in {path}", file=sys.stderr)
        print(f"  Expected: {ANCHOR!r}", file=sys.stderr)
        print("  vLLM version may be incompatible.", file=sys.stderr)
        sys.exit(1)

    lines = text.splitlines(keepends=True)
    lines = _ensure_os_import(lines)

    # Find anchor and insert patch after it
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if ANCHOR in line:
            new_lines.append(PATCH_LINES)

    path.write_text("".join(new_lines))

    # Clear bytecache
    pyc_dir = path.parent / "__pycache__"
    if pyc_dir.exists():
        for pyc in pyc_dir.glob("__init__*.pyc"):
            pyc.unlink()

    print(f"Patched: {path}")


def revert_patch(path: Path) -> None:
    if not is_patched(path):
        print(f"Not patched: {path}")
        return

    lines = path.read_text().splitlines(keepends=True)
    new_lines = []
    skip = False
    for line in lines:
        if MARKER in line:
            skip = True
            continue
        if skip:
            if (
                line.strip() == ""
                or line.strip().startswith("#")
                or line.strip().startswith("_override")
                or line.strip().startswith("try:")
                or line.strip().startswith("_current_platform")
                or line.strip().startswith("except")
                or line.strip().startswith("pass")
            ):
                continue
            skip = False
        new_lines.append(line)

    path.write_text("".join(new_lines))

    pyc_dir = path.parent / "__pycache__"
    if pyc_dir.exists():
        for pyc in pyc_dir.glob("__init__*.pyc"):
            pyc.unlink()

    print(f"Reverted: {path}")


def main():
    parser = argparse.ArgumentParser(description="Patch vLLM platform resolution")
    parser.add_argument("--check", action="store_true", help="Check if patched")
    parser.add_argument("--revert", action="store_true", help="Revert patch")
    args = parser.parse_args()

    path = _find_platforms_init()

    if args.check:
        status = "patched" if is_patched(path) else "not patched"
        print(f"{path}: {status}")
        sys.exit(0 if is_patched(path) else 1)

    if args.revert:
        revert_patch(path)
    else:
        apply_patch(path)


if __name__ == "__main__":
    main()
