# SPDX-License-Identifier: Apache-2.0
"""Install a .pth startup hook that sets VLLM_PLATFORM_CLASS for all processes.

vLLM spawns subprocesses (APIServer, EngineCore, Workers) using spawn mode.
The entry-point plugin system may fail to discover vllm-metal in these
subprocesses.  A .pth file runs at Python startup — before any user code —
ensuring the env var is set in every process.

Usage:
    python scripts/install_platform_hook.py          # install
    python scripts/install_platform_hook.py --remove # remove
"""

from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path

HOOK_FILENAME = "vllm_metal_platform.pth"

# .pth files support single-line import statements that execute at startup.
# This sets the env var before vLLM's platform detection runs.
HOOK_CONTENT = (
    "import os, sys; "
    "exec("
    '\'if sys.platform=="darwin": '
    'os.environ.setdefault("VLLM_PLATFORM_CLASS","vllm_metal.platform.MetalPlatform"); '
    'os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD","spawn"); '
    'os.environ.setdefault("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE","shm"); '
    'os.environ.setdefault("VLLM_USE_RAY_WRAPPED_PP_COMM","0"); '
    'os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE","0")'
    "')\n"
)


def _get_site_packages() -> Path:
    """Get the site-packages directory for the current environment."""
    paths = site.getsitepackages()
    if not paths:
        print("ERROR: No site-packages found.", file=sys.stderr)
        sys.exit(1)
    return Path(paths[0])


def install(site_dir: Path) -> None:
    hook_path = site_dir / HOOK_FILENAME
    hook_path.write_text(HOOK_CONTENT)
    print(f"Installed: {hook_path}")


def remove(site_dir: Path) -> None:
    hook_path = site_dir / HOOK_FILENAME
    if hook_path.exists():
        hook_path.unlink()
        print(f"Removed: {hook_path}")
    else:
        print(f"Not found: {hook_path}")


def main():
    parser = argparse.ArgumentParser(description="Install vllm-metal platform hook")
    parser.add_argument("--remove", action="store_true", help="Remove the hook")
    args = parser.parse_args()

    site_dir = _get_site_packages()

    if args.remove:
        remove(site_dir)
    else:
        install(site_dir)


if __name__ == "__main__":
    main()
