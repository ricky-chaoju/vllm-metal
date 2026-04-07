#!/bin/bash
# GPU-specific tests that require Metal hardware.
# Runs @pytest.mark.slow tests (Qwen3, Qwen3.5, etc.)

main() {
  set -eu -o pipefail

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  # shellcheck source=lib.sh disable=SC1091
  source "${script_dir}/lib.sh"

  setup_dev_env

  if [ "$(uname)" != "Darwin" ]; then
    echo "Skipping GPU tests on non-macOS platform"
    exit 0
  fi

  ./install.sh
  # shellcheck source=/dev/null
  source .venv-vllm-metal/bin/activate

  section "Running GPU model tests (slow)"
  pytest -m slow tests/ -v --tb=short
}

main "$@"
