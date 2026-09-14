#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/bootstrap.sh cpu|cu128. GPU mode must match the actual machine.
asr_torch_backend="${1:-cpu}"
case "$asr_torch_backend" in cpu|cu128) ;; *) echo "Expected cpu or cu128" >&2; exit 2;; esac
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url "https://download.pytorch.org/whl/$asr_torch_backend" \
  'torch==2.8.0' 'torchaudio==2.8.0'
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[model,service,dev]'
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python -m pip check
