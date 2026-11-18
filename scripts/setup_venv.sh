#!/usr/bin/env bash
# Creates .venv (Python 3.12) with CUDA torch 2.7.1+cu128, the vendored libraries and the server,
# enables the gitleaks hooks and installs the web UI dependencies. Linux VM only (llamacpp backend).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$1"; }

if [ ! -x .venv/bin/python ]; then
  step "Create Python 3.12 venv"
  uv venv -p 3.12 .venv
fi
step "Install CUDA torch"
uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt
step "Install RealtimeSTT[faster-whisper], RealtimeTTS[kokoro], server[dev]"
uv pip install --python .venv -e "RealtimeSTT[faster-whisper]" -e "RealtimeTTS[kokoro]" -e "server[dev]"
# A dependency may have pulled in another torch build; put the CUDA one back.
step "Re-assert CUDA torch"
uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt
step "Check CUDA"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not active'; print(torch.__version__, torch.cuda.get_device_name(0))"
step "Enable gitleaks hooks"
git config core.hooksPath .githooks
if [ -f web/package.json ]; then
  step "Install web dependencies"
  npm --prefix web install
fi
echo "Setup complete."
