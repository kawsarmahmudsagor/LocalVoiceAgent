#!/usr/bin/env bash
# Create venvs/bench-<family> from bench/families/<family>.txt and run the import self-test.
# Usage: scripts/bench_install.sh <family>
set -euo pipefail
cd "$(dirname "$0")/.."
family="${1:?usage: scripts/bench_install.sh <family>}"
req="bench/families/$family.txt"
[ -f "$req" ] || { echo "unknown family '$family' (no $req)" >&2; exit 2; }
venv="venvs/bench-$family"

[ -x "$venv/bin/python" ] || uv venv -p 3.12 "$venv"
uv pip install --python "$venv" -r "$req"
uv pip install --python "$venv" --no-deps -e server
if [ "$family" = kroko ]; then
  # kroko-onnx.md: Kroko-ONNX is built from source on Linux by RealtimeSTT's helper.
  "$venv/bin/stt-install-kroko" --build
fi
"$venv/bin/python" -m voicebot.bench.selftest --family "$family"
du -sh "$venv"
df -h ~ | tail -1
