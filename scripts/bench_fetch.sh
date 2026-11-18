#!/usr/bin/env bash
# Download the weights for one pool entry with its family's venv. Usage: scripts/bench_fetch.sh <slug>
set -euo pipefail
cd "$(dirname "$0")/.."
slug="${1:?usage: scripts/bench_fetch.sh <slug>}"
exec .venv/bin/python -m voicebot.bench.fetch --slug "$slug"
