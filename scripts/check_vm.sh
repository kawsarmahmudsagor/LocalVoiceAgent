#!/usr/bin/env bash
# Check the VM prerequisites (v2 spec §2.2). Prints OK, or each problem with its fix.
set -euo pipefail
exec python3 "$(dirname "$0")/../server/voicebot/vmcheck.py" "$@"
