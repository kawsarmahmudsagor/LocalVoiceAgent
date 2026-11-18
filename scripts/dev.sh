#!/usr/bin/env bash
# Starts the API in the background and the Vite dev server in the foreground; stops the API on exit.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

[ -x .venv/bin/python ] || { echo "No .venv: run scripts/setup_venv.sh first." >&2; exit 1; }
[ -d web/node_modules ] || { echo "No web/node_modules: run 'npm --prefix web install' first." >&2; exit 1; }
[ -f .env ] || echo "No .env found: using defaults (cp .env.example .env to change them)." >&2

(cd server && exec "$root/.venv/bin/python" -m voicebot.web) &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT

echo "API starting (first start loads the models; watch for 'Uvicorn running')."
echo "From your PC: ssh -L 8000:localhost:8000 -L 5173:localhost:5173 $USER@<vm-ip>"
echo "Then open http://localhost:5173"
npm --prefix web run dev
