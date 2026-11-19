#!/usr/bin/env bash
# Ollama in Docker for the RAG embedding backend (v3 spec §5).
#   scripts/ollama.sh start | pull [model] | stop
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME=voicebot-ollama

case "${1:-}" in
  start)
    if docker container inspect "$NAME" >/dev/null 2>&1; then
      docker start "$NAME"
    else
      mkdir -p "$REPO/models/ollama"
      docker run -d --gpus all --name "$NAME" -p 127.0.0.1:11434:11434 \
        -v "$REPO/models/ollama:/root/.ollama" ollama/ollama
    fi
    ;;
  pull)
    docker exec "$NAME" ollama pull "${2:-nomic-embed-text}"
    ;;
  stop)
    docker stop "$NAME"
    ;;
  *)
    echo "usage: scripts/ollama.sh start | pull [model] | stop" >&2
    exit 2
    ;;
esac
