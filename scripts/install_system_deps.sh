#!/usr/bin/env bash
# System packages for audio, espeak (Kokoro phonemes) and native builds. Needs sudo.
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y espeak-ng ffmpeg portaudio19-dev libasound2-dev build-essential
