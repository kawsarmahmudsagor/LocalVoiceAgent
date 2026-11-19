# Local Voice Bot

A test web app for talking to a local LLM by voice: RealtimeSTT (faster-whisper) hears you, Qwen3.5-9B answers, and Kokoro speaks the reply. Everything runs on one Windows PC, with a playground-style browser UI that shows the conversation, the generation settings and per-stage latency.

## Requirements

- Windows 10/11 with an NVIDIA GPU (8 GB VRAM is enough for the llama.cpp backend)
- [uv](https://docs.astral.sh/uv/) (creates the Python 3.12 environment; the system Python can be any version)
- Node.js 20 or newer, with npm
- [gitleaks](https://github.com/gitleaks/gitleaks) (`winget install Gitleaks.Gitleaks`), used by the git hooks
- About 15 GB of free disk space for the llama.cpp setup (CUDA torch, the 5.7 GB GGUF and llama.cpp itself)

## Setup

1. Create the environment. This installs CUDA torch, RealtimeSTT with faster-whisper, RealtimeTTS with Kokoro and the server, and enables the gitleaks git hooks:
   ```powershell
   .\scripts\setup_venv.ps1
   ```
2. Download the official llama.cpp Windows CUDA build into `tools\llama\`:
   ```powershell
   .\scripts\fetch_llama_server.ps1
   ```
3. Create your settings file and choose the two switches below:
   ```powershell
   copy .env.example .env
   ```
4. Install the web UI dependencies (step 1 already does this if `web\package.json` exists):
   ```powershell
   npm --prefix web install
   ```

The first start downloads the selected model into the Hugging Face cache: 5.7 GB for the GGUF.

## The two switches in `.env`

| Variable | Values | Meaning |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` (default), `transformers` | `llamacpp` runs `llama-server.exe` with `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`; the app starts and stops it. `transformers` loads `Qwen/Qwen3.5-9B` in process with 4-bit bitsandbytes. |
| `AUDIO_IO` | `browser` (default), `server` | `browser` uses the browser's microphone and speakers. `server` uses this PC's microphone and speakers; the browser only shows the conversation. |

Both take effect on restart. Every other setting is listed with its default in [.env.example](.env.example).

> **transformers backend:** implemented and unit-tested, but not yet run against the real 19.3 GB checkpoint. It loads the model text-only (`AutoModelForCausalLM`), which skips the vision encoder. Expect to fix things the first time you use it.

## Run

**One click:** `run-local.bat` in the repo root builds the UI, starts the API and opens the browser when it is ready. `run-local.bat dev` starts the API and the Vite dev server instead.

**Development** (hot reload for the UI):
```powershell
.\scripts\dev.ps1
```
The API starts in a new window (wait for `Uvicorn running`), then open http://localhost:5173.

**Production** (one process, UI served by FastAPI):
```powershell
npm --prefix web run build
cd server; ..\.venv\Scripts\python -m voicebot.web
```
Then open http://127.0.0.1:8000.

In browser mode, the microphone only works on `localhost` or over HTTPS.

## Linux VM

v2 runs on an Ubuntu 24.04 VM with an RTX 5090 (see `docs/superpowers/specs/2026-11-18-v2-model-switching-design.md` §2). The browser runs on your own PC and reaches the VM through an SSH tunnel.

Run these from the repo root, in this order:

1. `scripts/check_vm.sh` must print `OK`. If it reports a driver/library version mismatch, run `sudo reboot` and check again.
2. `scripts/install_system_deps.sh` installs espeak-ng, ffmpeg, the PortAudio and ALSA headers and build tools. It uses sudo.
3. `scripts/setup_venv.sh` creates `.venv` with `torch==2.7.1+cu128`, RealtimeSTT, RealtimeTTS (Kokoro) and the server. It also enables the gitleaks hooks and installs the web dependencies.
4. `cp .env.example .env`. The defaults on Linux are `LLM_BACKEND=llamacpp`, `AUDIO_IO=browser` and `LLAMA_RUNTIME=docker`.
5. `scripts/dev.sh` starts the API and the Vite dev server.

llama-server runs from the Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda`, and the app starts and stops the `voicebot-llama` container itself. If `logs/llama-server.log` shows `no kernel image is available`, the image has no `sm_120` kernels. In that case run `scripts/build_llama_image.sh`, then set `LLAMA_IMAGE=voicebot/llama-server:cuda12.8` in `.env`.

**Open the UI from your PC:**
```
ssh -L 8000:localhost:8000 -L 5173:localhost:5173 xr23@<vm-ip>
```
Then open http://localhost:5173 (dev) or http://localhost:8000 (production build). The browser treats `localhost` as a secure context, so the microphone works with no certificate.

If port 8000 is taken on the VM, set `WEB_PORT` in `.env` (for example `8001`). The Vite dev proxy and the tunnel command printed by `scripts/dev.sh` follow it; use that port in the `ssh -L` command and the production URL.

**HTTPS on the LAN address (optional):** run `scripts/make_cert.sh [ip]`, then set `WEB_HOST=0.0.0.0`, `WEB_SSL_CERTFILE=certs/cert.pem` and `WEB_SSL_KEYFILE=certs/key.pem`. Open `https://<vm-ip>:8000` and accept the certificate once.

**Deferred on the VM:** the `transformers` backend. The `Qwen/Qwen3.5-9B` checkpoint is downloaded into the Hugging Face cache, but running the backend on the VM is left for later; use `llamacpp`.

**Not available on the VM:** server audio (`AUDIO_IO=server`), because the VM has no microphone or speakers, and the app refuses to start with it.

## Latency

Every turn records a timestamp at each pipeline stage. The UI shows "Response: N ms" (end of speech, or a typed send, to first sound) with a per-stage breakdown; hover it for every mark and span. The server logs one `turn timing` line per turn and appends a JSON record to `logs/turns.jsonl` (for `latency_report`) and a readable block, with times in seconds and ms, to `logs/turns.log`.

| Span | What it measures |
|---|---|
| `silence_wait_ms` | end of speech until the silence wait (`STT_SILENCE_SEC`) ends the turn |
| `stt_ms` | final transcription |
| `llm_ttft_ms` | LLM time to first token |
| `filter_ms` | text dropped by the speech filter (`<think>`, markdown) |
| `tts_first_audio_ms` | Kokoro time to first audio |
| `playback_ms` | WebSocket plus browser player buffering (browser mode) |
| `perceived_ms` | what you feel: end of speech to first sound |
| `total_ms` | whole turn, including playback |

Summarize the log with medians and 90th percentiles per configuration:
```powershell
cd server; ..\.venv\Scripts\python -m voicebot.latency_report --by llm_backend,audio_io
```

### Measured latency

RTX 4060 8 GB, `llamacpp` backend, typed turns over the WebSocket (no STT), 2026-11-18:

| Audio mode | Turns | `llm_ttft_ms` median | `tts_first_audio_ms` median | `perceived_ms` median / p90 |
|---|---|---|---|---|
| browser | 33 | 188 | 316 | 705 / 953 |
| server | 16 | 224 | 295 | 553 / 804 |

Voice turns add `silence_wait_ms` (about 600 ms by default) and `stt_ms`. They have not been measured yet: run a few voice turns, then use `latency_report`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Startup names an invalid `.env` variable | Fix that value; allowed values are in `.env.example`. |
| `llama-server not found` | Run `scripts\fetch_llama_server.ps1`. |
| `port ... is already in use` | Change `LLAMA_PORT`, or set `LLAMA_AUTOSTART=false` to use a llama-server you started yourself. |
| llama-server `timeout` or `exited` (with its last 30 log lines) | Usually CUDA out of memory: close other GPU apps, lower `LLAMA_CTX`, or set `STT_DEVICE=cpu`. Full log: `logs\llama-server.log`. |
| `faster-whisper is not installed` / `cublas64_12.dll could not be loaded` | Re-run `scripts\setup_venv.ps1`, or set `STT_DEVICE=cpu`. |
| transformers: CUDA unavailable, bitsandbytes missing, or out of memory | Follow the message; `LLM_BACKEND=llamacpp` needs less VRAM. |
| "Microphone permission denied" | Allow the microphone for the site; typing still works. |
| UI shows "disconnected" | The server stopped or restarted; the UI reconnects by itself. |
| "Another browser tab is already connected" | Only one client at a time; close the other tab. |
| The bot hears itself in browser mode | Use headphones (half-duplex muting plus echo cancellation is not always enough). |

## Project layout

- `server/voicebot/`: FastAPI app, voice session, STT/TTS/LLM adapters, timing. Tests: `cd server; ..\.venv\Scripts\python -m pytest`
- `web/`: React + TypeScript UI (Vite). Checks: `npm run test`, `npm run typecheck`, `npm run build`
- `scripts/`: setup, llama.cpp download, dev launcher
- `docs/superpowers/`: design spec and implementation plan

## Vendored code

`RealtimeSTT/` and `RealtimeTTS/` are copies of the upstream repositories, installed in editable mode. They are not edited; two upstream issues are worked around at runtime. See [VENDORED.md](VENDORED.md) for versions, the workarounds and how to update them.
