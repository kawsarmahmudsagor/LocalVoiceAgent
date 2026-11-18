# Local Voice Bot v2 — Benchmarked STT/TTS Shortlist on a Linux GPU VM

- **Date:** 2026-11-18
- **Status:** Draft, awaiting review
- **Builds on:** [v1 design](2026-11-18-local-voice-bot-design.md) and [v1 plan](../plans/2026-11-18-local-voice-bot.md), including the stage timing in plan Task 7b. The v1 documents describe Windows. §2 of this spec defines how v1 runs on Linux, and it overrides v1 wherever the two differ.

## 1. Goal

The project's purpose is to measure the voice bot's latency using the resources in the cloned `RealtimeSTT` and `RealtimeTTS` repos. v2 does this in two phases, on a Linux VM with an RTX 5090:

1. **Phase A — benchmark.** Measure a curated pool of STT and TTS engines (§3), using the repos' own benchmark tools where they exist.
2. **Phase B — shortlist.** Pick 3–5 STT engines and 2–3 TTS engines from the results. Make only those switchable from the web UI, so whole voice turns can be compared with v1's stage timing.

Model slugs show the role of each STT model. Realtime slugs end in `-rt`.

### Success criteria

1. The whole project installs and runs on the VM (§2.1) from the cloned repo, using only the scripts in this repo.
2. Phase A produces `docs/v2-model-matrix.md`, with latency, accuracy and GPU memory for every pool entry, or the reason an entry could not run.
3. The shortlist follows the rule in §4.3, and the choice is recorded in the matrix.
4. In Phase B, final STT, realtime STT and TTS can be switched from the UI while idle, with no restart. A failed switch leaves the previous model running.
5. Every turn's `logs/turns.jsonl` record includes the active slugs, so `latency_report --by stt_final,tts_model` compares them directly.
6. Total disk used by v2 (venvs, model weights, Docker images, LLM) stays under 50 GB (raised from about 35 GB on 2026-11-18).
7. Unit tests pass without a GPU or downloaded models.

### Out of scope

- Windows support for v2.
- Cloud and network TTS engines.
- Engines outside the curated pool (§3.2).
- The `transformers` LLM backend on the VM. Its 19 GB checkpoint is outside the disk budget, so the VM uses `LLM_BACKEND=llamacpp` only.
- Server audio mode (`AUDIO_IO=server`), because the VM has no microphone or speakers.
- Choosing the LLM from the UI, barge-in, saving UI choices across restarts.

## 2. Platform: Linux port of v1

### 2.1 Target VM

| Item | Value |
|---|---|
| OS | Ubuntu 24.04 LTS, kernel 6.8, glibc 2.39 |
| GPU | NVIDIA RTX 5090, 32 GB, Blackwell (`sm_120`) |
| CPU / RAM | 28 cores / 49 GB |
| Python | 3.12 (system 3.12.3), managed with `uv` |
| Containers | Docker 29 with the NVIDIA Container Toolkit 1.19 |
| Host CUDA toolkit | `nvcc` 12.0. **Not used**, because it is too old for `sm_120` |
| Access | From a PC on the same LAN, over SSH |

### 2.2 Prerequisites, checked by `scripts/check_vm.sh`

1. `nvidia-smi` shows the RTX 5090, driver 570 or newer and CUDA 12.8 or newer.
   - The error "Driver/library version mismatch" means the VM needs a reboot after a driver update.
   - The script prints this hint and exits non-zero.
2. `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi` succeeds.
3. At least 45 GB of free disk on the filesystem holding the repo and `~/.cache`.
4. `uv` is on `PATH`.

### 2.3 What changes from v1

| v1 (Windows) | v2 on the VM |
|---|---|
| `scripts/setup_venv.ps1` | `scripts/setup_venv.sh`: `uv venv -p 3.12 .venv`, torch from `RealtimeSTT/requirements-gpu-torch.txt` (`torch==2.7.1+cu128`, which supports `sm_120`), editable installs, server deps |
| — | `scripts/install_system_deps.sh`: `sudo apt-get install -y espeak-ng ffmpeg portaudio19-dev libasound2-dev build-essential` |
| `scripts/fetch_llama_server.ps1` and `llama-server.exe` | `llama-server` runs from the official image `ghcr.io/ggml-org/llama.cpp:server-cuda` (§2.4) |
| `scripts/dev.ps1` | `scripts/dev.sh`: starts uvicorn in the background and runs `npm run dev`; stops uvicorn on exit |
| `.venv\Scripts\python` | `.venv/bin/python` |
| `LLM_BACKEND` `transformers` or `llamacpp` | `llamacpp` only on the VM |
| `AUDIO_IO` `browser` or `server` | `browser` only. `server` fails at startup with a clear message |
| Browser on the same PC | Browser on the user's PC, through an SSH tunnel (§2.5) |
| Windows Job Object for child processes | Linux: `start_new_session=True` plus `PR_SET_PDEATHSIG` (§4.5) |

Torch is the only CUDA dependency installed with pip. It brings its own CUDA 12.8 runtime, so nothing on the host is compiled against CUDA.

### 2.4 llama-server in Docker

`LlamaServer` (v1 plan Task 4) gets a Docker mode, which is the VM default:

```
docker run --rm --name voicebot-llama --gpus all \
  -p 127.0.0.1:${LLAMA_PORT}:8080 \
  -v ${HOME}/.cache/llama.cpp:/root/.cache/llama.cpp \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf ${LLAMA_HF_REPO} --no-mmproj --jinja -ngl 99 -c ${LLAMA_CTX} --host 0.0.0.0 --port 8080
```

- **New settings:**
  - `LLAMA_RUNTIME=docker|binary`, default `docker`
  - `LLAMA_IMAGE`, default `ghcr.io/ggml-org/llama.cpp:server-cuda`
- **Lifecycle:**
  - `start()` checks that `docker` exists, removes any stale `voicebot-llama` container, runs the command above and polls `/health` exactly as in v1.
  - `close()` runs `docker stop -t 5 voicebot-llama`.
  - Container logs go to `logs/llama-server.log` (`docker logs -f`).
- **If the image has no `sm_120` kernels** (the server log shows "no kernel image is available"):
  - `scripts/build_llama_image.sh` builds `voicebot/llama-server:cuda12.8` from llama.cpp's `.devops/cuda.Dockerfile`, with `CUDA_VERSION=12.8.0` and `CUDA_DOCKER_ARCH=120`.
  - Then set `LLAMA_IMAGE` to that tag. The build happens inside a container, so the host `nvcc` is still unused.
- **Disk:** the image is about 3.5 GB and the GGUF about 5.7 GB.

### 2.5 Access from the user's PC

- **Default:** keep `WEB_HOST=127.0.0.1` and connect with:
  ```
  ssh -L 8000:localhost:8000 xr23@<vm-ip>
  ```
  Then open `http://localhost:8000`. The browser treats `localhost` as a secure context, so microphone access works with no certificates.
- **Development:** add `-L 5173:localhost:5173` and open `http://localhost:5173`.
- **Optional HTTPS on the LAN address:**
  - `scripts/make_cert.sh` creates a self-signed certificate in `certs/` with the VM's IP in its subject alternative names.
  - Set `WEB_HOST=0.0.0.0`, `WEB_SSL_CERTFILE=certs/cert.pem` and `WEB_SSL_KEYFILE=certs/key.pem`, which are passed to uvicorn.
  - Open `https://<vm-ip>:8000` and accept the certificate once.

## 3. Phase A — benchmark the curated pool

### 3.1 Pool

Every entry must install against the shared `torch==2.7.1+cu128`. An entry whose package pins a different torch is dropped and recorded in the matrix. One torch version also keeps disk use low, because `uv` hardlinks a single cached copy into every venv.

`*` marks a true streaming engine. The other realtime entries re-transcribe the growing audio buffer.

| Role | Slug | Engine | Model | Device | Approx. size | Why it is in the pool |
|---|---|---|---|---|---|---|
| Final STT | `whisper-small-en` | `faster_whisper` | `small.en` | cuda | 0.5 GB | v1 baseline |
| Final STT | `whisper-large-v3-turbo` | `faster_whisper` | `large-v3-turbo` | cuda | 1.6 GB | best Whisper accuracy; fast on a 5090 |
| Final STT | `transcribe-cpp-parakeet-q4` | `transcribe_cpp` | Parakeet TDT 0.6B v3 Q4 GGUF | cuda | 0.8 GB | the repo's "lowest-latency warm GPU final"; official Linux CUDA provider wheel |
| Final STT | `sherpa-parakeet-v3` | `sherpa_onnx_parakeet` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | cpu | 0.7 GB | the repo's Linux production final engine |
| Realtime STT | `nemotron-rt`\* | `sherpa_onnx_nemotron` | `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` | cpu | 0.6 GB | the repo's Linux production live engine |
| Realtime STT | `kroko-en-128-rt`\* | `kroko_onnx` | `Kroko-EN-Community-128-L-Streaming-001.data` | cpu | 0.15 GB | small true-streaming engine |
| Realtime STT | `whisper-tiny-en-rt` | `faster_whisper` | `tiny.en` | cuda | 0.1 GB | classic re-decode baseline |
| TTS | `kokoro` | `KokoroEngine` | default voices | cuda | 0.4 GB | v1 baseline |
| TTS | `qwen-cuda` | `QwenEngine` | Qwen3-TTS 12 Hz 0.6B Base Q8_0 | cuda | 2.3 GB | the repo's recommended engine; CUDA 12.8 native wheel |
| TTS | `inflect-cuda` | `InflectEngine` (PyTorch) | Inflect-Micro-v2 | cuda | 0.3 GB | fast fixed English voice |
| TTS | `piper` | `PiperEngine` | one English Piper voice (`en_US-lessac-medium`) | cpu | 0.1 GB | fastest CPU engine; Linux `piper-tts` package, no manual binary |
| TTS | `chatterbox` | `ChatterboxEngine` | `chatterbox-tts` default | cuda | 3 GB | quality voice cloning; kept only if it resolves on torch 2.7.1 |
| Final STT | `qwen3-asr-1-7b` | `qwen3_asr` | `Qwen/Qwen3-ASR-1.7B` | cuda | ≈ 4.5 GB | **user request (2026-11-18)**; always on the shortlist via `bench/overrides.toml` |
| TTS | `coqui-xtts-v2` | `CoquiEngine` | XTTS v2 (v2.0.2), cloning `assets/ref_voice.wav` | cuda | ≈ 2 GB | **user request (2026-11-18)**; always on the shortlist via `bench/overrides.toml` |

Both user-requested entries resolve with `torch==2.7.1+cu128` (`uv pip compile`, 2026-11-18: `qwen-asr 0.0.6` with `transformers 4.57.6`; `coqui-tts 0.28.0`). They add about 7 GB; the user raised the v2 disk budget to 50 GB for them.

**Reference voice.** `qwen-cuda` and `chatterbox` clone a voice. The bench installer generates `assets/ref_voice.wav` once with Kokoro (`af_heart`), from the sentence in `assets/ref_voice.txt`, so the reference text matches the audio exactly.

Exact options come from each engine's doc under `RealtimeSTT/docs/engines/` and `RealtimeTTS/docs/engines/`.

### 3.2 Disk budget

| Item | Approx. size |
|---|---|
| torch + CUDA libraries (one cached copy, hardlinked) | 6 GB |
| Other Python packages across venvs | 5 GB |
| STT weights | 4.5 GB |
| TTS weights | 6.5 GB |
| LLM GGUF + llama.cpp image | 9.2 GB |
| Benchmark audio and outputs | < 1 GB |
| **Total** | **≈ 32–35 GB** |

Engines **not** in the pool, and why:

- **Covered by a pool entry:** NeMo Parakeet, parakeet-onnx, openai-whisper, whisper-cpp, the other Whisper sizes, Orukeet (European languages), the Moonshine variants, kroko-fp32.
- **Over the budget, or LLM-style ASR that is slower by design:** Granite Speech, Cohere Transcribe, FunASR, Omnilingual ASR, Higgs (SGLang image), Parler, Orpheus (needs a 3B LLM endpoint). Qwen3-ASR and Coqui XTTS were excluded here at first and added back on user request (§3.1).
- **Low value for a latency test:** OmniVoice, LuxTTS, NeuTTS, Pocket, Sopro, Soprano, MOSS, StyleTTS, ZipVoice, `system` (espeak quality on a headless VM), `qwen-cpu` (the GPU is free).

### 3.3 Method

**Test material.** All entries use the same inputs.
- `bench/clips/`: 20 English WAV clips, 16 kHz mono, 2–8 s each. The user records them once in the browser through the bot's own mic path, using `scripts/record_clips.py`, which saves what the server receives. `bench/clips/manifest.json` holds each clip's reference text.
- `bench/sentences.txt`: 10 typical bot replies of 1–3 sentences, for TTS.

**Conditions.** The llama-server container runs with the LLM loaded during every measurement, so GPU contention is realistic. Peak GPU memory comes from `nvidia-smi --query-gpu=memory.used --format=csv -lms 200`, sampled during each run.

**Environments.** `scripts/bench_install.sh <family>` creates `venvs/bench-<family>` with the matching RealtimeSTT or RealtimeTTS extra, and runs an import self-test. The families are:
- `faster-whisper`
- `transcribe-cpp`
- `sherpa` (parakeet, nemotron)
- `kroko`
- `kokoro`
- `qwen`
- `inflect`
- `piper`
- `chatterbox`

`scripts/bench_fetch.sh <slug>` downloads the weights for one entry:
- sherpa bundles through `stt-install-sherpa-models --root models/sherpa-onnx`
- Hugging Face files through `uv run huggingface-cli download`
- Kroko Community `.data` files
- Piper voice files

**STT final: repo tools only.**
1. In the entry's venv, start the repo's server: `stt-server-production --engine <engine> --model <model> [--engine-options <json>]`.
2. Run `RealtimeSTT/tools/benchmarks/benchmark_asr_ab.py --manifest bench/clips/manifest.json --repetitions 3`.
3. Record the median and p90 latency, the word error rate against the reference text, and peak GPU memory.

**STT realtime: repo tools only.**
1. Start the same server with `--realtime-engine <engine>`.
2. Run `RealtimeSTT/tools/benchmarks/benchmark_asr_streaming.py --manifest bench/clips/manifest.json --pace`.
3. Record the time to the first partial, the partial update interval and the final-text quality.

**TTS: one small script, because RealtimeTTS only has Qwen-specific benchmarks.** `scripts/bench_tts.py --engine <slug>` runs in the entry's venv and uses only the public RealtimeTTS API.
1. It builds `TextToAudioStream(engine)` and warms it up once.
2. For each sentence, it calls `feed(sentence)` and `play(muted=True, on_audio_chunk=cb)`.
3. It records the time to first audio and the real-time factor (synthesis time divided by audio duration).
4. It saves each sentence's audio to `bench/out/<slug>/`. The user copies the files to their PC with `scp` and gives an OK / poor listening verdict.

**Output.** `docs/v2-model-matrix.md` gets one row per entry:
- latency median and p90
- WER (STT) or the listening verdict (TTS)
- peak GPU memory and disk used
- install notes, or the reason the entry could not run

**Clean-up.** After Phase B is installed (§4.2), `scripts/bench_clean.sh` deletes the `venvs/bench-*` venvs and the weights of entries that didn't make the shortlist.

### 3.4 Shortlist rule

The v1 baselines (`whisper-small-en`, `kokoro`) are always on the shortlist, so every comparison has a reference point.

- **STT final:** up to 3 engines, including the baseline. Keep only engines whose WER is at most 2 percentage points worse than `whisper-small-en`, then pick the lowest median latency.
- **STT realtime:** up to 2 engines with the lowest time to first partial whose final-text quality is usable. `off` is always available as well.
- **TTS:** 2–3 engines, including the baseline. Keep only engines with an "OK" listening verdict, then pick the lowest median time to first audio.

The user may override the rule. Overrides are recorded in the matrix with a reason.

## 4. Phase B — the switchable shortlist

### 4.1 Catalog and slugs

`server/voicebot/catalog.toml` lists only the shortlisted entries. Each entry has these fields:

| Field | Meaning |
|---|---|
| `slug` | `<family>-<model>[-<variant>][-rt]`, kebab-case. Ends in `-rt` if and only if `role = "realtime"` |
| `kind`, `role` | `stt` with `final` or `realtime`, or `tts` |
| `engine`, `model`, `options` | Passed to the engine as-is |
| `streaming` | STT realtime only: `true` uses the streaming-session API |
| `device`, `vram_mb` | `vram_mb` is the peak measured in Phase A |
| `runtime` | `inprocess` or `worker` (§4.2) |
| `group` | Worker venv name; only used when `runtime = "worker"` |

### 4.2 In-process or workers

The choice is made once, after the shortlist is fixed:

1. Run `uv pip compile` on the v1 server requirements plus every shortlisted engine's extra.
2. **If it resolves:** install everything into the main `.venv`. Every entry is `runtime = "inprocess"`, and no worker code is built.
3. **If it does not:** find the smallest set of entries whose removal makes it resolve. Those entries become `runtime = "worker"`, each running from its own `venvs/<group>`, which can be the entry's Phase A venv. All other entries stay in-process.

There is one more check. After an in-process unload (§4.4), GPU memory must drop back to within 300 MB of its level before the load, measured with `nvidia-smi`. An engine that fails this check moves to `runtime = "worker"`, because only ending its process frees the memory.

### 4.3 How STT plugs in

The main server keeps `AudioToTextRecorder` for voice activity detection, turn-taking and v1's timing callbacks.

- **In-process:** the recorder is built with `transcription_engine`, `model` and `transcription_engine_options` for the final slug. When realtime is on, it also gets `enable_realtime_transcription=True` and `realtime_transcription_engine`, `realtime_model_type` and `realtime_transcription_engine_options`.
- **Worker:** the recorder gets `transcription_executor=` or `realtime_transcription_executor=` set to a proxy object (§4.5). RealtimeSTT's public executor API accepts any object with `transcribe()`. Streaming engines use the `StreamingTranscriptionSession` contract.

Realtime partials come from the recorder's `on_realtime_transcription_update` callback, which emits `user_partial`.

### 4.4 Switching

Switching is only allowed while idle; it is refused during a turn.

**In-process switch**
1. Stop the listen loop.
2. Shut down the old recorder (STT) or stream (TTS) with `shutdown()` where the class provides it, then `del` it and run `gc.collect()`. If `torch` is imported, also run `torch.cuda.empty_cache()`.
3. Build the new one.
4. If building fails, rebuild the previous one and report the failure.
5. Restart the listen loop if it was running.

**Worker switch**
1. Start the new worker and wait for `ready`.
2. Point the proxy at the new worker, then shut down the old one. With 32 GB of VRAM, old and new always fit together.
3. On failure, keep the old worker and report the failure.

A TTS switch can change the sample rate. The server then sends a new `config` with the new `output_format`, and the browser player resamples.

### 4.5 Worker mode (only for `runtime = "worker"` entries)

- **Start:** `venvs/<group>/bin/python -m voicebot_worker --slug <slug> --port 0`, launched with `start_new_session=True` and a `preexec_fn` that calls `prctl(PR_SET_PDEATHSIG, SIGTERM)`, so the worker dies if the server dies.
- **Handshake:** the worker binds to `127.0.0.1`, prints `PORT <n>`, loads the model, then sends `ready{info}` or `fatal{message}`. Logs go to `logs/worker-<slug>.log`.
- **Frames:** a 4-byte big-endian length followed by a UTF-8 JSON header with a `type` field, then a 4-byte length followed by an optional binary body. Audio is PCM s16le mono.
- **Messages:**
  - STT: `transcribe{sample_rate}` + PCM → `result{text}`.
  - Streaming STT: `session_open` / `accept` + PCM / `partial` / `finish` / `session_close`.
  - TTS: `synthesize{text}` (repeated), `end` → `audio` + PCM frames … `done{text}`; `stop` → `stopped`; `voices`; `set_voice`.
  - All: `ping` / `pong` every 2 s, `shutdown`.
- **Proxies:** `ProxyExecutor`, `ProxyStreamingEngine` and `ProxyTTSStream` give the session the same interface it uses in-process.
- **Clean-up:** on server shutdown, each worker gets `shutdown`, then `SIGTERM` to its process group after 5 s, then `SIGKILL`. A worker that misses heartbeats for 6 s is marked crashed and restarted once.

Worker IPC adds about 1–2 ms per call. Each turn's log record includes `runtime`, so worker overhead stays visible in the comparison.

### 4.6 Timing integration

v1's metrics record (plan Task 7b) gains `stt_final`, `stt_realtime`, `tts_model` and the `runtime` of each. Comparisons are then one command:

```
.venv/bin/python -m voicebot.latency_report --by stt_final,stt_realtime,tts_model --source voice
```

With realtime STT on, the `stt_start` and `stt_done` marks still refer to the final transcription. Partials are not timed per stage; Phase A covers their latency.

### 4.7 Settings

These `.env` variables replace v1's `STT_MODEL`, `STT_DEVICE` and `STT_COMPUTE_TYPE`. `TTS_VOICE` stays.

| Variable | Default | Meaning |
|---|---|---|
| `STT_FINAL` | `whisper-small-en` | startup final STT slug |
| `STT_REALTIME` | `off` | startup realtime slug, or `off` |
| `TTS_MODEL` | `kokoro` | startup TTS slug |
| `GPU_TOTAL_MB` | `32607` | for the VRAM meter (RTX 5090) |
| `LLM_VRAM_MB` | `7000` | LLM estimate for the VRAM meter (Q4_K_M with an 8k context) |
| `LLAMA_RUNTIME` | `docker` | §2.4 |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | §2.4 |
| `WEB_SSL_CERTFILE` / `WEB_SSL_KEYFILE` | empty | §2.5; HTTPS only when both are set |

### 4.8 Web UI

- **Pickers:** `Final STT`, `Realtime STT` (first option `off`) and `TTS`. Each option shows its slug, `cpu`/`cuda` and its measured VRAM.
- **VRAM meter:** sums the LLM and the three selections against `GPU_TOTAL_MB`, and turns amber when over budget.
- **Voice picker:** filled from the engine's `get_voices()`, or `default` if it has none.
- **While switching:** the pickers are locked and the status pill shows `loading <slug>`.
- **Live text:** a grey partial line shows the realtime text while the user speaks. The final message replaces it.
- **StatusBar:** v1's timing breakdown is unchanged, with the active slugs added to its tooltip.

### 4.9 WebSocket changes (additions to v1 §4.9)

| Direction | Type | Fields |
|---|---|---|
| C→S | `select_model` | `role`: `stt_final` / `stt_realtime` / `tts`; `slug` (`"off"` allowed for `stt_realtime`) |
| S→C | `catalog` | the shortlisted entries |
| S→C | `model_status` | `role`, `slug`, `state`: `loading` / `ready` / `failed`; `message?`, `load_ms?` |
| S→C | `user_partial` | `text` |
| S→C | `config` | adds `stt_final`, `stt_realtime`, `tts`, `gpu_total_mb`, `llm_vram_mb` |

`select_model` during a turn is refused with `error{code:"busy"}`. An unknown slug, or a slug in the wrong role, is refused with `error{code:"bad_model"}`.

### 4.10 Error handling

| Failure | Behaviour |
|---|---|
| `check_vm.sh` fails | Setup stops with the failing check and its fix (for example, reboot after a driver update) |
| llama-server image lacks `sm_120` kernels | Startup error points to `scripts/build_llama_image.sh` |
| Model load fails (import error, CUDA error, timeout) | `model_status failed` with the error, or the last 30 worker log lines; the previous model is restored |
| Worker crashes during a turn | `error` event and the v1 spoken apology; restart once, else revert to the default slug |
| Startup slug is not in the catalog | Fall back to `whisper-small-en` / `off` / `kokoro` with a warning |
| `AUDIO_IO=server` on the VM | Stop at startup: "server audio is not supported on the VM; use browser mode" |

### 4.11 Testing

- **Catalog:** schema; unique slugs; `-rt` if and only if `role = "realtime"`; `group` present when `runtime = "worker"`.
- **LlamaServer Docker mode:** the built `docker run` and `docker stop` commands; stale-container removal; the health-poll path. Subprocess calls are mocked.
- **Session:** `select_model` refused during a turn; in-process switch rebuilds and restores on failure, using fake recorder and stream factories; `user_partial` events; metrics records include the slugs.
- **Worker mode (only if any entry uses it):** frame round trip; fake-worker tests for ready, switch, rollback, heartbeat, and process-group kill on close.
- **Frontend:** the reducer handles `catalog`, `model_status` and `user_partial`; the VRAM sum; player sample-rate change.
- **Bench scripts:** `bench_tts.py` timing logic with a fake engine; `check_vm.sh` parsing of sample `nvidia-smi` outputs, including the version-mismatch error.
- **Manual:** 10 voice turns per shortlisted combination through the SSH tunnel, then `latency_report`, with results added to the matrix.

## 5. Risks

| Risk | Mitigation |
|---|---|
| A pool package has no `sm_120` kernels (ctranslate2 for faster-whisper, the transcribe.cpp CUDA wheel, the Qwen native wheel) | Phase A records the failure. For faster-whisper, try `device="cuda"` with `compute_type="float16"` first; if CUDA still fails, record it rather than building from source |
| `chatterbox-tts` pins an older torch | Dropped under the §3.1 rule and recorded |
| The official llama.cpp CUDA image lacks `sm_120` | `scripts/build_llama_image.sh` (§2.4) |
| The NVIDIA driver/library mismatch comes back after updates | `check_vm.sh` detects it and says to reboot |
| `stt-server-production` does not support an engine's options | Pass them through `--engine-options`; if that is not possible, benchmark that engine through the RealtimeSTT Python API with the same clips and note it |
| The SSH tunnel adds latency to browser audio | It runs on a LAN, so the effect is a few ms; `playback_ms` makes it visible |
