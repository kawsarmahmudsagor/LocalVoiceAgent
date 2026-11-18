# Local Voice Bot v2 — Switchable STT and TTS Models

- **Date:** 2026-11-18
- **Status:** Draft, awaiting review
- **Builds on:** [v1 design](2026-11-18-local-voice-bot-design.md)

## 1. Goal

v2 installs every local speech-to-text (STT) and text-to-speech (TTS) engine that ships in the cloned `RealtimeSTT` and `RealtimeTTS` repos. You can switch between them from the web UI while the app runs, with no restart.

STT has two roles:

- The **final** model produces the transcript that goes to the LLM.
- The optional **realtime** model shows live partial text while you speak.

Every model has a slug. Realtime slugs end in `-rt`.

### Success criteria

1. Every catalog entry that is installable on the current platform installs through one script, and its status is visible in the UI.
2. Final STT, realtime STT and TTS can each be switched from the UI while idle, with no restart.
3. A failed switch leaves the previous model running.
4. Switching a model frees its GPU memory.
5. With realtime STT on, partial text appears in the chat while the user speaks.
6. v1 behaviour (both LLM backends, both audio modes) still works.
7. All unit tests pass without a GPU or downloaded models.

### Out of scope

- Cloud and network TTS engines: OpenAI, Azure, ElevenLabs, CAMB, MiniMax, Cartesia, Typecast, ModelsLab, Speechify, gTTS and Edge. v2 stays fully local.
- Choosing the LLM from the UI. The LLM stays selected by `.env`, as in v1.
- Interrupting the bot by voice (barge-in).
- Saving UI model choices across restarts.
- Non-English UI flows. Multilingual models can still be selected.

## 2. Why worker processes

Two facts rule out loading every engine into the main server process:

1. **Dependency conflicts.** `realtimetts[luxtts]` pins `transformers<=4.57.6`, while the v1 `transformers` LLM backend needs a current `transformers` to load Qwen3.5. Coqui, Chatterbox and Parler also pin specific `torch` or `transformers` versions. One venv with every engine cannot resolve.
2. **GPU memory.** The GPU has 8 GB, and the LLM uses about 6 GB of it. CUDA and torch keep cached memory after a model is released. Ending the process is the only reliable way to free it.

So each STT or TTS engine runs in its own child process (a **worker**), using the venv of its **dependency group**.

## 3. Architecture

```
 Browser ──/ws──► Main server (v1 venv: FastAPI, VoiceSession, LLM, AudioToTextRecorder/VAD)
                     │
                     ├─ ProxyExecutor ─────────IPC──► STT worker  (venv of the final slug's group)
                     ├─ ProxyStreamingEngine ──IPC──► STT worker  (venv of the -rt slug's group)
                     └─ ProxyTTSStream ────────IPC──► TTS worker  (venv of the TTS slug's group)
```

- **The main server** keeps all of v1: `VoiceSession`, the LLM backends, the WebSocket and the audio I/O. It still runs `AudioToTextRecorder` for voice activity detection and turn-taking, but loads no STT or TTS model itself.
- **STT models plug in through RealtimeSTT's public API.**
  - `AudioToTextRecorder(transcription_executor=..., realtime_transcription_executor=...)` accepts any object with a `transcribe()` method.
  - Streaming engines use the `StreamingTranscriptionSession` contract: `accept_audio`, `decode`, `get_result`, `finish`, `close`.
  - The main server passes proxy objects that forward these calls to workers.
- **Workers.** Each worker loads exactly one engine. At most three run at once: final STT, realtime STT and TTS.
- **Audio out.** The TTS worker always returns PCM. The main server sends it to the browser (browser mode) or plays it on the PC speakers (server mode). Both v1 audio modes keep working.

### 3.1 Repository layout (additions to v1)

```
LocalVoiceAgent/
├─ server/
│  ├─ voicebot/
│  │  ├─ catalog.toml               every STT/TTS entry (§4)
│  │  ├─ catalog.py                 loads and validates the catalog, merges install status
│  │  ├─ workers/
│  │  │  ├─ framing.py              length-prefixed frame encode/decode
│  │  │  ├─ manager.py              WorkerManager: spawn, ready, switch, rollback, heartbeat
│  │  │  ├─ stt_proxy.py            ProxyExecutor, ProxyStreamingEngine, ProxyStreamingSession
│  │  │  └─ tts_proxy.py            ProxyTTSStream
│  │  └─ (v1 files; stt.py and tts.py now build proxies)
│  └─ groups/<group>.txt            requirements per dependency group
├─ worker/
│  ├─ pyproject.toml                package voicebot_worker, stdlib-only core
│  └─ voicebot_worker/
│     ├─ __main__.py                CLI: --slug, --port, --selftest
│     ├─ framing.py                 same frame format as the server
│     ├─ stt_host.py                loads a RealtimeSTT engine, answers STT requests
│     └─ tts_host.py                loads a RealtimeTTS engine, answers TTS requests
├─ venvs/<group>/                   one uv venv per group (git-ignored)
├─ venvs/status.json                install result per slug (git-ignored)
├─ assets/ref_voice.wav, ref_voice.txt   reference voice for cloning engines
└─ scripts/install_models.ps1
```

`framing.py` is duplicated in the worker package so that the worker has no dependency on the server package. A test checks that the two copies are identical.

## 4. Model catalog

`server/voicebot/catalog.toml` is the single source of truth for which models exist.

### 4.1 Entry fields

| Field | Example | Meaning |
|---|---|---|
| `slug` | `sherpa-parakeet-v3` | ID used in the UI, `.env` and logs |
| `kind` | `stt` / `tts` | |
| `role` | `final` / `realtime` | STT only. Picks which UI picker shows the entry |
| `engine` | `sherpa_onnx_parakeet` | RealtimeSTT `transcription_engine` or RealtimeTTS engine class |
| `model` | path, HF id or model name | Passed to the engine as-is |
| `options` | table | Engine options (`transcription_engine_options` or constructor kwargs) |
| `streaming` | `true` / `false` | STT realtime only. `true` uses the streaming-session API |
| `group` | `stt-onnx` | Dependency group, which is also the venv name |
| `device` | `cpu` / `cuda` | |
| `vram_mb` | `0`, `1200` | Estimate, shown in the UI |
| `platforms` | `["win32", "linux"]` | Entries for other platforms are listed but disabled |
| `setup` | `auto` / `manual` | `manual` needs a binary, checkout, server or other step the installer cannot do |
| `setup_notes` | text | Shown in the UI for `manual` entries |
| `prefetch` | `sherpa:parakeet`, `hf:<repo>`, `kroko:<file>`, `none` | How the installer downloads weights |

### 4.2 Slug rule

- Format: `<family>-<model>[-<variant>][-rt]`, in lowercase kebab-case.
- A slug ends in `-rt` if and only if `role = "realtime"`. The catalog loader enforces this.
- A model that can serve both roles gets two entries, for example `whisper-tiny-en` and `whisper-tiny-en-rt`.
- Realtime "off" is not a catalog entry. It is the value `off`.

### 4.3 STT entries

A `*` marks a true streaming engine. The other realtime entries re-transcribe the growing audio buffer.

**Final**

| Slug | Engine | Model | Device |
|---|---|---|---|
| `whisper-tiny-en` | `faster_whisper` | `tiny.en` | cuda |
| `whisper-base-en` | `faster_whisper` | `base.en` | cuda |
| `whisper-small-en` | `faster_whisper` | `small.en` | cuda |
| `whisper-medium-en` | `faster_whisper` | `medium.en` | cuda |
| `whisper-large-v3-turbo` | `faster_whisper` | `large-v3-turbo` | cuda |
| `whispercpp-small-en-q5` | `whisper_cpp` | `small.en-q5_1` | cpu |
| `openai-whisper-small-en` | `openai_whisper` | `small.en` | cuda |
| `transcribe-cpp-parakeet-q4` | `transcribe_cpp` | Parakeet Q4 GGUF (manual download) | cuda |
| `parakeet-onnx-v3` | `parakeet_onnx` | Parakeet TDT 0.6B v3 | cpu |
| `sherpa-parakeet-v3` | `sherpa_onnx_parakeet` | `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | cpu |
| `sherpa-orukeet` | `sherpa_onnx_parakeet` | `oruk/orukeet` | cpu |
| `sherpa-moonshine-tiny-en` | `sherpa_onnx_moonshine` | `sherpa-onnx-moonshine-tiny-en-int8` | cpu |
| `kroko-fp32-en` | `kroko_fp32` | English FP32 bundle | cpu |
| `hf-moonshine-medium` | `moonshine` | `UsefulSensors/moonshine-streaming-medium` | cuda |
| `granite-speech-2b` | `granite_speech` | `ibm-granite/granite-speech-4.1-2b` | cuda |
| `qwen3-asr-1.7b` | `qwen3_asr` | `Qwen/Qwen3-ASR-1.7B` | cuda |
| `cohere-transcribe` | `cohere_transcribe` | `CohereLabs/cohere-transcribe-03-2026` | cuda |
| `funasr-sensevoice-small` | `funasr` | `iic/SenseVoiceSmall` | cuda |
| `funasr-nano` | `funasr` | `FunAudioLLM/Fun-ASR-Nano-2512` | cuda |
| `nemo-parakeet-v3` | `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | cuda, Linux only |
| `omnilingual-ctc-1b` | `omnilingual_asr` | `omniASR_CTC_1B_v2` | cuda, Linux only |
| `omnilingual-llm-1b` | `omnilingual_asr` | `omniASR_LLM_1B_v2` | cuda, Linux only |

**Realtime**

| Slug | Engine | Model | Device |
|---|---|---|---|
| `whisper-tiny-en-rt` | `faster_whisper` | `tiny.en` | cuda |
| `whisper-base-en-rt` | `faster_whisper` | `base.en` | cuda |
| `whispercpp-tiny-en-rt` | `whisper_cpp` | `tiny.en` | cpu |
| `nemotron-rt`\* | `sherpa_onnx_nemotron` | `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` | cpu |
| `nemotron-ort-rt`\* | `nemotron_ort` | Nemotron ORT export | cpu |
| `kroko-en-64-rt`\* | `kroko_onnx` | `Kroko-EN-Community-64-L-Streaming-001.data` | cpu |
| `kroko-en-128-rt`\* | `kroko_onnx` | `Kroko-EN-Community-128-L-Streaming-001.data` | cpu |
| `hf-moonshine-streaming-rt` | `moonshine_streaming` | `UsefulSensors/moonshine-streaming-medium` | cuda |
| `sherpa-moonshine-tiny-en-rt` | `sherpa_onnx_moonshine` | `sherpa-onnx-moonshine-tiny-en-int8` | cpu |
| `omnilingual-ctc-1b-rt` | `omnilingual_asr` | `omniASR_CTC_1B_v2` | cuda, Linux only |

`openai_api` is a placeholder in RealtimeSTT and has no entry.

### 4.4 TTS entries (local only)

| Slug | Engine | Setup | Notes |
|---|---|---|---|
| `kokoro` | `KokoroEngine` | auto | v1 default, CPU |
| `piper` | `PiperEngine` | manual | needs the Piper binary (`PIPER_PATH`) and a voice model |
| `system` | `SystemEngine` | auto | Windows SAPI voices |
| `inflect-onnx` | `InflectEngine` (ONNX) | auto | CPU, fixed English voice |
| `inflect-cuda` | `InflectEngine` (PyTorch) | auto | GPU, fixed English voice |
| `qwen-cuda` | `QwenEngine` | auto | GPU, voice cloning |
| `qwen-cpu` | `QwenCpuEngine` | auto | CPU, voice cloning |
| `coqui-xtts` | `CoquiEngine` | auto | GPU, voice cloning |
| `parler` | `ParlerEngine` | auto | GPU |
| `chatterbox` | `ChatterboxEngine` | auto | reference WAV |
| `omnivoice` | `OmniVoiceEngine` | auto | reference audio and text |
| `luxtts` | `LuxTTSEngine` | auto | voice cloning |
| `neutts` | `NeuTTSEngine` | auto | reference audio |
| `pocket` | `PocketTTSEngine` | auto | CPU, voice cloning |
| `pocket-gpu` | `PocketTTSGpuEngine` | auto | GPU fork |
| `sopro` | `SoproTTSEngine` | auto | reference audio |
| `soprano` | `SopranoEngine` | auto | single English voice |
| `moss-nano` | `MossTTSEngine` | manual | needs runtime assets |
| `styletts` | `StyleTTSEngine` | manual | needs a StyleTTS2 checkout and assets |
| `zipvoice` | `ZipVoiceEngine` | manual | needs a ZipVoice checkout |
| `orpheus` | `OrpheusEngine` | manual | needs a local OpenAI-compatible endpoint |
| `higgs` | `HiggsEngine` | manual | needs a separate SGLang-Omni server (Linux) |

**Reference voice.** Voice-cloning engines need a reference clip. The installer generates `assets/ref_voice.wav` once with Kokoro (`af_heart`), from the sentence in `assets/ref_voice.txt`, so the reference text matches the audio exactly. The user can replace both files.

**Voices.** If an engine exposes `get_voices()`, its voices fill the Voice picker. Otherwise the picker shows only `default`.

### 4.5 Filling in details

The exact model IDs, file names, options and `vram_mb` values are taken from each engine's doc under `RealtimeSTT/docs/engines/` and `RealtimeTTS/docs/engines/` during implementation. The tables above are the required set of entries. An entry may be dropped only if its engine cannot run in a worker at all, and the reason must be recorded in `docs/v2-model-matrix.md`.

## 5. Dependency groups and installer

### 5.1 Groups

`server/groups/<group>.txt` lists each group's requirements. Every group also installs `-e RealtimeSTT` or `-e RealtimeTTS` with the matching extras, plus `-e worker`.

| Group | Engines |
|---|---|
| `stt-whisper` | `faster_whisper`, `whisper_cpp`, `openai_whisper` |
| `stt-onnx` | sherpa (parakeet, orukeet, moonshine, nemotron), `nemotron_ort`, `parakeet_onnx`, `kroko_onnx`, `kroko_fp32`, `transcribe_cpp` |
| `stt-hf` | `moonshine`, `granite_speech`, `qwen3_asr`, `cohere_transcribe` |
| `stt-funasr` | `funasr` |
| `stt-nemo` | `parakeet` (Linux only) |
| `stt-omni` | `omnilingual_asr` (Linux only, Python 3.11) |
| `tts-light` | `kokoro`, `piper`, `system`, `inflect` |
| `tts-qwen` | `qwen-cuda`, `qwen-cpu` |
| one group each | `coqui`, `parler`, `chatterbox`, `omnivoice`, `luxtts`, `neutts`, `pocket`, `pocket-gpu`, `sopro`, `soprano`, `moss`, `styletts`, `zipvoice`, `orpheus`, `higgs` |

- **Splitting rule:** if a group does not resolve, split it until it does, and update the catalog's `group` fields.
- **Python version:** 3.12 for every group, except where an engine requires another version on its supported platform (`stt-omni`).
- **CUDA:** GPU groups install the CUDA build of torch, using the same index URL as v1's `setup_venv.ps1`.

### 5.2 `scripts/install_models.ps1`

Arguments: `-All`, `-Group <name>` or `-Slug <slug>` (repeatable), plus `-Prefetch` and `-Force`.

For each selected group, the script:

1. Runs `uv venv -p <python> venvs/<group>` if the venv is missing.
2. Runs `uv pip install --python venvs/<group> -r server/groups/<group>.txt`.
3. For each slug in the group, runs `venvs/<group>/Scripts/python -m voicebot_worker --slug <slug> --selftest`. This checks that the engine imports, but does not load the model.
4. With `-Prefetch`, downloads the weights according to `prefetch`:
   - `sherpa:*` through `stt-install-sherpa-models --root models/sherpa-onnx`
   - `hf:*` through `huggingface-cli download`
   - `kroko:*` through the Kroko Community download
5. Writes the result into `venvs/status.json`.

`status.json` maps each slug to one of these states:

- `installed`
- `failed`, with a reason
- `manual`, with the `setup_notes`
- `unsupported_platform`

Further rules:

- Groups that are already installed and pass the selftest are skipped, unless `-Force` is given.
- The script never deletes user files or model caches.
- Before installing any group, it generates the reference voice (§4.4) using the `tts-light` venv.

## 6. Worker protocol

### 6.1 Process

- **Start:** `venvs/<group>/Scripts/python -m voicebot_worker --slug <slug> --port 0`.
- **Handshake:**
  1. The worker binds a socket on `127.0.0.1` and prints `PORT <n>` on stdout.
  2. The main server connects.
  3. The worker loads the model.
  4. The worker sends `ready`, with `info`: `sample_rate`, `channels` and `voices` for TTS, plus `load_ms`.
- **Load failure:** the worker sends `fatal{message}` and exits with a non-zero code.
- **Logs:** stderr and Python logging go to `logs/worker-<slug>.log`.

### 6.2 Frames

Each frame has:

1. A 4-byte big-endian length, followed by that many bytes of UTF-8 JSON header. The header always has a `type` field.
2. A 4-byte big-endian length for the body, followed by the body. The length is 0 when there is no body.

All audio bodies are PCM s16le mono.

### 6.3 Messages

| Worker kind | Request | Response |
|---|---|---|
| STT final, STT realtime (re-decode) | `transcribe{sample_rate: 16000, language: "en"}` + PCM | `result{text}` |
| STT realtime (streaming) | `session_open{language}` | `ok{session_id}` |
| | `accept{session_id}` + PCM | `ok` |
| | `partial{session_id}` | `result{text}` |
| | `finish{session_id}` | `result{text}` |
| | `session_close{session_id}` | `ok` |
| TTS | `synthesize{text}` (sent repeatedly as the LLM streams) | none |
| | `end` (no more text) | a stream of `audio` + PCM frames, then `done{text}` |
| | `stop` | `stopped` (no further `audio` frames after it) |
| | `voices` | `voices{list}` |
| | `set_voice{name}` | `ok` |
| All | `ping` | `pong` |
| | `shutdown` | the worker exits |
| All, on failure | any | `error{message}` (the worker keeps running) |

The TTS worker starts sending `audio` frames as soon as it has audio. It does not wait for `end`. `end` only marks that no more text will come.

### 6.4 Main-server proxies

- **`ProxyExecutor`** implements `transcribe(audio, language=None, use_prompt=True, **kwargs)` and returns `TranscriptionResult(text=...)`. It is used as `transcription_executor`, and as `realtime_transcription_executor` for re-decode realtime slugs.
- **`ProxyStreamingEngine`** has `supports_streaming = True`. Its `create_streaming_session()` returns a `ProxyStreamingSession`, whose methods map one-to-one to the streaming messages.
- **`ProxyTTSStream`** has `feed(iterator)`, `play(on_audio_chunk=None, muted=...)`, `stop()`, `text()` and `engine`. `engine` has `get_voices()`, `set_voice()` and `get_stream_info()`. This matches how v1's `VoiceSession` and `audio_io` use `TextToAudioStream`, so `session.py` and `audio_io.py` need only small changes. In server mode, `play` writes PCM to PyAudio in the main server.

### 6.5 WorkerManager

- Owns the three role slots: `stt_final`, `stt_realtime`, `tts`.
- **`switch(role, slug)`:**
  1. If the old and new slugs both use the GPU, and `LLM + others + new` would exceed `GPU_TOTAL_MB` while the old one is still loaded, it stops the old worker first. Otherwise it keeps the old worker running.
  2. It starts the new worker and waits for `ready`, up to `WORKER_START_TIMEOUT`.
  3. On success, it points the proxy at the new worker and shuts down the old one.
  4. On failure, it keeps the old worker. If the old worker was already stopped, it restarts it, then reports the failure.
- **Heartbeat:** `ping` every 2 s. No reply for 6 s marks the worker crashed.
- **Clean-up:** workers are assigned to a Windows Job Object with kill-on-close, so they die with the main server. On Linux, the server uses `prctl(PR_SET_PDEATHSIG)`.
- **Recorder rebuild:** if the realtime role switches between `off` and a slug, or between re-decode and streaming, `AudioToTextRecorder` must be rebuilt with different arguments. The manager rebuilds it while idle, and the listen loop restarts.

## 7. Settings

New `.env` variables. They replace v1's `STT_MODEL`, `STT_DEVICE` and `STT_COMPUTE_TYPE`, and v1's `TTS_VOICE` stays.

| Variable | Default | Meaning |
|---|---|---|
| `STT_FINAL` | `whisper-small-en` | startup final STT slug |
| `STT_REALTIME` | `off` | startup realtime STT slug, or `off` |
| `TTS_MODEL` | `kokoro` | startup TTS slug |
| `GPU_TOTAL_MB` | `8192` | used by the VRAM meter and the switch logic |
| `LLM_VRAM_MB` | `6000` | LLM estimate for the VRAM meter |
| `WORKER_START_TIMEOUT` | `300` | seconds; the first run may download weights |

Choices made in the UI last until restart.

## 8. Web UI

**SettingsPanel additions**

- Three pickers: `Final STT`, `Realtime STT` (first option `off`) and `TTS`. Options are grouped by family.
- Each option shows its slug, a `cpu`/`cuda` badge and its VRAM estimate.
- Options whose status is not `installed` are disabled. Hovering shows the reason, the manual setup steps, or "Linux only".
- A **VRAM meter** shows LLM + final STT + realtime STT + TTS against `GPU_TOTAL_MB`. It turns amber when over budget. Over-budget choices are still allowed.
- The **Voice** picker is filled from the TTS worker's `voices`. It is reset when the TTS model changes.
- While any switch is in progress, all pickers are locked, and the status pill shows `loading <slug>`.

**Chat:** with realtime STT on, a grey partial line shows the live text. It is replaced by the final user message when the turn ends.

## 9. WebSocket protocol changes

These are additions to v1 §4.9.

| Direction | Type | Fields |
|---|---|---|
| C→S | `select_model` | `role`: `stt_final` / `stt_realtime` / `tts`; `slug` (`"off"` allowed for `stt_realtime`) |
| S→C | `catalog` | `entries`: catalog entries merged with `status` and `status_detail` |
| S→C | `model_status` | `role`, `slug`, `state`: `loading` / `ready` / `failed`; `message?`, `load_ms?` |
| S→C | `user_partial` | `text` |
| S→C | `config` | adds `stt_final`, `stt_realtime`, `tts`, `gpu_total_mb`, `llm_vram_mb` |

- On connect, the server sends `config` and then `catalog`. `GET /api/catalog` returns the same data as the `catalog` event.
- `select_model` is refused with `error{code:"busy"}` during a turn, and with `error{code:"bad_model"}` for an unknown slug or a slug in the wrong role.
- After a TTS switch, the server sends a new `config`, whose `output_format` may have a different `sample_rate`. The browser player resamples to the new rate.

## 10. Error handling

| Failure | Behaviour |
|---|---|
| Selected slug is not `installed` | `model_status failed` with the status detail; the current model is kept |
| Worker fails to load (import error, CUDA OOM, `fatal`, timeout) | `model_status failed` with the last 30 log lines; roll back to the previous slug |
| Worker crashes during a turn | `error` event; spoken apology through `kokoro` (or text only if that fails); the selected slug is restarted once; if that fails, `model_status failed` and the previous default is used |
| Startup default slug unavailable | Fall back to `whisper-small-en` / `off` / `kokoro` with a warning; if those fail too, stop at startup and name the slug |
| `status.json` missing | Every slug is treated as not installed except the v1 defaults, which are checked by starting them |
| Main server exits | The Job Object kills all workers |

## 11. Testing

**Python (pytest, no GPU, no models)**

- `test_catalog.py`: schema, unique slugs, `-rt` ⇔ `role = "realtime"`, every `group` has a requirements file, `streaming` set only on realtime entries, status merging.
- `test_framing.py`: encode/decode round trip, empty body, frames split across reads; the server and worker copies are identical.
- `test_worker_manager.py`: uses a fake worker, a stdlib-only script speaking the protocol. Covers ready, switch, rollback on `fatal`, rollback on timeout, the stop-old-first path when over budget, heartbeat timeout, and kill on close.
- `test_stt_proxy.py`: `ProxyExecutor` and `ProxyStreamingSession` against an in-process fake worker.
- `test_tts_proxy.py`: `feed`/`play` stream PCM to the callback, `stop` stops audio, `text()` returns the spoken text.
- `test_session.py` (extended): `user_partial` events, `select_model` refused mid-turn, recorder rebuilt when realtime toggles.
- `test_web.py` (extended): `catalog` on connect, `select_model` routing, `bad_model`.
- `voicebot_worker` tests: `--selftest` with a fake engine module; each message type with a fake engine.

**Frontend (vitest)**

- Reducer handles `catalog`, `model_status` and `user_partial`.
- VRAM sum and the over-budget flag.
- Player sample-rate change.

**Manual**

For every `installed` slug, one voice turn with the default LLM backend. Record `load_ms`, `ttft_ms`, `first_audio_ms` and any failure in `docs/v2-model-matrix.md`.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Disk use: each GPU group venv holds its own torch (~2–3 GB) | Install per group or per slug; document sizes in the matrix |
| Some engine packages do not install on Windows | The entry gets `failed` with a reason; it stays visible but disabled |
| GPU models do not fit beside the 6 GB LLM | VRAM meter warning; OOM rollback; CPU entries available for every role |
| IPC adds latency per realtime call | Localhost TCP with binary bodies; measured in the matrix |
| Recorder rebuild on realtime toggle drops a turn | Only allowed while idle; listen loop restarts automatically |
