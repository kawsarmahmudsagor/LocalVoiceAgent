# Local Voice Bot — Design

- **Date:** 2026-11-18
- **Status:** Draft, awaiting review
- **Location:** `c:\Users\BS23-DESKTOP-00038\Projects\LocalVoiceAgent`

## 1. Goal

A test web app for talking to a local LLM by voice. The user speaks, Qwen3.5-9B answers, and the reply is spoken back. A playground-style browser UI shows the conversation and exposes the generation settings.

The app runs entirely on one Windows PC. Two choices are made in `.env` and take effect on restart:

- `LLM_BACKEND=transformers|llamacpp` — how Qwen3.5-9B runs.
- `AUDIO_IO=browser|server` — whether the microphone and speaker are the browser's or the PC's.

### Success criteria

1. With either backend and either audio mode, a spoken English question gets a spoken answer.
2. Speech starts within about 1–2 s after the user stops talking (with the llamacpp backend on the RTX 4060).
3. Switching backend or audio mode needs only a `.env` edit and a restart.
4. The model's reasoning (`<think>` content) is never shown or spoken.
5. All unit tests pass without a GPU or downloaded models. `tsc --noEmit` and `npm run build` pass.

### Out of scope (v1)

- Interrupting the bot by voice (barge-in). The Stop button is the only interrupt.
- More than one simultaneous user.
- Languages other than English.
- Switching backend or audio mode at runtime.
- Authentication, HTTPS, and deployment beyond localhost or a LAN.

## 2. Constraints and verified facts

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA RTX 4060, 8 GB VRAM | `nvidia-smi` |
| System Python | 3.14 (too new) | `python --version` |
| Required Python | 3.12 | RealtimeSTT `setup.py`: `python_requires='>=3.11,<3.13'` |
| LLM (transformers) | `Qwen/Qwen3.5-9B`, Apache-2.0, not gated, 19.3 GB BF16, architecture `Qwen3_5ForConditionalGeneration` (vision + text) | HF Hub API |
| LLM (llamacpp) | `unsloth/Qwen3.5-9B-GGUF`, file `Qwen3.5-9B-Q4_K_M.gguf`, 5.68 GB. The repo also has `mmproj-*.gguf` (0.92 GB or more), which must **not** be loaded | HF Hub API |
| Thinking mode | On by default. The chat template disables it only when `enable_thinking` is explicitly `false` | `chat_template` in `tokenizer_config` |
| STT external audio | `AudioToTextRecorder(use_microphone=False)` + `feed_audio(bytes, original_sample_rate=16000)`, 16-bit mono PCM | `RealtimeSTT/docs/external-audio.md` |
| STT mic muting | `AudioToTextRecorder.set_microphone(bool)` | `RealtimeSTT/RealtimeSTT/audio_recorder.py:899` |
| TTS output to code | `stream.play(on_audio_chunk=cb, muted=True)` | `RealtimeTTS/docs/output-and-files.md:57-65` |
| Kokoro audio format | int16, mono, 24000 Hz | `KokoroEngine.get_stream_info()` |

### VRAM budget (8 GB)

| Component | Device | Approx. VRAM |
|---|---|---|
| Qwen3.5-9B, 4-bit (either backend) | GPU | ~6 GB |
| faster-whisper `small.en`, int8 | GPU by default; `STT_DEVICE=cpu` moves it | ~0.5–1 GB |
| Kokoro | CPU | 0 |

## 3. Architecture

```
 Browser (React + TS)                 Python server (FastAPI, 3.12 venv)
 ┌──────────────────────┐   /ws      ┌───────────────────────────────────────────┐
 │ SettingsPanel        │◄──JSON────►│ web.py ── VoiceSession (one at a time)     │
 │ ChatTranscript       │            │             │                              │
 │ Composer + mic       │──PCM16────►│   AudioIO ──┤  BrowserAudio | ServerAudio  │
 │ playback worklet     │◄─PCM16─────│             │                              │
 └──────────────────────┘            │   STT (RealtimeSTT, faster-whisper small.en)│
                                     │   LLM backend ── transformers (in process)  │
                                     │               └─ llamacpp ──► llama-server  │
                                     │   speech_filter                             │
                                     │   TTS (RealtimeTTS, KokoroEngine on CPU)    │
                                     └───────────────────────────────────────────┘
```

### 3.1 Repository layout

```
LocalVoiceAgent/
├─ RealtimeSTT/  RealtimeTTS/          cloned upstream repos, installed with pip install -e
├─ .env.example                        every setting, with comments
├─ server/
│  ├─ pyproject.toml
│  ├─ voicebot/
│  │  ├─ config.py                     Settings: loads and validates .env
│  │  ├─ llm/
│  │  │  ├─ base.py                    LLMBackend protocol, GenerationParams
│  │  │  ├─ transformers_backend.py
│  │  │  ├─ llamacpp_backend.py
│  │  │  ├─ llama_server.py            llama-server.exe process manager
│  │  │  └─ factory.py                 picks the backend from settings, with lazy imports
│  │  ├─ speech_filter.py              removes <think> blocks and markdown symbols from the stream
│  │  ├─ conversation.py               system prompt and trimmed history
│  │  ├─ stt.py                        builds AudioToTextRecorder
│  │  ├─ tts.py                        builds TextToAudioStream(KokoroEngine)
│  │  ├─ audio_io.py                   BrowserAudio, ServerAudio
│  │  ├─ session.py                    VoiceSession: the turn loop, emits events
│  │  ├─ web.py                        FastAPI app: /ws, /api/config, static web/dist
│  │  └─ smoke_llm.py                  text-only check of the selected backend
│  └─ tests/
├─ web/
│  ├─ package.json  vite.config.ts  tsconfig.json  index.html
│  └─ src/
│     ├─ main.tsx  App.tsx  styles.css
│     ├─ protocol.ts                   message types shared by the UI
│     ├─ ws.ts                         WebSocket client with reconnect
│     ├─ state.ts                      reducer for conversation and status
│     ├─ components/  SettingsPanel.tsx  ChatTranscript.tsx  Composer.tsx  StatusBar.tsx
│     └─ audio/  pcm.ts  mic-worklet.ts  player-worklet.ts  mic.ts  player.ts
├─ scripts/
│  ├─ setup_venv.ps1                   uv venv -p 3.12, CUDA torch, editable installs, server deps
│  ├─ fetch_llama_server.ps1           downloads the official llama.cpp Windows CUDA release into tools/llama/
│  └─ dev.ps1                          starts uvicorn and npm run dev together
└─ docs/superpowers/specs/             this document
```

## 4. Components

### 4.1 `config.py` — Settings

The one place that reads `.env` (via `python-dotenv`). It builds a frozen dataclass and fails at startup with the variable's name when a value is missing or invalid.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` | `transformers` or `llamacpp` |
| `AUDIO_IO` | `browser` | `browser` or `server` |
| `HF_MODEL_ID` | `Qwen/Qwen3.5-9B` | transformers model |
| `HF_LOAD_IN_4BIT` | `true` | bitsandbytes NF4 |
| `LLAMA_AUTOSTART` | `true` | the app starts and stops llama-server |
| `LLAMA_SERVER_PATH` | `tools/llama/llama-server.exe` | binary location |
| `LLAMA_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | passed to `-hf` |
| `LLAMA_HOST` / `LLAMA_PORT` | `127.0.0.1` / `8081` | server address |
| `LLAMA_BASE_URL` | derived from host and port | used when `LLAMA_AUTOSTART=false` |
| `LLAMA_CTX` | `8192` | `-c` |
| `LLAMA_START_TIMEOUT` | `900` | seconds; the first run downloads 5.7 GB |
| `STT_MODEL` | `small.en` | faster-whisper model |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` | `cuda` / `int8` | |
| `STT_SILENCE_SEC` | `0.6` | `post_speech_silence_duration` |
| `TTS_VOICE` | `af_heart` | Kokoro voice |
| `SYSTEM_PROMPT` | `You are a friendly voice assistant. Reply in one to three short spoken sentences. Never use lists, markdown, code blocks or emoji.` | default for the UI |
| `LLM_TEMPERATURE` / `LLM_TOP_P` / `LLM_MAX_TOKENS` | `0.7` / `0.8` / `256` | defaults for the UI |
| `HISTORY_TURNS` | `10` | user+assistant pairs kept |
| `WEB_HOST` / `WEB_PORT` | `127.0.0.1` / `8000` | FastAPI |

### 4.2 LLM backends

```python
@dataclass
class GenerationParams:
    temperature: float
    top_p: float
    max_tokens: int

class LLMBackend(Protocol):
    name: str
    model_label: str
    def start(self) -> None: ...                    # load the model or start the server; blocks until ready
    def stream(self, messages: list[dict], params: GenerationParams,
               stop_event: threading.Event) -> Iterator[str]: ...
    def close(self) -> None: ...
```

`stream()` yields only non-empty text. When `stop_event` is set, it stops within one token or chunk.

**`factory.make_backend(settings)`** imports the selected backend module inside the function. With `llamacpp` selected, `torch`, `transformers` and `bitsandbytes` are never imported. A test checks this.

**`TransformersBackend`**
- Loads `AutoProcessor` and `AutoModelForImageTextToText` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` and `device_map="cuda:0"`.
- Builds prompts with `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt")`.
- Runs `model.generate` on a worker thread with `TextIteratorStreamer(skip_prompt=True, skip_special_tokens=True)` and a `StoppingCriteria` that returns `True` once `stop_event` is set.
- Startup checks:
  - `torch.cuda.is_available()` must be true.
  - `bitsandbytes` must import.
  - A CUDA out-of-memory error during load is re-raised with the hint "set STT_DEVICE=cpu or use LLM_BACKEND=llamacpp".
- **Open item for implementation:** check whether `AutoModelForCausalLM` can load only the text part of this checkpoint, which would save the vision encoder's VRAM. If it can't, keep `AutoModelForImageTextToText`.

**`LlamaCppBackend`**
- Uses the `openai` client (`base_url=f"{LLAMA_BASE_URL}/v1"`, `api_key="local"`) with `chat.completions.create(stream=True, temperature, top_p, max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": False}})`.
- Closes the response stream when `stop_event` is set.
- `start()`:
  - If `LLAMA_AUTOSTART=true`, it owns a `LlamaServer`.
  - Otherwise it only polls `GET {base}/health` until it returns 200 or the timeout passes.

**`LlamaServer`**
- Command:
  `llama-server.exe -hf <LLAMA_HF_REPO> --no-mmproj --jinja -ngl 99 -c <LLAMA_CTX> --host <host> --port <port>`
- Server output goes to `logs/llama-server.log`.
- `start()`:
  - Fails fast if the binary is missing (pointing at `scripts/fetch_llama_server.ps1`) or the port is already in use.
  - Polls `/health` until it returns 200. While the model loads, the server returns 503, and polling continues.
  - If the process exits early or the timeout passes, it kills the process and includes the last 30 log lines in the error.
- `close()` terminates the process, waits 5 s, then kills it. Shutdown is also registered with `atexit`.

### 4.3 `speech_filter.py`

`filter_stream(chunks: Iterator[str]) -> Iterator[str]` is a small state machine:

1. Drops everything between `<think>` and `</think>`, including tags split across chunks. An unclosed `<think>` drops the rest of the reply.
2. Removes the markdown symbols `*`, `#` and backticks, and turns list bullets at the start of a line into plain text.
3. Never yields an empty string.

The same filtered text is sent to both TTS and the UI, so the screen shows what is spoken.

### 4.4 `conversation.py`

- Holds the system prompt and a list of `{role, content}` turns.
- `messages(user_text)` returns the system message, the last `HISTORY_TURNS` user+assistant pairs, and the new user message.
- `add_exchange(user, assistant)` saves a turn.
- `clear()` empties the history.
- The default system prompt asks for short conversational spoken replies with no lists, markdown or emoji.

### 4.5 `stt.py` and `tts.py`

- **`make_recorder(settings, use_microphone)`** builds `AudioToTextRecorder` with:
  - `model=STT_MODEL`, `language="en"`, `device`, `compute_type`
  - `use_microphone=use_microphone`
  - `post_speech_silence_duration=STT_SILENCE_SEC`
  - `spinner=False`, `enable_realtime_transcription=False`
- **`make_tts(settings)`** builds `TextToAudioStream(KokoroEngine(voice=TTS_VOICE))`. It warms up once with a short phrase played muted and discarded. `set_voice(name)` changes the voice for the next turn.

### 4.6 `audio_io.py`

Both classes have the same interface, which `VoiceSession` uses:

| Method | `BrowserAudio` | `ServerAudio` |
|---|---|---|
| `recorder` | `make_recorder(use_microphone=False)` | `make_recorder(use_microphone=True)` |
| `on_client_audio(bytes)` | `recorder.feed_audio(b, original_sample_rate=16000)`, ignored while muted | ignored |
| `mute_input(bool)` | sets an ignore flag | `recorder.set_microphone(not muted)` |
| `play(stream, on_done)` | `stream.play(on_audio_chunk=send_binary, muted=True)` | `stream.play()` on the PC speakers |
| `output_format` | `{"sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}` from `engine.get_stream_info()` | `null` (the browser plays nothing) |

### 4.7 `session.py` — VoiceSession

Owns the backend, the audio I/O, the TTS stream, the `Conversation`, the current `GenerationParams` and voice, and one `stop_event` per turn. It runs the listen loop on a worker thread and sends events to the WebSocket through a thread-safe callback (`loop.call_soon_threadsafe`).

**Turn loop**

```
while listening:
  status(listening)
  text = recorder.text()                          # blocks; VAD ends the turn
  if is_noise(text): continue                     # "", ".", "you", fewer than 2 letters
  run_turn(text)

run_turn(text):                                    # also called directly by send_text
  emit user_transcript(text)
  audio.mute_input(True); status(thinking)
  stop_event = new Event()
  chunks = tee(filter_stream(backend.stream(conversation.messages(text), params, stop_event)))
      → each chunk: emit assistant_delta(chunk), and record time to first token
  stream.feed(chunks); status(speaking) on first audio; record time to first audio
  audio.play(stream)                              # blocks until done or stopped
  conversation.add_exchange(text, stream.text())
  emit assistant_done(stream.text(), interrupted=stop_event.is_set())
  emit metrics(ttft_ms, first_audio_ms, total_ms)
  audio.mute_input(False)
```

**Commands**

| Command | Behavior |
|---|---|
| `start_listening` / `stop_listening` | Start or end the listen loop. Stopping calls `recorder.abort()` so a blocked `text()` returns. |
| `stop_speaking` | Sets `stop_event` and calls `stream.stop()`. The partial reply is kept in history. |
| `send_text` | Runs `run_turn` with typed text. Ignored with an `error` event while a turn is already running. |
| `update_settings` | Updates `GenerationParams`, the system prompt and the voice, starting from the next turn. |
| `clear` | Empties the conversation. |

**Errors**
- An exception from the LLM during a turn is logged and sent as an `error` event.
- The bot then speaks "Sorry, something went wrong." and the loop continues.

### 4.8 `web.py` — FastAPI

- **Startup:** `Settings.load()` → `make_backend().start()` → create the `VoiceSession` resources (recorder and Kokoro load once).
- **Shutdown:** stop the session, `recorder.shutdown()`, `backend.close()`.
- **`GET /api/config`** returns:
  - `llm_backend`, `audio_io`, `model_label`
  - `voices` (from `KokoroEngine.get_voices()`)
  - default settings and `output_format`
- **`WS /ws`:**
  - Only one client at a time. A second connection receives `error{code:"session_busy"}` and is closed.
  - On connect, the server sends `config`.
  - Text frames are JSON commands. Binary frames are mic PCM (browser mode).
  - Server events are JSON, and TTS audio is sent as binary frames.
- **Static files:** `web/dist` is served at `/` when it exists.

### 4.9 WebSocket protocol

Every JSON frame has a `type` field. `web/src/protocol.ts` mirrors these types.

**Client → server**

| type | fields |
|---|---|
| `start_listening` | — |
| `stop_listening` | — |
| `stop_speaking` | — |
| `send_text` | `text` |
| `update_settings` | `system_prompt?`, `temperature?`, `top_p?`, `max_tokens?`, `voice?` |
| `clear` | — |
| *(binary)* | mic audio: PCM s16le, mono, 16 kHz, about 20–100 ms per frame |

**Server → client**

| type | fields |
|---|---|
| `config` | same as `/api/config` |
| `status` | `state`: `idle` / `listening` / `thinking` / `speaking` |
| `user_transcript` | `text` |
| `assistant_delta` | `text` |
| `assistant_done` | `text`, `interrupted` |
| `metrics` | `ttft_ms`, `first_audio_ms`, `total_ms` |
| `error` | `code`, `message` |
| *(binary)* | TTS audio in `output_format` (browser mode only) |

## 5. Frontend (`web/`)

- **Stack:** npm, Vite, React 18, TypeScript (strict), plain CSS, no UI library.
- **Look:** dark, playground-style layout. The design is original; no OpenAI branding or assets.

**Layout**
- **Left sidebar (SettingsPanel):**
  - Read-only badges for backend, audio mode and model.
  - System prompt textarea.
  - Sliders for temperature (0–2), top_p (0–1) and max tokens (32–1024).
  - Voice dropdown.
  - Changes are sent as `update_settings`, debounced by 300 ms.
- **Center (ChatTranscript):** user and assistant messages. The assistant message grows with each `assistant_delta`. Interrupted replies are marked.
- **Bottom (Composer and StatusBar):**
  - Mic toggle (start or stop listening), text input with Send (Enter), Stop button, Clear.
  - Status pill and the last turn's latency numbers.

**Audio (browser mode)**
- **`mic.ts`:** `getUserMedia({audio:{echoCancellation:true, noiseSuppression:true, channelCount:1}})`, passed to an `AudioWorklet` (`mic-worklet.ts`). It converts to 16 kHz Int16 (`pcm.ts`) and sends ~40 ms binary frames.
- **`player.ts`:** a playback `AudioWorklet` with a queue. It converts incoming s16le at the server's `sample_rate` into Float32 at the `AudioContext` rate. `stop_speaking` and `clear` flush the queue.
- **Secure context:** the browser only allows mic access on `localhost` or HTTPS. That is acceptable for this test app.
- **Server mode:** the UI does not touch the mic or speakers. The mic toggle sends only `start_listening` / `stop_listening`.

**State:** a single `useReducer` in `state.ts` applies server events to `{status, messages, settings, config, metrics, error}`.

**Dev and build**
- `npm run dev`: Vite on :5173, proxying `/api` and `/ws` to `127.0.0.1:8000`.
- `npm run build`: creates `web/dist`, which FastAPI serves.
- `npm run test`: vitest.
- `npm run typecheck`: `tsc --noEmit`.

## 6. Error handling

| Failure | Behavior |
|---|---|
| Invalid or missing `.env` value | Stop at startup and name the variable |
| `llama-server.exe` missing | Stop and point at `scripts/fetch_llama_server.ps1` |
| llama-server port in use | Stop and suggest `LLAMA_AUTOSTART=false` or a different `LLAMA_PORT` |
| llama-server not ready within `LLAMA_START_TIMEOUT`, or exited early | Kill it and show the last 30 log lines |
| CUDA unavailable, or bitsandbytes missing (transformers) | Stop and say which |
| CUDA out of memory | Stop and suggest `STT_DEVICE=cpu` or `LLM_BACKEND=llamacpp` |
| LLM error during a turn | `error` event, spoken apology, loop continues |
| Mic permission denied (browser) | Show an error banner; typed input still works |
| WebSocket drops | UI shows "disconnected" and reconnects with backoff; the server stops the session's listen loop |
| Second browser client | `session_busy`, then close |

## 7. Testing

**Python (pytest, no GPU, no model downloads)**
- `test_config.py`: defaults, parsing, an invalid `LLM_BACKEND` or `AUDIO_IO`, and the derived `LLAMA_BASE_URL`.
- `test_factory.py`: picks the right class. With `llamacpp`, `torch` is not in `sys.modules`.
- `test_speech_filter.py`: `<think>` split across chunks, an unclosed tag, markdown removal, no empty strings.
- `test_conversation.py`: history trimming and clear.
- `test_llama_server.py`: the command line it builds (including `--no-mmproj --jinja`), polling 503 then 200, timeout and early exit with the log tail, and terminate-then-kill on close. Subprocess and HTTP are mocked.
- `test_llamacpp_backend.py`: sends `chat_template_kwargs.enable_thinking=false`, skips empty deltas, stops on `stop_event`. Uses a fake OpenAI stream.
- `test_session.py`: one voice turn and one typed turn with a fake recorder, TTS and backend. Checks the event order, that input is muted before playback and unmuted after, that `stop_speaking` marks the reply interrupted and keeps the partial text, and that noise transcripts are skipped.
- `test_web.py`: FastAPI `TestClient` with a fake session. Covers `config` on connect, JSON commands reaching the session, a second client getting `session_busy`, and binary frames reaching `on_client_audio`.

**Frontend (vitest)**
- `pcm.test.ts`: Float32 → Int16 clamping, resampling 48 kHz → 16 kHz length and values, s16le → Float32.
- `state.test.ts`: applying the reducer to each server event.
- `typecheck` and `build` must pass.

**Manual checks**
1. `python -m voicebot.smoke_llm` with each `LLM_BACKEND`: one prompt, streamed reply printed, no `<think>` text.
2. A full voice turn in each `AUDIO_IO` mode with each backend (4 combinations), noting `metrics`.

## 8. Setup flow (developer)

1. `scripts/setup_venv.ps1`:
   - `uv venv -p 3.12 .venv`
   - install CUDA torch
   - `pip install -e RealtimeSTT -e "RealtimeTTS[kokoro]" -e server[dev]` (plus `bitsandbytes`, `transformers`, `accelerate`, `openai`, `fastapi`, `uvicorn[standard]`, `python-dotenv`)
2. `scripts/fetch_llama_server.ps1`: download the latest official llama.cpp Windows CUDA build into `tools/llama/`.
3. `copy .env.example .env` and choose `LLM_BACKEND` and `AUDIO_IO`.
4. `cd web && npm install`.
5. `scripts/dev.ps1`, then open http://localhost:5173.

On the first run, the selected backend downloads its weights: 5.7 GB for the GGUF or 19.3 GB for the transformers checkpoint, cached under the Hugging Face cache.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Qwen3.5's new architecture is unsupported by the installed `transformers` or llama.cpp build | Pin a recent `transformers`; fetch the latest llama.cpp release; `smoke_llm` checks each backend early |
| 4-bit model plus STT exceeds 8 GB | `STT_DEVICE=cpu`; skip mmproj for GGUF; the open item to load text-only in transformers |
| Python 3.12 wheels missing for an audio dependency (e.g. `pyaudio`, `webrtcvad`) | Use the binary wheels pinned by RealtimeSTT; document any manual install in the setup script |
| Browser echo triggers VAD in browser mode | Half-duplex muting plus browser echo cancellation; headphones recommended |
| bitsandbytes on Windows | Recent bitsandbytes ships Windows CUDA wheels; the startup check reports clearly if the import fails |
