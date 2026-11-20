# LocalVoiceAgent — Specifications and Engineering

This document explains what the LocalVoiceAgent project was built against (its specifications) and what engineering was done to solve the problems that came up along the way. It covers every phase now on `main`: v1 to v4.

- **Part 1** is a plain-language summary for anyone.
- **Part 2** is the engineering record: architecture, the problems each phase met and how they were solved, the measurement design, and the numbers that have been recorded.

The detailed design documents remain the source of truth. They are listed in §2.

---

# Part 1 — Summary

## What the project is

A voice assistant that runs entirely on local hardware, built to **measure latency**: how long each step takes between the moment a person stops talking and the moment they hear the reply.

One voice turn goes through these steps:

1. **Listen.** The browser streams the microphone to the server. A voice-activity detector decides when the person has finished speaking.
2. **Transcribe (STT).** A speech-to-text model turns the audio into text.
3. **(Optional) Check the speaker (v4).** A diarization model decides who spoke. Only the main user is answered.
4. **(Optional) Look up documents (v3, RAG).** Relevant passages from the user's uploaded files are found and given to the language model.
5. **Answer (LLM).** Qwen3.5-9B writes the reply, streamed word by word.
6. **Speak (TTS).** A text-to-speech model turns the reply into audio, which is streamed back to the browser.

Each step is timed, and every turn is written to a log, so different models and settings can be compared with numbers rather than impressions.

A hard rule shaped the whole project: use only the engines and benchmark tools in two open-source libraries that are copied into the repo, **RealtimeSTT** and **RealtimeTTS**, and do not modify them.

## What was built, phase by phase

| Phase | What it added | Why |
|---|---|---|
| **v1** | The working voice bot on a Windows PC (RTX 4060, 8 GB), with a playground-style web UI and per-step timing | Get an end-to-end baseline and a way to measure it |
| **v2** | A port to a Linux GPU server (RTX 5090, 32 GB). Tooling to benchmark 14 STT/TTS engines with the libraries' own tools. Model switching from the UI without a restart | Find which engines are fastest at acceptable quality, then compare them inside real voice turns |
| **v3 — RAG** | Answers grounded in the user's documents (`.txt`, `.md`, `.pdf`, `.docx`), with two local embedding backends and a RAG settings tab | Measure what document lookup costs in latency |
| **v3 — Barge-in** | The user can interrupt the bot by speaking. Two trigger modes. The conversation history keeps only the words the user actually heard | Measure how fast an interruption takes effect, and how often it misfires |
| **v4 — Diarization** | The bot answers, and can be interrupted by, only its main user, using NVIDIA's Nemotron 3 diarization model | Stop other people in the room, and the bot's own echo, from triggering turns; measure the cost |

## Main problems and how they were solved, in brief

- **The model hardware was new (Blackwell GPU).** Every Python environment uses one exact PyTorch build that supports it (`2.7.1+cu128`). The language-model server runs in a Docker image, so the host's outdated CUDA toolkit is never used.
- **The libraries could hang, crash or misbehave on the edges.** Small runtime workarounds live in our own code and are documented, so the upstream code stays untouched.
- **Timing had to be honest.** The real end of speech is taken from the right callback, and every step has its own start and end marks. Waits that a user feels (silence detection, the speaker check, document lookup) are counted in the latency the user perceives.
- **Optional features must never break the basic bot.** Diarization, RAG and barge-in are all off by default, fail open (an error never blocks a turn), and are not even imported when switched off.
- **Several things happen at once** (microphone, transcription, model, playback, diarizer). Background work runs on separate threads with a fixed lock order, and every change was reviewed specifically for race conditions.

## Status

All four phases are merged to `main`. Automated tests: 386 server tests and 48 web tests pass. The one failing test, `test_catalog_toml_round_trips`, is a known Windows-only encoding issue unrelated to this work. Manual checks that need a person speaking (v3 barge-in, v4 diarization) are still to be done on the VM. See §9.

---

# Part 2 — Engineering

## 2. Specifications used

### 2.1 Design documents in this repo

| Phase | Specification | Implementation plan |
|---|---|---|
| v1 | `docs/superpowers/specs/2026-11-18-local-voice-bot-design.md` | `docs/superpowers/plans/2026-11-18-local-voice-bot.md` |
| v2 | `docs/superpowers/specs/2026-11-18-v2-model-switching-design.md` | `docs/superpowers/plans/2026-11-18-v2-linux-benchmark-shortlist.md` |
| v3 RAG | `docs/superpowers/specs/2026-11-19-v3-rag-design.md` | `docs/superpowers/plans/2026-11-19-v3-rag.md` |
| v3 Barge-in | `docs/superpowers/specs/2026-11-20-barge-in-design.md` | `docs/superpowers/plans/2026-11-20-barge-in.md` |
| v4 Diarization | `docs/superpowers/specs/2026-11-20-v4-diarization-design.md` | `docs/superpowers/plans/2026-11-20-v4-diarization.md` |

`docs/SPECIFICATIONS.md` is a plain-language overview of v1. `CLAUDE.md` records the decisions already made and the reasons for them. `VENDORED.md` records the upstream versions and every workaround.

Each spec ends with numbered **success criteria**. Each plan breaks the work into test-first tasks, and its "Review Focus" names the riskiest behaviours. The spec is the authority; where a plan and its spec disagreed during implementation, the spec won, and the ruling is recorded in the commit history.

### 2.2 External sources the specs rely on

| Source | Used for |
|---|---|
| **RealtimeSTT** (vendored, commit `3574c24`, Apache-2.0) | Voice activity detection, turn detection, STT engines, external-audio feeding, and the benchmark tools `benchmark_asr_ab.py` and `benchmark_asr_streaming.py` |
| **RealtimeTTS** (vendored, commit `7443092`, Apache-2.0) | TTS engines and the streaming playback API |
| `Qwen/Qwen3.5-9B` and `unsloth/Qwen3.5-9B-GGUF` (Q4_K_M) | The LLM: the transformers checkpoint (19.3 GB) or the llama.cpp GGUF (5.68 GB) |
| llama.cpp `llama-server` | The OpenAI-compatible LLM server. Windows binary in v1; Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda` on the VM |
| Chroma, sentence-transformers (`BAAI/bge-small-en-v1.5`), Ollama (`nomic-embed-text`) | v3 RAG vector store and embedders |
| `nvidia/Nemotron-3-Diarization` (model card, transformers docs) | v4 streaming speaker diarization |

### 2.3 Facts verified in the code before designing

The specs only build on behaviour that was checked in the vendored source, not assumed:

| Fact | Where | Consequence |
|---|---|---|
| `post_speech_silence_duration` defaults to 0.6 s | `RealtimeSTT/docs/configuration.md` | Every voice turn pays a fixed 0.6 s silence wait; it is timed as `silence_wait_ms` |
| `on_vad_stop` fires *after* the silence wait | `RealtimeSTT/core/recording.py` (~line 474) | The true end of speech is `on_turn_detection_start`, which v1's timing uses |
| A truthy return from `on_transcription_start` aborts the transcription | `core/transcription_api.py` (~line 53) | Our callback always returns `None` |
| After the first `text()`, the recorder keeps listening and queues recordings (`continuous_listening`) | `core/lifecycle.py`, `core/recording.py` | Barge-in needs no library patch: keep the mic unmuted during a turn |
| `on_turn_detection_start` needs `min_length_of_recording` (0.5 s) plus ~0.16 s of silence | `core/recording.py` (lines 398–421) | A short noise cannot cancel an onset barge-in below ~650 ms; false fires are logged instead |
| Qwen3.5's chat template thinks by default unless `enable_thinking` is explicitly `false` | `tokenizer_config` chat template | Both backends pass `enable_thinking=False`; a stream filter also removes any `<think>` text |

## 3. Architecture

```
 Browser (React + TypeScript)              Python server (FastAPI, Python 3.12)
 ┌───────────────────────────┐   /ws      ┌──────────────────────────────────────────────┐
 │ Tabs: Playground |        │◄──JSON────►│ web.py ── VoiceSession (one client at a time) │
 │   System Prompt | RAG     │            │   audio_io  BrowserAudio ─┬─► RealtimeSTT      │
 │ Settings, transcript,     │──PCM16────►│                           └─► Diarizer (v4)    │
 │   latency breakdown       │  16 kHz    │   rag/ (v3)  Chroma + HF / Ollama embedder     │
 │ mic + playback worklets   │◄─PCM16─────│   llm/       llama.cpp (Docker) | transformers │
 └───────────────────────────┘            │   speech_filter ─► RealtimeTTS engine          │
                                          │   timing.py ─► logs/turns.jsonl, turns.txt     │
                                          └──────────────────────────────────────────────┘
```

Main modules in `server/voicebot/`:

| Module | Responsibility |
|---|---|
| `config.py` | Reads and validates `.env`; fails at startup and names any bad variable |
| `session.py` | The turn loop, commands, barge-in, the diarization lifecycle and checks |
| `audio_io.py` | Browser and server audio; the mic tap to the diarizer |
| `stt.py`, `tts.py`, `tts_engines.py`, `catalog.py`, `catalog.toml` | Build STT and TTS from catalog entries; runtime shims for vendored issues |
| `llm/` | LLM backends behind one protocol, imported lazily |
| `speech_filter.py` | Removes `<think>` blocks and markdown from the token stream |
| `rag/` | Loaders, chunker, embedders, Chroma store, document library, indexing service |
| `diarizer.py` | Streaming diarizer, 10 ms speaker timeline, Nemotron adapter, smoke CLI |
| `timing.py`, `latency_report.py` | Per-stage marks and spans; grouped latency reports |

## 4. Engineering by phase: problems and solutions

### 4.1 v1 — the voice bot (Windows, RTX 4060, 8 GB)

| Problem | Engineering response |
|---|---|
| System Python was 3.14; RealtimeSTT requires `>=3.11,<3.13` | A dedicated `uv` venv on Python 3.12 |
| An 8 GB GPU must hold a 9B LLM and STT | 4-bit models (~6 GB); the GGUF's vision projector is skipped with `--no-mmproj`; llama-server runs with a single slot; `STT_DEVICE=cpu` is available as a fallback |
| Two LLM runtimes must not load each other's heavy libraries | `llm/factory.py` imports the selected backend inside the function. A test checks that `torch` is not imported when llamacpp is used |
| Qwen3.5 "thinks" by default, and thinking text must never be spoken | `enable_thinking=False` in both backends, plus a streaming state machine (`speech_filter.py`) that drops `<think>` blocks even when tags are split across chunks, and strips markdown |
| The screen and the speech could disagree | The same filtered text goes to TTS and to the UI |
| Startup could hang forever if STT failed to load its model | `stt.preflight` checks the known causes first and fails with a fix hint; `expose_cuda_dlls` puts torch's CUDA DLLs on the path for CTranslate2 |
| Pressing Stop could crash the process (`0xc0000374` heap corruption) | RealtimeTTS closes the PyAudio stream without a lock. A runtime shim serialises `AudioStream.close_stream` with a class lock. Recorded in `VENDORED.md`; no vendored file is edited |
| Stop froze the socket; audio from a stopped turn leaked into the next one | `stream.stop()` runs off the event loop; the audio adapter drops chunks from a stopped turn |
| Kokoro synthesises faster than real time, so the server ended turns while the browser still had seconds of audio queued | The turn stays open until the audio's estimated end; Stop ends the wait |
| A quick Mic off/on could run two listen loops | Each listen run gets a generation number |
| llama-server could be orphaned on shutdown | Shutdown never blocks on RealtimeSTT's abort and always closes the backend; `atexit` also stops the server |
| Where does the time go? | `TurnTimer` with named marks (§5); every turn is appended to `logs/turns.jsonl` |

### 4.2 v2 — Linux VM, benchmarking and model switching (RTX 5090, 32 GB)

| Problem | Engineering response |
|---|---|
| The Blackwell GPU (`sm_120`) needs a recent CUDA; the host's `nvcc` 12.0 is too old | `torch==2.7.1+cu128` in every venv (it brings its own CUDA runtime). llama-server runs from a Docker image, with `scripts/build_llama_image.sh` available to build `sm_120` kernels in a container. The host toolkit is never used |
| A driver update left `nvidia-smi` failing with "Driver/library version mismatch" | `scripts/check_vm.sh` detects it and says to reboot |
| The VM has no microphone, speakers or sound card | Browser audio only; server audio mode is rejected at startup on Linux; no audio output device is opened in browser mode |
| The browser on the user's PC needs a secure context for the mic | An SSH tunnel to `localhost` (no certificates needed); optional self-signed HTTPS |
| Engines that need different torch versions would break each other and use more disk | One torch for every engine. An engine that needs another torch is dropped and recorded, not worked around. `uv` hardlinks one cached torch into every venv |
| Choosing engines fairly | A curated pool of 14 entries (spec §3.1) chosen for latency per GB, plus two added on user request. Benchmarks use the libraries' own tools (`stt-server-production` with `benchmark_asr_ab.py` / `benchmark_asr_streaming.py`); TTS uses a small script built only on the public RealtimeTTS API. The rule for picking the shortlist is written down in advance (spec §3.4) |
| `kroko-onnx` installs its own ONNX Runtime (1.17.1) into sherpa-onnx's folder, so `sherpa_onnx` fails to import in the same venv | Kroko was removed from the server catalog and kept only in its own bench venv |
| sherpa Parakeet voice turns took ~300 ms | The CPU configuration measured in RealtimeSTT's docs is used: sherpa-onnx 1.13.8, ONNX Runtime thread spinning disabled (`assets/sherpa-cpu-session.conf`). The docs report ~71 ms for 3 s of audio and ~107 ms for 5 s |
| llama.cpp re-downloaded the GGUF on every container run; a missing image made the health check time out | The download cache is mounted into the container; the image is pulled before the health timeout starts |
| Switching models at runtime without leaking GPU memory or breaking a turn | Switches are allowed only while idle. The old model is shut down, then the new one is built; a failure rebuilds the previous one. TTS switches resend `output_format` so the browser resamples. Worker processes were specified but only to be built if needed; the shortlist runs in process |
| Comparing models inside real turns | Every turn record carries the active slugs, so `latency_report --by stt_final,tts_model` compares them directly |

### 4.3 v3 — RAG

| Problem | Engineering response |
|---|---|
| Retrieval must not slow down chat turns that have nothing to retrieve | Retrieval runs on every turn while RAG is on, but context is injected only for chunks above a similarity threshold. Turns with no match pay only `retrieval_ms` |
| Retrieval must be measured as its own stage | `retrieval_start` / `retrieval_done` marks; `llm_ttft_ms` starts at `retrieval_done`, so it stays the LLM's own time |
| A voice turn could wait for a whole indexing job | The indexing worker used an unfair lock. Queries now register first, and the worker yields before each batch of 32 chunks, so a query waits for at most one batch |
| Large documents failed to index | Writes are sliced by Chroma's maximum batch size |
| Switching embedders while indexing could mix vectors | In-flight index jobs are invalidated on an embedder switch; one collection per embedder |
| A retrieval failure (for example Ollama down) must not fail a turn | The error is logged and shown, and the turn answers without context |
| Context must not pollute history | Context goes into the system message only; history keeps the raw user text and the reply |

### 4.4 v3 — Barge-in

| Problem | Engineering response |
|---|---|
| Interrupting by voice without patching RealtimeSTT | The library already keeps listening and queues recordings (§2.3), so the mic simply stays unmuted during an armed turn. With mode `off`, the v1 behaviour is unchanged |
| A fast trigger vs a robust one | Two runtime modes: `onset` (voice onset plus a minimum speech time) and `words` (first non-noise realtime partial). Both are measured |
| A short noise can fire an `onset` barge-in, because RealtimeSTT cannot report the end of a short sound below ~650 ms | Kept the fast trigger. A fired barge-in whose transcript turns out to be noise is written as a `barge_false_fire` event, so the false-fire rate is measured |
| After an interrupt, history should hold only what the user heard | The browser reveals each word as its audio plays. After a cut, it reports the revealed text (`playback_cut`); the server stores that as the reply. No acknowledgement means the full reply is kept |
| Measuring the interruption itself | `barge_detect_ms` (trigger cost) and `barge_cut_ms` (onset to audio stopped in the browser) |

### 4.5 v4 — Speaker diarization

| Problem | Engineering response |
|---|---|
| Does the model run in the server's environment? | Checked on the VM before building: transformers 5.23.0, already in the server venv, contains the model, with `torch==2.7.1+cu128`. No new dependency conflict. Result: in process, no worker venv |
| Output format unknown (the docs disagree between "logits" and "probabilities") | Probed on the GPU: values ranged from about −45.8 to +7.7, so they are logits; the adapter applies a sigmoid |
| Feeding the model the same audio STT hears, on the same clock | The browser mic frames (16 kHz mono) are copied to the diarizer in `BrowserAudio`. The diarizer's sample counter is the shared clock; the utterance window is `[recording start − 0.2 s pre-roll, recording stop]` |
| The model must never block the audio path | `feed()` only enqueues (bounded queue). A dedicated thread runs the model. 50 consecutive drops switch diarization off (`overflow`). Dropped frames are replaced by silence so the timeline stays aligned with the clock |
| Who is the main user? | Speaker 0 = the first voice since the last reset. Clear, a model switch, a mode change and switching diarization off all reset it |
| A slow, failing or missing diarizer must not block the user | Everything fails open: a timeout, an unknown speaker, an error, a reset during a decision → `"unknown"`, which is answered. A failure during a wait releases the turn within one 20 ms poll |
| Clear or a model switch during an utterance mixes the old and new clocks | Every reset bumps an epoch and clears the stored windows; a decision on a window from an older epoch is `"unknown"` |
| The next utterance starting before the previous one is checked could turn an ignored speaker into "unknown" | The completed window is snapshotted when recording stops, so a new recording cannot overwrite it |
| Barge-in must accept only Speaker 0 without blocking RealtimeSTT's callbacks | The check runs on its own thread: share of Speaker 0 ≥ 0.5 fires; below 0.5 is rejected and counted (`diar_rejects`); not yet covered waits up to `diar_wait_ms`, then fires; a covered but silent window fires at once |
| A resumed utterance with a 0 ms minimum was checked over an empty window | The resume onset reaches back by the 0.2 s pre-roll, like the recording start |
| After an overflow error, switching back on silently disabled later error reports | The error handler stops the old diarizer on a separate thread, and switch-on waits for that stop before starting a fresh run |
| Diarization off must be exactly v3 | `voicebot.diarizer` and `transformers` are not imported when it is off (a test checks this); records gain only `diarization: false` and null fields |
| Measuring the cost | `diar_start` / `diar_done` marks, the `diar_wait_ms` span (included in `perceived_ms`), plus `diar_lag_ms`, `diar_rtf`, `diar_rejects` and on/off/ignored events |

## 5. Measurement design

### 5.1 Stages (marks on the turn timer)

`speech_end` → `endpoint` → `stt_start` → `stt_done` → `diar_start` → `diar_done` → `retrieval_start` → `retrieval_done` → `llm_first_token` → `first_text` → `tts_first_audio` → `playback_started` → `llm_done` → `play_done`. Typed turns start at `input`. Barge-in adds `barge_onset`, `barge_in` and `playback_cut`.

### 5.2 Spans (milliseconds, per turn)

| Span | Meaning |
|---|---|
| `silence_wait_ms` | End of speech to endpoint: the VAD silence wait (0.6 s by default) |
| `stt_ms` | Final transcription |
| `diar_wait_ms` | The speaker decision (v4) |
| `retrieval_ms` | Document lookup (v3) |
| `llm_ttft_ms` | LLM time to first token, starting after the decision and retrieval |
| `filter_ms`, `tts_first_audio_ms`, `playback_ms` | Speech filter, TTS first audio, browser playback start |
| `perceived_ms` | End of speech to first audio heard: the number a user feels |
| `llm_total_ms`, `total_ms` | Whole LLM stream; whole turn |
| `barge_detect_ms`, `barge_cut_ms` | Barge-in trigger cost; onset to audio stopped |

Each turn record also carries its context: active model slugs, RAG settings, barge-in mode and diarization fields. `python -m voicebot.latency_report --by <fields>` groups and compares them (for example `--by diarization,diar_mode`). Event records (`barge_false_fire`, `diarizer_on`, `diarizer_off`, `diar_ignored`) are skipped by the reply summaries.

## 6. Cross-cutting engineering rules

| Rule | Why |
|---|---|
| Never edit `RealtimeSTT/` or `RealtimeTTS/`; work around issues in `server/voicebot/` and record them in `VENDORED.md` | The project measures these libraries as they are; local patches would distort the results and block upstream updates |
| One torch build (`2.7.1+cu128`) in every venv | It supports the RTX 5090, avoids conflicts, and keeps disk use low |
| Lazy imports for optional heavy features | A feature that is off costs nothing and cannot break startup |
| Fail open | A measurement tool must not lose turns to a helper model |
| Features off by default | Existing runs and timings stay comparable |
| Test-first, with no GPU, network, Docker or models in automated tests | Fakes stand in for models; real hardware is checked by smoke commands and manual runs |
| A separate review after each task, plus a final whole-branch review | During v3 barge-in and v4, per-task reviews caught real race conditions and clock bugs before merge |
| A fixed lock order in `session.py`: `_diar_lock` → `_state_lock` → the diarizer's internal lock | Prevents deadlocks between the audio thread, the listen loop, the barge check and the diarizer thread |
| Secrets stay out of git (`.env` ignored; gitleaks pre-commit and pre-push hooks) | Safe sharing of the repo |

## 7. Recorded numbers

Only numbers recorded in the repo's specs and commit history are listed here.

| Item | Value | Source |
|---|---|---|
| RealtimeSTT default silence wait | 0.6 s per voice turn | `RealtimeSTT/docs/configuration.md` |
| Model sizes | Qwen3.5-9B GGUF Q4_K_M 5.68 GB; transformers checkpoint 19.3 GB | v1 spec §2 |
| sherpa Parakeet (CPU, tuned) | ~71 ms for 3 s of audio, ~107 ms for 5 s (RealtimeSTT docs); ~300 ms observed before tuning | commit `84c8e37` |
| Nemotron diarization load (RTX 5090) | 33.7 s first run (includes download); 7.9 s warm | v4 spec §8 |
| Nemotron streaming, probe RTF (bf16) | 0.175 at 0.32 s latency; 0.051 at 0.64 s; 0.032 at 1.04 s | v4 spec §8 |
| Nemotron through the full `Diarizer` (0.32 s mode) | RTF 0.141; 13.20 s of a 13.5 s clip covered; both voices labelled correctly | v4 spec §8 |
| Automated tests on `main` | 386 server tests and 48 web tests pass (1 known Windows-only failure) | 2026-11-20 run |

The v2 benchmark results matrix (`docs/v2-model-matrix.md`, spec §3.3) is not committed to the repo yet, so per-engine STT and TTS benchmark numbers are not listed here. Per-turn latencies live in `logs/turns.jsonl` on the machine that ran them.

## 8. Known limits

- **Speaker 0 is "the first voice".** If the bot's own echo is the first voice after a reset (for example, Clear pressed while the bot is speaking through loudspeakers), the echo becomes Speaker 0. The UI hint says to Clear while the bot is quiet.
- **One completed utterance window is kept.** If two utterances both finish before the first one's speaker check, the first is judged on the second's window. This is rare.
- **The barge check reads `covered()` after `share()`.** A diarizer step landing between the two calls can let a barge fire without its speaker check. The window is microseconds per 40 ms poll; it is a one-line follow-up.
- **Onset barge-in below ~650 ms cannot be cancelled by a short noise** (RealtimeSTT limit). False fires are logged.
- **A noise and a real barge-in in one turn:** the noise's timing marks win, so the next turn can show `null` for `silence_wait_ms` and `stt_ms`.
- **Pre-existing dependency conflicts in the server venv:** gradio and gradio-client want `huggingface-hub<2.0` (2.4.0 installed); qwen-asr wants transformers 4.57.6 (5.23.0 installed). None was introduced by v4.
- **The transformers LLM backend on the VM is deferred** by user decision; the VM uses llama.cpp.

## 9. Open items

| Item | Owner |
|---|---|
| Run the full server test suite on the VM for v4 (spec §8 says "pending on VM") | User, then record the result in spec §8 |
| v3 barge-in manual checks on the VM | User |
| v4 manual checks: one speaker; two people; speakers instead of headphones; each latency mode compared with `latency_report --by diarization,diar_mode` | User |
| Commit the v2 benchmark matrix (`docs/v2-model-matrix.md`) | Whoever holds the Phase A results |

## 10. Where to find things

| Looking for | Go to |
|---|---|
| Why a decision was made | `CLAUDE.md` ("Decisions already made"), then the phase's spec |
| How a feature should behave | The phase's spec (§2.1 above) |
| What changed and why | `git log` on `main`; commit messages explain each fix |
| Workarounds for vendored code | `VENDORED.md` |
| Timing definitions | `server/voicebot/timing.py` |
| Per-turn measurements | `logs/turns.jsonl` and `logs/turns.txt`; compare with `python -m voicebot.latency_report --by ...` |
| VM setup | `scripts/check_vm.sh`, `scripts/setup_venv.sh`, `scripts/dev.sh`; access through the SSH tunnel in v2 spec §2.5 |
