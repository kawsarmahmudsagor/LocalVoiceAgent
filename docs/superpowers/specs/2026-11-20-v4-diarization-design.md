# Local Voice Bot v4 — Speaker Diarization

- **Date:** 2026-11-20
- **Status:** Implemented; manual checks open
- **Branch:** `feat/v4-diarization` (from `feat/v3-barge-In` at `6fd9312`)
- **Builds on:** [v1 design](2026-11-18-local-voice-bot-design.md), [v2 design](2026-11-18-v2-model-switching-design.md), [v3 RAG design](2026-11-19-v3-rag-design.md) and [v3 barge-in design](2026-11-20-barge-in-design.md). v2 §2 (Linux VM, browser audio) still applies.

## 1. Goal

Make the bot talk only to its user when other people are in the room, and measure what that costs in latency. A streaming diarization model labels who is speaking. Speech from anyone but the main speaker does not start a turn and cannot interrupt the bot. The diarization stage is timed like STT, LLM and TTS, so its cost shows in the turn logs.

Diarization tells *who spoke when*, with anonymous labels (Speaker 0, 1, …). It does not identify a person.

### What the user asked for

- Speaker diarization, not speaker verification: no voice enrolment.
- The model: NVIDIA Nemotron 3 Diarization from Hugging Face (`nvidia/Nemotron-3-Diarization`).
- The rule against editing `RealtimeSTT/` and `RealtimeTTS/` is lifted for this change, but nothing may break.
- The turn log shows whether the diarization model was on.

### The model (checked 2026-11-20)

| Item | Value |
|---|---|
| Checkpoint | `nvidia/Nemotron-3-Diarization`, 100M parameters, released 2026-09-23 |
| Licence | OpenMDW 1.1, commercial use permitted |
| Speakers | Up to 8. Output channels are ordered by each speaker's first arrival |
| Input | 16 kHz mono float32 |
| Output | Per-speaker activity probability for each 10 ms frame, shape `(1, frames, 8)`. Overlap allowed |
| Runtimes | Hugging Face transformers (`AutoProcessor`, `AutoModelForAudioFrameClassification`; streaming with `processor.set_streaming_mode(...)` and a `speaker_cache` passed between calls) or NVIDIA NeMo |
| Streaming modes (input-buffer latency) | `ultra_low_latency` 0.32 s, `very_low_latency` 0.64 s, `low_latency` 1.04 s. 0.32 s is the lowest recommended |
| Speed (model card, bf16, batch 1, eager / compiled) | RTFx 12.5 / 54 at 0.32 s |
| Accuracy (DIHARD III full, DER) | 12.73 offline-style, 13.55 at 0.32 s |
| Not stated by NVIDIA | Minimum transformers version, torch requirement, GPU memory, threshold guidance, minimum speech before a speaker is assigned |

Sources: [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization), [transformers docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron3_diarization.md), [ASR integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md), [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-diarization).

### Decisions made during design (2026-11-20)

| Topic | Decision | Reason |
|---|---|---|
| What diarization controls | Speaker labels, the turn check and the barge-in check (§3) | The full "talk only to my user" behaviour |
| Main speaker | Speaker 0 (the first voice in the session) is always the main speaker. Clear resets it | Simplest rule. The model already orders speakers by arrival. The bot's echo arrives after the user speaks, so it gets another label and is filtered too |
| Runtime | transformers, in the server process if it resolves with the server's dependencies, else a worker process with its own venv (Task 1 decides) | Same rule as v2. No pinned transformers version is documented, so this must be checked on the VM |
| Audio source | A copy of the browser frames at `BrowserAudio.on_client_audio` | The browser already sends 16 kHz mono int16. RealtimeSTT does not need to change |
| Vendored code | Allowed to change for this feature, but no change is planned. Any change is recorded in `VENDORED.md` and the vendored repo's own tests must pass | The user lifted the rule; "nothing breaks" still holds |
| Failure behaviour | Fail open: a slow, failing or missing diarizer never blocks the user | A measurement tool must not lose turns to a helper model |
| Default | Off | Existing runs and timings are unchanged unless diarization is switched on |
| Audio mode | Browser mode only | Server mode is Windows v1 and out of scope |

### Success criteria

1. With diarization on and one person speaking, every turn runs as before, labelled Speaker 0, and the record shows `diarization: true` with `diar_wait_ms`.
2. With a second person speaking, their utterances do not start turns. The UI shows them greyed out as "Speaker N (ignored)" and they are not in history.
3. With barge-in on, only Speaker 0 can interrupt the bot. Another voice or the bot's own echo does not.
4. With diarization off, behaviour and timings are exactly those of v3, and no diarization code is imported.
5. `turns.jsonl` has `diarizer_on` / `diarizer_off` events when the model is switched, and every turn record says whether diarization was on and in which mode.
6. Every existing server and web test passes, every existing optional extra still installs with `torch==2.7.1+cu128`, and the benchmark tools are untouched.
7. Any diarizer failure switches diarization off, shows an error in the UI, and voice turns carry on as in v3.

### Non-goals

- Identifying who a speaker is, or voice enrolment.
- Choosing a different main speaker from the UI (Clear is the reset).
- More than 8 speakers.
- Server audio mode.
- Using diarization to split overlapping speech into separate transcripts.

## 2. Runtime and data flow

1. **Dependency check (plan Task 1, on the VM).** Install a transformers version that contains `nemotron3_diarization` into the server venv with `torch==2.7.1+cu128` pinned. The check passes if:
   - the resolve keeps `torch==2.7.1+cu128`;
   - every existing extra (`transformers`, `rag`, the TTS/STT engines on the shortlist) still installs;
   - the full server test suite passes;
   - a smoke script loads the model on CUDA and diarizes a 10 s clip.

   If it passes, the diarizer runs in the server process. If not, it runs in a worker process with its own venv (`.venvs/diarizer`), reached over a local socket with the same `Diarizer` interface. The result and the chosen versions are recorded in this spec's §8.
2. **`server/voicebot/diarizer.py`** holds a `Diarizer` class:
   - It loads `nvidia/Nemotron-3-Diarization` with `AutoProcessor` and `AutoModelForAudioFrameClassification` in bf16 on CUDA, on first use, off the event loop (like the word aligner).
   - `feed(pcm: bytes)` puts a copy of a client frame on a bounded queue and never blocks.
   - A diarizer thread converts frames to float32, cuts them into the processor's streaming chunks, runs the model with the `speaker_cache` from the previous call, and appends the chunk's 10 ms frames to a timeline.
   - Each timeline frame stores the set of speakers with probability ≥ `DIAR_THRESHOLD` (0.5) and the audio sample index it covers.
   - `position() -> int`: the number of samples fed so far (the shared clock).
   - `covered() -> int`: the sample index up to which the timeline has frames.
   - `speaker_between(start, end) -> int | None`: the speaker active in the most frames between two sample indices, or `None` when no frame has an active speaker.
   - `share(speaker, start, end) -> float | None`: the fraction of diarized frames in the window where that speaker is active; `None` when the window is not yet covered.
   - `reset()` clears the speaker cache, the timeline and the clock. The next voice becomes Speaker 0.
   - `rtf()`: compute time divided by audio duration since the last call.
3. **Audio tap.** `BrowserAudio.on_client_audio` passes each frame to the recorder as today and, when a diarizer is attached, also to `diarizer.feed`. It skips frames while input is muted (mode `off` turns), like the recorder.
4. **Clock.** The recorder and the diarizer see the same frames in the same order, so the diarizer's sample counter is the shared clock:
   - On `on_recording_start`, the session records `diarizer.position()` minus RealtimeSTT's `pre_recording_buffer_duration` (converted to samples) as the utterance start.
   - On `on_recording_stop`, it records `diarizer.position()` as the utterance end.
5. **Streaming mode** is a setting: `ultra_low_latency` (0.32 s, default), `very_low_latency` (0.64 s), `low_latency` (1.04 s). A change resets the diarizer.
6. **Reset.** Clear, any model switch, a mode change and switching diarization off all call `reset()`.

## 3. Checks

1. **Who spoke an utterance.** When the listen loop has a transcript that is not noise:
   - Wait until `diarizer.covered()` reaches the utterance end, for at most `diar_wait_ms`, checking every 20 ms.
   - Then `speaker_between(start, end)` gives a speaker number, or `None`.
   - `None`, or a timeout, gives `"unknown"`.
2. **Turn check.**
   - Speaker 0 or `"unknown"`: the turn runs as normal. `"unknown"` passes so that a lagging or confused diarizer never blocks the user.
   - Any other speaker: no turn. The session emits `user_transcript` with `speaker` and `ignored: true`, does not add it to history, and writes a `diar_ignored` event (§4).
3. **Barge-in check** (only when barge-in is on).
   - `onset` mode: when the min-ms timer fires, the session computes `share(0, onset, now)`.
     - At least 0.5: fire.
     - Below 0.5: do not fire. Count a `diar_rejects` and let the onset re-arm on the next pause and resume, as in v3.
     - `None` (the diarizer has not reached the window yet): re-check every 40 ms, up to `diar_wait_ms`, then fire (fail open).
   - `words` mode: the same check runs when the first non-noise partial arrives.
   - The wait is part of `barge_detect_ms`, so its cost is measured.
4. **Settings** (`update_settings`, reported in `config.defaults`):

   | Setting | Values | Default |
   |---|---|---|
   | `diarization` | bool | `false` |
   | `diar_mode` | `"ultra_low_latency"`, `"very_low_latency"`, `"low_latency"` | `"ultra_low_latency"` |
   | `diar_wait_ms` | int, 0–2000 | 600 |

   Switching `diarization` on loads the model and emits `{"type": "diarization", "state": "loading" | "ready" | "off" | "error", "message"?}`, like `word_align`. `config` adds `diarization_available` (false in server audio mode).
5. **UI.**
   - Playground settings get a **Speaker diarization** toggle, a **Diarization latency** select (0.32 / 0.64 / 1.04 s) and a **Max wait (ms)** slider (0–2000, step 50), with a status line (loading, ready, error).
   - The controls are disabled when `diarization_available` is false.
   - User messages show a small `S0`, `S1`, … tag.
   - Ignored utterances appear greyed out with "Speaker N (ignored)".

## 4. Metrics and logs

New marks: `diar_start` (the session starts the speaker decision), `diar_done` (the decision is made).

New spans:

| Span | From → to | Meaning |
|---|---|---|
| `diar_wait_ms` | `diar_start` → `diar_done` | The cost of the turn check |

`perceived_ms` and `total_ms` include it, because the LLM starts after the decision.

New turn record fields:

| Field | Value |
|---|---|
| `diarization` | `true` / `false` |
| `diar_mode` | the streaming mode, or `null` when off |
| `diar_speaker` | int, `"unknown"`, or `null` when off |
| `diar_lag_ms` | how far the timeline trailed the utterance end when the decision started |
| `diar_rejects` | int: barge-in checks that rejected another speaker in this turn |
| `diar_rtf` | the diarizer's real-time factor over this turn |

New event records in `turns.jsonl` (they carry `"event"`, so `latency_report`'s reply summaries skip them, as with `barge_false_fire`):

| Event | Fields |
|---|---|
| `diarizer_on` | `ts`, `diar_mode`, `load_ms` |
| `diarizer_off` | `ts`, `reason` (`"user"`, `"error"`, `"overflow"`) |
| `diar_ignored` | `ts`, `speaker`, `text`, `diar_wait_ms`, `diar_mode` |

The server log also prints a line when the diarizer turns on or off. The text log (`turns.txt`) prints one line per event and adds the diarization fields to each turn block. `latency_report --by diarization,diar_mode` compares runs; `diar_wait_ms` is added to the span table.

## 5. Errors

All of these switch diarization off, emit `diarization{state: "error", message}`, write `diarizer_off` and leave voice turns working as in v3:

- the model fails to load (missing dependency, download failure, CUDA OOM);
- an exception in the diarizer thread;
- the frame queue overflows (frames are dropped and counted; 50 drops in a row switch it off with reason `"overflow"`).

A turn that is waiting on the diarizer when it fails gets `"unknown"` at once.

## 6. Compatibility ("nothing breaks")

- With `diarization` off, no diarization module is imported and the turn, barge-in and audio paths behave exactly as in v3.
- The full server and web test suites pass.
- The dependency check in §2.1 passes, or the diarizer moves to a worker venv.
- `bench/` and the vendored benchmark tools are untouched.
- No change to `RealtimeSTT/` or `RealtimeTTS/` is planned. If one becomes necessary, it is recorded in `VENDORED.md` and that repo's own unit tests (`RealtimeSTT/tests/unit`) are run.

## 7. Testing

No GPU, model or network in automated tests.

**Server**

- `test_diarizer.py`, with a fake model object returning scripted logits:
  - chunking follows the processor's chunk sizes;
  - the timeline maps frames to sample indices;
  - `speaker_between`, `share` and `covered` give the expected values;
  - `reset` clears the cache, timeline and clock;
  - queue overflow is counted and triggers the overflow switch-off.
- `test_session.py`, with a `FakeDiarizer` whose timeline is scripted:
  - Speaker 0 is answered and tagged.
  - Speaker 1 is ignored, not added to history, and logged as `diar_ignored`.
  - `"unknown"` and a timeout pass.
  - Barge-in: Speaker 0 fires; Speaker 1 is rejected and counted; a lagging diarizer is waited for; a diarizer that never catches up fails open.
  - Clear, a model switch and a mode change reset the diarizer.
  - A diarizer error switches diarization off and emits the error event.
  - Settings validation; `diarizer_on` / `diarizer_off` events.
  - With diarization off, the v3 tests pass unchanged.
- `test_timing.py` and `test_latency_report.py`: the new span, fields and events.
- `test_audio_io.py`: the tap feeds the diarizer only when attached and not muted.

**Web**

- State: speaker tags, ignored transcripts, the `diarization` status event, the new config fields.
- Settings panel: the controls, disabled when `diarization_available` is false.

**Manual (the user, on the VM through the SSH tunnel)**

1. One speaker with diarization on: turns are answered and tagged S0. Compare `perceived_ms` with diarization off.
2. Two people: the second person's speech is shown as ignored and not answered.
3. Speakers instead of headphones with barge-in on: the bot's echo does not interrupt it.
4. Each latency mode: compare `diar_wait_ms`, `diar_lag_ms` and accuracy with `latency_report --by diarization,diar_mode`.

## 8. Dependency check result

Measured on 2026-11-20 on the Linux VM (RTX 5090 32 GB, NVIDIA driver 580.178.04, Ubuntu 24.04).

**Runtime: in process.** The server venv `.venv` already had transformers **5.23.0** (the latest PyPI release), which contains `nemotron3_diarization`, with torch **2.7.1+cu128**. No upgrade and no git pin were needed. The extra `diarization = ["transformers==5.23.0", "accelerate"]` pins it, and `scripts/setup_venv.sh` installs it.

**`uv pip check` on `.venv`:** three incompatibilities, all present before v4 and not caused by it. gradio and gradio-client want huggingface-hub<2.0 (2.4.0 is installed). qwen-asr wants transformers==4.57.6 (5.23.0 is installed). Criterion applied: no new conflict was introduced.

**Test suite:** pending on VM. The branch is not yet on the VM. The server suite passes on the Windows dev machine (370 passed, 1 known Windows-only failure).

**Load time:** 33.7 s on the first run (includes the model download), 7.9 s warm.

**Output is logits**, not probabilities. Values range from about -45.8 to +7.7, so the wrapper applies a sigmoid (`OUTPUT_IS_LOGITS = True`).

**Forward call:** `model(**inputs.to(device, dtype=model.dtype), speaker_cache=cache)` works as written. It returns `.logits` with shape (1, n, 8) and `.speaker_cache`.

**Probe clip:** 13.5 s, two voices (A 0–5 s, B 5.25–10.25 s, A 10.5–13.5 s), 16 kHz mono, built from RealtimeTTS test WAVs. All three modes labelled A as Speaker 0, B as Speaker 1, and A again as Speaker 0. `ultra_low_latency` also produced one 0.11 s spurious Speaker 1 blip at 1.09 s.

| mode | latency ms | num_samples_first_audio_chunk | num_samples_per_audio_chunk | num_mel_frames_per_step | audio_chunk_start(mel_per_step) | output frames per step | probe RTF, eager bf16 |
|---|---|---|---|---|---|---|---|
| ultra_low_latency | 320 | 5160 | 5520 | 24 | 3584 | 24 | 0.175 |
| very_low_latency | 640 | 10280 | 10640 | 48 | 7424 | 48 | 0.051 |
| low_latency | 1040 | 16680 | 17040 | 72 | 11264 | 72 | 0.032 |

**Task 4 smoke** (`python -m voicebot.diarizer --smoke`, `ultra_low_latency` through the `Diarizer`, 20 ms pieces): load 7.9 s, RTF 0.141, covered 13.20 s of 13.5 s, speakers per second correct.
