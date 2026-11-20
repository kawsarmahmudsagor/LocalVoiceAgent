# v4 Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label who is speaking with `nvidia/Nemotron-3-Diarization`, answer and accept barge-ins only from Speaker 0, and time the diarization stage like STT, LLM and TTS.

**Architecture:** A copy of each browser mic frame goes to a `Diarizer` (new `server/voicebot/diarizer.py`), which runs the model in streaming mode on its own thread and keeps a 10 ms speaker timeline indexed by sample count. The session reads that timeline at the recording start and stop callbacks to decide who spoke an utterance (turn check) and who is talking over the bot (barge-in check). Everything fails open, and with diarization off no diarization code is imported.

**Tech Stack:** Python 3.12, transformers (`AutoProcessor`, `AutoModelForAudioFrameClassification`), torch 2.7.1+cu128, numpy, pytest; React + TypeScript + vitest.

**Spec:** `docs/superpowers/specs/2026-11-20-v4-diarization-design.md`

## Global Constraints

- Branch `feat/v4-diarization`. Stage files by explicit path, never `git add -A`, never `--no-verify`. Every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push only when the user asks.
- `torch==2.7.1+cu128` in every venv. A transformers version that needs another torch is not used.
- Default off. With `diarization` off, behaviour and timings are exactly those of v3, and neither `voicebot.diarizer` nor `transformers` is imported by the server.
- Fail open: a slow, failing or missing diarizer never blocks a turn or a barge-in.
- Speaker 0 (first voice since the last reset) is the main speaker. Clear, any model switch, a mode change and switching diarization off reset the diarizer.
- Exact values: `DIAR_THRESHOLD = 0.5`; frame 10 ms = 160 samples at 16 kHz; turn-check poll 20 ms; barge-check poll 40 ms; barge share threshold 0.5; overflow after 50 consecutive drops; `diar_wait_ms` int 0–2000 default 600; `diar_mode` one of `ultra_low_latency` (default), `very_low_latency`, `low_latency`.
- No change to `RealtimeSTT/` or `RealtimeTTS/` is planned. If one becomes necessary, record it in `VENDORED.md` and run `RealtimeSTT/tests/unit`.
- `bench/` and the vendored benchmark tools stay untouched.
- Tests run without GPU, model or network. Server: from `server/`, `../.venv/bin/python -m pytest -q` (Windows: `../.venv/Scripts/python`). Web: from `web/`, `npm run test && npm run typecheck && npm run build`. Known unrelated failure on Windows only: `tests/test_bench_resolve.py::test_catalog_toml_round_trips`.
- Tasks 1, 4 (step 6) and 11 run on the VM. If executing off the VM, do the other tasks first and mark these as waiting for the VM in the ledger. Task 9 runs only if Task 1 fails.

## Review Focus

1. The diarizer fails while the listen loop is waiting on it: the turn must get `"unknown"` at once and run, not wait out `diar_wait_ms` (Task 7 test `test_diarizer_failure_during_wait_passes_at_once`).
2. Clear during a recording: the utterance window now points before or past the reset clock. The decision must be `"unknown"` without an exception (Task 3 `test_queries_on_bad_windows`, Task 7 `test_reset_mid_utterance_gives_unknown`).
3. Diarization switched off while its model is still loading: it must stay off, with no `diarizer_on` event (Task 6 `test_off_while_loading_stays_off`).
4. Frames dropped on a full queue: the timeline must stay aligned with the sample clock (Task 3 `test_dropped_frames_keep_clock_aligned`).
5. A typed turn with diarization on: no wait, `diar_speaker` is `null`, and the record still says `diarization: true` (Task 7 `test_typed_turn_records_diarization_without_speaker`).

---

### Task 1: Dependency check on the VM

Decides in-process vs worker (spec §2.1) and records the model's streaming facts for Task 4.

**Files:**
- Modify: `server/pyproject.toml` (in-process result only)
- Modify: `scripts/setup_venv.sh` (in-process result only)
- Modify: `docs/superpowers/specs/2026-11-20-v4-diarization-design.md` §8

**Interfaces:**
- Produces: the `diarization` optional extra (exact transformers pin), and §8 facts: per mode `num_samples_first_audio_chunk`, `num_samples_per_audio_chunk`, `num_mel_frames_per_step`; logits value range (logits or probabilities); output frames per step; load time.

- [ ] **Step 1: Check the GPU.** Run `nvidia-smi`. If it fails, stop and ask the user to reboot the VM.

- [ ] **Step 2: Build a scratch venv that copies the server venv.** Do not touch `.venv` yet.

```bash
uv venv -p 3.12 .venvs/diarcheck
uv pip install --python .venvs/diarcheck -r RealtimeSTT/requirements-gpu-torch.txt
uv pip install --python .venvs/diarcheck -e "RealtimeSTT[faster-whisper]" -e "RealtimeTTS[kokoro]" -e "server[dev,rag,transformers]"
uv pip install --python .venvs/diarcheck -r RealtimeSTT/requirements-gpu-torch.txt
```

Also install every in-process engine extra that `.venv` has today: compare `uv pip freeze --python .venv` with the scratch venv and install the missing packages at the same versions.

- [ ] **Step 3: Find a transformers that has the model.** Try the newest release first, then a pinned git commit:

```bash
uv pip install --python .venvs/diarcheck -U transformers accelerate
uv pip install --python .venvs/diarcheck -r RealtimeSTT/requirements-gpu-torch.txt
.venvs/diarcheck/bin/python -c "import transformers, torch; from transformers.models import nemotron3_diarization; print(transformers.__version__, torch.__version__)"
```

If the import fails, use `"transformers @ git+https://github.com/huggingface/transformers@<sha>"` with the `main` commit of the day, and record the sha. Expected output: torch `2.7.1+cu128`.

- [ ] **Step 4: Run the checks.** `uv pip check --python .venvs/diarcheck` reports no conflict. From `server/`: `../.venvs/diarcheck/bin/python -m pytest -q` passes.

- [ ] **Step 5: Probe the model.** Run this with any 16 kHz mono WAV of about 10 s with two voices (one of the recorded clips under `bench/` is fine):

```python
import sys, time, numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, AutoModelForAudioFrameClassification
audio, sr = sf.read(sys.argv[1], dtype="float32"); assert sr == 16000 and audio.ndim == 1
t0 = time.monotonic()
proc = AutoProcessor.from_pretrained("nvidia/Nemotron-3-Diarization")
model = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization", device_map="cuda", dtype=torch.bfloat16)
print("load_s", round(time.monotonic() - t0, 2), "dtype", model.dtype)
for mode in ("ultra_low_latency", "very_low_latency", "low_latency"):
    proc.set_streaming_mode(mode)
    print(mode, proc.streaming_latency_ms, proc.num_samples_first_audio_chunk, proc.num_samples_per_audio_chunk,
          proc.num_mel_frames_per_step, proc.audio_chunk_start(proc.num_mel_frames_per_step))
    cache, outs, steps, mel = None, [], [], proc.num_mel_frames_per_step
    chunks = [proc(audio[:proc.num_samples_first_audio_chunk], sampling_rate=16000, is_streaming=True, is_first_audio_chunk=True)]
    start = proc.audio_chunk_start(mel)
    while start + proc.num_samples_per_audio_chunk <= len(audio):
        chunks.append(proc(audio[start:start + proc.num_samples_per_audio_chunk], sampling_rate=16000,
                           is_streaming=True, is_first_audio_chunk=False))
        mel += proc.num_mel_frames_per_step; start = proc.audio_chunk_start(mel)
    t1 = time.monotonic()
    with torch.inference_mode():
        for x in chunks:
            o = model(**x.to(model.device, dtype=model.dtype), speaker_cache=cache)
            cache = o.speaker_cache; outs.append(o.logits.float().cpu()); steps.append(o.logits.shape[1])
    y = torch.cat(outs, 1)
    print("  steps", len(steps), "frames/step", sorted(set(steps)), "shape", tuple(y.shape),
          "min", float(y.min()), "max", float(y.max()), "rtf", round((time.monotonic() - t1) / (len(audio) / 16000), 3))
    print("  ", proc.extract_speaker_dict(y)[0][:6])
```

If the forward fails because `dtype=model.dtype` casts a non-float input, move only the float tensors and record that.

- [ ] **Step 6: Decide.**
  - In process if steps 3–5 all pass. Apply the same upgrade to `.venv` (steps 3–4 against `.venv`), add `diarization = ["transformers==<version or git URL>", "accelerate"]` to `[project.optional-dependencies]` in `server/pyproject.toml`, and change `server[dev,rag]` to `server[dev,rag,diarization]` in `scripts/setup_venv.sh`.
  - Otherwise worker: keep `.venvs/diarcheck`, rename it `.venvs/diarizer` and run Task 9 after Task 8.
  - Ask the user before deleting `.venvs/diarcheck`.

- [ ] **Step 7: Record §8.** Replace the §8 placeholder with: transformers version (or sha), torch version, runtime, `uv pip check` result, load time, and a table of the three modes with the probe's numbers. State whether the output is logits (values outside 0–1) or probabilities. Set the spec status to "Approved; Task 1 done".

- [ ] **Step 8: Commit.**

```bash
git add docs/superpowers/specs/2026-11-20-v4-diarization-design.md server/pyproject.toml scripts/setup_venv.sh
git commit -m "build(diar): record the Nemotron diarization dependency check"
```

---

### Task 2: Timing, turn log and report

**Files:**
- Modify: `server/voicebot/timing.py`
- Modify: `server/voicebot/latency_report.py` (docstring only, if needed)
- Modify: `web/src/protocol.ts`, `web/src/components/StatusBar.tsx`
- Test: `server/tests/test_timing.py`, `server/tests/test_latency_report.py`

**Interfaces:**
- Produces: stages `"diar_start"`, `"diar_done"`; span `"diar_wait_ms"` in `SPANS` right after `"stt_ms"`; turn-log keys `diarization`, `diar_mode`, `diar_speaker`, `diar_lag_ms`, `diar_rejects`, `diar_rtf`; event lines for `diarizer_on`, `diarizer_off`, `diar_ignored`.

- [ ] **Step 1: Write the failing tests** in `server/tests/test_timing.py`:

```python
def test_diar_wait_span_and_llm_starts_after_the_decision():
    clock = iter([0.0, 0.1, 0.2, 0.5, 0.9, 1.4]).__next__
    t = TurnTimer(clock=clock)
    for s in ("speech_end", "endpoint", "stt_done", "diar_start", "diar_done", "llm_first_token"):
        t.mark(s)
    sp = t.report()["spans"]
    assert sp["diar_wait_ms"] == 400 and sp["llm_ttft_ms"] == 500   # LLM time starts at diar_done
    assert SPANS.index("diar_wait_ms") == SPANS.index("stt_ms") + 1

def test_format_turn_prints_diarization_fields_and_events():
    rec = {"turn_id": 3, "source": "voice", "spans": {}, "diarization": True, "diar_mode": "ultra_low_latency",
           "diar_speaker": 0, "diar_lag_ms": 120, "diar_rejects": 1, "diar_rtf": 0.08}
    text = format_turn(rec)
    for part in ("diarization=True", "diar_mode=ultra_low_latency", "diar_speaker=0", "diar_lag_ms=120",
                 "diar_rejects=1", "diar_rtf=0.08", "diar_wait"):
        assert part in text
    on = format_turn({"event": "diarizer_on", "ts": "2026-11-20T10:00:00.000+00:00", "diar_mode": "low_latency", "load_ms": 4200})
    assert on.startswith("=== diarizer_on | low_latency | load 4200 ms |") and "turn" not in on
    off = format_turn({"event": "diarizer_off", "ts": "x", "reason": "overflow"})
    assert "diarizer_off | overflow" in off
    ign = format_turn({"event": "diar_ignored", "ts": "x", "speaker": 1, "text": "hello there", "diar_mode": "low_latency"})
    assert 'speaker 1 | "hello there"' in ign
    ff = format_turn({"event": "barge_false_fire", "ts": "x", "turn_id": 2, "barge_in_mode": "onset", "barge_in_min_ms": 200})
    assert "barge_false_fire | turn 2 | onset 200 ms" in ff          # v3 line unchanged
```

In `server/tests/test_latency_report.py`:

```python
def test_diar_events_are_not_turns_and_diar_wait_is_reported(tmp_path, capsys):
    from voicebot.latency_report import summarize_barge
    turn = {**rec("llamacpp", 1000, 3000), "diarization": True, "diar_mode": "ultra_low_latency"}
    turn["spans"]["diar_wait_ms"] = 350
    events = [{"event": "diarizer_on", "diar_mode": "ultra_low_latency", "load_ms": 1},
              {"event": "diar_ignored", "speaker": 1, "text": "x"}, {"event": "diarizer_off", "reason": "user"}]
    rows = summarize([turn, *events], ("diarization", "diar_mode"))
    assert len(rows) == 1 and rows[0]["n"] == 1 and rows[0]["spans"]["diar_wait_ms"] == {"median": 350, "p90": 350}
    assert summarize_barge(events, ("diarization",)) == []
```

- [ ] **Step 2: Run them to see them fail.** `../.venv/bin/python -m pytest tests/test_timing.py tests/test_latency_report.py -q` → FAIL (`unknown stage 'diar_start'`, missing span).

- [ ] **Step 3: Implement in `timing.py`.**
  - Add `"diar_start"` and `"diar_done"` to `STAGES` after `"stt_done"` (comments: "speaker decision starts", "speaker decision made").
  - Add `"diar_wait_ms"` to `SPANS` after `"stt_ms"`, computed as `_ms(g("diar_start"), g("diar_done"))`.
  - LLM start: `retrieval_done` if marked, else `diar_done` if marked, else `inp`.
  - Append the six diarization keys to `_CONTEXT_KEYS`.
  - Event line: `head = [event]`, then `turn N` only when `turn_id` is present, then the existing barge part, then `diar_mode`, `load {load_ms} ms`, `reason`, `speaker {speaker}` and `"{text}"` when present.
  - Update `_LOG_SPANS` in `session.py` to include `"diar_wait_ms"` after `"stt_ms"`.

- [ ] **Step 4: Web.** In `protocol.ts` add `"diar_wait_ms"` to `SPANS` after `"stt_ms"`. In `StatusBar.tsx` add `["diar_wait_ms", "diar"]` to `BREAKDOWN` after `stt`.

- [ ] **Step 5: Run the tests.** Server tests above PASS. Full server suite passes. `npm run test && npm run typecheck` pass.

- [ ] **Step 6: Commit.**

```bash
git add server/voicebot/timing.py server/voicebot/session.py server/tests/test_timing.py server/tests/test_latency_report.py web/src/protocol.ts web/src/components/StatusBar.tsx
git commit -m "feat(timing): diar_wait_ms span, diarization turn fields and event lines"
```

---

### Task 3: The `Diarizer` (timeline, queue, thread)

The model is injected, so this task needs no torch or transformers.

**Files:**
- Create: `server/voicebot/diarizer.py`
- Test: `server/tests/test_diarizer.py`

**Interfaces:**
- Consumes: a model adapter with `load() -> None`, `ready: bool`, and `open(mode: str) -> stream`, where `stream.push(audio: np.ndarray) -> np.ndarray` takes 16 kHz mono float32 and returns the new frames' per-speaker probabilities, shape `(n, 8)`, `n` may be 0.
- Produces (module `voicebot.diarizer`, top-level imports stdlib + numpy only):
  - Constants `SAMPLE_RATE = 16000`, `FRAME_SAMPLES = 160`, `DIAR_THRESHOLD = 0.5`, `MODES = ("ultra_low_latency", "very_low_latency", "low_latency")`, `DROP_LIMIT = 50`, `QUEUE_MAX = 500`.
  - `class Diarizer(model, *, on_error: Callable[[str, str], None] | None = None, mode: str = "ultra_low_latency", queue_max: int = QUEUE_MAX)`.
  - `load() -> None`, `ready -> bool` (property), `mode -> str` (property), `set_mode(mode) -> None` (resets), `start() -> None`, `stop() -> None` (never joins the calling thread), `process(timeout: float = 0.0) -> bool` (handle one queued frame; the thread loops on it), `feed(pcm: bytes) -> None`, `position() -> int`, `covered() -> int`, `speaker_between(start: int, end: int) -> int | None`, `share(speaker: int, start: int, end: int) -> float | None`, `reset() -> None`, `rtf() -> float | None`.
  - `on_error(reason, message)` with reason `"error"` or `"overflow"`, called at most once per `start()`, never while holding the diarizer's lock.

Design decisions (not in the spec text):
- The timeline is a `bytearray`, one byte per 10 ms frame, bit `k` set when speaker `k` has probability ≥ `DIAR_THRESHOLD`. Frame `i` covers samples `[i*160, (i+1)*160)`. `covered() == len(timeline) * 160`.
- Each queued item carries `(generation, start_sample, pcm)`. `reset()` bumps the generation, clears timeline, clock and queue, and drops the open stream (the next `process` opens a fresh one with `model.open(mode)`). Frames of an old generation are discarded, and results computed during a reset are not appended.
- A dropped frame still advances `position()`. Before pushing a frame whose `start_sample` is past the samples already pushed, `process` pushes zeros for the gap, so timeline and clock stay aligned.
- `share` denominator: frames in the window where at least one speaker is active. It returns `None` when `covered() < end` or no frame in the window has an active speaker. Ruling: counting silent frames would let the pre-roll silence pull a real Speaker 0 onset below 0.5.
- `speaker_between`: the speaker active in the most frames of the covered part of the window; ties go to the lower index; `None` when `start >= end`, the window is empty or no speaker is active.
- `rtf()`: model compute seconds divided by audio seconds pushed since the last call; `None` with no audio.
- A `process` exception calls `on_error("error", str(e))` and ends the thread. `DROP_LIMIT` consecutive drops call `on_error("overflow", ...)` once; a successful enqueue resets the drop count.

- [ ] **Step 1: Write the failing tests** in `server/tests/test_diarizer.py`:

```python
import numpy as np, pytest, threading, time
from voicebot.diarizer import Diarizer, DIAR_THRESHOLD, FRAME_SAMPLES

def pcm(n_samples, value=1000):
    return (np.full(n_samples, value, dtype=np.int16)).tobytes()

class FakeStream:
    """One frame per 160 samples; script(frame_index) -> {speaker: prob}. Holds back `delay` samples like look-ahead."""
    def __init__(self, script, delay=0): self.script, self.delay, self.buf, self.done, self.pushed = script, delay, 0, 0, []
    def push(self, audio):
        self.pushed.append(audio.copy()); self.buf += len(audio)
        n = max(0, (self.buf - self.delay) // FRAME_SAMPLES - self.done)
        out = np.zeros((n, 8), dtype=np.float32)
        for j in range(n):
            for spk, p in self.script(self.done + j).items(): out[j, spk] = p
        self.done += n; return out

class FakeModel:
    def __init__(self, script=lambda i: {0: 0.9}, delay=0, fail=None):
        self.script, self.delay, self.fail, self.ready, self.opened = script, delay, fail, False, []
    def load(self): self.ready = True
    def open(self, mode):
        self.opened.append(mode); s = FakeStream(self.script, self.delay)
        if self.fail:
            def boom(a): raise RuntimeError(self.fail)
            s.push = boom
        self.stream = s; return s

def drain(d):
    while d.process(0): pass

def test_timeline_maps_frames_to_samples():
    d = Diarizer(FakeModel(lambda i: {0: 0.9} if i < 50 else {1: 0.8}))
    d.feed(pcm(16000)); drain(d)
    assert d.position() == 16000 and d.covered() == 16000
    assert d.speaker_between(0, 8000) == 0 and d.speaker_between(8000, 16000) == 1
    assert d.share(0, 0, 16000) == pytest.approx(0.5) and d.share(1, 8000, 16000) == 1.0

def test_threshold_and_silence():
    d = Diarizer(FakeModel(lambda i: {0: DIAR_THRESHOLD - 0.01} if i < 10 else {}))
    d.feed(pcm(3200)); drain(d)
    assert d.speaker_between(0, 3200) is None and d.share(0, 0, 3200) is None

def test_share_is_none_until_covered():
    d = Diarizer(FakeModel(delay=1600))
    d.feed(pcm(3200)); drain(d)
    assert d.covered() == 1600 and d.share(0, 0, 3200) is None and d.share(0, 0, 1600) == 1.0

def test_queries_on_bad_windows():                                   # Review Focus 2
    d = Diarizer(FakeModel()); d.feed(pcm(1600)); drain(d)
    assert d.speaker_between(1600, 0) is None and d.speaker_between(-500, -1) is None
    assert d.speaker_between(50_000, 60_000) is None

def test_reset_clears_timeline_clock_queue_and_stream():
    m = FakeModel(); d = Diarizer(m); d.feed(pcm(1600)); drain(d); first = m.stream
    d.feed(pcm(1600)); d.reset()
    assert d.position() == 0 and d.covered() == 0 and not d.process(0)   # the queued frame is gone
    d.feed(pcm(320)); drain(d)
    assert m.stream is not first and d.covered() == 320

def test_set_mode_resets_and_opens_in_the_new_mode():
    m = FakeModel(); d = Diarizer(m); d.feed(pcm(320)); drain(d)
    d.set_mode("low_latency"); assert d.mode == "low_latency" and d.position() == 0
    d.feed(pcm(320)); drain(d); assert m.opened[-1] == "low_latency"
    with pytest.raises(ValueError): d.set_mode("fast")

def test_dropped_frames_keep_clock_aligned():                        # Review Focus 4
    m = FakeModel(); d = Diarizer(m, queue_max=1)
    d.feed(pcm(160)); d.feed(pcm(160)); d.feed(pcm(160))             # 2nd and 3rd dropped
    drain(d); d.feed(pcm(160)); drain(d)
    assert d.position() == 640 and d.covered() == 640                # gap filled with zeros
    assert sum(len(a) for a in m.stream.pushed) == 640 and not m.stream.pushed[1].any()

def test_overflow_reports_once():
    got = []; d = Diarizer(FakeModel(), on_error=lambda r, m: got.append(r), queue_max=1)
    for _ in range(1 + 50 + 10): d.feed(pcm(160))
    assert got == ["overflow"]

def test_model_error_reports_and_stops_the_thread():
    got = threading.Event(); reasons = []
    d = Diarizer(FakeModel(fail="cuda oom"), on_error=lambda r, m: (reasons.append((r, m)), got.set()))
    d.start(); d.feed(pcm(320)); assert got.wait(2)
    assert reasons == [("error", "cuda oom")]
    d.stop()

def test_thread_processes_fed_audio_and_stop_is_safe_from_on_error():
    d = Diarizer(FakeModel()); d.start(); d.feed(pcm(1600))
    end = time.monotonic() + 2
    while d.covered() < 1600 and time.monotonic() < end: time.sleep(0.01)
    assert d.covered() == 1600; d.stop(); d.stop()

def test_rtf_since_last_call():
    d = Diarizer(FakeModel()); assert d.rtf() is None
    d.feed(pcm(16000)); drain(d); r = d.rtf()
    assert r is not None and r >= 0 and d.rtf() is None

def test_module_imports_no_torch():
    import subprocess, sys
    code = "import voicebot.diarizer, sys; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
```

- [ ] **Step 2: Run them to see them fail.** `../.venv/bin/python -m pytest tests/test_diarizer.py -q` → FAIL (`No module named 'voicebot.diarizer'`).

- [ ] **Step 3: Implement `voicebot/diarizer.py`** with the interface and design decisions above. PCM to float: reuse the formula of `aligner.pcm16_to_float` inline (int16 / 32768), do not import `voicebot.aligner`. Queue: `queue.Queue(maxsize=queue_max)` with `put_nowait`. Thread name `"diarizer"`, daemon.

- [ ] **Step 4: Run the tests.** All of `tests/test_diarizer.py` PASS; the full server suite passes.

- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/diarizer.py server/tests/test_diarizer.py
git commit -m "feat(diar): streaming Diarizer with a 10 ms speaker timeline"
```

---

### Task 4: `NemotronModel` adapter and VM smoke

**Files:**
- Modify: `server/voicebot/diarizer.py`
- Test: `server/tests/test_diarizer.py`

**Interfaces:**
- Consumes: Task 1 §8 facts (chunking calls, logits or probabilities); Task 3's adapter contract.
- Produces: `class NemotronModel(device: str = "cuda", model_id: str = "nvidia/Nemotron-3-Diarization", *, processor=None, model=None)` with `load()`, `ready`, `open(mode) -> NemotronStream`; `OUTPUT_IS_LOGITS: bool` module constant (from §8); CLI `python -m voicebot.diarizer --smoke <wav>`.

Adapter rules:
- `load()` imports torch and transformers inside the method, loads the processor and model in bf16 on `device` (CPU with a warning when CUDA is unavailable), and `model.eval()`. Injected `processor`/`model` skip loading (tests).
- `NemotronStream` keeps the audio received since `open` in a growing buffer with an absolute offset, a `first` flag, `mel` (the processor's mel frame index) and the `speaker_cache`. Each processor instance holds one streaming mode, so `open(mode)` calls `processor.set_streaming_mode(mode)`; the session only ever has one stream open.
- `push(audio)`: append; then while enough samples exist, build the next chunk exactly as Task 1's probe does (first chunk `[:num_samples_first_audio_chunk]` with `is_first_audio_chunk=True`; then `[start : start + num_samples_per_audio_chunk]` with `start = processor.audio_chunk_start(mel)`, `mel += num_mel_frames_per_step` after each), run the model under `torch.inference_mode()` with `speaker_cache`, collect `outputs.logits[0]` as float32 numpy; apply sigmoid when `OUTPUT_IS_LOGITS`. Trim buffer samples before the next chunk start. Return all new frames concatenated, shape `(n, 8)`.
- Never send `is_last_audio_chunk`: the session's stream has no end.

- [ ] **Step 1: Write the failing test** with fake processor and model objects that record the slices they get:

```python
class FakeProc:
    num_samples_first_audio_chunk, num_samples_per_audio_chunk, num_mel_frames_per_step = 800, 640, 4
    def __init__(self): self.calls, self.mode = [], None
    def set_streaming_mode(self, m): self.mode = m
    def audio_chunk_start(self, mel): return mel * 160 - 160          # overlapping windows, like look-back
    def __call__(self, audio, sampling_rate, is_streaming, is_first_audio_chunk):
        self.calls.append((len(audio), is_first_audio_chunk)); return FakeInputs()

class FakeInputs:
    def to(self, *a, **k): return self
    def keys(self): return []
    def __getitem__(self, k): raise KeyError(k)

class FakeNet:
    dtype, device = "bf16", "cpu"
    def __init__(self): self.caches = []
    def __call__(self, speaker_cache=None, **kw):
        import torch
        self.caches.append(speaker_cache)
        return type("O", (), {"logits": torch.zeros(1, 4, 8), "speaker_cache": len(self.caches)})()

def test_nemotron_stream_chunks_like_the_processor():
    pytest.importorskip("torch")
    from voicebot.diarizer import NemotronModel
    p, net = FakeProc(), FakeNet()
    m = NemotronModel(processor=p, model=net); m.load(); s = m.open("very_low_latency")
    assert p.mode == "very_low_latency"
    assert s.push(np.zeros(700, np.float32)).shape == (0, 8)               # not enough for the first chunk
    out = s.push(np.zeros(1000, np.float32))                               # 1700 samples: first + one more chunk
    assert p.calls == [(800, True), (640, False)] and out.shape == (8, 8)
    assert net.caches == [None, 1]                                         # the cache links the chunks
```

`importorskip("torch")` keeps the test running in the server venv (torch is installed there) and skipping elsewhere. Adjust `FakeNet.__call__` to the forward signature §8 recorded if it differs.

- [ ] **Step 2: Run it to see it fail.** → FAIL (`cannot import name 'NemotronModel'`).

- [ ] **Step 3: Implement `NemotronModel`, `NemotronStream`, `OUTPUT_IS_LOGITS`** and the `--smoke` CLI: load, open `ultra_low_latency`, push the WAV in 20 ms pieces through a `Diarizer` (no thread, `process` loop), then print load time, RTF, `covered()` and speaker per second of audio.

- [ ] **Step 4: Run the tests.** PASS; full suite passes; `test_module_imports_no_torch` still passes.

- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/diarizer.py server/tests/test_diarizer.py
git commit -m "feat(diar): Nemotron streaming adapter and smoke CLI"
```

- [ ] **Step 6 (VM):** `cd server && ../.venv/bin/python -m voicebot.diarizer --smoke <the Task 1 clip>`. Expected: the speakers match the clip, `covered()` within one chunk of the clip length, RTF below 0.2. Fix and recommit if not.

---

### Task 5: Audio tap

**Files:**
- Modify: `server/voicebot/audio_io.py`
- Test: `server/tests/test_audio_io.py`

**Interfaces:**
- Produces: `BrowserAudio.diarizer` attribute (default `None`); when set and input is not muted, `on_client_audio` calls `diarizer.feed(data)` after `recorder.feed_audio`. `ServerAudio` gets no tap.

- [ ] **Step 1: Write the failing test:**

```python
def test_browser_audio_taps_the_diarizer_only_when_attached_and_unmuted():
    class D:
        def __init__(self): self.fed = []
        def feed(self, b): self.fed.append(b)
    r, d = Rec(), D()
    a = BrowserAudio(r, {"sample_rate": 24000})
    a.on_client_audio(b"\x01\x02"); assert a.diarizer is None
    a.diarizer = d; a.on_client_audio(b"\x03\x04")
    a.mute_input(True); a.on_client_audio(b"\x05\x06")
    assert d.fed == [b"\x03\x04"] and len(r.fed) == 2
```

- [ ] **Step 2: Run it.** → FAIL (`'BrowserAudio' object has no attribute 'diarizer'`).
- [ ] **Step 3: Implement** (read `self.diarizer` once into a local).
- [ ] **Step 4: Run the tests.** PASS.
- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/audio_io.py server/tests/test_audio_io.py
git commit -m "feat(diar): tap browser mic frames into the diarizer"
```

---

### Task 6: Session settings, lifecycle, events and wiring

**Files:**
- Modify: `server/voicebot/session.py`, `server/voicebot/web.py`
- Test: `server/tests/test_session.py`, `server/tests/test_web.py`

**Interfaces:**
- Consumes: Task 3 `Diarizer` API; Task 5 `audio.diarizer`.
- Produces on `VoiceSession`:
  - Constructor kwarg `diarizer_factory: Callable[[Callable[[str, str], None]], Diarizer] | None = None` (called once, lazily, with the session's `_on_diar_error`).
  - Attributes `diarization: bool` (False), `diar_mode: str` ("ultra_low_latency"), `diar_wait_ms: int` (600).
  - `diarization_available() -> bool`: factory set and `audio.output_format is not None`.
  - `_active_diarizer()`: the diarizer when `diarization` is on and attached to the audio, else `None`.
  - `_set_diarization(on: bool)`, `_on_diar_error(reason: str, message: str)`, `_write_event(record: dict)` (adds `ts`, `**info` and model slugs, then calls `metrics_sink`; `_log_false_fire` is rewritten on top of it).
  - `config()["defaults"]` gains `diarization`, `diar_mode`, `diar_wait_ms`; `config()` gains `diarization_available`.
  - Events: `{"type": "diarization", "state": "loading" | "ready" | "off" | "error", "message"?}`.
- Produces in `web.py`: `_build_diarizer_factory(s) -> Callable | None` returning `None` unless `s.audio_io == "browser"`; the factory imports `voicebot.diarizer` inside its body and returns `Diarizer(NemotronModel(device="cuda"), on_error=on_error)`.

Behaviour:
- Validation (all `bad_settings`): `diarization` must be bool; `diar_mode` in `MODES` (copy the tuple into the session; do not import `voicebot.diarizer`); `diar_wait_ms` int (not bool) 0–2000.
- On: not available → `diarization` `error` "Speaker diarization is not available on this server (browser audio only)." Build via the factory on first use (factory exception → `error`). If not `ready`: emit `loading`, load on a `_spawn`ed thread timing `load_ms`; a load failure emits `error`, logs, writes `diarizer_off` with `reason: "error"`. When loaded and still wanted (or already ready): `set_mode(self.diar_mode)`, `start()`, `audio.diarizer = d`, `diarization = True`, emit `ready`, `log.info("diarizer on: mode=%s load_ms=%d", ...)`, write `{"event": "diarizer_on", "diar_mode", "load_ms"}`.
- Off: detach (`audio.diarizer = None`), `stop()`, `reset()`, `diarization = False`, emit `off`, `log.info("diarizer off: user")`, write `diarizer_off` `reason: "user"` (only if it was on).
- `_on_diar_error`: ignored unless on; detach, `diarization = False`, emit `error` with the message, `log.warning("diarizer off: %s (%s)", reason, message)`, write `diarizer_off` with the reason. Do not call `stop()` from inside it (it runs on the diarizer thread, which is already ending).
- `diar_mode` change: store; if active, `set_mode(mode)` (which resets).
- Reset: `_clear` and `_run_switch` (after any switch, success or failure) call `reset()` on the active diarizer. `close()` stops it.

- [ ] **Step 1: Write the failing tests.** Add a fake and fixture to `test_session.py`:

```python
class FakeDiarizer:
    """Scripted timeline. share() pops from shares (None: not covered yet) and then repeats the last value."""
    def __init__(self, ready=True, fail_load=None):
        self.ready, self.fail_load, self.mode = ready, fail_load, "ultra_low_latency"
        self.pos = self.cov = 0; self.speaker = 0; self.shares = [1.0]; self.windows = []
        self.loads = self.resets = 0; self.started = False; self.on_error = None; self.load_gate = None
    def load(self):
        self.loads += 1
        if self.load_gate: self.load_gate.wait(2)
        if self.fail_load: raise RuntimeError(self.fail_load)
        self.ready = True
    def set_mode(self, m): self.mode = m; self.reset()
    def start(self): self.started = True
    def stop(self): self.started = False
    def feed(self, pcm): pass
    def position(self): return self.pos
    def covered(self): return self.cov
    def speaker_between(self, s, e): self.windows.append((s, e)); return self.speaker
    def share(self, spk, s, e):
        self.windows.append((s, e)); return self.shares.pop(0) if len(self.shares) > 1 else self.shares[0]
    def reset(self): self.resets += 1
    def rtf(self): return 0.1

@pytest.fixture
def diar(): return FakeDiarizer()

def diar_session(make, fd, **extra):
    def factory(on_error): fd.on_error = on_error; return fd
    return make(diarizer_factory=factory, **extra)

def diar_events(events): return [e for e in events if e["type"] == "diarization"]

def turn_on(s, events):
    s.handle({"type": "update_settings", "diarization": True})
    wait_for(lambda: diar_events(events) and diar_events(events)[-1]["state"] == "ready")
```

Tests:

```python
def test_diar_settings_validated_and_reported(make, diar, events):
    s = diar_session(make, diar)
    for bad in ({"diarization": "yes"}, {"diar_mode": "fast"}, {"diar_wait_ms": 2001}, {"diar_wait_ms": True}):
        s.handle({"type": "update_settings", **bad})
    assert [e["code"] for e in events if e["type"] == "error"] == ["bad_settings"] * 4
    s.handle({"type": "update_settings", "diar_mode": "low_latency", "diar_wait_ms": 0})
    c = s.config(); d = c["defaults"]
    assert d["diarization"] is False and d["diar_mode"] == "low_latency" and d["diar_wait_ms"] == 0
    assert c["diarization_available"] is True

def test_diar_defaults_and_unavailable_without_factory(session, events):
    c = session.config()
    assert c["defaults"]["diarization"] is False and c["defaults"]["diar_mode"] == "ultra_low_latency"
    assert c["defaults"]["diar_wait_ms"] == 600 and c["diarization_available"] is False
    session.handle({"type": "update_settings", "diarization": True})
    assert diar_events(events)[-1]["state"] == "error" and session.diarization is False

def test_diar_on_off_attaches_logs_and_writes_events(make, diar, events, audio, sink, caplog):
    diar.ready = False; s = diar_session(make, diar)
    with caplog.at_level(logging.INFO, logger="voicebot.session"):
        turn_on(s, events)
        assert [e["state"] for e in diar_events(events)] == ["loading", "ready"]
        assert audio.diarizer is diar and diar.started and s.config()["defaults"]["diarization"] is True
        s.handle({"type": "update_settings", "diarization": False})
    assert audio.diarizer is None and not diar.started and diar_events(events)[-1]["state"] == "off"
    on, off = [r for r in sink if "event" in r]
    assert on["event"] == "diarizer_on" and on["diar_mode"] == "ultra_low_latency" and isinstance(on["load_ms"], int)
    assert off["event"] == "diarizer_off" and off["reason"] == "user" and "ts" in off
    assert "diarizer on: mode=ultra_low_latency" in caplog.text and "diarizer off: user" in caplog.text

def test_off_while_loading_stays_off(make, diar, events, audio, sink):          # Review Focus 3
    diar.ready = False; diar.load_gate = threading.Event(); s = diar_session(make, diar)
    s.handle({"type": "update_settings", "diarization": True})
    s.handle({"type": "update_settings", "diarization": False}); diar.load_gate.set()
    time.sleep(0.1)
    assert getattr(audio, "diarizer", None) is None and s.diarization is False
    assert not any(r.get("event") == "diarizer_on" for r in sink)

def test_load_failure_reports_error(make, events, sink):
    fd = FakeDiarizer(ready=False, fail_load="no cuda"); s = diar_session(make, fd)
    s.handle({"type": "update_settings", "diarization": True})
    wait_for(lambda: diar_events(events)[-1]["state"] == "error")
    assert "no cuda" in diar_events(events)[-1]["message"] and s.diarization is False
    assert sink[-1]["event"] == "diarizer_off" and sink[-1]["reason"] == "error"

def test_diarizer_error_switches_off(make, diar, events, audio, sink):
    s = diar_session(make, diar); turn_on(s, events)
    diar.on_error("overflow", "50 frames dropped")
    assert s.diarization is False and audio.diarizer is None
    assert diar_events(events)[-1] == {"type": "diarization", "state": "error", "message": "50 frames dropped"}
    assert sink[-1]["event"] == "diarizer_off" and sink[-1]["reason"] == "overflow"

def test_clear_switch_and_mode_change_reset(make, diar, events):
    s = diar_session(make, diar); turn_on(s, events); base = diar.resets
    s.handle({"type": "clear"}); assert diar.resets == base + 1
    s.handle({"type": "update_settings", "diar_mode": "low_latency"})
    assert diar.mode == "low_latency" and diar.resets == base + 2

def test_model_switch_resets_the_diarizer(session_with_models, diar, events):
    s = session_with_models
    def factory(on_error): diar.on_error = on_error; return diar
    s.diarizer_factory = factory
    turn_on(s, events); base = diar.resets
    s.handle({"type": "select_model", "role": "tts", "slug": "kokoro"})
    wait_for(lambda: not s._switching.locked() and diar.resets > base)
```

If `session_with_models` builds the session with a fixed constructor, give it a `diarizer_factory` the same way the `make` fixture does instead of setting the attribute.

In `test_web.py`:

```python
def test_server_does_not_import_diarization_when_off():
    import subprocess, sys
    code = ("import sys, voicebot.web, voicebot.session, voicebot.audio_io; "
            "bad = {'voicebot.diarizer', 'transformers'} & set(sys.modules); assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)

def test_diarizer_factory_only_in_browser_mode():
    from voicebot.web import _build_diarizer_factory
    from voicebot.config import Settings
    assert _build_diarizer_factory(Settings.load({"AUDIO_IO": "server"}, platform="win32")) is None
    assert callable(_build_diarizer_factory(Settings.load({}, platform="linux")))
```

- [ ] **Step 2: Run them to see them fail.** → FAIL (unexpected keyword `diarizer_factory`).
- [ ] **Step 3: Implement** the behaviour above in `session.py`, and in `web.py` pass `diarizer_factory=_build_diarizer_factory(s)` to `VoiceSession`.
- [ ] **Step 4: Run the tests.** New tests PASS; the full server suite passes unchanged.
- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/session.py server/voicebot/web.py server/tests/test_session.py server/tests/test_web.py
git commit -m "feat(diar): diarization settings, load/unload lifecycle and on/off events"
```

---

### Task 7: Turn check

**Files:**
- Modify: `server/voicebot/session.py`
- Test: `server/tests/test_session.py`

**Interfaces:**
- Consumes: Task 6 `_active_diarizer()`, `_write_event`, `diar_wait_ms`; Task 2 stages.
- Produces: `_utt_start` / `_utt_end` (sample indices, `None` when unknown, guarded by `_state_lock`); `_pre_samples()` = `int(getattr(audio.recorder, "pre_recording_buffer_duration", 0.2) * 16000)`; `_decide_speaker() -> dict | None`; `_run_turn(text, *, source, diar_fields: dict | None = None)`; `user_transcript` gains `speaker` (only when diarization decided it).

Behaviour:
- `on_recording_start`: before its early returns, if a diarizer is active set `_utt_start = max(0, d.position() - _pre_samples())` and `_utt_end = None`. `on_recording_stop`: `_utt_end = d.position()`. Read `d.position()` while holding `_state_lock` is fine (the diarizer never takes the session's locks).
- Listen loop, after the noise check, inside `_turn_lock` after `stt_done`: `diar = self._decide_speaker()`. If `diar` and `diar["diar_speaker"]` is neither `0` nor `"unknown"`: emit `{"type": "user_transcript", "text": text, "turn_id": None, "speaker": n, "ignored": True}`, `log.info("ignored speaker %d: %r", ...)`, `_write_event({"event": "diar_ignored", "speaker", "text", "diar_wait_ms", "diar_mode"})` (wait from the timer's marks), `timer.reset()`, `continue`. Otherwise `_run_turn(..., diar_fields=diar)`.
- `_decide_speaker()`: `None` when no diarizer is active. Else mark `diar_start`; `lag = max(0, end - d.covered())`; poll every 20 ms until `covered() >= end`, `diar_wait_ms` passes, or `_active_diarizer()` becomes `None`; speaker = `speaker_between(start, end)` when covered, `"unknown"` otherwise or when it returns `None` or when `start`/`end` is `None` or `start >= end`; mark `diar_done`. Return `{"diarization": True, "diar_mode": d.mode, "diar_speaker": speaker, "diar_lag_ms": round(lag / 16)}`.
- `_run_turn`: the `user_transcript` event carries `"speaker": diar_fields["diar_speaker"]` when `diar_fields` is set. The record gets `diarization` (current state), `diar_mode` (or `None` when off), `diar_speaker` (`None` for typed turns and when off), `diar_lag_ms` (`None` likewise), `diar_rejects` (from Task 8; `0` until then), `diar_rtf` (`d.rtf()` when active, else `None`).

- [ ] **Step 1: Write the failing tests:**

```python
def voice_turn(s, recorder, events, text="Hello there"):
    s.on_recording_start(); s.on_recording_stop(); recorder.texts.append(text)
    return wait_for(lambda: next((e for e in events if e["type"] == "user_transcript"), None))

def listening_diar_session(make, diar, events):
    s = diar_session(make, diar); turn_on(s, events)
    s.handle({"type": "start_listening"}); wait_for(lambda: s.audio.recorder.waiting)
    return s

def test_speaker_zero_is_answered_and_tagged(make, diar, events, recorder, sink, conv):
    diar.pos, diar.cov = 16000, 16000
    s = listening_diar_session(make, diar, events); ev = voice_turn(s, recorder, events)
    assert ev["speaker"] == 0 and ev["turn_id"] == 1 and "ignored" not in ev
    r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert r["diarization"] is True and r["diar_speaker"] == 0 and r["diar_mode"] == "ultra_low_latency"
    assert isinstance(r["spans"]["diar_wait_ms"], int) and r["diar_rtf"] == 0.1 and r["diar_lag_ms"] == 0
    assert diar.windows[0] == (16000 - 3200, 16000)                    # 0.2 s pre-roll at 16 kHz

def test_other_speaker_is_ignored_and_logged(make, diar, events, recorder, sink, conv, backend):
    diar.pos = diar.cov = 16000; diar.speaker = 1
    s = listening_diar_session(make, diar, events); ev = voice_turn(s, recorder, events, "Not you")
    assert ev == {"type": "user_transcript", "text": "Not you", "turn_id": None, "speaker": 1, "ignored": True}
    rec = wait_for(lambda: next((x for x in sink if x.get("event") == "diar_ignored"), None))
    assert rec["speaker"] == 1 and rec["text"] == "Not you" and isinstance(rec["diar_wait_ms"], int)
    time.sleep(0.05); assert backend.calls == [] and conv.messages("x")[1:-1] == []

def test_unknown_passes(make, diar, events, recorder, backend):
    diar.pos = diar.cov = 16000; diar.speaker = None
    s = listening_diar_session(make, diar, events); assert voice_turn(s, recorder, events)["speaker"] == "unknown"
    wait_for(lambda: backend.calls)

def test_timeout_passes_as_unknown(make, diar, events, recorder, sink):
    diar.pos, diar.cov = 16000, 0                                     # the timeline never catches up
    s = listening_diar_session(make, diar, events)
    s.handle({"type": "update_settings", "diar_wait_ms": 100})
    assert voice_turn(s, recorder, events)["speaker"] == "unknown"
    r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert 90 <= r["spans"]["diar_wait_ms"] <= 400 and r["diar_lag_ms"] == 1000

def test_diarizer_failure_during_wait_passes_at_once(make, diar, events, recorder):   # Review Focus 1
    diar.pos, diar.cov = 16000, 0
    s = listening_diar_session(make, diar, events); s.handle({"type": "update_settings", "diar_wait_ms": 2000})
    s.on_recording_start(); s.on_recording_stop(); recorder.texts.append("Hello there")
    time.sleep(0.1); t0 = time.monotonic(); diar.on_error("error", "boom")
    ev = wait_for(lambda: next((e for e in events if e["type"] == "user_transcript"), None))
    assert ev["speaker"] == "unknown" and time.monotonic() - t0 < 0.5

def test_reset_mid_utterance_gives_unknown(make, diar, events, recorder):            # Review Focus 2
    diar.pos = diar.cov = 16000
    s = listening_diar_session(make, diar, events); s.on_recording_start()
    diar.pos = diar.cov = 800; s.handle({"type": "clear"})                # clock restarted below the start
    s.on_recording_stop(); recorder.texts.append("Hello there")
    assert wait_for(lambda: next((e for e in events if e["type"] == "user_transcript"), None))["speaker"] == "unknown"

def test_typed_turn_records_diarization_without_speaker(make, diar, events, sink):     # Review Focus 5
    s = diar_session(make, diar); turn_on(s, events)
    s.handle({"type": "send_text", "text": "Hi"})
    r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert r["diarization"] is True and r["diar_speaker"] is None and r["spans"]["diar_wait_ms"] is None
    assert "speaker" not in next(e for e in events if e["type"] == "user_transcript")

def test_diarization_off_records_and_events_like_v3(session, events, sink):
    session.handle({"type": "send_text", "text": "Hi"}); r = wait_for(lambda: sink and sink[-1])
    assert r["diarization"] is False and r["diar_mode"] is None and r["diar_speaker"] is None
    assert r["diar_rtf"] is None and r["diar_rejects"] == 0
    assert "speaker" not in next(e for e in events if e["type"] == "user_transcript")
```

The `FakeRecorder` calls `on_stop` (timing marks) itself; the tests call `on_recording_start/stop` on the session directly, standing in for the session callbacks.

- [ ] **Step 2: Run them to see them fail.** → FAIL (no `speaker` in `user_transcript`).
- [ ] **Step 3: Implement** the behaviour above.
- [ ] **Step 4: Run the tests.** New tests PASS; all v3 session tests pass unchanged.
- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/session.py server/tests/test_session.py
git commit -m "feat(diar): answer only Speaker 0; log ignored speakers"
```

---

### Task 8: Barge-in check

**Files:**
- Modify: `server/voicebot/session.py`
- Test: `server/tests/test_session.py`

**Interfaces:**
- Consumes: Task 7 `_utt_start`, `_pre_samples()`; v3 `_fire_barge(turn_id)`.
- Produces: `_confirm_barge(turn_id)` replacing every direct `_fire_barge` call in the onset timer's `due()`, the `min_ms <= 0` paths in `on_recording_start` / `on_speech_resume` / `_run_turn`, and `on_partial`. State under `_state_lock`: `_barge_onset_sample: int | None`, `_barge_checking: bool`, `_barge_words_rejected: bool`, `_diar_rejects: int` (reset when a turn arms; read into the record's `diar_rejects`).

Behaviour:
- No active diarizer: `_confirm_barge` calls `_fire_barge` directly (v3 path, same thread).
- Wherever `barge_onset` is marked, set `_barge_onset_sample`: `_utt_start` in `on_recording_start` and in the mid-utterance arm in `_run_turn`, `d.position()` in `on_speech_resume`.
- Active diarizer: if `_barge_checking`, return; set it and `_spawn` a check thread (never block RealtimeSTT's callback thread). The check fixes `end = d.position()` once, then loops every 40 ms up to `diar_wait_ms`:
  - turn disarmed, already fired, or diarizer gone → fire if gone, else stop;
  - `share(0, onset, end)` ≥ 0.5 → `_fire_barge`;
  - below 0.5 → reject: `_diar_rejects += 1`, `timer.unmark("barge_onset")`; in words mode set `_barge_words_rejected`; stop;
  - `None` → keep waiting; at the deadline fire (fail open).
  - Always clear `_barge_checking` at the end.
- `on_speech_resume` does not arm while `_barge_checking`. `on_recording_start` clears `_barge_words_rejected`. `on_partial` does not trigger while it is set.

- [ ] **Step 1: Write the failing tests** (reuse `armed()` and `FakeDiarizer`; the diarizer must be on before `armed()` starts the turn):

```python
def armed_diar(make, diar, events, mode, min_ms=0, wait_ms=600):
    def factory(on_error): diar.on_error = on_error; return diar
    extra = {"diarizer_factory": factory}
    s = make(models=FakeModels(), **extra) if mode == "words" else make(**extra)
    turn_on(s, events); s.handle({"type": "update_settings", "diar_wait_ms": wait_ms})
    s.handle({"type": "update_settings", "barge_in_mode": mode, "barge_in_min_ms": min_ms})
    s.handle({"type": "start_listening"}); wait_for(lambda: s.audio.recorder.waiting)
    s.backend.gate = threading.Event()
    s.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "assistant_delta" in types(events))
    return s

def test_barge_speaker_zero_fires(make, diar, events, sink):
    diar.pos = 32000; diar.shares = [0.9]
    s = armed_diar(make, diar, events, "onset"); s.on_recording_start()
    wait_for(lambda: "barge_in" in types(events))
    r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert r["diar_rejects"] == 0 and r["interrupted"] is True

def test_barge_other_speaker_rejected_and_counted(make, diar, events, sink, backend):
    diar.pos = 32000; diar.shares = [0.2]
    s = armed_diar(make, diar, events, "onset"); s.on_recording_start(); time.sleep(0.2)
    assert "barge_in" not in types(events)
    backend.gate.set(); r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert r["diar_rejects"] == 1 and r["interrupted"] is False

def test_barge_waits_for_a_lagging_diarizer(make, diar, events, sink):
    diar.pos = 32000; diar.shares = [None, None, None, 0.8]
    s = armed_diar(make, diar, events, "onset"); s.on_recording_start()
    wait_for(lambda: "barge_in" in types(events))
    r = wait_for(lambda: next((x for x in sink if "event" not in x), None))
    assert r["spans"]["barge_detect_ms"] >= 100                           # three 40 ms polls before the decision

def test_barge_fails_open_when_the_diarizer_never_catches_up(make, diar, events):
    diar.pos = 32000; diar.shares = [None]
    s = armed_diar(make, diar, events, "onset", wait_ms=150); s.on_recording_start()
    wait_for(lambda: "barge_in" in types(events), timeout=1.0)

def test_barge_rejected_onset_rearms_on_resume(make, diar, events):
    diar.pos = 32000; diar.shares = [0.1, 0.9]
    s = armed_diar(make, diar, events, "onset"); s.on_recording_start(); time.sleep(0.15)
    assert "barge_in" not in types(events)
    s.on_speech_pause(); s.on_speech_resume(); wait_for(lambda: "barge_in" in types(events))

def test_words_mode_reject_holds_until_next_recording(make, diar, events):
    diar.pos = 32000; diar.shares = [0.1, 0.9]
    s = armed_diar(make, diar, events, "words"); s.on_recording_start(); s.on_partial("hello there")
    time.sleep(0.15); s.on_partial("hello there again"); time.sleep(0.1)
    assert "barge_in" not in types(events)
    s.on_recording_stop(); s.on_recording_start(); s.on_partial("stop please")
    wait_for(lambda: "barge_in" in types(events))

def test_barge_without_diarization_is_v3(make, events):
    s = armed(make, "onset", 0); s.on_recording_start()
    assert "barge_in" in types(events)                                   # same thread, no check thread
```

- [ ] **Step 2: Run them to see them fail.** → FAIL (Speaker 1 still fires).
- [ ] **Step 3: Implement** the behaviour above.
- [ ] **Step 4: Run the tests.** New tests PASS; every v3 barge-in test passes unchanged.
- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/session.py server/tests/test_session.py
git commit -m "feat(diar): only Speaker 0 can barge in; count rejects"
```

---

### Task 9 (only if Task 1 failed): Diarizer worker process

**Files:**
- Create: `server/voicebot/diar_worker.py`, `scripts/setup_diarizer_venv.sh`
- Modify: `server/voicebot/diarizer.py` (add `RemoteNemotronModel`), `server/voicebot/web.py` (factory)
- Test: `server/tests/test_diarizer.py`

**Interfaces:**
- Produces: `RemoteNemotronModel(python: Path, *, port: int = 0)` with the same `load()` / `ready` / `open(mode)` contract as `NemotronModel`. `load()` starts `<python> -m voicebot.diar_worker --port <p>` (the worker venv has the server installed with `--no-deps -e server` plus its own transformers), waits for its `ready` line on stdout, and connects over TCP on `127.0.0.1`. `open(mode)` sends an `open` message; `push(audio)` sends float32 bytes and receives `(n, 8)` float32 bytes.
- Wire format: each message is a 4-byte big-endian length, a 1-byte kind (`o` open with UTF-8 mode, `a` audio, `f` frames), then the payload.
- `web.py`: when `REPO_ROOT / ".venvs/diarizer/bin/python"` exists and the server venv lacks the model, the factory uses `RemoteNemotronModel`. A dead worker raises in `push`, which the `Diarizer` turns into `on_error("error", ...)`.

- [ ] **Step 1: Write the failing test:** run `voicebot.diar_worker` in-process on a thread with an injected fake model (Task 3's `FakeModel`), connect a `RemoteNemotronModel` to its port, push 1600 samples, and assert 10 frames come back with the scripted speaker; then kill the worker and assert `push` raises.
- [ ] **Step 2: Run it.** → FAIL.
- [ ] **Step 3: Implement** the worker, the remote adapter, the factory switch and `scripts/setup_diarizer_venv.sh` (creates `.venvs/diarizer` with the CUDA torch, the Task 1 transformers and `--no-deps -e server`).
- [ ] **Step 4: Run the tests.** PASS. On the VM, `python -m voicebot.diarizer --smoke` through the remote adapter gives the same speakers as Task 1.
- [ ] **Step 5: Commit.**

```bash
git add server/voicebot/diar_worker.py server/voicebot/diarizer.py server/voicebot/web.py scripts/setup_diarizer_venv.sh server/tests/test_diarizer.py
git commit -m "feat(diar): run the diarizer in its own venv when transformers conflicts"
```

---

### Task 10: Web UI and docs

**Files:**
- Modify: `web/src/protocol.ts`, `web/src/state.ts`, `web/src/App.tsx`, `web/src/components/ChatTranscript.tsx`, `web/src/components/SettingsPanel.tsx`, `web/src/styles.css`
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-11-20-v4-diarization-design.md` (status line)
- Test: `web/src/state.test.ts`, `web/src/components/SettingsPanel.test.tsx`

**Interfaces:**
- Produces (protocol):
  - `export type DiarMode = "ultra_low_latency" | "very_low_latency" | "low_latency"`; `export type DiarState = "off" | "loading" | "ready" | "error"`.
  - `Defaults` gains `diarization?: boolean; diar_mode?: DiarMode; diar_wait_ms?: number`; `ServerConfig` gains `diarization_available?: boolean`.
  - `user_transcript` becomes `{ type: "user_transcript"; text: string; turn_id: number | null; speaker?: number | "unknown"; ignored?: boolean }`.
  - New event `{ type: "diarization"; state: DiarState; message?: string }`.
- Produces (state): `ChatMessage` gains `speaker?: number | "unknown"; ignored?: boolean`; `AppState.diarization: { state: DiarState; message: string | null }`, initial `{ state: "off", message: null }`, synced from `config` exactly like `wordAlign`.

Behaviour:
- Reducer: an ignored transcript adds a user message with `speaker` and `ignored: true`, clears `partial`, and leaves `turnId` unchanged. A normal one stores `speaker` on the message as today plus the tag.
- `App.tsx`: the `user_transcript` branch that resets the player and `heard` runs only when `!e.ignored && e.turn_id != null`. Pass `diarization={state.diarization}` to `SettingsPanel`.
- `ChatTranscript`: user role label `You` plus a `<span className="speaker-tag">S0</span>` (`S?` for `"unknown"`) when `speaker` is set; ignored messages get class `msg-ignored` and role label `Speaker N (ignored)`.
- `SettingsPanel` (shown when `config.defaults.diarization !== undefined`): checkbox **Speaker diarization** (checked when `ready` or `loading`; disabled when `!config.diarization_available`, no settings, or loading); select `id="diar-mode"` **Diarization latency** with options `ultra_low_latency` "0.32 s", `very_low_latency` "0.64 s", `low_latency` "1.04 s"; slider **Max wait (ms)** 0–2000 step 50; mode and wait disabled when `!config.diarization_available`. Status line: error message (role alert) or a hint: off "Answers everyone.", loading "Loading the diarization model…", ready "Answers Speaker 0 only. Clear (while the bot is quiet) makes the next voice Speaker 0."
- CSS: `.msg-ignored { opacity: 0.5; }`, `.speaker-tag` small muted pill next to the role label.

- [ ] **Step 1: Write the failing tests.** In `state.test.ts`:

```ts
it("tags speakers, keeps ignored transcripts out of the turn, tracks diarization", () => {
  let s = srv(initialState, { ...cfg, defaults: { ...cfg.defaults, diarization: true } });
  expect(s.diarization.state).toBe("ready");
  s = srv(s, { type: "user_transcript", text: "hi", turn_id: 4, speaker: 0 });
  expect(s.messages[0]).toMatchObject({ text: "hi", speaker: 0 }); expect(s.turnId).toBe(4);
  s = srv(s, { type: "user_partial", text: "not" });
  s = srv(s, { type: "user_transcript", text: "not me", turn_id: null, speaker: 1, ignored: true });
  expect(s.messages[1]).toMatchObject({ text: "not me", speaker: 1, ignored: true });
  expect(s.turnId).toBe(4); expect(s.partial).toBeNull();
  s = srv(s, { type: "diarization", state: "error", message: "oom" });
  expect(s.diarization).toEqual({ state: "error", message: "oom" });
  s = srv(s, cfg); expect(s.diarization.state).toBe("error");          // a reconnect keeps the error
});
```

In `SettingsPanel.test.tsx` (pass `diarization={{ state: "off", message: null }}` in `render`):

```tsx
describe("SettingsPanel diarization", () => {
  it("shows the controls and disables them when unavailable", () => {
    const html = render({ diarization_available: false }, { diarization: false, diar_mode: "ultra_low_latency", diar_wait_ms: 600 });
    expect(html).toContain("Speaker diarization"); expect(html).toContain('id="diar-mode"');
    expect(html).toContain("Max wait (ms)"); expect(html).toMatch(/<select id="diar-mode"[^>]*disabled/);
  });
  it("hides the controls for an older server", () => {
    expect(render({}, {})).not.toContain("Speaker diarization");
  });
});
```

- [ ] **Step 2: Run them to see them fail.** `npm run test` → FAIL.
- [ ] **Step 3: Implement** the behaviour above.
- [ ] **Step 4: Run the web checks.** `npm run test && npm run typecheck && npm run build` pass.
- [ ] **Step 5: Docs.** In `CLAUDE.md` add a status row: "v4: speaker diarization | Built on branch `feat/v4-diarization`. Manual checks on the VM still open. | Spec `docs/superpowers/specs/2026-11-20-v4-diarization-design.md`, plan `docs/superpowers/plans/2026-11-20-v4-diarization.md`". Set the spec status to "Implemented; manual checks open".
- [ ] **Step 6: Commit.**

```bash
git add web/src/protocol.ts web/src/state.ts web/src/App.tsx web/src/components/ChatTranscript.tsx web/src/components/SettingsPanel.tsx web/src/styles.css web/src/state.test.ts web/src/components/SettingsPanel.test.tsx CLAUDE.md docs/superpowers/specs/2026-11-20-v4-diarization-design.md
git commit -m "feat(ui): diarization controls, speaker tags and ignored transcripts"
```

---

### Task 11: Manual checks (the user, on the VM)

Not code. Hand these to the user and record the results in spec §8 when they come back.

1. One speaker, diarization on: turns answered and tagged S0. Compare `perceived_ms` with diarization off.
2. Two people: the second person's speech shows as "Speaker 1 (ignored)" and gets no answer.
3. Speakers instead of headphones, barge-in `onset`: the bot's own echo does not interrupt it; the user still can. Press Clear and speak first, before letting the bot talk through the speakers: after a reset the first voice becomes Speaker 0, and that must be you, not the echo.
4. Each latency mode: `cd server && ../.venv/bin/python -m voicebot.latency_report --by diarization,diar_mode` and compare `diar_wait_ms`, plus `diar_lag_ms` and the labels in `logs/turns.txt`.
