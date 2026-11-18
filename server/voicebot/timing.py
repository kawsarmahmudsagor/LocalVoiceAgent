"""Per-turn stage timing. See plan Task 7b.

Each turn marks a timestamp at each pipeline stage; report() turns them into
per-stage durations ("spans") so the source of latency is visible.
Stdlib only.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

STAGES: tuple[str, ...] = (
    "speech_end",        # VAD first hears silence after speech (cleared if speech resumes)
    "endpoint",          # silence wait over, recording stops
    "stt_start",         # final transcription starts
    "stt_done",          # recorder.text() returned
    "input",             # typed send_text received (typed turns only)
    "llm_first_token",   # first raw backend chunk
    "first_text",        # first speakable chunk after speech_filter
    "tts_first_audio",   # first TTS audio chunk produced
    "playback_started",  # browser player started output (server mode: = tts_first_audio)
    "llm_done",          # backend stream ended
    "play_done",         # audio.play() returned
)

SPANS: tuple[str, ...] = (
    "silence_wait_ms", "stt_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
    "playback_ms", "perceived_ms", "llm_total_ms", "total_ms",
)

_ORIGIN_STAGES = ("speech_end", "endpoint", "stt_done", "input")


def _ms(a: float | None, b: float | None) -> int | None:
    return None if a is None or b is None else round((b - a) * 1000)


class TurnTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}

    @staticmethod
    def _check(stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")

    def mark(self, stage: str) -> None:
        """Record now for stage; the first mark wins until unmark()/reset()."""
        self._check(stage)
        now = self._clock()
        with self._lock:
            self._marks.setdefault(stage, now)

    def unmark(self, stage: str) -> None:
        self._check(stage)
        with self._lock:
            self._marks.pop(stage, None)

    def get(self, stage: str) -> float | None:
        self._check(stage)
        with self._lock:
            return self._marks.get(stage)

    def has(self, stage: str) -> bool:
        return self.get(stage) is not None

    def reset(self) -> None:
        with self._lock:
            self._marks.clear()

    def report(self) -> dict:
        with self._lock:
            m = dict(self._marks)
        g = m.get
        origin = next((m[s] for s in _ORIGIN_STAGES if s in m), None)
        if origin is None:
            return {"marks": {s: None for s in STAGES}, "spans": {s: None for s in SPANS}}
        inp = g("input") if "input" in m else g("stt_done")
        heard = g("playback_started") if "playback_started" in m else g("tts_first_audio")
        spans = {
            "silence_wait_ms": _ms(g("speech_end"), g("endpoint")),
            "stt_ms": _ms(g("endpoint"), g("stt_done")),
            "llm_ttft_ms": _ms(inp, g("llm_first_token")),
            "filter_ms": _ms(g("llm_first_token"), g("first_text")),
            "tts_first_audio_ms": _ms(g("first_text"), g("tts_first_audio")),
            "playback_ms": _ms(g("tts_first_audio"), g("playback_started")),
            "perceived_ms": _ms(origin, heard),
            "llm_total_ms": _ms(inp, g("llm_done")),
            "total_ms": _ms(origin, g("play_done")),
        }
        return {"marks": {s: _ms(origin, g(s)) for s in STAGES}, "spans": spans}


def append_jsonl(path: Path, record: dict) -> None:
    """Append one compact JSON line. Never raises on I/O errors: logging must not fail a turn."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
    except OSError:
        log.exception("could not append to %s", path)
