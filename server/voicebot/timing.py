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
    "retrieval_start",   # RAG retrieval starts (only when RAG is on)
    "retrieval_done",    # retriever returned or failed
    "llm_first_token",   # first raw backend chunk
    "first_text",        # first speakable chunk after speech_filter
    "tts_first_audio",   # first TTS audio chunk produced
    "playback_started",  # browser player started output (server mode: = tts_first_audio)
    "llm_done",          # backend stream ended
    "play_done",         # audio.play() returned
    "barge_onset",       # user speech detected while the bot was speaking
    "barge_in",          # barge-in confirmed
    "playback_cut",      # bot audio stopped
)

SPANS: tuple[str, ...] = (
    "silence_wait_ms", "stt_ms", "retrieval_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
    "playback_ms", "perceived_ms", "llm_total_ms", "total_ms",
)

BARGE_SPANS: tuple[str, ...] = ("barge_detect_ms", "barge_cut_ms")

# Marks made by the STT of the *next* utterance; routed to a pending dict while held.
INPUT_STAGES: tuple[str, ...] = ("speech_end", "endpoint", "stt_start")

_ORIGIN_STAGES = ("speech_end", "endpoint", "stt_done", "input")


def _ms(a: float | None, b: float | None) -> int | None:
    return None if a is None or b is None else round((b - a) * 1000)


class TurnTimer:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._marks: dict[str, float] = {}
        self._pending: dict[str, float] = {}
        self._held = False

    def _dict(self, stage: str) -> dict[str, float]:
        return self._pending if self._held and stage in INPUT_STAGES else self._marks

    @staticmethod
    def _check(stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")

    def mark(self, stage: str) -> None:
        """Record now for stage; the first mark wins until unmark()/reset()."""
        self._check(stage)
        now = self._clock()
        with self._lock:
            self._dict(stage).setdefault(stage, now)

    def unmark(self, stage: str) -> None:
        self._check(stage)
        with self._lock:
            self._dict(stage).pop(stage, None)

    def get(self, stage: str) -> float | None:
        self._check(stage)
        with self._lock:
            return self._dict(stage).get(stage)

    def has(self, stage: str) -> bool:
        return self.get(stage) is not None

    def now(self) -> float:
        """The timer's clock, for waiting relative to recorded marks."""
        return self._clock()

    def hold(self) -> None:
        """Route INPUT_STAGES marks to a pending dict until the next reset()."""
        with self._lock:
            self._held = True

    def reset(self) -> None:
        with self._lock:
            self._marks.clear()
            if self._held:
                self._marks.update(self._pending)
                self._pending.clear()
                self._held = False

    def report(self) -> dict:
        with self._lock:
            m = dict(self._marks)
        g = m.get
        origin = next((m[s] for s in _ORIGIN_STAGES if s in m), None)
        if origin is None:
            return {"marks": {s: None for s in STAGES}, "spans": {s: None for s in SPANS + BARGE_SPANS}}
        inp = g("input") if "input" in m else g("stt_done")
        llm_start = g("retrieval_done") if "retrieval_done" in m else inp  # LLM spans are the LLM's own time
        heard = g("playback_started") if "playback_started" in m else g("tts_first_audio")
        spans = {
            "silence_wait_ms": _ms(g("speech_end"), g("endpoint")),
            "stt_ms": _ms(g("endpoint"), g("stt_done")),
            "retrieval_ms": _ms(g("retrieval_start"), g("retrieval_done")),
            "llm_ttft_ms": _ms(llm_start, g("llm_first_token")),
            "filter_ms": _ms(g("llm_first_token"), g("first_text")),
            "tts_first_audio_ms": _ms(g("first_text"), g("tts_first_audio")),
            "playback_ms": _ms(g("tts_first_audio"), g("playback_started")),
            "perceived_ms": _ms(origin, heard),
            "llm_total_ms": _ms(llm_start, g("llm_done")),
            "total_ms": _ms(origin, g("play_done")),
            "barge_detect_ms": _ms(g("barge_onset"), g("barge_in")),
            "barge_cut_ms": _ms(g("barge_onset"), g("playback_cut")),
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


# The selected catalog slugs come first: stt_model below is the .env default, not the active engine.
_MODEL_KEYS = ("stt_final", "stt_realtime", "tts_model")
_CONTEXT_KEYS = ("llm_backend", "model_label", "audio_io", "stt_model", "stt_device", "voice",
                 "temperature", "top_p", "max_tokens", "reply_chars",
                 "kokoro_trim_silence", "kokoro_extra_end_ms", "tts_first_fragment_words",
                 "tts_fast_sentence_fragment", "tts_sentence_silence_ms", "kokoro_speed",
                 "rag_enabled", "rag_embedder", "rag_top_k", "rag_min_score", "rag_max_context_chars",
                 "rag_chunk_size", "rag_chunk_overlap", "rag_chunks_used", "rag_context_chars", "rag_error")


def fmt_ms(ms: int | None) -> str:
    """1234 -> '1.234 s (1234 ms)'; None -> '-'."""
    return "-" if ms is None else f"{ms / 1000:.3f} s ({ms} ms)"


def format_turn(record: dict) -> str:
    """Render one turn record as a readable text block, times in seconds and ms."""
    ts = str(record.get("ts", ""))
    ts = ts.replace("T", " ").replace("+00:00", " UTC") if ts else "-"
    head = [f"Turn {record.get('turn_id', '?')}", str(record.get("source", "?"))]
    if record.get("interrupted"):
        head.append("interrupted")
    lines = [f"=== {' | '.join(head)} | {ts} ==="]
    models = [f"{k}={record[k]}" for k in _MODEL_KEYS if record.get(k) is not None]
    if models:
        lines.append("  " + "  ".join(models))
    ctx = [f"{k}={record[k]}" for k in _CONTEXT_KEYS if record.get(k) is not None]
    if ctx:
        lines.append("  " + "  ".join(ctx))
    for label, key in (("heard", "transcript"), ("reply", "reply")):
        if record.get(key) is not None:
            lines.append(f'  {label}: "{record[key]}"')
    spans = record.get("spans") or {}
    for span in SPANS:
        lines.append(f"  {span.removesuffix('_ms'):<18}{fmt_ms(spans.get(span))}")
    return "\n".join(lines) + "\n\n"


def append_turn_text(path: Path, record: dict) -> None:
    """Append one readable turn block. Never raises on I/O errors: logging must not fail a turn."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(format_turn(record))
    except OSError:
        log.exception("could not append to %s", path)
