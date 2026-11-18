"""Record benchmark clips through the bot's own mic path (v2 spec §3.3, plan Task 7).

With CLIP_RECORD_DIR set, each utterance the recorder is about to transcribe is saved as the next
clip-NNN.wav (16 kHz mono int16), so the clips are exactly what the server receives.
"""
from __future__ import annotations

import logging
import re
import wave
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)
CLIP_RE = re.compile(r"^clip-(\d+)\.wav$")
SAMPLE_RATE = 16000


def merge_callbacks(*dicts: dict[str, Callable]) -> dict[str, Callable]:
    """Combine callback dicts; a key in several dicts calls each function in order and returns None.

    Returning None matters: a truthy on_transcription_start return aborts the transcription.
    """
    merged: dict[str, list[Callable]] = {}
    for d in dicts:
        for key, fn in d.items():
            merged.setdefault(key, []).append(fn)

    def chain(fns: list[Callable]) -> Callable:
        def call(*args, **kwargs) -> None:
            for fn in fns:
                fn(*args, **kwargs)
            return None
        return call

    return {key: chain(fns) for key, fns in merged.items()}


def _next_index(out_dir: Path) -> int:
    nums = [int(m[1]) for p in out_dir.glob("clip-*.wav") if (m := CLIP_RE.match(p.name))]
    return max(nums, default=0) + 1


def to_int16(audio) -> np.ndarray:
    a = np.asarray(audio)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.int16)
    return np.clip(np.round(a.astype(np.float64) * 32767), -32767, 32767).astype(np.int16)


def clip_callbacks(out_dir: Path) -> dict[str, Callable]:
    out_dir = Path(out_dir)

    def on_transcription_start(audio) -> None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"clip-{_next_index(out_dir):03d}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(to_int16(audio).tobytes())
            log.info("saved benchmark clip %s", path)
        except Exception:
            log.exception("could not save the benchmark clip")
        return None  # a truthy return makes RealtimeSTT abort the transcription

    return {"on_transcription_start": on_transcription_start}
