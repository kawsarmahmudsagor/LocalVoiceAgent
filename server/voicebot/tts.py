"""Builds the RealtimeTTS stream with Kokoro on CPU. See spec §4.5."""
from __future__ import annotations

import functools
import threading

from voicebot.config import Settings


def serialize_method(cls, name: str) -> None:
    """Wrap cls.<name> so calls from different threads run one at a time (idempotent)."""
    original = getattr(cls, name)
    if getattr(original, "_voicebot_serialized", False):
        return
    lock = threading.RLock()

    @functools.wraps(original)
    def serialized(*args, **kwargs):
        with lock:
            return original(*args, **kwargs)
    serialized._voicebot_serialized = True  # type: ignore[attr-defined]
    setattr(cls, name, serialized)


def install_realtimetts_shims() -> None:
    """Runtime workaround for an upstream race (see VENDORED.md).

    AudioStream.close_stream() checks and closes the PyAudio stream without a lock.
    Stop (immediate player stop, on a helper thread) and play()'s own cleanup (draining
    player stop) can both reach it, and a concurrent double close corrupts the heap
    (Windows 0xc0000374). Only the close is serialized: locking StreamPlayer.stop as a
    whole would make Stop wait until the draining stop had played out the buffer.
    """
    from RealtimeTTS.stream_player import AudioStream

    serialize_method(AudioStream, "close_stream")


def kokoro_pacing(settings: Settings) -> dict:
    """KokoroEngine trims the silence around every synthesized fragment by default, which removes the
    pauses after commas and full stops (speech runs on "in one breath"). KOKORO_TRIM_SILENCE=false keeps
    them; with trimming on, KOKORO_EXTRA_END_MS keeps a short pause at each fragment's end."""
    return {"trim_silence": settings.kokoro_trim_silence, "extra_end_ms": settings.kokoro_extra_end_ms}


def make_tts(settings: Settings, *, engine_cls=None, stream_cls=None, entry=None, module=None):
    """Kokoro from settings (v1), or the engine a catalog entry names (v2 plan Task 12)."""
    if entry is not None and entry.engine == "KokoroEngine":
        from dataclasses import replace

        entry = replace(entry, options={**kokoro_pacing(settings), **entry.options})
    if entry is not None:
        from voicebot.config import REPO_ROOT
        from voicebot.tts_engines import build_engine

        if module is None:
            import RealtimeTTS as module

            install_realtimetts_shims()
        if stream_cls is None:
            from RealtimeTTS import TextToAudioStream as stream_cls
        return stream_cls(build_engine(entry, root=REPO_ROOT, module=module))
    if engine_cls is None or stream_cls is None:
        from RealtimeTTS import KokoroEngine, TextToAudioStream

        install_realtimetts_shims()
        engine_cls = engine_cls or KokoroEngine
        stream_cls = stream_cls or TextToAudioStream
    return stream_cls(engine_cls(voice=settings.tts_voice, **kokoro_pacing(settings)))


def warm_up(stream) -> None:
    """Synthesize one short phrase muted, so the first real turn is not slowed by lazy init."""
    stream.feed("Warming up.")
    stream.play(muted=True)


def list_voices(stream) -> list[str]:
    return [getattr(v, "name", str(v)) for v in stream.engine.get_voices()]


def set_voice(stream, name: str) -> None:
    stream.engine.set_voice(name)


def drain_audio_queue(stream) -> int:
    """Discard synthesized audio still queued in the engine (left over by a stopped turn).

    The queue may be wrapped (RealtimeTTS DecodedAudioQueue keeps the original in
    .wrapped), so both are emptied. Returns the number of chunks dropped.
    """
    dropped = 0
    q = getattr(getattr(stream, "engine", None), "queue", None)
    seen = set()
    while q is not None and id(q) not in seen and hasattr(q, "get_nowait"):
        seen.add(id(q))
        while True:
            try:
                q.get_nowait()
            except Exception:  # queue.Empty, or a wrapper that raises its own
                break
            dropped += 1
        q = getattr(q, "wrapped", None)
    return dropped


def output_format(stream) -> dict:
    _fmt, channels, rate = stream.engine.get_stream_info()
    return {"sample_rate": rate, "channels": channels, "encoding": "pcm_s16le"}
