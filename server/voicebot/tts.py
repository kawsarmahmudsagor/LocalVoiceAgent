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


def make_tts(settings: Settings, *, engine_cls=None, stream_cls=None):
    if engine_cls is None or stream_cls is None:
        from RealtimeTTS import KokoroEngine, TextToAudioStream
        from RealtimeTTS.stream_player import StreamPlayer

        # Runtime shim (see VENDORED.md): StreamPlayer.stop() closes the PyAudio stream
        # without a lock. Stop (on a helper thread) and play()'s own cleanup both call it,
        # and a concurrent double close corrupts the heap (0xc0000374).
        serialize_method(StreamPlayer, "stop")
        engine_cls = engine_cls or KokoroEngine
        stream_cls = stream_cls or TextToAudioStream
    return stream_cls(engine_cls(voice=settings.tts_voice))


def warm_up(stream) -> None:
    """Synthesize one short phrase muted, so the first real turn is not slowed by lazy init."""
    stream.feed("Warming up.")
    stream.play(muted=True)


def list_voices(stream) -> list[str]:
    return [getattr(v, "name", str(v)) for v in stream.engine.get_voices()]


def set_voice(stream, name: str) -> None:
    stream.engine.set_voice(name)


def output_format(stream) -> dict:
    _fmt, channels, rate = stream.engine.get_stream_info()
    return {"sample_rate": rate, "channels": channels, "encoding": "pcm_s16le"}
