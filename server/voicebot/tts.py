"""Builds the RealtimeTTS stream with Kokoro on CPU. See spec §4.5."""
from __future__ import annotations

from voicebot.config import Settings


def make_tts(settings: Settings, *, engine_cls=None, stream_cls=None):
    if engine_cls is None or stream_cls is None:
        from RealtimeTTS import KokoroEngine, TextToAudioStream
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
