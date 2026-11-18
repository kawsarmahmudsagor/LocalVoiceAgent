"""Generate assets/ref_voice.wav with Kokoro (af_heart) from assets/ref_voice.txt (v2 spec §3.1).

    venvs/bench-kokoro/bin/python -m voicebot.bench.ref_voice

qwen-cuda and chatterbox clone this voice, and its text matches the audio exactly.
"""
from __future__ import annotations

import sys
import wave

from voicebot.config import REPO_ROOT

TEXT = REPO_ROOT / "assets" / "ref_voice.txt"
OUT = REPO_ROOT / "assets" / "ref_voice.wav"


def main() -> int:
    from RealtimeTTS import KokoroEngine, TextToAudioStream  # noqa: PLC0415  (bench venv only)

    text = TEXT.read_text(encoding="utf-8").strip()
    engine = KokoroEngine(voice="af_heart")
    _, channels, rate = engine.get_stream_info()
    chunks: list[bytes] = []
    stream = TextToAudioStream(engine, muted=True)  # no output device on the VM; chunks still arrive
    stream.feed(text)
    stream.play(muted=True, on_audio_chunk=chunks.append)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(chunks))
    seconds = sum(map(len, chunks)) / (2 * channels * rate)
    print(f"wrote {OUT} ({seconds:.1f} s at {rate} Hz)")
    engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
