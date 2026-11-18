"""Time one TTS pool entry with the public RealtimeTTS API (v2 spec §3.3, plan Task 8).

    python -m voicebot.bench.run_tts --slug S [--pool ...] [--sentences bench/sentences.txt]

For each sentence: time to first audio and real-time factor. Writes bench/results/<slug>.json
and saves each sentence's audio to bench/out/<slug>/NN.wav for the listening check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

from voicebot.bench.gpu import GpuSampler
from voicebot.catalog import Entry, by_slug, load_catalog
from voicebot.config import REPO_ROOT
from voicebot.tts_engines import build_engine as _build_engine

DEFAULT_POOL = REPO_ROOT / "bench" / "pool.toml"
DEFAULT_SENTENCES = REPO_ROOT / "bench" / "sentences.txt"
WARMUP_TEXT = "Warming up."


def build_engine(entry: Entry):
    return _build_engine(entry, root=REPO_ROOT)


def measure(stream, sentences: list[str], clock=time.perf_counter, out_dir: Path | None = None) -> list[dict]:
    _, channels, rate = stream.engine.get_stream_info()
    rows = []
    for i, text in enumerate(sentences, start=1):
        first: list[float] = []
        chunks: list[bytes] = []

        def cb(chunk: bytes) -> None:
            if not first:
                first.append(clock())
            chunks.append(chunk)

        t0 = clock()
        stream.feed(text)
        stream.play(muted=True, on_audio_chunk=cb)
        t_end = clock()
        audio = b"".join(chunks)
        audio_ms = round(len(audio) / (rate * 2 * channels) * 1000)
        rows.append({"text": text,
                     "first_audio_ms": round((first[0] - t0) * 1000) if first else None,
                     "audio_ms": audio_ms,
                     "rtf": (t_end - t0) / (audio_ms / 1000) if audio_ms else None})
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_dir / f"{i:02d}.wav"), "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(audio)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.run_tts", description=__doc__.splitlines()[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    args = ap.parse_args(argv)

    from RealtimeTTS import TextToAudioStream  # noqa: PLC0415  (bench venv only)

    entry = by_slug(load_catalog(args.pool))[args.slug]
    if entry.kind != "tts":
        raise SystemExit(f"{entry.slug} is not a TTS entry")
    sentences = [l.strip() for l in args.sentences.read_text(encoding="utf-8").splitlines() if l.strip()]
    with GpuSampler() as gpu:
        engine = build_engine(entry)
        stream = TextToAudioStream(engine)
        measure(stream, [WARMUP_TEXT])  # warm-up, excluded
        rows = measure(stream, sentences, out_dir=REPO_ROOT / "bench" / "out" / entry.slug)
    out = REPO_ROOT / "bench" / "results" / f"{entry.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"slug": entry.slug, "rows": rows, "peak_vram_mb": gpu.peak_mb}, indent=2) + "\n")
    firsts = sorted(r["first_audio_ms"] for r in rows if r["first_audio_ms"] is not None)
    print(f"{entry.slug}: median first_audio_ms={firsts[len(firsts) // 2] if firsts else '-'} "
          f"peak_vram_mb={gpu.peak_mb} -> {out}")
    try:
        engine.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
