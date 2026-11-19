"""Forced word alignment for TTS engines that report no word timings (Piper, Qwen, Inflect, Coqui).

Uses RealtimeTTS's torchaudio MMS forced aligner (vendored, unchanged): it finds where each character of
the known sentence text sits in the synthesized audio. RealtimeTTS's default omniASR aligner needs
fairseq2, which brings its own torch build, so this project uses the torchaudio one: it runs on the
torch 2.7.1+cu128 already installed.

Fetch the weights once (about 1.2 GB, torch hub cache): `python -m voicebot.aligner --fetch`."""
from __future__ import annotations

import logging
import sys
import threading
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    from RealtimeTTS.alignment.torchaudio_fa import normalize_mms_fa_text  # what the aligner aligns
    return normalize_mms_fa_text(text)


def word_starts_from_characters(text: str, characters: Sequence) -> list[float] | None:
    """Start time of each whitespace word of `text`, from the aligner's per-character times.

    The aligner keeps only letters and apostrophes, so "U.S." is two short words, and "—" or "42"
    vanish: a word left without characters takes the previous word's time. Returns None when the
    characters don't cover the text (the aligner gave up part way)."""
    out: list[float] = []
    i, prev = 0, 0.0
    for word in text.split():
        start = None
        for sub in _normalize(word).split():
            if i + len(sub) > len(characters):
                return None
            if start is None:
                start = float(characters[i].start_time)
            i += len(sub)
        if start is not None:
            prev = start
        out.append(prev)
    return out if i == len(characters) else None


def pcm16_to_float(pcm: bytes, channels: int = 1) -> np.ndarray:
    """s16le PCM to mono float32 in [-1, 1)."""
    a = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        a = a[: len(a) - len(a) % channels].reshape(-1, channels).mean(axis=1)
    return a


class WordAligner:
    """Loads the MMS model on first use (or load()) and aligns one sentence at a time."""

    def __init__(self, device: str = "cuda", *, char_aligner=None) -> None:
        self.device = device
        self._aligner = char_aligner
        self._lock = threading.Lock()  # one alignment at a time; the model is not shared across threads

    def load(self) -> None:
        """Download (first time) and load the model. Slow: call it off the event loop and turn path."""
        with self._lock:
            if self._aligner is None:
                from RealtimeTTS.alignment.torchaudio_fa import TorchaudioMMSFACharacterAligner
                device = self.device
                if device.startswith("cuda"):
                    import torch
                    if not torch.cuda.is_available():
                        log.warning("word alignment: CUDA is not available, using the CPU")
                        device = "cpu"
                aligner = TorchaudioMMSFACharacterAligner(device=device)
                aligner._load()  # the vendored aligner loads lazily; do it now, not on the first sentence
                self._aligner = aligner

    @property
    def ready(self) -> bool:
        return self._aligner is not None

    def word_starts(self, text: str, pcm: bytes, sample_rate: int, channels: int) -> list[float] | None:
        """Each whitespace word's start, in seconds from the start of `pcm`; None if it can't align."""
        if self._aligner is None or not pcm or not text.split():
            return None
        audio = pcm16_to_float(pcm, channels)
        try:
            with self._lock:
                chars = self._aligner.align_characters(text, audio, sample_rate)
        except Exception:
            log.exception("word alignment failed for %r", text)
            return None
        return word_starts_from_characters(text, chars)


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        logging.basicConfig(level=logging.INFO)
        WordAligner(device="cpu").load()
        print("MMS forced-alignment model is in the torch hub cache.")
    else:
        print(__doc__)
