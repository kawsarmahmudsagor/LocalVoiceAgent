from types import SimpleNamespace as NS

import numpy as np
import pytest

from voicebot.aligner import WordAligner, pcm16_to_float, word_starts_from_characters


def chars(*spec):
    """spec: (character, start) pairs in order, as the MMS character aligner returns them."""
    return [NS(character=c, start_time=t) for c, t in spec]


def test_word_starts_are_each_words_first_character():
    out = word_starts_from_characters("Hi there.", chars(("h", 0.1), ("i", 0.15), ("t", 0.3), ("h", 0.32),
                                                        ("e", 0.34), ("r", 0.36), ("e", 0.38)))
    assert out == [0.1, 0.3]


def test_words_without_letters_take_the_previous_time_and_split_abbreviations_still_count_once():
    # MMS keeps only a-z and apostrophes: "—" and "42" vanish, "U.S." becomes two words "u s".
    text = "Hi — the U.S. 42 it's"
    c = chars(("h", 0.0), ("i", 0.05), ("t", 0.2), ("h", 0.21), ("e", 0.22), ("u", 0.4), ("s", 0.5),
              ("i", 0.8), ("t", 0.81), ("'", 0.82), ("s", 0.83))
    assert word_starts_from_characters(text, c) == [0.0, 0.0, 0.2, 0.4, 0.4, 0.8]


def test_leading_symbol_starts_at_zero_and_a_short_alignment_gives_none():
    assert word_starts_from_characters("— ok", chars(("o", 0.3), ("k", 0.35))) == [0.0, 0.3]
    assert word_starts_from_characters("hi there", chars(("h", 0.0), ("i", 0.1))) is None


def test_pcm16_to_float_downmixes_to_mono():
    stereo = np.array([16384, -16384, 8192, 8192], dtype=np.int16).tobytes()
    assert pcm16_to_float(stereo, channels=2).tolist() == pytest.approx([0.0, 0.25])


class FakeCharAligner:
    def __init__(self): self.calls = []
    def align_characters(self, text, audio, sample_rate):
        self.calls.append((text, len(audio), sample_rate))
        return chars(*[(ch, i * 0.1) for i, ch in enumerate(c for c in text.lower() if c.isalpha())])


def test_word_aligner_aligns_pcm_and_returns_relative_word_starts():
    fake = FakeCharAligner()
    wa = WordAligner(device="cpu", char_aligner=fake)
    pcm = np.zeros(2400, dtype=np.int16).tobytes()
    assert wa.word_starts("ab cd", pcm, sample_rate=24000, channels=1) == pytest.approx([0.0, 0.2])
    assert fake.calls == [("ab cd", 2400, 24000)]
    assert wa.word_starts("ab", b"", sample_rate=24000, channels=1) is None     # no audio, nothing to align


def test_word_aligner_returns_none_when_alignment_fails():
    class Broken:
        def align_characters(self, *a): raise RuntimeError("boom")
    assert WordAligner(device="cpu", char_aligner=Broken()).word_starts("ab", b"\0\0" * 10, 24000, 1) is None
