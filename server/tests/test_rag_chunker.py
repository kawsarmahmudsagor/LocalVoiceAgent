import pytest

from voicebot.rag.chunker import chunk


def test_short_text_is_one_chunk():
    assert chunk("Hello world.", 800, 120) == ["Hello world."]


def test_prefers_paragraph_breaks():
    text = "A" * 300 + "\n\n" + "B" * 300
    assert chunk(text, 400, 0) == ["A" * 300, "B" * 300]


def test_no_chunk_exceeds_size_and_overlap_is_shared():
    text = " ".join(f"word{i}." for i in range(500))
    out = chunk(text, 200, 40)
    assert all(len(c) <= 200 for c in out) and len(out) > 1
    assert all(a[-20:] in b for a, b in zip(out, out[1:]))   # tail of each chunk reappears in the next


def test_unbroken_text_is_hard_split():          # Review Focus 3
    out = chunk("x" * 1000, 300, 50)
    assert all(len(c) <= 300 for c in out) and "".join(out).count("x") >= 1000


def test_whitespace_normalised_and_no_empty_chunks():
    assert chunk("a   b\n\n\n\n   \n\nc", 800, 0) == ["a b\n\nc"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, -1), (100, 51)])
def test_bad_parameters(size, overlap):
    with pytest.raises(ValueError): chunk("text", size, overlap)
