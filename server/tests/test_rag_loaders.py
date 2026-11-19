from pathlib import Path

import pytest

from voicebot.rag.loaders import NoText, UnsupportedType, load_text

FIX = Path(__file__).parent / "fixtures" / "rag"


def test_txt_and_md_read_as_text(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Title\n\nBody", encoding="utf-8")
    assert load_text(tmp_path / "a.txt") == "hello"
    assert load_text(tmp_path / "b.md") == "# Title\n\nBody"


def test_txt_falls_back_to_cp1252(tmp_path):
    (tmp_path / "a.txt").write_bytes("café".encode("cp1252"))
    assert load_text(tmp_path / "a.txt") == "café"


def test_docx_paragraphs_and_tables(tmp_path):
    import docx
    d = docx.Document(); d.add_paragraph("First para."); t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Cell A", "Cell B"; d.save(tmp_path / "x.docx")
    out = load_text(tmp_path / "x.docx")
    assert "First para." in out and "Cell A" in out and "Cell B" in out


def test_pdf_text_and_image_only():
    assert "Canberra is the capital of Australia." in load_text(FIX / "text.pdf")
    with pytest.raises(NoText): load_text(FIX / "image-only.pdf")


def test_empty_and_unsupported(tmp_path):
    (tmp_path / "e.txt").write_text("  \n", encoding="utf-8")
    with pytest.raises(NoText): load_text(tmp_path / "e.txt")
    (tmp_path / "x.doc").write_bytes(b"x")
    with pytest.raises(UnsupportedType): load_text(tmp_path / "x.doc")
