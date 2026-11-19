"""Read uploaded documents (txt, md, pdf, docx) into plain text."""
from pathlib import Path

SUPPORTED: frozenset[str] = frozenset({".txt", ".md", ".pdf", ".docx"})


class UnsupportedType(ValueError):
    """The file extension is not one of SUPPORTED."""


class NoText(ValueError):
    """The file holds no extractable text (empty, or a scanned/image-only PDF)."""


def _read_plain(path: Path) -> str:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            lines.extend(cell.text for cell in row.cells)
    return "\n".join(lines)


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedType(f"unsupported file type: {suffix or path.name}")
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        text = _read_plain(path)
    text = text.strip()
    if not text:
        raise NoText(f"no extractable text in {path.name}")
    return text
