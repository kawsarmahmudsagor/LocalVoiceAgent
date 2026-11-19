"""Split text into overlapping chunks, preferring natural boundaries."""
import re

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split(text: str, size: int, seps: list[str], hard: int) -> list[str]:
    """Split into pieces of at most `size` chars (hard splits use `hard`, leaving room for overlap); separators stay attached to the piece before."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep == "":
            return [text[j:j + hard] for j in range(0, len(text), hard)]
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
            out: list[str] = []
            for piece in pieces:
                if piece:
                    out.extend(_split(piece, size, seps[i + 1:], hard))
            return out
    return [text]


def _tail(prev: str, overlap: int, room: int) -> str:
    """Last `overlap` chars of prev (at most `room`), starting at a word boundary if possible."""
    n = min(overlap, room)
    if n <= 0:
        return ""
    tail = prev[-n:]
    if len(prev) > n and " " in tail:
        tail = tail.split(" ", 1)[1]
    return tail


def chunk(text: str, size: int, overlap: int) -> list[str]:
    if size < 1 or overlap < 0 or overlap > size // 2:
        raise ValueError(f"bad chunk parameters: size={size}, overlap={overlap}")
    text = _normalise(text)
    if not text:
        return []
    chunks: list[str] = []
    cur = ""
    for piece in _split(text, size, _SEPARATORS, size - overlap):
        if len(cur) + len(piece) <= size:
            cur += piece
            continue
        chunks.append(cur)
        cur = _tail(cur, overlap, size - len(piece)) + piece
    chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]
