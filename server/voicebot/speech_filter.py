"""Streaming cleanup of LLM output before it is spoken and shown. See spec §4.3.

- Drops everything inside <think>...</think>, even when tags are split across
  chunks. An unclosed <think> drops the rest of the reply.
- Removes the markdown symbols * # and backticks, and line-start bullets ("- ", "* ").
- Never yields an empty string, and yields as soon as text is safe to emit.
"""
from __future__ import annotations

from typing import Iterable, Iterator

OPEN, CLOSE = "<think>", "</think>"
_MARKDOWN = set("*#`")


def _partial_tag_len(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is a proper prefix of tag."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0


class _Cleaner:
    """Stateful markdown remover; state carries across chunks."""

    def __init__(self) -> None:
        self.at_line_start = True
        self.held = ""  # a "-" or "*" at line start, waiting to see if a space follows

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self.held:
                held, self.held = self.held, ""
                if ch == " ":  # it was a bullet: drop marker and space
                    self.at_line_start = False
                    continue
                if held == "-":
                    out.append("-")
                    self.at_line_start = False
            if self.at_line_start and ch in "-*":
                self.held = ch
                continue
            if ch in _MARKDOWN:
                continue
            out.append(ch)
            self.at_line_start = ch == "\n"
        return "".join(out)

    def finish(self) -> str:
        held, self.held = self.held, ""
        return "-" if held == "-" else ""


def filter_stream(chunks: Iterable[str]) -> Iterator[str]:
    cleaner = _Cleaner()
    pending = ""
    in_think = False
    for chunk in chunks:
        if not chunk:
            continue
        pending += chunk
        while True:
            tag = CLOSE if in_think else OPEN
            idx = pending.find(tag)
            if idx >= 0:
                before, pending = pending[:idx], pending[idx + len(tag):]
                if not in_think:
                    out = cleaner.feed(before)
                    if out:
                        yield out
                in_think = not in_think
                continue
            keep = _partial_tag_len(pending, tag)
            ready, pending = pending[: len(pending) - keep], pending[len(pending) - keep:]
            if not in_think:
                out = cleaner.feed(ready)
                if out:
                    yield out
            break
    if not in_think:
        out = cleaner.feed(pending) + cleaner.finish()
        if out:
            yield out
