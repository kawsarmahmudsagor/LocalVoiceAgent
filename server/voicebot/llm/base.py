"""Shared LLM backend types. See spec §4.2."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass(frozen=True)
class GenerationParams:
    temperature: float
    top_p: float
    max_tokens: int


class LLMError(RuntimeError):
    """Any failure of the LLM backend, at startup or during a turn."""


class LLMBackend(Protocol):
    name: str
    model_label: str

    def start(self) -> None:
        """Load the model or start the server; blocks until ready."""

    def stream(self, messages: list[dict], params: GenerationParams,
               stop_event: threading.Event) -> Iterator[str]:
        """Yield non-empty text; stop within one token/chunk once stop_event is set."""

    def close(self) -> None: ...
