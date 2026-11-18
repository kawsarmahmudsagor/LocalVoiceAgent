"""Text-only check of the backend selected in .env.

    python -m voicebot.smoke_llm

Prints the model label, the streamed raw reply and timings. Exits 1 if the
reply contains <think> (thinking mode was not disabled), otherwise 0.
"""
from __future__ import annotations

import sys
import threading
import time

from voicebot.config import Settings
from voicebot.llm.base import GenerationParams
from voicebot.llm.factory import make_backend

PROMPT = "In one sentence, what is a voice assistant?"


def main(argv: list[str] | None = None, *, backend_factory=make_backend) -> int:
    settings = Settings.load()
    backend = backend_factory(settings)
    try:
        print(f"backend: {backend.name}  model: {backend.model_label}")
        t0 = time.perf_counter()
        backend.start()
        print(f"ready in {time.perf_counter() - t0:.1f}s")
        messages = [{"role": "system", "content": settings.system_prompt}, {"role": "user", "content": PROMPT}]
        params = GenerationParams(settings.llm_temperature, settings.llm_top_p, settings.llm_max_tokens)
        start, first, parts = time.perf_counter(), None, []
        for text in backend.stream(messages, params, threading.Event()):
            if first is None:
                first = time.perf_counter()
            parts.append(text)
            print(text, end="", flush=True)
        end = time.perf_counter()
        raw = "".join(parts)
        ttft = round((first - start) * 1000) if first is not None else None
        print(f"\nttft_ms={ttft} total_ms={round((end - start) * 1000)}")
        if "<think>" in raw:
            print("FAIL: reply contains <think>; thinking mode is not disabled", file=sys.stderr)
            return 1
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
