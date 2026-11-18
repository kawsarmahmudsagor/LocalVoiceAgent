"""Builds the backend chosen by LLM_BACKEND. Imports are lazy so the llamacpp
path never loads torch, transformers or bitsandbytes."""
from __future__ import annotations

from voicebot.config import Settings
from voicebot.llm.base import LLMBackend


def make_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "transformers":
        from voicebot.llm.transformers_backend import TransformersBackend
        return TransformersBackend(settings)
    if settings.llm_backend == "llamacpp":
        from voicebot.llm.llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend(settings)
    raise ValueError(f"unknown LLM_BACKEND {settings.llm_backend!r}")
