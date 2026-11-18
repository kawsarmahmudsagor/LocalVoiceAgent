"""Streams chat completions from llama-server's OpenAI-compatible API. See spec §4.2."""
from __future__ import annotations

import threading
import time
from typing import Iterator

import httpx
import openai

from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.llm.llama_server import LlamaServer, http_status, wait_for_health

THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, settings: Settings, *, client=None, server_factory=LlamaServer, status=http_status,
                 sleep=time.sleep, clock=time.monotonic) -> None:
        self.settings = settings
        self.model_label = settings.llama_hf_repo
        self.base_url = settings.llama_base_url
        self._client = client
        self._server_factory = server_factory
        self._status, self._sleep, self._clock = status, sleep, clock
        self._server = None

    def start(self) -> None:
        s = self.settings
        if s.llama_autostart:
            server = self._server_factory(s)
            server.start()
            self._server = server
        else:
            wait_for_health(self.base_url, s.llama_start_timeout, status=self._status,
                            sleep=self._sleep, clock=self._clock)
        if self._client is None:
            self._client = openai.OpenAI(base_url=f"{self.base_url}/v1", api_key="local")

    def stream(self, messages: list[dict], params: GenerationParams,
               stop_event: threading.Event) -> Iterator[str]:
        if self._client is None:
            raise LLMError("llama-server backend used before start()")
        try:
            response = self._client.chat.completions.create(
                model="local", messages=messages, stream=True, temperature=params.temperature,
                top_p=params.top_p, max_tokens=params.max_tokens, extra_body=THINKING_OFF)
            try:
                for chunk in response:
                    if stop_event.is_set():
                        return
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        yield text
                        if stop_event.is_set():
                            return
            finally:
                response.close()
        except (openai.APIError, httpx.HTTPError) as e:
            raise LLMError(f"llama-server at {self.base_url} failed: {e}") from e

    def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
