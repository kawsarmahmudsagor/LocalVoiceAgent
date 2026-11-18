"""Settings: the one place that reads .env. See spec §4.1."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TypeVar

from dotenv import dotenv_values

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Reply in one to three short spoken sentences. "
    "Never use lists, markdown, code blocks or emoji."
)

T = TypeVar("T")


class ConfigError(ValueError):
    """Invalid or missing setting. The message always names the variable."""


def _bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError("expected true/false/1/0/yes/no")


def _choice(*allowed: str) -> Callable[[str], str]:
    def parse(raw: str) -> str:
        v = raw.strip().lower()
        if v not in allowed:
            raise ValueError(f"expected one of {', '.join(allowed)}")
        return v
    return parse


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    audio_io: str
    hf_model_id: str
    hf_load_in_4bit: bool
    llama_autostart: bool
    llama_server_path: Path
    llama_hf_repo: str
    llama_host: str
    llama_port: int
    llama_base_url: str
    llama_ctx: int
    llama_start_timeout: float
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_silence_sec: float
    tts_voice: str
    system_prompt: str
    llm_temperature: float
    llm_top_p: float
    llm_max_tokens: int
    history_turns: int
    web_host: str
    web_port: int
    llama_runtime: str = "binary"
    llama_image: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    web_ssl_certfile: Path | None = None
    web_ssl_keyfile: Path | None = None

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None, env_file: Path | None = None, *,
             platform: str = sys.platform) -> "Settings":
        if env is None:
            values = {k: v for k, v in dotenv_values(env_file or REPO_ROOT / ".env").items() if v is not None}
            values.update(os.environ)
        else:
            values = dict(env)

        def get(name: str, default: str, parse: Callable[[str], T] = str) -> T:  # type: ignore[assignment]
            raw = values.get(name)
            if raw is None or raw.strip() == "":
                raw = default
            try:
                return parse(raw)
            except ValueError as e:
                raise ConfigError(f"{name}={raw!r} is invalid: {e}") from None

        server_path = Path(get("LLAMA_SERVER_PATH", "tools/llama/llama-server.exe"))
        if not server_path.is_absolute():
            server_path = REPO_ROOT / server_path
        linux = platform.startswith("linux")
        audio_io = get("AUDIO_IO", "browser", _choice("browser", "server"))
        if linux and audio_io == "server":
            raise ConfigError("AUDIO_IO=server: server audio is not supported on the VM; use browser mode")

        def path_or_none(name: str) -> Path | None:
            raw = get(name, "")
            if not raw:
                return None
            p = Path(raw)
            return p if p.is_absolute() else REPO_ROOT / p

        certfile, keyfile = path_or_none("WEB_SSL_CERTFILE"), path_or_none("WEB_SSL_KEYFILE")
        if (certfile is None) != (keyfile is None):
            missing = "WEB_SSL_KEYFILE" if keyfile is None else "WEB_SSL_CERTFILE"
            raise ConfigError(f"{missing} is required when the other WEB_SSL_* variable is set")

        host = get("LLAMA_HOST", "127.0.0.1")
        port = get("LLAMA_PORT", "8081", int)

        return cls(
            llm_backend=get("LLM_BACKEND", "llamacpp", _choice("transformers", "llamacpp")),
            audio_io=audio_io,
            hf_model_id=get("HF_MODEL_ID", "Qwen/Qwen3.5-9B"),
            hf_load_in_4bit=get("HF_LOAD_IN_4BIT", "true", _bool),
            llama_autostart=get("LLAMA_AUTOSTART", "true", _bool),
            llama_server_path=server_path,
            llama_hf_repo=get("LLAMA_HF_REPO", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"),
            llama_host=host,
            llama_port=port,
            llama_base_url=get("LLAMA_BASE_URL", f"http://{host}:{port}").rstrip("/"),
            llama_ctx=get("LLAMA_CTX", "8192", int),
            llama_start_timeout=get("LLAMA_START_TIMEOUT", "900", float),
            stt_model=get("STT_MODEL", "small.en"),
            stt_device=get("STT_DEVICE", "cuda"),
            stt_compute_type=get("STT_COMPUTE_TYPE", "int8"),
            stt_silence_sec=get("STT_SILENCE_SEC", "0.6", float),
            tts_voice=get("TTS_VOICE", "af_heart"),
            system_prompt=get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            llm_temperature=get("LLM_TEMPERATURE", "0.7", float),
            llm_top_p=get("LLM_TOP_P", "0.8", float),
            llm_max_tokens=get("LLM_MAX_TOKENS", "256", int),
            history_turns=get("HISTORY_TURNS", "10", int),
            web_host=get("WEB_HOST", "127.0.0.1"),
            web_port=get("WEB_PORT", "8000", int),
            llama_runtime=get("LLAMA_RUNTIME", "docker" if linux else "binary", _choice("docker", "binary")),
            llama_image=get("LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda"),
            web_ssl_certfile=certfile,
            web_ssl_keyfile=keyfile,
        )
