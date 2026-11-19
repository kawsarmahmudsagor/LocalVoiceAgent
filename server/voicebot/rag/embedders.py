"""Embedding backends (Hugging Face sentence-transformers, Ollama) behind one protocol."""
from __future__ import annotations

import math
from typing import Callable, Literal, Protocol

import httpx

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_HF = "hf:BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA = "ollama:nomic-embed-text"


class EmbedderError(RuntimeError):
    pass


class Embedder(Protocol):
    slug: str
    dim: int | None

    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> list[list[float]]: ...
    def close(self) -> None: ...


def _httpx_post(url: str, json: dict, timeout: float) -> tuple[int, dict]:
    r = httpx.post(url, json=json, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r.status_code, body


def _httpx_get(url: str, timeout: float) -> tuple[int, dict]:
    r = httpx.get(url, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r.status_code, body


class HFEmbedder:
    def __init__(self, model_id: str, device: str = "cuda", *, st_cls=None):
        if st_cls is None:
            from sentence_transformers import SentenceTransformer as st_cls  # lazy
        self.slug = f"hf:{model_id}"
        self._model = st_cls(model_id, device=device)
        self.dim: int | None = self._model.get_sentence_embedding_dimension()

    def embed(self, texts, kind):
        if kind == "query":
            texts = [QUERY_PREFIX + t for t in texts]
        out = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in out]

    def close(self) -> None:
        self._model = None


class OllamaEmbedder:
    def __init__(self, model: str, url: str, *, post: Callable | None = None, timeout: float = 10.0):
        self.slug = f"ollama:{model}"
        self._model = model
        self._url = url.rstrip("/")
        self._post = post or _httpx_post
        self._timeout = timeout
        self.dim: int | None = None

    def embed(self, texts, kind):
        try:
            status, body = self._post(
                f"{self._url}/api/embed", {"model": self._model, "input": list(texts)}, self._timeout
            )
        except httpx.HTTPError as exc:
            raise EmbedderError(f"Ollama not reachable at {self._url}") from exc
        if status != 200:
            raise EmbedderError(body.get("error", f"HTTP {status}"))
        out = []
        for v in body["embeddings"]:
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        if out and self.dim is None:
            self.dim = len(out[0])
        return out

    def close(self) -> None:
        pass


def make_embedder(slug: str, *, device: str, ollama_url: str) -> Embedder:
    if slug.startswith("hf:"):
        return HFEmbedder(slug[3:], device)
    if slug.startswith("ollama:"):
        return OllamaEmbedder(slug[7:], ollama_url)
    raise ValueError(f"unknown embedder slug: {slug!r}")


def list_embedders(ollama_url: str, *, get: Callable | None = None) -> list[dict]:
    get = get or _httpx_get
    url = ollama_url.rstrip("/")
    out = [{"slug": DEFAULT_HF, "available": True, "reason": ""}]
    try:
        status, body = get(f"{url}/api/tags", 3.0)
        if status != 200:
            raise httpx.HTTPError(f"HTTP {status}")
    except httpx.HTTPError:
        out.append({"slug": DEFAULT_OLLAMA, "available": False, "reason": f"Ollama not reachable at {ollama_url}"})
        return out
    names = []
    for m in body.get("models", []):
        name = m.get("name", "")
        if name.endswith(":latest"):
            name = name[: -len(":latest")]
        if "bert" in (m.get("details") or {}).get("family", "") or "embed" in name:
            names.append(name)
    for n in names:
        out.append({"slug": f"ollama:{n}", "available": True, "reason": ""})
    if DEFAULT_OLLAMA[7:] not in names:
        out.append({"slug": DEFAULT_OLLAMA, "available": False,
                    "reason": "not pulled: run scripts/ollama.sh pull nomic-embed-text"})
    return out
