import httpx
import pytest

from voicebot.rag.embedders import (
    QUERY_PREFIX,
    EmbedderError,
    HFEmbedder,
    OllamaEmbedder,
    list_embedders,
    make_embedder,
)


class FakeST:
    def __init__(self, model_id, device): self.model_id, self.device, self.seen = model_id, device, []
    def get_sentence_embedding_dimension(self): return 3
    def encode(self, texts, normalize_embeddings): self.seen.append((list(texts), normalize_embeddings)); return [[1.0, 0, 0] for _ in texts]


def test_hf_prefixes_queries_only():
    e = HFEmbedder("BAAI/bge-small-en-v1.5", "cpu", st_cls=FakeST)
    assert (e.slug, e.dim) == ("hf:BAAI/bge-small-en-v1.5", 3)
    e.embed(["doc"], "document"); e.embed(["q"], "query")
    assert e._model.seen == [(["doc"], True), ([QUERY_PREFIX + "q"], True)]


def test_ollama_request_and_normalisation():
    calls = []
    post = lambda url, json, timeout: calls.append((url, json, timeout)) or (200, {"embeddings": [[3.0, 4.0]]})
    e = OllamaEmbedder("nomic-embed-text", "http://h:11434", post=post)
    assert e.embed(["x"], "document") == [[0.6, 0.8]] and e.dim == 2
    assert calls == [("http://h:11434/api/embed", {"model": "nomic-embed-text", "input": ["x"]}, 10.0)]


def test_ollama_errors_raise_embedder_error():
    e = OllamaEmbedder("missing", "http://h", post=lambda *a, **k: (404, {"error": "model 'missing' not found"}))
    with pytest.raises(EmbedderError, match="not found"): e.embed(["x"], "query")
    def down(*a, **k): raise httpx.ConnectError("refused")
    with pytest.raises(EmbedderError, match="not reachable"): OllamaEmbedder("m", "http://h", post=down).embed(["x"], "query")


def test_make_embedder_parses_prefix():
    assert isinstance(make_embedder("ollama:nomic-embed-text", device="cpu", ollama_url="http://h"), OllamaEmbedder)
    with pytest.raises(ValueError): make_embedder("openai:x", device="cpu", ollama_url="http://h")


def test_list_embedders_with_and_without_ollama():
    tags = {"models": [{"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
                       {"name": "qwen3:8b", "details": {"family": "qwen3"}}]}
    up = list_embedders("http://h", get=lambda url, timeout: (200, tags))
    assert [x["slug"] for x in up] == ["hf:BAAI/bge-small-en-v1.5", "ollama:nomic-embed-text"]
    assert all(x["available"] for x in up)
    def down(*a, **k): raise httpx.ConnectError("refused")
    off = list_embedders("http://h", get=down)
    assert off[1] == {"slug": "ollama:nomic-embed-text", "available": False, "reason": "Ollama not reachable at http://h"}
